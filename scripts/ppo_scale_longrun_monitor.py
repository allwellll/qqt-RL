"""Read-only supervision and durable milestone receipts; never starts training."""
import json
import subprocess
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts.ppo_scale_longrun_run import B, notify, read


def counts():
    lines = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
    return {name: sum(line.endswith(f'-m scripts.{name}') or
                     line.endswith(f'-m scripts.{name} formal') for line in lines)
            for name in ['ppo_scale_longrun_run', 'ppo_scale_longrun_train',
                         'ppo_scale_longrun_cpu']}


def snapshot():
    status = read(B / 'STATUS.json') if (B / 'STATUS.json').exists() else {}
    completed = []
    for p in sorted((B / 'formal').glob('*_s*/COMPLETE.json')):
        d = read(p)
        final = p.parent / f'u{d["update"]:03d}'
        assert sha(final / 'state.pkl') == d['state_sha256']
        assert sha(final / 'policy.pt') == d['policy_sha256']
        completed.append(dict(run=p.parent.name, **d))
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True)
    devices = [[int(x.strip()) for x in line.split(',')] for line in raw.strip().splitlines()]
    return dict(unix=time.time(), processes=counts(),
                last_atomic={k: status.get(k) for k in ['arm', 'seed', 'update',
                    'finite', 'environment_steps', 'steps_per_second']},
                committed_status_age_seconds=time.time()-(B/'STATUS.json').stat().st_mtime,
                complete_runs=completed, gpu_devices=devices,
                training_complete=(B/'TRAINING_COMPLETE.json').exists(),
                cpu_complete=(B/'CPU_EVALUATION_COMPLETE.json').exists(),
                technical_failure=(B/'TECHNICAL_FAILURE.json').exists(),
                cpu_technical_failure=(B/'CPU_TECHNICAL_FAILURE.json').exists(),
                old_queues_resume=False, launches_training=False)


def owner():
    unique = lock(B / 'read_only_monitor.lock')
    seen = set()
    report_signature = None
    while not (B / 'MONITOR_STOP.json').exists():
        s = snapshot()
        atomic_json(B / 'LIVE_SUPERVISION.json', s)
        markers = sorted((B / 'formal').glob('*/u*/retention/COMPLETE.json'))
        signature = (tuple((str(p), p.stat().st_mtime_ns) for p in markers),
                     s['training_complete'], s['cpu_complete'])
        if signature != report_signature:
            from scripts.ppo_scale_longrun_report import summarize
            summarize()
            report_signature = signature
        for run in s['complete_runs']:
            marker = B / 'milestones' / (run['run'] + '.json')
            if not marker.exists():
                atomic_json(marker, dict(unix=time.time(), **run,
                    atomic_hash_verified=True, queued_successor_controls_gpu_owner=True))
        messages = []
        if s['processes']['ppo_scale_longrun_run'] != 1 and not s['training_complete']:
            messages.append('GPU owner不再唯一/缺失；最近原子点保留，请处理真实运行状态。')
        if s['processes']['ppo_scale_longrun_cpu'] != 1 and not s['cpu_complete']:
            messages.append('独立CPU evaluator缺失/重复，GPU不受此状态控制。')
        if s['technical_failure']:
            messages.append('正式GPU技术故障已记录，需从最近原子点诊断恢复。')
        if s['cpu_technical_failure']:
            messages.append('固定CPU评估技术故障已记录，需补齐，不能暂停健康GPU。')
        for message in messages:
            if message not in seen:
                notify('PPO scale监督告警：' + message)
                seen.add(message)
        if s['training_complete'] and s['cpu_complete']:
            atomic_json(B / 'EXPERIMENT_READY_FOR_FINAL_ANALYSIS.json', dict(
                passed=True, unix=time.time(), completed_runs=len(s['complete_runs']),
                milestone_hashes_verified=True, old_queues_resume=False,
                further_budget_requires_separate_frozen_question=True))
            notify('PPO scale R1 六run及独立CPU/JS完整证据终态已就绪；需综合尾部/最差点选择下个有界PPO问题，旧队列未恢复。')
            return
        time.sleep(30)


if __name__ == '__main__':
    owner()
