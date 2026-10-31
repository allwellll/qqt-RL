"""Inactive three-run, matched-control goal-kind qualification experiment."""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from scripts.population_run_state import atomic_json, sha, lock

R = Path(__file__).resolve().parents[1]
NAMESPACE = os.environ.get('DAGGER_RL_R17_NAMESPACE', 'r17_goal_kind')
B = R/'runs/dagger_rl_20261025'/NAMESPACE
PRIOR = R/'runs/dagger_rl_20261025/r16_retention_schedule'
read = lambda p: json.loads(Path(p).read_text())


class UserStopped(Exception):
    pass


def validate_protocol(p):
    assert p['arms'] == ['sequential_anchor']
    assert p['paired_seeds'] == [2026102661, 2026102662, 2026102663]
    assert p['updates'] == 32 and p['switch_update'] == 16
    assert p['total_environment_steps'] == 1572864 and p['total_optimizer_steps'] == 3072
    assert p['goal_kind_guard'] is True and not p['whole_legacy_active_guard']
    assert p['parent_input_sha256'] == read(PRIOR/'PROTOCOL.json')['parent_input_sha256']
    assert p.get('namespace', NAMESPACE) == NAMESPACE


def authorize():
    if (B/'USER_STOP.json').exists() or (PRIOR/'USER_STOP.json').exists():
        raise UserStopped('explicit user atomic stop')
    p = read(B/'PROTOCOL.json'); validate_protocol(p)
    a = read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert a['authorized_by'] == 'explicit_user_instruction' and not a['old_continuous_resume']
    assert a['protocol_sha256'] == sha(B/'PROTOCOL.json')
    assert a['old_global_pause_sha256'] == sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    assert a.get('namespace', NAMESPACE) == NAMESPACE
    assert sha(B/'inputs/parent.pkl') == p['parent_input_sha256']
    for name in ['GPU_COMPLETE.json', 'COMPLETE_EVIDENCE_CONSUMED.json',
                 'SUPPLEMENTAL_EVIDENCE_CONSUMED.json']:
        assert read(PRIOR/name)['passed']
    assert read(PRIOR/'FIXED_DEV_DIRECTION_AUDIT_v2.json')['completed'] == 19
    for name in ['TRAIN_PROJECTION_AUDIT.json', 'GOAL0_REPLAY_WITNESS_v2.json',
                 'REAL_GOAL_GUARD_CHECK.json', 'GOAL_KIND_GUARD_PREPARATION.json']:
        assert read(PRIOR/name)['passed']
    assert read(PRIOR/'REAL_GOAL_GUARD_CHECK.json')['kind_only']
    assert sha(R/'scripts/dagger_rl_route_goal_eligibility.py') == p['goal_guard_source_sha256']
    selection = read(B/'SCIENTIFIC_SELECTION.json')
    assert selection['execute_goal_kind'] and selection['protocol_sha256'] == sha(B/'PROTOCOL.json')
    assert selection['R16_comparison_sha256'] == sha(PRIOR/'FINAL_COMPARISON.json')
    assert selection['R16_supplement_sha256'] == sha(PRIOR/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json')
    assert selection['matched_controls_sha256'] == sha(B/'MATCHED_CONTROLS.json')


def notify(message):
    print(message, flush=True)
    atomic_json(B/'notifications'/f'{time.time_ns()}.json', dict(message=message, unix=time.time()))
    try:
        subprocess.run(['jdme-notify', 'send', message], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def owner():
    unique = lock(B/'owner.lock'); authorize()
    p = read(B/'PROTOCOL.json')
    for control in read(B/'MATCHED_CONTROLS.json')['controls']:
        cp = R/control['checkpoint']
        assert sha(cp/'state.pkl') == control['state_sha256']
        assert sha(cp/'policy.pt') == control['policy_sha256']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', JAX_COMPILATION_CACHE_DIR=str(B/'cache'))

    def child(args, stage):
        authorize()
        for name, expected in read(B/'SOURCE_SEAL.json')['sources'].items():
            assert sha(R/name) == expected, 'sealed source changed: '+name
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True).strip():
            authorize(); atomic_json(B/'RESOURCE_WAIT.json', dict(stage=stage, no_preempt=True)); time.sleep(5)
        atomic_json(B/'OWNER_STAGE.json', dict(stage=stage, unix=time.time()))
        with (B/f'{stage.split("/")[0]}.log').open('a') as f:
            worker = subprocess.Popen([sys.executable, '-m']+args, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT)
            while worker.poll() is None:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True)
                with (B/'GPU_ACTIVITY.jsonl').open('a') as g:
                    g.write(json.dumps(dict(unix=time.time(), stage=stage,
                        devices=[[int(v.strip()) for v in row.split(',')] for row in raw.strip().splitlines()]))+'\n')
                time.sleep(5)
            if (B/'USER_STOP.json').exists() or (PRIOR/'USER_STOP.json').exists():
                raise UserStopped('saved atomic boundary')
            assert worker.returncode == 0, f'{stage} exit={worker.returncode}'
    try:
        if not (B/'TECHNICAL_SMOKE.json').exists():
            child(['scripts.dagger_rl_r17_train', 'smoke'], 'smoke')
        assert read(B/'TECHNICAL_SMOKE.json')['passed']
        records = []
        for seed in p['paired_seeds']:
            cp = B/'formal'/f'sequential_anchor_s{seed}'
            if not (cp/'COMPLETE.json').exists():
                child(['scripts.dagger_rl_r17_train', 'formal', 'sequential_anchor', str(seed)],
                      f'formal/sequential_anchor_s{seed}')
            x = read(cp/'COMPLETE.json'); assert x['update'] == 32
            assert x['state_sha256'] == sha(cp/'u032/state.pkl'); records.append(x)
        atomic_json(B/'TRAINING_COMPLETE.json', dict(passed=True, runs=records,
            environment_steps=1572864, optimizer_steps=3072, reused_controls=3, old_continuous_resume=False))
        if not (B/'NATIVE_BANK_COMPLETE.json').exists():
            child(['scripts.dagger_rl_r17_bank'], 'bank')
        assert read(B/'NATIVE_BANK_COMPLETE.json')['passed']
        atomic_json(B/'GPU_COMPLETE.json', dict(passed=True, old_continuous_resume=False))
        notify('R17三配对新处理run及固定Bank GPU阶段完成；匹配R16顺序对照复用，CPU独立，旧队列不恢复。')
    except UserStopped as e:
        atomic_json(B/'PAUSED_OWNER.json', dict(safe_atomic=True, reason=str(e), old_continuous_resume=False))
        notify('R17已在原子边界暂停，禁止下一更新/run/smoke，完整恢复状态保留。')
    except Exception as e:
        atomic_json(B/'TECHNICAL_FAILURE.json', dict(reason=str(e).replace(str(R), '<WORKTREE>'), unix=time.time()))
        notify('R17技术故障：保留原子证据并等待修复，不恢复旧队列。'); raise
    finally:
        atomic_json(B/'OWNER_EXIT.json', dict(unix=time.time(), old_continuous_resume=False))


def complete():
    unique = lock(B/'completion.lock')
    while not all((B/n).exists() for n in ['GPU_COMPLETE.json', 'CPU_EVALUATION_COMPLETE.json', 'FIXED_FIT_COMPLETE.json']):
        if any((B/n).exists() for n in ['USER_STOP.json', 'TECHNICAL_FAILURE.json', 'CPU_TECHNICAL_FAILURE.json']):
            return
        time.sleep(5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (B/'analysis.log').open('a') as f:
        subprocess.run([sys.executable, '-m', 'scripts.dagger_rl_r17_report'], cwd=R, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    atomic_json(B/'COMPLETE_EVIDENCE_CONSUMED.json', dict(passed=True,
        comparison_sha256=sha(B/'FINAL_COMPARISON.json'), new_formal_runs=3, reused_matched_runs=3))
    notify('R17完整配对/父点/保留/真实道具链结果已消费；无自动晋升，依据固定规则决定下一项有界问题。')


if __name__ == '__main__':
    complete() if len(sys.argv) > 1 and sys.argv[1] == 'complete' else owner()
