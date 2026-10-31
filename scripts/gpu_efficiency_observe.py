"""Read-only, PID-redacted sampling of the frozen D worker and evaluators."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'runs/expert_reward_20261025/D_ablation_v1'


def role_for_command(command):
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or not Path(words[0]).name.startswith('python') or '-m' not in words:
        return None
    module_index = words.index('-m') + 1
    if module_index >= len(words):
        return None
    module = words[module_index]
    args = words[module_index + 1:]
    if module == 'scripts.expert_reward_ablation_resume' and args[:1] == ['worker']:
        return 'D_worker'
    if module == 'scripts.expert_reward_ablation_resume' and args[:1] == ['owner']:
        return 'D_owner'
    if module == 'scripts.expert_reward_ablation_cpu':
        return 'CPU_JS_evaluator'
    if module == 'scripts.gpu_efficiency_observe':
        return 'passive_observer'
    if module == 'scripts.gpu_efficiency_performance_probe':
        return 'isolated_performance_probe'
    return None


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def process_snapshot():
    """Identifiers stay in memory; saved rows contain stable role labels only."""
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            cmd = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
            io = dict(line.split(':', 1) for line in (path/'io').read_text().splitlines())
            processes[int(path.name)] = dict(parent=int(stat[1]), role=role_for_command(cmd),
                cpu_ticks=int(stat[11])+int(stat[12]), rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),
                read_bytes=int(io['read_bytes']), write_bytes=int(io['write_bytes']))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    for _ in range(5):
        for process in processes.values():
            if process['role'] is None:
                parent = processes.get(process['parent'], {})
                if parent.get('role') == 'CPU_JS_evaluator':
                    process['role'] = 'CPU_JS_evaluator_child'
    return processes


def summarize_processes(current, previous, elapsed):
    totals = {}
    hz = os.sysconf('SC_CLK_TCK')
    for identity, process in current.items():
        role = process['role']
        if role is None:
            continue
        result = totals.setdefault(role, dict(count=0, cpu_percent_one_core=0., rss_bytes=0,
                                              interval_read_bytes=0, interval_write_bytes=0))
        result['count'] += 1
        result['rss_bytes'] += process['rss_bytes']
        old = previous.get(identity)
        if old is not None and old['role'] == role and elapsed > 0:
            result['cpu_percent_one_core'] += max(0,process['cpu_ticks']-old['cpu_ticks'])/hz/elapsed*100
            for field in ('read_bytes', 'write_bytes'):
                result['interval_'+field] += max(0,process[field]-old[field])
    return totals


def gpu_snapshot(processes):
    query = ['nvidia-smi', '--format=csv,noheader,nounits']
    raw = subprocess.check_output(query+['--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw'], text=True)
    gpu_rows, identifiers = [], {}
    for line in raw.splitlines():
        index, identity, util, memory, power = [v.strip() for v in line.split(',')]
        identifiers[identity] = int(index)
        gpu_rows.append(dict(index=int(index), utilization_percent=float(util), memory_used_mib=float(memory),
                             power_watts=None if power in ('[N/A]','N/A') else float(power), compute_roles=[]))
    apps = subprocess.check_output(query+['--query-compute-apps=pid,gpu_uuid'], text=True)
    for line in apps.splitlines():
        identity, device = [v.strip() for v in line.split(',')]
        index = identifiers.get(device)
        if index is not None:
            role = processes.get(int(identity), {}).get('role') or 'unrelated_compute'
            next(g for g in gpu_rows if g['index'] == index)['compute_roles'].append(role)
    return gpu_rows


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=int, default=360)
    parser.add_argument('--interval', type=float, default=1.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # The observer cannot launch, signal, or mutate any trainer.
    import fcntl
    owner_lock = (args.output/'observer.lock').open('a')
    fcntl.flock(owner_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.output/'SAMPLES.jsonl').exists():
        raise RuntimeError('Refusing to overwrite an existing observation')
    task = ROOT/'.hermes/gpu_efficiency_followup.md'
    atomic_json(args.output/'TASK_CONSUMPTION.json',dict(task='.hermes/gpu_efficiency_followup.md',
        task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),consumed=True,controls_gpu=False,
        changes_training=False,stage='OBSERVING',duration_seconds=args.seconds,interval_seconds=args.interval))
    started = time.monotonic()
    previous, last = {}, started
    sample_count = 0
    try:
        with (args.output/'SAMPLES.jsonl').open('x') as stream:
            while time.monotonic()-started < args.seconds:
                begin = time.monotonic()
                processes = process_snapshot()
                status = read_json(D/'STATUS.json')
                saved_status = {k:status[k] for k in ('stage','seed','new_update','environment_steps','finite',
                    'steps_per_second','last_committed_new_update','unix') if k in status}
                row = dict(unix=time.time(),elapsed_seconds=begin-started,
                    process_roles=summarize_processes(processes,previous,begin-last),
                    gpus=gpu_snapshot(processes),training=saved_status)
                previous, last = processes, begin
                row['sampling_seconds'] = time.monotonic()-begin
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                sample_count += 1
                if sample_count % 10 == 0:
                    atomic_json(args.output/'STATUS.json',dict(stage='OBSERVING',samples=sample_count,
                        elapsed_seconds=time.monotonic()-started,controls_gpu=False))
                time.sleep(max(0,args.interval-(time.monotonic()-begin)))
        atomic_json(args.output/'STATUS.json',dict(stage='OBSERVATION_COMPLETE',samples=sample_count,
            elapsed_seconds=time.monotonic()-started,controls_gpu=False,changes_training=False))
    except BaseException as exc:
        atomic_json(args.output/'STATUS.json',dict(stage='FAILED',samples=sample_count,
            failure_type=type(exc).__name__,controls_gpu=False))
        raise


if __name__ == '__main__':
    main()
