"""Attach a durable terminal owner to an already-running H8 PBRS worker."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMESPACE = 'runs/je09_bun_20261017/stage_h8_route_pbrs_v35'
BASE = ROOT / NAMESPACE
ARM = 'bun392_bridge_pfsp8'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from scripts.population_run_state import atomic_json
    atomic_json(path, value)


def main():
    launch = read(BASE / 'TRAIN_LAUNCH.json')
    worker_pid = int(launch['pid'])
    write(BASE / 'SUPERVISOR_ATTACH.json', dict(
        attached=True, worker_pid=worker_pid, devices=launch['devices'],
        fixed_generations=launch['fixed_generations'], unix=time.time()))
    exit_path = BASE / 'formal' / ARM / 'TRAINER_EXIT.json'
    worker_proc = Path('/proc') / str(worker_pid)
    while not exit_path.exists():
        if (BASE / 'TECHNICAL_FAILURE.json').exists():
            return 2
        if not worker_proc.exists():
            write(BASE / 'TECHNICAL_FAILURE.json', dict(
                error='formal worker exited before TRAINER_EXIT', worker_pid=worker_pid,
                unix=time.time()))
            return 2
        time.sleep(10)

    result = read(exit_path)
    if result.get('generation') != 32 or result.get('fresh_ticks') != 67108864:
        write(BASE / 'TECHNICAL_FAILURE.json', dict(
            error='formal worker ended outside the fixed budget', worker_pid=worker_pid,
            trainer_exit=result, unix=time.time()))
        return 2

    checkpoint = f'formal/{ARM}/gen000032'
    env = dict(os.environ, JE09_NAMESPACE=NAMESPACE, CUDA_VISIBLE_DEVICES='',
               JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
    with (BASE / 'supervised_final_restore.log').open('a') as log:
        subprocess.run([
            sys.executable, 'scripts/je09_stage_h8_pbrs.py', 'restore',
            '--run', NAMESPACE, '--resume', f'{NAMESPACE}/{checkpoint}'],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(BASE / 'CAMPAIGN_COMPLETE.json', dict(
        completed=True, generation=32, fresh_ticks=67108864,
        cpu_restore=True, terminal_owner='attached supervisor', unix=time.time()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
