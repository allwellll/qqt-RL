"""Restart gate consumed by the existing sole training owner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = BASE / 'reboot_resume_20261030'


def read(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def dispatch(slot, entry, children, live, free, affinity):
    plan = read(OUT / 'PLAN.json', {})
    spec = plan.get('slots', {}).get(str(slot))
    if not spec or spec['run'] != entry.get('run'):
        return False
    dest = OUT / f'slot{slot:02d}'
    dest.mkdir(exist_ok=True)
    status = read(dest / 'STATUS.json', {})
    if status.get('status') == 'READY_FOR_FORMAL':
        return False
    if status.get('status') == 'BLOCKED':
        return True
    key = f'prewarm_reboot_slot{slot:02d}'
    if key in children or entry['run'] in live:
        return True
    # Successful child leaves an independently verifiable phase receipt.
    stage = status.get('stage')
    if stage:
        proof = dest / ('BOOTSTRAP.json' if stage == 'bootstrap' else 'SMOKE.json' if stage == 'smoke' else 'RESTORE.json')
        value = read(proof, {})
        passed = value.get('passed', value.get('status') == 'PASS')
        if not passed:
            atomic_json(dest / 'STATUS.json', dict(status='BLOCKED', stage=stage,
                reason='recovery process exited without a passing phase receipt; inspect stage log', unix=time.time()))
            return True
        if stage == 'bootstrap':
            stage = 'restore'
        elif stage == 'restore':
            atomic_json(dest / 'INITIAL_RESTORE.json', value)
            if spec['already_complete']:
                atomic_json(dest / 'STATUS.json', dict(status='READY_FOR_FORMAL',
                    note='completed chunk preserved; sole owner must continue next same-recipe chunk', unix=time.time()))
                return False
            stage = 'smoke'
        elif stage == 'smoke':
            stage = 'postrestore'
        else:
            atomic_json(dest / 'STATUS.json', dict(status='READY_FOR_FORMAL',
                exact_resume_update=spec['update'],smoke=read(dest / 'SMOKE.json'),
                fresh_restore=read(dest / 'RESTORE.json'),unix=time.time()))
            return False
    else:
        stage = 'restore' if spec['probe_present'] else 'bootstrap'
    if slot not in free:
        return True
    log = (dest / f'{stage}.log').open('ab', buffering=0)
    protocol = read(ROOT / spec['protocol'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot), JAXBOMB_DEVICE_OVERRIDE=str(slot),
        JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_COMPILATION_CACHE_DIR=str(BASE / 'iter2/r1_attack_20261030/items_v2/cache'),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    args = [spec['protocol'], entry['run'], str(spec['seed']), stage]
    proc = subprocess.Popen(['taskset', '-c', affinity, sys.executable, '-m',
        'scripts.bomb_threat_reboot_worker'] + args, cwd=ROOT, env=env,
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    children[key] = proc, log
    atomic_json(dest / 'STATUS.json', dict(status='RUNNING', stage=stage, run=entry['run'],
        original_checkpoint=spec['checkpoint'], original_protocol_sha256=sha(ROOT / spec['protocol']),
        gpu=slot, unix=time.time()))
    return True
