"""CPU-only takeover queue. Training files and their SHA-pinned protocols stay intact.

Newest missing 80-update point first; scores never imply automatic promotion.
One lock per arm, shared cc_strong_eval CPU slots, restart-safe completed shards.
"""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.cc_fixed_eval_v1 import CPU_ENV
from scripts.model_launch import command, manifest
from scripts.population_run_state import atomic_json

ROOT = Path('runs/cc_train_20261105')
FORMAL = ROOT / 'training/formal'
OUT = ROOT / 'cx_takeover_20261106'
POINTS = ROOT / 'eval_strong_v1/points'


def pending(arm):
    return sorted((int(p.parent.name[1:]), p.parent)
                  for p in (FORMAL / arm).glob('u*/COMMITTED.json')
                  if (int(p.parent.name[1:]) - 18262) % 80 == 0
                  and not (POINTS / f'{arm}_{p.parent.name}__seats01.json').exists())


def main():
    arm, = sys.argv[1:]
    assert arm in ('L2_FTN', 'L2_ATK', 'L2_S05h', 'L2_SP')
    OUT.mkdir(parents=True, exist_ok=True)
    guard = (OUT / f'{arm}.eval.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    failed = {}
    while not (OUT / f'{arm}.EVAL_STOP').exists():
        todo = [(u, cp) for u, cp in pending(arm) if time.time() >= failed.get(u, 0)]
        state = dict(owner='CX', arm=arm, cpu_only=True, unix=time.time(),
                     pending=[u for u, _ in pending(arm)],
                     selection='same-checkpoint new-seed retest with same-batch G_frozen; no automatic promotion')
        if not todo:
            atomic_json(OUT / f'{arm}.EVAL.json', dict(state, status='WAITING_FOR_COMMITTED_POINT'))
            time.sleep(30)
            continue
        u, cp = todo[-1]
        cfg = manifest(dict(module='scripts.cc_strong_eval_v1',
                            args=['point', str(cp), '--seats', '0,1', '--opponents', 'G_frozen'], env=CPU_ENV))
        with (OUT / f'{arm}.evaluation.log').open('a') as log:
            process = subprocess.Popen(command('evaluation', cfg), stdout=log, stderr=subprocess.STDOUT)
            atomic_json(OUT / f'{arm}.EVAL.json', dict(state, status='RUNNING', update=u,
                        child_pid=process.pid, config=cfg, checkpoint=str(cp)))
            code = process.wait()
        exists = (POINTS / f'{arm}_u{u:06d}__seats01.json').exists()
        row = dict(state, status='COMPLETE' if code == 0 and exists else 'RETRY_COOLDOWN',
                   update=u, code=code, point_exists=exists, finished_unix=time.time())
        atomic_json(OUT / f'{arm}.EVAL.json', row)
        with (OUT / 'evaluation_events.jsonl').open('a') as log:
            log.write(json.dumps(row) + '\n')
        if code or not exists:
            failed[u] = time.time() + 300


if __name__ == '__main__':
    main()
