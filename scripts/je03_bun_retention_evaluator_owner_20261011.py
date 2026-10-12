"""Retention-v2 CPU evaluator owner; isolated from the sealed GPU trainer."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', JAXBOMB_RULE='bun',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import je03_bun831_20261011 as candidate
from scripts import je03_bun831_evaluator_20261011 as authority
from scripts.population_run_state import atomic_json, sha, lock
from scripts.je03_bun_stage1_20261011 import read, verify_checkpoint

RUN = candidate.RUN
OUT = RUN / 'authority_bun831_v1'


def owner():
    candidate.verify_seal()
    OUT.mkdir(parents=True, exist_ok=True)
    authority.OUT = OUT
    authority.RUN = RUN
    authority.POINTS = candidate.POINTS
    # je03_bun831_evaluator_20261011 delegates to the lower-level authority
    # module, whose globals must be rebound as well for namespace isolation.
    authority.authority.OUT = OUT
    authority.authority.RUN = RUN
    authority.setup = getattr(authority, 'setup', authority.setup)
    # Reuse the authority's protocol/spawn setup, but keep retention labels.
    if not (OUT / 'PROTOCOL.json').exists():
        atomic_json(OUT / 'PROTOCOL.json', dict(
            namespace=OUT.name, paired=True, hunter_games_per_tier_per_arm=64,
            tactical_games_per_arm=64, hunter_primary=300, hunter_residual=64,
            tactical_primary=300, checkpoints=candidate.POINTS,
            native_bun_tasks=authority.TASKS, native_task_games=16,
            native_task_ticks=300, cuda_visible_devices='', stage2_enabled=False,
            sources={f: sha(Path(candidate.ROOT) / f) for f in authority.authority.FILES}))
    if not (OUT / 'spawns.json').exists():
        atomic_json(OUT / 'spawns.json', read(authority.authority.SPAWNS))
    authority.OUT = OUT
    own = lock(OUT / 'owner_retention_v2.lock')
    atomic_json(OUT / 'OWNER_RETENTION_V2.json',
                dict(pid=os.getpid(), cuda_visible_devices='', unix=time.time()))
    labels = ['bun392_retention_control', 'bun392_retention_teacher']
    while True:
        jobs = [('original_bun_it4000', candidate.BUN_SOURCE,
                 dict(reference_sha256=sha(candidate.BUN_SOURCE)))]
        for g in candidate.POINTS:
            for arm in labels:
                cp = RUN / 'formal' / arm / f'gen{g:06d}'
                if (cp / 'COMMITTED.json').exists():
                    m = verify_checkpoint(cp, sha(RUN / 'plan.json'))
                    jobs.append((f'{arm}_g{g:06d}', cp / 'policy.pt',
                                 dict(generation=g,
                                      committed_sha256=sha(cp / 'COMMITTED.json'))))
        complete, errors = [], []
        for label, policy, proof in jobs:
            try:
                authority.evaluate(label, policy, proof)
                complete.append(label)
            except Exception as exc:
                atomic_json(OUT / f'TECHNICAL_STOP_{label}_{time.time_ns()}.json',
                            dict(error=repr(exc), unix=time.time()))
                errors.append(label)
        atomic_json(OUT / 'STATE.json', dict(pid=os.getpid(), completed=complete,
                    errors=errors, phase='CPU_ASYNC_RETENTION_V2', unix=time.time()))
        if all((RUN / f'TRAINING_COMPLETE_{a}.json').exists() for a in labels) and \
                len(complete) == 1 + len(labels) * len(candidate.POINTS):
            atomic_json(OUT / 'EVALUATION_COMPLETE.json',
                        dict(passed=True, completed=complete, unix=time.time()))
            atomic_json(RUN / 'AB_COMPLETE.json', dict(
                passed=True, technical_only=True, stage1_passed=False,
                stage2_enabled=False, unix=time.time()))
            return
        time.sleep(5)


if __name__ == '__main__':
    owner()
