"""Existing CPU protocols only; observation has no training control channel."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.update(JE09_NAMESPACE='runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15',
    CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', JAXBOMB_RULE='bun',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
    NODE_OPTIONS='--require ./scripts/je09_stage_h_metrics.js --require ./scripts/je09_stage_h_eval_cache.js')
from scripts.je09_stage_h2 import BASE, ARM, POINTS, configure
original_popen = subprocess.Popen


def patched_popen(args, *rest, **kwargs):
    names = {'je05_web_hunter.js': 'je09_stage_h_hunter.js', 'je08_web_diagnostics.js': 'je09_stage_h_diagnostics.js'}
    if isinstance(args, (list, tuple)) and len(args)>1 and Path(args[0]).name=='node' and Path(args[1]).name in names:
        args = [args[0], 'scripts/'+names[Path(args[1]).name], *args[2:]]
    return original_popen(args, *rest, **kwargs)


subprocess.Popen = patched_popen


def point(g):
    from scripts.je08_bun_authority import export
    out = BASE/'authority_je09_v1'/f'{ARM}_g{g:06d}'; out.mkdir(parents=True, exist_ok=True)
    model = export(BASE/'formal'/ARM/f'gen{g:06d}'/'policy.pt', out/'model')
    for script, modes in [('scripts/je09_stage_e_controlled.js', [('evaluation','STAGE_B.json'),('holdout','STAGE_B_HOLDOUT.json')]),
                          ('scripts/je09_stage_f2_stage_a.js', [('evaluation','STAGE_A.json'),('holdout','STAGE_A_HOLDOUT.json')])]:
        for mode, name in modes:
            if not (out/name).exists(): subprocess.run(['node',script,str(model),str(out/name),mode],check=True)
    from scripts import je09_stage_e as ev
    ev.POINTS = POINTS; ev.evaluate(g)


def queue():
    from scripts.population_run_state import atomic_json
    pending = POINTS.copy(); jobs = {}; done = []; errors = []
    def job(g):
        with (BASE/f'evaluate_g{g:06d}.log').open('a') as log:
            subprocess.run([sys.executable, 'scripts/je09_stage_h2_evaluate.py', str(g)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        return g
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while pending or jobs:
            for g in pending[:]:
                if len(jobs)>=2: break
                if (BASE/'formal'/ARM/f'gen{g:06d}/COMMITTED.json').exists():
                    jobs[pool.submit(job,g)] = g; pending.remove(g)
            for future in list(jobs):
                if future.done():
                    g = jobs.pop(future)
                    try: done.append(future.result())
                    except Exception: errors.append(g)
            atomic_json(BASE/'authority_je09_v1/QUEUE.json', dict(completed=done, pending=pending,
                running=list(jobs.values()), failed=errors, observation_only=True, controls_gpu=False))
            end = BASE/'formal'/ARM/'TRAINER_EXIT.json'
            if end.exists(): pending = [g for g in pending if g<=json.loads(end.read_text())['generation']]
            time.sleep(3)
    atomic_json(BASE/'authority_je09_v1/EVALUATION_COMPLETE.json', dict(completed=not errors,
        completed_generations=sorted(done), failed=errors, assist_off=True, observation_only=True))


if __name__ == '__main__':
    configure()
    if len(sys.argv)>1: point(int(sys.argv[1]))
    else: queue()
