"""Existing assist-off protocols only; CPU queue cannot hold a GPU worker."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.update(JE09_NAMESPACE='runs/je09_bun_20261016/stage_h_carry_item_events_pfsp8_v14',
                  CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  JAXBOMB_RULE='bun', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  NODE_OPTIONS='--require ./scripts/je09_stage_h_metrics.js')
from scripts.je09_stage_h import BASE, ARM, POINTS, configure


def wait_checkpoint(g):
    cp = BASE/'formal'/ARM/f'gen{g:06d}'
    while not (cp/'COMMITTED.json').exists():
        end = BASE/'formal'/ARM/'TRAINER_EXIT.json'
        if end.exists() and json.loads(end.read_text())['generation'] < g: return None
        time.sleep(3)
    return cp


def monitor(g):
    cp = wait_checkpoint(g)
    if cp is None: return None
    from scripts.je08_bun_authority import export
    out = BASE/'async_monitor_v1'/f'g{g:06d}'; out.mkdir(parents=True, exist_ok=True)
    model = export(cp/'policy.pt', out/'model')
    with (out/'evaluate.log').open('a') as log:
        subprocess.run(['node', 'scripts/je09_web_basic_retention.js', str(model),
                        str(out/'BASIC_RETENTION.json'), str(BASE/'regions.json')],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    return g


def job(g):
    cp = wait_checkpoint(g)
    if cp is None: return None
    from scripts.je08_bun_authority import export
    out = BASE/'authority_je09_v1'/f'{ARM}_g{g:06d}'; out.mkdir(parents=True, exist_ok=True)
    model = export(cp/'policy.pt', out/'model')
    with (BASE/f'evaluate_g{g:06d}.log').open('a') as log:
        for mode, name in [('evaluation', 'STAGE_B.json'), ('holdout', 'STAGE_B_HOLDOUT.json')]:
            if not (out/name).exists():
                subprocess.run(['node', 'scripts/je09_stage_e_controlled.js', str(model), str(out/name), mode],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
        for mode, name in [('evaluation', 'STAGE_A.json'), ('holdout', 'STAGE_A_HOLDOUT.json')]:
            if not (out/name).exists():
                subprocess.run(['node', 'scripts/je09_stage_f2_stage_a.js', str(model), str(out/name), mode],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, 'scripts/je09_stage_h_evaluate.py', 'point', str(g)],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    return g


if __name__ == '__main__':
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'point':
        from scripts import je09_stage_e as evaluation
        evaluation.POINTS = POINTS; evaluation.evaluate(int(sys.argv[2]))
    else:
        from scripts.population_run_state import atomic_json
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as light, concurrent.futures.ThreadPoolExecutor(max_workers=3) as full:
            small = [light.submit(monitor, g) for g in [0, 4, 8, 12, 16]]
            jobs = [full.submit(job, g) for g in POINTS]
            done = [future.result() for future in jobs]
            monitored = [future.result() for future in small]
        end = json.loads((BASE/'formal'/ARM/'TRAINER_EXIT.json').read_text())['generation']
        if end not in done: done.append(job(end))
        atomic_json(BASE/'authority_je09_v1/EVALUATION_COMPLETE.json', dict(completed=True,
                    completed_generations=[g for g in done if g is not None], monitored_generations=monitored,
                    assist_off=True, no_new_web_mode_protocol=True, ability_barrier=False))
