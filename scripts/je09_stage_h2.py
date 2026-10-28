"""H2 fixed 32-generation campaign; CPU evaluations cannot end its budget."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('JE09_NAMESPACE', 'runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15')
os.environ.setdefault('JAXBOMB_RULE', 'bun')
BASE = ROOT/os.environ['JE09_NAMESPACE']; ARM = 'bun392_bridge_pfsp8'
from scripts.je09_stage_h2_policy import POINTS, TRAINING_GENERATIONS, hard_safety


def configure():
    from scripts import je09_stage_h as parent
    ctl = parent.configure()
    original = ctl.continuation_plan
    def plan(source):
        p = original(source)
        d = json.loads((BASE/'ROUND_DECLARATION.json').read_text())
        p['campaign'].update(stage='H2', total_budget_generations=32,
            total_budget_fresh_ticks=67108864, stopping='fixed budget; no capability trend stopper')
        p['item_state_coverage'] = d['item_state_coverage']
        p['state_schema'] = 'je09-eight-device-stage-h2-item-coverage-v15'
        return p
    ctl.continuation_plan = plan
    ctl.SCHEMA = 'je09-eight-device-stage-h2-item-coverage-v15'
    ctl.POINTS = POINTS; ctl.ENTRY = 'scripts/je09_stage_h2.py'; ctl.SMOKE_NAME = 'production_smoke_stage_h2'
    from scripts import je09_stage_h2_worker as worker
    sys.modules['scripts.je09_bun_worker'] = worker
    return ctl


def prepare(ctl):
    from scripts import je09_stage_h as parent
    from scripts.population_run_state import sha, atomic_json
    parent.prepare(ctl)
    seal = json.loads((BASE/'SOURCE_SEAL.json').read_text())
    for name in ['scripts/je09_stage_h2.py', 'scripts/je09_stage_h2_policy.py',
                 'scripts/je09_stage_h2_coverage.py', 'scripts/je09_stage_h2_ops.py',
                 'scripts/je09_stage_h2_worker.py', 'scripts/je09_stage_h2_evaluate.py',
                 'scripts/je09_stage_h2_declare.py', 'tests/test_je09_stage_h2.py',
                 'scripts/je09_stage_h_hunter.js', 'scripts/je09_stage_h_diagnostics.js',
                 'scripts/je09_stage_h_eval_cache.js']:
        seal['sources'][name] = sha(ROOT/name)
        target = BASE/'inputs/sealed_sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    atomic_json(BASE/'SOURCE_SEAL.json', seal)
    prepared = json.loads((BASE/'PREPARED.json').read_text())
    prepared['source_seal_sha256'] = sha(BASE/'SOURCE_SEAL.json'); atomic_json(BASE/'PREPARED.json', prepared)
    atomic_json(BASE/'FROZEN_BEFORE_SMOKE.json', dict(round_sha256=sha(BASE/'ROUND_DECLARATION.json'),
        plan_sha256=sha(BASE/'plan.json'), source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))


def run(ctl):
    from scripts.population_run_state import atomic_json, sha
    from scripts.je03_bun831_20261011 import child_env
    ctl.check_seal(); smoke = BASE/ctl.SMOKE_NAME
    assert not (smoke/'formal'/ARM/'gen000000/COMMITTED.json').exists()
    (smoke/'inputs').mkdir(parents=True, exist_ok=True)
    for name in ['initial.pkl', 'initial_pool.pkl', 'frozen.pkl', 'regions.pkl', 'retention_parent.pkl']:
        shutil.copyfile(BASE/'inputs'/name, smoke/'inputs'/name)
    shutil.copyfile(BASE/'regions.json', smoke/'regions.json')
    p = json.loads((BASE/'plan.json').read_text())
    p.update(generations=1, checkpoints=[0, 1], seed=json.loads((BASE/'ROUND_DECLARATION.json').read_text())['seeds']['technical'])
    atomic_json(smoke/'plan.json', p)
    assert ctl.launch('worker', smoke, log_name='smoke.log').wait() == 0
    row = json.loads((smoke/'formal'/ARM/'events.jsonl').read_text().splitlines()[0])
    assert row['finite'] and row['nonzero_update'] and ctl.verify(smoke/'formal'/ARM/'gen000001')['replicas_exact']
    atomic_json(BASE/'PRODUCTION_SMOKE_COMPLETE.json', dict(passed=True, devices=8,
        finite_nonzero=True, replicas_exact=True, source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))
    assert ctl.launch('restore', smoke, smoke/'formal'/ARM/'gen000001', cpu=True, log_name='restore.log').wait() == 0
    assert ctl.launch('replay', smoke, smoke/'formal'/ARM/'gen000000', log_name='replay.log').wait() == 0
    assert json.loads((smoke/'state/FRESH_GPU_REPLAY.json').read_text())['next_update_exact']
    assert not (BASE/'formal'/ARM/'gen000000/COMMITTED.json').exists()
    evaluation = subprocess.Popen(['taskset', '-c', '80-127', sys.executable,
        'scripts/je09_stage_h2_evaluate.py'], cwd=ROOT, env=child_env('', True),
        stdout=(BASE/'async_evaluate.log').open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    training = ctl.launch('worker', BASE, log_name='formal.log')
    atomic_json(BASE/'TRAIN_LAUNCH.json', dict(started=True, devices=list(range(8)),
        budget_generations=32, budget_fresh_ticks=67108864, capability_early_stop=False,
        async_evaluator_started=evaluation.poll() is None, independent_namespace=True))
    seen = set()
    while training.poll() is None:
        state = BASE/'formal'/ARM/'STATE.json'
        if state.exists():
            row = json.loads(state.read_text()); g = row['generation']
            if g not in seen:
                seen.add(g); check = hard_safety(row)
                atomic_json(BASE/'state/HARD_SAFETY_MONITOR.json', dict(generation=g, **check))
                if check['stop']:
                    atomic_json(BASE/'state/HARD_SAFETY_STOP.json', dict(generation=g, **check))
                    training.send_signal(signal.SIGTERM)
        time.sleep(2)
    assert training.wait() == 0, 'technical training failure; preserve logs'
    final = json.loads((BASE/'formal'/ARM/'TRAINER_EXIT.json').read_text())['generation']
    assert final == TRAINING_GENERATIONS or (BASE/'state/HARD_SAFETY_STOP.json').exists()
    assert ctl.launch('restore', BASE, BASE/'formal'/ARM/f'gen{final:06d}', cpu=True, log_name='final_restore.log').wait() == 0
    atomic_json(BASE/'CAMPAIGN_COMPLETE.json', dict(completed=True, actual_generation=final,
        fresh_ticks=final*2097152, final_fresh_restore=True, capability_early_stop=False,
        evaluations_asynchronous=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'worker', 'restore', 'replay'])
    parser.add_argument('--run', default=os.environ['JE09_NAMESPACE']); parser.add_argument('--resume')
    args = parser.parse_args(); ctl = configure()
    if args.command == 'prepare': prepare(ctl)
    elif args.command == 'run': run(ctl)
    elif args.command == 'restore': ctl.restore(ROOT/args.run, ROOT/args.resume)
    else:
        from scripts.je09_stage_h2_coverage import install
        install(json.loads((BASE/'ROUND_DECLARATION.json').read_text())['item_state_coverage']['fraction'])
        ctl.run_worker(ROOT/args.run, ROOT/args.resume if args.resume else None, args.command == 'replay')
