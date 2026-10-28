"""JE09 Stage H3: fixed-budget natural combat coverage experiment."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The controller imports JAX for configuration but must stay CPU-only. The
# continuation launcher overrides these values for the actual GPU workers.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('JE09_NAMESPACE', 'runs/je09_bun_20261016/stage_h3_natural_combat_coverage_pfsp8_v17')
os.environ.setdefault('JAXBOMB_RULE', 'bun')
BASE = ROOT/os.environ['JE09_NAMESPACE']
ARM = 'bun392_bridge_pfsp8'
POINTS = [0,4,8,12,16,24,32]

def configure():
    from scripts import je09_stage_h2 as parent
    ctl = parent.configure()
    from scripts import je09_bun_migration as m
    source_ns = 'runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15'
    m.NAMESPACE = os.environ['JE09_NAMESPACE']; m.SOURCE_NAMESPACE = source_ns
    m.SOURCE_CHECKPOINT = source_ns+'/formal/'+ARM+'/gen000000'
    m.SOURCE_EVALUATION = 'formal/'+ARM+'/gen000000/COMMITTED.json'
    m.SCHEMA = 'je09-eight-device-stage-h3-natural-combat-v17'
    import scripts.je09_bun_continuation as continuation
    continuation.NAMESPACE = m.NAMESPACE; continuation.SOURCE_NAMESPACE = m.SOURCE_NAMESPACE
    continuation.SOURCE = ROOT/source_ns; continuation.SOURCE_CHECKPOINT = m.SOURCE_CHECKPOINT
    continuation.SOURCE_EVALUATION = m.SOURCE_EVALUATION; continuation.SCHEMA = m.SCHEMA
    def source_audit():
        from scripts.population_run_state import sha
        cp = ROOT/m.SOURCE_CHECKPOINT; marker = json.loads((cp/'COMMITTED.json').read_text())
        assert marker['generation'] == 0 and marker['passed']
        assert sha(cp/'state.pkl') == marker['state_sha256'] and sha(cp/'policy.pt') == marker['policy_sha256']
        return cp, marker
    continuation.source_audit = source_audit
    from scripts import je09_stage_h3_ops as h3ops
    from scripts import je09_stage_h2_worker as worker
    sys.modules['scripts.je09_stage_h2_ops'] = h3ops
    import scripts as scripts_package
    setattr(scripts_package, 'je09_stage_h2_ops', h3ops)
    sys.modules['scripts.je09_bun_worker'] = worker
    continuation.POINTS = POINTS; continuation.ENTRY = 'scripts/je09_stage_h3.py'; continuation.SMOKE_NAME = 'production_smoke_stage_h3'
    ctl = continuation
    return ctl

def prepare(ctl):
    from scripts import je09_stage_h2 as parent
    from scripts.population_run_state import atomic_json, sha
    parent.prepare(ctl)
    plan = json.loads((BASE/'plan.json').read_text())
    declaration = json.loads((BASE/'ROUND_DECLARATION.json').read_text())
    plan['campaign'].update(stage='H3', total_budget_generations=32, total_budget_fresh_ticks=67108864,
                            change=declaration['change'], stopping='fixed budget; no capability trend stopper')
    plan['full_scene_coverage'] = declaration['full_scene_coverage']
    plan['state_schema'] = ctl.SCHEMA
    plan['parent_selection']['reason'] = 'H2 g0, equivalent to G g4; strongest combat retention while controlled return remains available'
    atomic_json(BASE/'plan.json', plan)
    seal = json.loads((BASE/'SOURCE_SEAL.json').read_text()); seal['plan_sha256'] = sha(BASE/'plan.json')
    for name in ['scripts/je09_stage_h3.py','scripts/je09_stage_h3_ops.py','scripts/je09_stage_h3_evaluate.py',
                 'scripts/je09_stage_h3_declare.py','scripts/je09_stage_handoff.py','tests/test_je09_stage_h3.py']:
        seal['sources'][name] = sha(ROOT/name)
        target = BASE/'inputs/sealed_sources'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
    atomic_json(BASE/'SOURCE_SEAL.json', seal)
    atomic_json(BASE/'PREPARED.json', dict(passed=True, parent_generation=0,
        plan_sha256=sha(BASE/'plan.json'), source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))
    atomic_json(BASE/'FROZEN_BEFORE_SMOKE.json', dict(round_sha256=sha(BASE/'ROUND_DECLARATION.json'),
        plan_sha256=sha(BASE/'plan.json'), source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))

def run(ctl):
    from scripts.population_run_state import atomic_json, sha
    ctl.check_seal()
    assert not (BASE/'formal'/ARM/'gen000000/COMMITTED.json').exists(), 'fresh formal namespace required'
    smoke = BASE/ctl.SMOKE_NAME; smoke.mkdir(parents=True, exist_ok=True)
    assert not (smoke/'formal'/ARM/'gen000000/COMMITTED.json').exists(), 'preserve previous smoke attempts'
    (smoke/'inputs').mkdir(parents=True, exist_ok=True)
    for name in ['initial.pkl','initial_pool.pkl','frozen.pkl','regions.pkl','retention_parent.pkl']:
        shutil.copyfile(BASE/'inputs'/name, smoke/'inputs'/name)
    shutil.copyfile(BASE/'regions.json', smoke/'regions.json')
    p = json.loads((BASE/'plan.json').read_text()); p.update(generations=1, checkpoints=[0,1], seed=json.loads((BASE/'ROUND_DECLARATION.json').read_text())['seeds']['technical'])
    atomic_json(smoke/'plan.json', p)
    assert ctl.launch('worker', smoke, log_name='smoke.log').wait() == 0
    event = json.loads((smoke/'formal'/ARM/'events.jsonl').read_text().splitlines()[0])
    assert event['finite'] and event['nonzero_update'] and ctl.verify(smoke/'formal'/ARM/'gen000001')['replicas_exact']
    atomic_json(BASE/'PRODUCTION_SMOKE_COMPLETE.json', dict(passed=True, devices=8, finite_nonzero=True,
        replicas_exact=True, source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))
    assert ctl.launch('restore', smoke, smoke/'formal'/ARM/'gen000001', cpu=True, log_name='restore.log').wait() == 0
    assert ctl.launch('replay', smoke, smoke/'formal'/ARM/'gen000000', log_name='replay.log').wait() == 0
    assert json.loads((smoke/'state/FRESH_GPU_REPLAY.json').read_text())['next_update_exact']
    observer = subprocess.Popen(['taskset','-c','80-127',sys.executable, 'scripts/je09_stage_h3_evaluate.py'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JE09_NAMESPACE=os.environ['JE09_NAMESPACE']),
        stdout=(BASE/'async_observer.log').open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    training = ctl.launch('worker', BASE, log_name='formal.log')
    atomic_json(BASE/'TRAIN_LAUNCH.json', dict(started=True, devices=list(range(8)), budget_generations=32,
        budget_fresh_ticks=67108864, capability_early_stop=False, async_observer_started=observer.poll() is None,
        independent_namespace=True, parent='formal/bun392_bridge_pfsp8/gen000000'))
    from scripts.je09_stage_h2_policy import hard_safety
    seen = set()
    while training.poll() is None:
        file = BASE/'formal'/ARM/'STATE.json'
        if file.exists():
            state = json.loads(file.read_text()); generation = state['generation']
            if generation not in seen:
                seen.add(generation); safety = hard_safety(state)
                atomic_json(BASE/'state/HARD_SAFETY_MONITOR.json',dict(generation=generation, **safety))
                if safety['stop']:
                    atomic_json(BASE/'state/HARD_SAFETY_STOP.json',dict(generation=generation, **safety))
                    training.send_signal(signal.SIGTERM)
        time.sleep(3)
    assert training.wait() == 0
    final = json.loads((BASE/'formal'/ARM/'TRAINER_EXIT.json').read_text())['generation']
    assert final == 32 or (BASE/'state/HARD_SAFETY_STOP.json').exists()
    assert ctl.launch('restore', BASE, BASE/'formal'/ARM/f'gen{final:06d}', cpu=True, log_name='final_restore.log').wait() == 0
    atomic_json(BASE/'CAMPAIGN_COMPLETE.json', dict(completed=True, actual_generation=final,
        fresh_ticks=final*2097152, final_fresh_restore=True, capability_early_stop=False,
        evaluations_asynchronous=True, handoff_managed=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare','run','worker','restore','replay']); parser.add_argument('--run', default=os.environ['JE09_NAMESPACE']); parser.add_argument('--resume')
    args = parser.parse_args(); ctl = configure()
    if args.command == 'prepare': prepare(ctl)
    elif args.command == 'run': run(ctl)
    elif args.command == 'restore': ctl.restore(ROOT/args.run, ROOT/args.resume)
    else:
        from scripts.je09_stage_h2_coverage import install
        install(json.loads((BASE/'ROUND_DECLARATION.json').read_text())['item_state_coverage']['fraction'])
        ctl.run_worker(ROOT/args.run, ROOT/args.resume if args.resume else None, args.command == 'replay')
