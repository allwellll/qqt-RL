"""Independent Stage H, direct bounded training with asynchronous old protocols."""
import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('JE09_NAMESPACE', 'runs/je09_bun_20261016/stage_h_carry_item_events_pfsp8_v14')
os.environ.setdefault('JAXBOMB_RULE', 'bun')
BASE = ROOT/os.environ['JE09_NAMESPACE']
ARM = 'bun392_bridge_pfsp8'
POINTS = [0, 1, 4, 8, 12, 16]


def configure():
    from scripts import je09_bun_migration as m
    m.NAMESPACE = os.environ['JE09_NAMESPACE']; m.STAGE_B = m.STAGE_C = m.STAGE_D = True
    m.SOURCE_NAMESPACE = 'runs/je09_bun_20261016/stage_g_native_combat_targets_pfsp8_v13'
    m.SOURCE_CHECKPOINT = m.SOURCE_NAMESPACE+'/formal/'+ARM+'/gen000004'
    m.SOURCE_EVALUATION = 'formal/'+ARM+'/gen000004/COMMITTED.json'
    m.SCHEMA = 'je09-eight-device-stage-h-carry-item-events-v14'
    def plan(source):
        import copy
        from scripts.population_run_state import sha
        p = copy.deepcopy(source); d = json.loads((BASE/'ROUND_DECLARATION.json').read_text())
        p.update(experiment_id=d['experiment_id'], state_schema=m.SCHEMA, seed=d['seed'], recovery_every=4)
        p['campaign'].update(stage='G', change=d['change'], promotion='observation only; no ability barriers',
                             total_budget_generations=16, total_budget_fresh_ticks=33554432)
        p['return_replay_sha256'] = sha(BASE/'inputs/return_replay.pkl')
        p['carry_item_events'] = d['item_events']
        p['distance_curriculum']['advance'] = 'fixed time schedule; no CPU dependency'
        p['return_coverage']['advance'] = 'fixed time schedule; no CPU dependency'
        p['continuation'].update(source_checkpoint=m.SOURCE_CHECKPOINT, source_generation=4)
        return p
    m.continuation_plan = plan
    from scripts import je09_bun_continuation as ctl
    from scripts import je09_stage_h_worker as worker
    sys.modules['scripts.je09_bun_worker'] = worker
    ctl.POINTS = POINTS; ctl.ENTRY = 'scripts/je09_stage_h.py'; ctl.SMOKE_NAME = 'production_smoke_stage_h'
    def source_audit():
        from scripts.population_run_state import sha
        cp = ROOT/m.SOURCE_CHECKPOINT; marker = json.loads((cp/'COMMITTED.json').read_text())
        assert marker['generation'] == 4 and marker['passed'] and sha(cp/'state.pkl') == marker['state_sha256']
        assert sha(cp/'policy.pt') == marker['policy_sha256']
        return cp, marker
    ctl.source_audit = source_audit
    from scripts.je09_stage_e_coverage import labels, gate, teacher
    from scripts import je09_stage_h_ops as ops
    import types
    ops.return_expert = types.SimpleNamespace(**{k: getattr(teacher, k) for k in ['state', 'refresh', 'supervised_loss']}, gate=gate, labels=labels)
    return ctl


def prepare(ctl):
    from scripts.population_run_state import atomic_json, sha
    assert not (BASE/'PREPARED.json').exists(), 'new namespace only'
    (BASE/'inputs').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ctl.SOURCE/'inputs/return_replay.pkl', BASE/'inputs/return_replay.pkl')
    ctl.prepare()
    p = json.loads((BASE/'plan.json').read_text())
    p['parent_selection']['reason'] = 'G g4 balanced intermediate parent; raw Actor, same-point Critic and complete 11-slot pool; fixed32 remains weak'
    atomic_json(BASE/'plan.json', p)
    seal = json.loads((BASE/'SOURCE_SEAL.json').read_text()); seal['plan_sha256'] = sha(BASE/'plan.json')
    for name in ['scripts/je09_stage_h.py', 'scripts/je09_stage_h_items.py', 'scripts/je09_stage_h_reward.py', 'scripts/je09_stage_h_ops.py',
                 'scripts/je09_stage_h_worker.py', 'scripts/je09_stage_h_proof.py', 'scripts/je09_stage_h_evaluate.py', 'scripts/je09_stage_h_metrics.js',
                 'tests/test_je09_stage_h.py', 'scripts/je09_stage_g_combat.py', 'scripts/je09_stage_f_mode.py', 'scripts/je09_stage_f_reward.py',
                 'jax_bomb/bun_jax_bots.py']:
        seal['sources'][name] = sha(ROOT/name)
        target = BASE/'inputs/sealed_sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    atomic_json(BASE/'SOURCE_SEAL.json', seal)
    atomic_json(BASE/'PREPARED.json', dict(passed=True, parent_generation=4, plan_sha256=sha(BASE/'plan.json'), source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))
    atomic_json(BASE/'FROZEN_BEFORE_SMOKE.json', dict(round_sha256=sha(BASE/'ROUND_DECLARATION.json'),
                plan_sha256=sha(BASE/'plan.json'), source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))


def run(ctl):
    from scripts.population_run_state import atomic_json, sha
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
    event = json.loads((smoke/'formal'/ARM/'events.jsonl').read_text().splitlines()[0])
    assert event['finite'] and event['nonzero_update'] and ctl.verify(smoke/'formal'/ARM/'gen000001')['replicas_exact']
    atomic_json(BASE/'PRODUCTION_SMOKE_COMPLETE.json', dict(passed=True, devices=8, finite_nonzero=True,
                replicas_exact=True, source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))
    assert ctl.launch('restore', smoke, smoke/'formal'/ARM/'gen000001', cpu=True, log_name='restore.log').wait() == 0
    assert ctl.launch('replay', smoke, smoke/'formal'/ARM/'gen000000', log_name='replay.log').wait() == 0
    assert json.loads((smoke/'state/FRESH_GPU_REPLAY.json').read_text())['next_update_exact']
    assert not (BASE/'formal'/ARM/'gen000000/COMMITTED.json').exists()
    training = ctl.launch('worker', BASE, log_name='formal.log'); observed = []; best = None; flat = 0
    while training.poll() is None:
        for g in [0, 4, 8, 12, 16]:
            file = BASE/'async_monitor_v1'/f'g{g:06d}'/'BASIC_RETENTION.json'
            if g in observed or not file.exists(): continue
            results = json.loads(file.read_text())['results']
            # Existing primary300 small-sample diagnostic; never a final safe-win metric.
            kills = sum(results['hunter_'+tier]['kills'] for tier in ['easy', 'normal', 'hard'])
            deaths = sum(results['hunter_'+tier]['deaths'] for tier in ['easy', 'normal', 'hard'])
            captures = sum(results['carry_'+bucket]['capture'] for bucket in ['near', 'mid', 'far'])
            score = kills - .25*deaths + 2*captures
            flat = flat+1 if best is not None and score <= best else 0
            best = max(best, score) if best is not None else score; observed.append(g)
            atomic_json(BASE/'state/ASYNC_TREND.json', dict(observed_generations=observed, latest_score=score,
                        best_score=best, consecutive_without_improvement=flat, small_sample_only=True,
                        ability_barrier=False, score='existing basic retention: kills-.25*deaths+2*carry captures'))
            if flat >= 2 and training.poll() is None:
                atomic_json(BASE/'state/ASYNC_EARLY_STOP.json', dict(generation_observed=g,
                            reason='two consecutive existing-protocol directional checkpoints without improvement', technical_failure=False))
                training.send_signal(signal.SIGTERM)
        time.sleep(3)
    assert training.wait() == 0
    final = json.loads((BASE/'formal'/ARM/'TRAINER_EXIT.json').read_text())['generation']
    assert ctl.launch('restore', BASE, BASE/'formal'/ARM/f'gen{final:06d}', cpu=True, log_name='final_restore.log').wait() == 0
    atomic_json(BASE/'CAMPAIGN_COMPLETE.json', dict(completed=True, actual_generation=final,
                fresh_ticks=final*2097152, final_fresh_restore=True, no_cpu_gate=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['prepare', 'run', 'worker', 'restore', 'replay'])
    ap.add_argument('--run', default=os.environ['JE09_NAMESPACE']); ap.add_argument('--resume')
    a = ap.parse_args(); ctl = configure()
    if a.command == 'prepare': prepare(ctl)
    elif a.command == 'run': run(ctl)
    elif a.command == 'restore': ctl.restore(ROOT/a.run, ROOT/a.resume)
    else: ctl.run_worker(ROOT/a.run, ROOT/a.resume if a.resume else None, a.command == 'replay')
