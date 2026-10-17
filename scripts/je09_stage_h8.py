"""JE09 H8 native courier FSM DAgger campaign."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = os.environ.get('JE09_NAMESPACE',
    'runs/je09_bun_20261017/stage_h8_native_courier_fsm_dagger_v34')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('JE09_NAMESPACE', NAMESPACE)
os.environ.setdefault('JE09_H6_VERIFIER_SLOTS', '4')
sys.path.insert(0, str(ROOT))
from scripts import je09_stage_h6 as h6
from scripts import je09_stage_h5c as h5c
BASE = ROOT / NAMESPACE
h6.NAMESPACE = NAMESPACE
h6.BASE = BASE


def configure():
    ctl = h6.configure()
    ctl.SCHEMA = 'je09-eight-device-stage-h8-native-courier-fsm-dagger-v34'
    ctl.ENTRY = 'scripts/je09_stage_h8.py'
    ctl.SMOKE_NAME = 'production_smoke_stage_h8_v34_attempt06'
    original = ctl.configure_worker

    def configure_worker(run):
        worker = original(run)
        from scripts import je09_stage_h8_ops as ops
        from scripts import je09_stage_h5c_ops as base_ops
        worker.ops = ops
        base_ops.collect = ops.collect
        original_aggregate = worker.dagger.aggregate

        def aggregate(old, batch, key, generation):
            updated, replay, stats = original_aggregate(old, batch, key, generation)
            context = batch[-1].reshape((-1,) + batch[-1].shape[2:])
            target = context[:, 72].astype('int32')
            scene = context[:, 0] == 8
            for index, name in [(73, 'own_recovery_pickup'),
                                (74, 'enemy_carried_drop'),
                                (75, 'enemy_origin_pickup'),
                                (76, 'courier_enter_learner_cage'),
                                (77, 'courier_stock_theft'),
                                (78, 'courier_learner_origin_custody')]:
                if context.shape[-1] > index:
                    stats['target_a_' + name] = ((target == 1) & scene
                        & (context[:, index] > 0)).sum().astype('float32')
            return updated, replay, stats

        worker.dagger.aggregate = aggregate
        module = sys.modules['scripts.je09_stage_h5c_worker']
        module.ops = ops
        return worker
    ctl.configure_worker = configure_worker
    launch = ctl.launch

    class SmokeAcceptance:
        def __init__(self, process, run):
            self.process = process
            self.run = Path(run)

        def wait(self):
            code = self.process.wait()
            if code != 0 or self.run.name != ctl.SMOKE_NAME:
                return code
            event_path = self.run / 'formal/bun392_bridge_pfsp8/events.jsonl'
            events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
            dagger_rows = [event['dagger'] for event in events]
            target_a_rows = [dagger['targeted_course']['1'] for dagger in dagger_rows]
            transitions = [sum(row['native_transitions'][i] for row in target_a_rows)
                           for i in range(8)]
            total = lambda key: sum(row.get(key, 0) for row in dagger_rows)
            audit = dict(
                passed=(total('target_a_enemy_carried_drop') > 0
                        and total('target_a_own_recovery_pickup') > 0
                        and total('target_a_courier_stock_theft') > 0
                        and total('target_a_courier_learner_origin_custody') > 0
                        and transitions[1] > 0
                        and sum(row['query'] for row in target_a_rows) > 0),
                generations=[event['generation'] for event in events],
                weak_courier_custody_ticks=sum(row['opponent_course']['1']['opponent_native_custody_ticks'] for row in dagger_rows),
                attributed_kill=transitions[1], native_loose_pickup=transitions[3],
                inventory_or_native_steal=transitions[4], attributed_custody_start=transitions[5],
                path_a_enemy=total('path_a_enemy'),
                path_a_own_recovery=total('path_a_own_recovery'),
                target_a_native_transitions=transitions,
                target_a_enemy_carried_drop=total('target_a_enemy_carried_drop'),
                target_a_own_recovery_pickup=total('target_a_own_recovery_pickup'),
                target_a_enemy_origin_pickup=total('target_a_enemy_origin_pickup'),
                target_a_cage_entry=total('target_a_courier_enter_learner_cage'),
                target_a_stock_theft=total('target_a_courier_stock_theft'),
                target_a_enemy_custody_ticks=total('target_a_courier_learner_origin_custody'),
                target_a_query=sum(row['query'] for row in target_a_rows),
                own_recovery_and_enemy_carried_drop_are_distinct=True)
            from scripts.population_run_state import atomic_json
            from scripts.je09_stage_h8_audit import audit_smoke_funnel
            strict_audit = audit_smoke_funnel(audit)
            audit['strict_reaudit'] = strict_audit
            audit['passed'] = strict_audit['passed']
            atomic_json(BASE / 'state/H8_SMOKE_A_FUNNEL.json', audit)
            return 0 if audit['passed'] else 97

        def poll(self):
            return self.process.poll()

    def checked_launch(command, run, *args, **kwargs):
        if command == 'worker' and Path(run).name == ctl.SMOKE_NAME:
            plan_path = Path(run) / 'plan.json'
            plan = json.loads(plan_path.read_text())
            generations = json.loads((BASE / 'ROUND_DECLARATION.json').read_text())['smoke_generations']
            plan.update(generations=generations, checkpoints=[0, *range(1, generations + 1)],
                        total_fresh_ticks=generations * 2097152)
            from scripts.population_run_state import atomic_json
            atomic_json(plan_path, plan)
        process = launch(command, run, *args, **kwargs)
        if command == 'worker' and Path(run).name == ctl.SMOKE_NAME:
            return SmokeAcceptance(process, run)
        return process

    ctl.launch = checked_launch
    return ctl


def prepare(ctl):
    from scripts.population_run_state import atomic_json, sha
    h6.prepare(ctl)
    plan_path = BASE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    declaration = json.loads((BASE / 'ROUND_DECLARATION.json').read_text())
    plan['campaign'].update(stage='H8', change=declaration['change'])
    plan['state_schema'] = ctl.SCHEMA
    plan['dagger_opponent_course'] = declaration['dagger_opponent_course']
    atomic_json(plan_path, plan)
    seal_path = BASE / 'SOURCE_SEAL.json'
    seal = json.loads(seal_path.read_text())
    for name in ['scripts/je09_stage_h8.py', 'scripts/je09_stage_h8_ops.py',
                 'scripts/je09_stage_h8_course.py', 'scripts/je09_stage_h8_declare.py',
                 'tests/test_je09_stage_h8.py']:
        seal['sources'][name] = sha(ROOT / name)
        dest = BASE / 'inputs/sealed_sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    seal['plan_sha256'] = sha(plan_path)
    atomic_json(seal_path, seal)
    atomic_json(BASE / 'PREPARED.json', dict(passed=True, plan_sha256=sha(plan_path),
        source_seal_sha256=sha(seal_path), parent=declaration['parent']))
    atomic_json(BASE / 'FROZEN_BEFORE_SMOKE.json', dict(
        round_sha256=sha(BASE / 'ROUND_DECLARATION.json'), plan_sha256=sha(plan_path),
        source_seal_sha256=sha(seal_path)))


def run(ctl):
    smoke = BASE / ctl.SMOKE_NAME
    smoke_marker = BASE / 'PRODUCTION_SMOKE_COMPLETE.json'
    a_audit_path = BASE / 'state/H8_SMOKE_A_FUNNEL.json'
    if not smoke_marker.exists() or not a_audit_path.exists() or not json.loads(a_audit_path.read_text()).get('passed'):
        raise RuntimeError('H8 smoke acceptance unavailable; refusing formal launch')

    from scripts.population_run_state import atomic_json
    ctl.check_seal()
    cp = smoke / 'formal/bun392_bridge_pfsp8/gen000001'
    if not (smoke / 'state/RESTORE_MONITOR.json').exists():
        assert ctl.launch('restore', smoke, cp, cpu=True, log_name='restore.log').wait() == 0
    if not (smoke / 'state/FRESH_GPU_REPLAY.json').exists():
        assert ctl.launch('replay', smoke, smoke / 'formal/bun392_bridge_pfsp8/gen000000',
                          log_name='replay.log').wait() == 0
    assert json.loads((smoke / 'state/FRESH_GPU_REPLAY.json').read_text())['next_update_exact']
    # Formal namespace is independent; the completed smoke namespace is preserved read-only.

    observer = subprocess.Popen(['taskset', '-c', '80-127', sys.executable,
        'scripts/je09_stage_h3_evaluate.py'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                 JE09_NAMESPACE=str(NAMESPACE)),
        stdout=(BASE / 'async_observer.log').open('a'), stderr=subprocess.STDOUT,
        start_new_session=True)
    training = ctl.launch('worker', BASE, log_name='formal.log')
    declaration = json.loads((BASE / 'ROUND_DECLARATION.json').read_text())
    atomic_json(BASE / 'TRAIN_LAUNCH.json', dict(started=True, devices=list(range(8)),
        budget_generations=32, budget_fresh_ticks=67108864, capability_early_stop=False,
        async_observer_started=observer.poll() is None, independent_namespace=True,
        parent=declaration['parent']))
    from scripts.je09_stage_h2_policy import hard_safety
    seen = set()
    while training.poll() is None:
        state_path = BASE / 'formal/bun392_bridge_pfsp8/STATE.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            generation = state['generation']
            if generation not in seen:
                seen.add(generation)
                safety = hard_safety(state)
                atomic_json(BASE / 'state/HARD_SAFETY_MONITOR.json',
                            dict(generation=generation, **safety))
                if safety['stop']:
                    atomic_json(BASE / 'state/HARD_SAFETY_STOP.json',
                                dict(generation=generation, **safety))
                    training.send_signal(signal.SIGTERM)
        time.sleep(3)
    assert training.wait() == 0
    final = json.loads((BASE / 'formal/bun392_bridge_pfsp8/TRAINER_EXIT.json').read_text())['generation']
    assert final == 32 or (BASE / 'state/HARD_SAFETY_STOP.json').exists()
    assert ctl.launch('restore', BASE, BASE / 'formal/bun392_bridge_pfsp8' /
                      f'gen{final:06d}', cpu=True, log_name='final_restore.log').wait() == 0
    atomic_json(BASE / 'CAMPAIGN_COMPLETE.json', dict(completed=True,
        actual_generation=final, fresh_ticks=final * 2097152, final_fresh_restore=True,
        capability_early_stop=False, evaluations_asynchronous=True, handoff_managed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['declare', 'prepare', 'run', 'worker', 'restore', 'replay'])
    parser.add_argument('--run', default=NAMESPACE)
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.command == 'declare':
        from scripts.je09_stage_h8_declare import declare
        declare()
        raise SystemExit(0)
    ctl = configure()
    if args.command == 'prepare':
        prepare(ctl)
    elif args.command == 'run':
        run(ctl)
    elif args.command == 'restore':
        ctl.restore(ROOT / args.run, ROOT / args.resume)
    else:
        from scripts import je09_stage_h5c_worker as worker
        from scripts import je09_stage_h8_ops as ops
        worker.ops = ops
        ctl.run_worker(ROOT / args.run, ROOT / args.resume if args.resume else None,
                       args.command == 'replay')
