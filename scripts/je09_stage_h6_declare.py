"""Freeze H6 identities and immutable experiment declaration."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

NAMESPACE = os.environ.get('JE09_NAMESPACE',
    'runs/je09_bun_20261016/stage_h6_full_trajectory_fragment_dagger_v28')
PARENT = os.environ.get('JE09_PARENT',
    'runs/je09_bun_20261016/stage_h5cr_native_course_dagger_replica_pfsp8_v27/formal/bun392_bridge_pfsp8/gen000032')
EXPERIMENT_ID = os.environ.get('JE09_EXPERIMENT_ID',
    'JE09-STAGE-H6-FULL-TRAJECTORY-FRAGMENT-DAGGER-20261016-v28')


def lineage_version(experiment_id):
    return next((version for version in ('v32', 'v31', 'v30', 'v29', 'v28')
                 if version in experiment_id), 'v28')


def parent_version(experiment_id):
    return {'v32': 'v31', 'v31': 'v30', 'v30': 'v29', 'v29': 'v28'}.get(
        lineage_version(experiment_id), 'v27')


def successor_version(experiment_id):
    return {'v31': 'v32', 'v32': 'v33'}.get(
        lineage_version(experiment_id), 'replica_v30')


def declare():
    root = Path(NAMESPACE)
    assert not (root / 'TRAIN_LAUNCH.json').exists(), 'H6 training is already launched'
    parent = json.loads(Path('reports/je09_bun_20261015/stage_h5cr_v27_declaration_20261016.json').read_text())
    declaration = dict(parent)
    declaration.update(experiment_id=EXPERIMENT_ID,
        namespace=NAMESPACE, parent=PARENT, parent_generation=32,
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    names = list(declaration['seeds'])
    identities = {}; seeds = {}; used = set()
    existing_path = root / 'ROUND_DECLARATION.json'
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else None
    if existing and existing.get('experiment_id') == EXPERIMENT_ID:
        seeds.update(existing['seeds']); identities.update(existing['seed_derivation'])
        used.update(seeds.values())
    for path in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior = json.loads(path.read_text())
        used.update(prior.get('seeds', {}).values())
        if 'seed' in prior: used.add(prior['seed'])
    for name in names:
        if name in seeds: continue
        identity = declaration['experiment_id'] + ':' + name
        nonce = 0
        while True:
            seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
            if seed not in used: break
            nonce += 1
        seeds[name] = seed; identities[name] = dict(identity=identity, nonce=nonce); used.add(seed)
    declaration['seeds'] = seeds; declaration['seed'] = seeds['train']; declaration['seed_derivation'] = identities
    round_version = lineage_version(EXPERIMENT_ID)
    if round_version == 'v32':
        declaration['change'] = ('Only change H6 fragment teacher-verifier capacity: increase independently selected '
            'strict-verifier slots from one to four per objective fragment per learner rollout. Preserve target-stratified '
            'native weak-opponent sampling, all verifier rules, DAgger aggregation, reward, network, pool and evaluation.')
    else:
        declaration['change'] = ('Only change H6 target-stratified weak-opponent sampling: assign weak native courier/guard episodes '
            'independently across A loose recovery, B cage stock, carry return and cage delivery while preserving H6 learner-state '
            'DAgger, opponent difficulty annealing, reward, network, pool and evaluation protocols.')
    declaration['full_scene_coverage'] = dict(scene=8, fraction=.35, combat_fraction=.40,
        controlled_bridge_fraction=.25, source='complete native full games; learner drives both seats; no initialized bun/drop/pickup')
    source_version = parent_version(EXPERIMENT_ID)
    declaration['parent_selection'] = dict(generation=32,
        checkpoint=PARENT,
        reason=f'H6 {source_version} fixed32 completed with finite/nonzero updates; use its g32 actor/critic as the parameter parent. Fresh optimizer, RNG, environment and empty DAgger aggregation buffer.')
    if source_version == 'v29':
        declaration['parent_status'] = 'H6 v29 fixed32 complete; A/B/return/delivery queries reached 735/1028/1228/94 by g28, COMBAT query remained zero, and full evaluation is asynchronous.'
    elif source_version == 'v30':
        declaration['parent_status'] = 'H6 v30 fixed32 complete; all four objective fragments queried with zero noneligible gradient pollution; full assist-off evaluation remains asynchronous.'
    elif source_version == 'v31':
        declaration['parent_status'] = ('H6 v31 fixed32 complete; A/B/return/delivery queried 564/805/1368/47 with zero '
            'noneligible pollution. Weak-target queries were 84/24/72/14, indicating target sampling reached each fragment '
            'but teacher verification capacity remained sparse; use v31 g32 as the parameter parent.')
    declaration['dagger']['failure_fragments'] = [
        'A_attributed_loose_recovery', 'B_enemy_cage_stock_rush',
        'CARRY_RETURN', 'CAGE_DELIVERY']
    declaration['dagger']['combat_policy'] = (
        'COMBAT_CONTROL is quality-audit-only and receives no H6 DAgger replay query; '
        'combat learning remains existing PPO, native safe-attack expert, and historical retention.')
    declaration['dagger']['query_policy'] = ('Only learner-visited states in scene 8. A, B, carry return, and cage delivery '
        'each retain their own first currently failing state per episode, so a sparse earlier fragment cannot starve later fragments. '
        'Only native sugar/death windows, attributed loose pickup, real custody, and delivery boundary are eligible. '
        'Unreachable, unsafe, terminal, dead, trapped, free-enemy objective, glue, and banana states are rejected.')
    declaration['dagger']['stratified_query'] = dict(
        per_episode_first_deviation=['A', 'B', 'CARRY_RETURN', 'CAGE_DELIVERY'],
        verifier_slots_per_rollout=({name: 4 for name in ('A', 'B', 'CARRY_RETURN', 'CAGE_DELIVERY')}
            if round_version == 'v32' else dict(A=1, B=1, CARRY_RETURN=1, CAGE_DELIVERY=1)),
        combat_control_dagger_slots=0,
        source='current learner native full-game interactions only; no state injection or action override')
    declaration['dagger']['iteration'] = ('At every generation the updated learner samples fresh complete games; current states are queried and '
        'aggregated with prior learner rounds; next generation uses the updated learner. No expert action takes over env.step.')
    declaration['dagger']['replay_fragment_mix'] = 'equal current and historical valid fragments; fallback to available side; invalid rows exact zero gradient'
    declaration['dagger_opponent_course']['targeted_fragment_sampling'] = dict(
        weak_episode_targets={'A_LOOSE_RECOVERY': .25, 'B_CAGE_STOCK': .25,
                              'CARRY_RETURN': .25, 'CAGE_DELIVERY': .25},
        target_role={'A_LOOSE_RECOVERY': 'weak_courier', 'B_CAGE_STOCK': 'weak_guard',
                     'CARRY_RETURN': 'weak_courier', 'CAGE_DELIVERY': 'weak_guard'},
        weakness_schedule=declaration['dagger_opponent_course']['weak_episode_fraction'],
        difficulty_independent=True, target_metadata_actor_input=False,
        state_source='learner actual native full-game interactions only; no injected custody/drop/death/window')
    declaration['budget'] = dict(formal_generations=32, fresh_ticks=67108864, fixed=True,
        evaluation='existing assist-off authority/fixed32/controlled/Hunter/Tactical/carry/full2400 asynchronously; observational only')
    next_version = successor_version(EXPERIMENT_ID)
    declaration['terminal_relay'] = dict(successor=f'runs/je09_bun_20261017/stage_h6_target_stratified_fragment_dagger_{next_version}',
        policy='single owner directly starts independent fixed32 continuation and verifies first finite nonzero formal STATE',
        no_capability_stop=True)
    declaration['training_owner'] = 'unique H6 relay; notify on startup, every terminal, and any technical failure'
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(declaration, indent=2) + '\n'
    (root / 'ROUND_DECLARATION.json').write_text(text)
    report_name = (f'stage_h6_{round_version}_declaration_20261017.json'
                   if round_version in ('v32', 'v31', 'v30', 'v29')
                   else 'stage_h6_v28_declaration_20261016.json')
    Path('reports/je09_bun_20261015', report_name).write_text(text)


if __name__ == '__main__': declare()
