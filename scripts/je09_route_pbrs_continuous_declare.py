"""Freeze the continuous-cell route-potential H8 v36 experiment."""
import hashlib
import json
import subprocess
from pathlib import Path

NAMESPACE = 'runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02'
PARENT = ('runs/je09_bun_20261017/stage_h8_native_courier_fsm_dagger_v34/'
          'formal/bun392_bridge_pfsp8/gen000032')
EXPERIMENT_ID = 'JE09-STAGE-H8-ROUTE-PBRS-CONTINUOUS-20261017-v36-attempt02'


def _seed(identity, used):
    nonce = 0
    while True:
        value = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
        if value not in used:
            used.add(value)
            return value, nonce
        nonce += 1


def declare():
    root = Path(NAMESPACE)
    assert not root.exists(), 'v36 namespace must be new and empty'
    parent = json.loads(Path('runs/je09_bun_20261017/stage_h8_native_courier_fsm_dagger_v34/ROUND_DECLARATION.json').read_text())
    used = set()
    for path in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        old = json.loads(path.read_text())
        used.update(old.get('seeds', {}).values())
        if 'seed' in old:
            used.add(old['seed'])
    roles = ['train', 'technical', 'evaluation_main', 'evaluation_holdout',
             'tactical_v2', 'hunter_hard', 'hunter_normal', 'hunter_easy',
             'route_stock', 'route_enemy_drop', 'route_carry', 'route_wall', 'full2400']
    seeds = {}; derivation = {}
    for role in roles:
        value, nonce = _seed(EXPERIMENT_ID + ':' + role, used)
        seeds[role] = value
        derivation[role] = dict(identity=EXPERIMENT_ID + ':' + role, nonce=nonce)
    parent.update(experiment_id=EXPERIMENT_ID, namespace=NAMESPACE, parent=PARENT,
        parent_generation=32, seeds=seeds, seed=seeds['train'], seed_derivation=derivation,
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    parent['change'] = ('Only replace integer-cell route distance in the existing strict PBRS base potential '
        'with topology-respecting continuous-to-cell-center weighted path distance; lambda=.05 and gamma=.995 remain fixed.')
    parent['route_pbrs'] = dict(enabled=True, gamma=.995, lambda_value=.05,
        distance='min of continuous Manhattan connector to current/axis-adjacent traversable cells plus production grid shortest path',
        topology='production weighted grid; hard walls excluded; enter-cell brick cost 1+3; no diagonal/corner cuts',
        phases=['native_enemy_death_or_sugar_stock', 'attributed_enemy_origin_drop',
                'enemy_bun_carry_return', 'cage_delivery_contact'],
        terminal_phi=0, phase_identity_offset_continuity=True, retired_identity_ledger=True,
        unsafe_topology_change_rebased=True, no_placement_reward=True,
        actor_observation_unchanged=True, coefficient_scan=False)
    parent['budget'] = dict(formal_generations=32, fresh_ticks=67108864, fixed=True,
        save_generations=[0,4,8,12,16,24,32], technical_smoke_generations=1,
        evaluation='existing assist-off protocols asynchronously; observational only')
    parent['stop'] = 'Stop only technical corruption, nonfinite update, OOM, or checkpoint/replica integrity failure.'
    parent['terminal_relay'] = dict(successor=None, policy='notify on terminal; no automatic algorithm change')
    root.mkdir(parents=True)
    (root/'ROUND_DECLARATION.json').write_text(json.dumps(parent, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    declare()
