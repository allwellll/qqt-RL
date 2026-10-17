"""Immutable H8 declaration derived from H6 v32, with disjoint identities."""
import hashlib
import json
import subprocess
from pathlib import Path

NAMESPACE = 'runs/je09_bun_20261017/stage_h8_native_courier_fsm_dagger_v34'
PARENT = ('runs/je09_bun_20261017/stage_h7_native_courier_fsm_dagger_v33/'
          'formal/bun392_bridge_pfsp8/gen000032')
EXPERIMENT_ID = 'JE09-STAGE-H8-NATIVE-DROP-RECOVERY-DAgger-20261017-v34'


def declare():
    root = Path(NAMESPACE)
    assert not root.exists(), 'H8 namespace must be fresh'
    source = json.loads(Path('runs/je09_bun_20261017/stage_h7_native_courier_fsm_dagger_v33/ROUND_DECLARATION.json').read_text())
    d = dict(source)
    d.update(experiment_id=EXPERIMENT_ID, namespace=NAMESPACE, parent=PARENT,
             parent_generation=32, parent_commit=subprocess.check_output(
                 ['git', 'rev-parse', 'HEAD'], text=True).strip())
    used = set()
    for path in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        old = json.loads(path.read_text())
        used.update(old.get('seeds', {}).values())
        if 'seed' in old:
            used.add(old['seed'])
    seeds, derivation = {}, {}
    for name in source['seeds']:
        identity = EXPERIMENT_ID + ':' + name
        nonce = 0
        while True:
            seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
            if seed not in used:
                break
            nonce += 1
        seeds[name] = seed
        derivation[name] = dict(identity=identity, nonce=nonce)
        used.add(seed)
    d['seeds'] = seeds
    d['seed'] = seeds['train']
    d['seed_derivation'] = derivation
    d['change'] = ('Only repair the weak native A courier episode FSM (return to its own cage after native stock pickup) '
        'and rebalance weak target sampling to A/B/return/delivery = 50/20/15/15. Preserve H6 strict labels/verifier, '
        'reward, loss, network, opponent pool and evaluation protocols.')
    d['parent_selection'] = dict(generation=32, checkpoint=PARENT,
        reason='Use v32 g32 as requested; fixed32 budget completed with finite nonzero updates and no completed same-protocol superior parent established.')
    d['dagger_opponent_course']['targeted_fragment_sampling'] = dict(
        weak_episode_targets={'A_LOOSE_RECOVERY': .55, 'B_CAGE_STOCK': .10,
                              'CARRY_RETURN': .05, 'CAGE_DELIVERY': .30},
        target_role={'A_LOOSE_RECOVERY': 'weak_courier', 'B_CAGE_STOCK': 'weak_guard',
                     'CARRY_RETURN': 'weak_courier', 'CAGE_DELIVERY': 'weak_guard'},
        weakness_schedule=d['dagger_opponent_course']['weak_episode_fraction'],
        courier_fsm={'empty': 'native path into learner cage for stock pickup',
                     'carrying': 'native path back to opponent own cage; carry move duty [.2,.3,.4,.55]'},
        difficulty_independent=True, target_metadata_actor_input=False,
        state_source='learner actual native full-game interactions only; no injected custody/drop/death/window')
    d['budget'] = dict(formal_generations=32, fresh_ticks=67108864, fixed=True,
        evaluation='existing assist-off protocols asynchronously; observational only')
    d['terminal_relay'] = dict(successor=None,
        policy='at fixed terminal notify and directly launch the predeclared independent H8 replication after acceptance; no marker-only handoff')
    d['training_owner'] = 'unique H8 controller; notify on startup, terminal, or any technical failure'
    d['parent_status'] = 'H6 v32 fixed32 completed; use its raw g32 Actor/Critic and full pool. Parent selection follows current request; no later completed same-protocol point is established as superior.'
    d['smoke_attempts'] = [dict(attempt=1, seed=seeds['technical'],
        target_weights={'A': .50, 'B': .20, 'return': .15, 'delivery': .15},
        carry_move_duty=[.75, .8, .85, .9], result='failed: A target had zero native loose pickup and zero A query'),
        dict(attempt=2, seed_identity=EXPERIMENT_ID + ':technical_smoke02',
        target_weights={'A': .55, 'B': .10, 'return': .05, 'delivery': .30},
        carry_move_duty=[.2, .3, .4, .55],
        acceptance='A-target native theft/custody, attributed learner kill, learner loose pickup, and A query all positive')]
    root.mkdir(parents=True)
    text = json.dumps(d, indent=2) + '\n'
    (root / 'ROUND_DECLARATION.json').write_text(text)
    Path('reports/je09_bun_20261015/stage_h8_v34_declaration_20261017.json').write_text(text)


def amend_attempt2():
    path = Path(NAMESPACE) / 'ROUND_DECLARATION.json'
    d = json.loads(path.read_text())
    assert d['experiment_id'] == EXPERIMENT_ID
    old_seed = d['seeds']['technical']
    used = set()
    for declaration in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior = json.loads(declaration.read_text())
        used.update(prior.get('seeds', {}).values())
        if 'seed' in prior:
            used.add(prior['seed'])
    identity = EXPERIMENT_ID + ':technical_smoke02'
    nonce = 0
    while True:
        seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
        if seed not in used:
            break
        nonce += 1
    d['smoke_attempts'] = [
        dict(attempt=1, seed=old_seed,
             target_weights={'A': .50, 'B': .20, 'return': .15, 'delivery': .15},
             carry_move_duty=[.75, .8, .85, .9],
             result='finite/nonzero; A target had zero native loose pickup and zero A query'),
        dict(attempt=2, seed=seed,
             seed_identity=identity, nonce=nonce,
             target_weights={'A': .55, 'B': .10, 'return': .05, 'delivery': .30},
             carry_move_duty=[.2, .3, .4, .55],
             acceptance='A-target learner-origin carrier death/drop and own-recovery pickup, attributed kill, and A query all positive')]
    d['seeds']['technical'] = seed
    d['seed_derivation']['technical'] = dict(identity=identity, nonce=nonce)
    d['dagger_opponent_course']['targeted_fragment_sampling'] = dict(
        weak_episode_targets={'A_LOOSE_RECOVERY': .55, 'B_CAGE_STOCK': .10,
                              'CARRY_RETURN': .05, 'CAGE_DELIVERY': .30},
        target_role={'A_LOOSE_RECOVERY': 'weak_courier', 'B_CAGE_STOCK': 'weak_guard',
                     'CARRY_RETURN': 'weak_courier', 'CAGE_DELIVERY': 'weak_guard'},
        weakness_schedule=d['dagger_opponent_course']['weak_episode_fraction'],
        courier_fsm={'empty': 'native path into learner cage for stock pickup',
                     'carrying': 'native path back to opponent own cage; carry move duty [.2,.3,.4,.55]'},
        difficulty_independent=True, target_metadata_actor_input=False,
        state_source='learner actual native full-game interactions only; no injected custody/drop/death/window')
    d['parent_status'] = ('H6 v32 fixed32 completed; use its raw g32 Actor/Critic and full pool. '
                          'No later completed same-protocol point is established as superior.')
    text = json.dumps(d, indent=2) + '\n'
    path.write_text(text)
    Path('reports/je09_bun_20261015/stage_h8_v34_declaration_20261017.json').write_text(text)


def amend_attempt3():
    path = Path(NAMESPACE) / 'ROUND_DECLARATION.json'
    d = json.loads(path.read_text())
    used = set()
    for declaration in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior = json.loads(declaration.read_text())
        used.update(prior.get('seeds', {}).values())
        if 'seed' in prior:
            used.add(prior['seed'])
    identity = EXPERIMENT_ID + ':technical_smoke03'
    nonce = 0
    while True:
        seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
        if seed not in used:
            break
        nonce += 1
    d['smoke_attempts'].append(dict(attempt=3, seed=seed, seed_identity=identity,
        nonce=nonce, target_weights={'A': .55, 'B': .10, 'return': .05, 'delivery': .30},
        carry_move_duty=[.2, .3, .4, .55], bomb_period=[64, 32, 16, 8],
        courier_route='direct scene topology distance, switched by native empty/carrying state',
        acceptance='A-target native stock theft/custody, attributed learner kill, loose drop/recovery, and query all positive'))
    d['seeds']['technical'] = seed
    d['seed_derivation']['technical'] = dict(identity=identity, nonce=nonce)
    d['dagger_opponent_course']['targeted_fragment_sampling']['courier_fsm'].update(
        route='direct scene topology distance; learner cage while empty, own cage while carrying',
        bomb_period=[64, 32, 16, 8])
    d['parent_status'] = ('H6 v32 fixed32 completed; use its raw g32 Actor/Critic and full pool. '
                          'No later completed same-protocol point is established as superior.')
    text = json.dumps(d, indent=2) + '\n'
    path.write_text(text)
    Path('reports/je09_bun_20261015/stage_h8_v34_declaration_20261017.json').write_text(text)


def amend_attempt4():
    path = Path(NAMESPACE) / 'ROUND_DECLARATION.json'
    d = json.loads(path.read_text())
    used = set()
    for declaration in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior = json.loads(declaration.read_text())
        used.update(prior.get('seeds', {}).values())
        if 'seed' in prior:
            used.add(prior['seed'])
    identity = EXPERIMENT_ID + ':technical_smoke04'
    nonce = 0
    while True:
        seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
        if seed not in used:
            break
        nonce += 1
    d['smoke_attempts'].append(dict(attempt=4, seed=seed, seed_identity=identity,
        nonce=nonce, target_weights={'A': .70, 'B': .05, 'return': .05, 'delivery': .20},
        carry_move_duty=[.2, .3, .4, .55], bomb_period=[64, 32, 16, 8],
        courier_route='direct scene topology distance; native empty/carrying state switch',
        acceptance='A-target native theft/custody, attributed learner kill, loose drop, own-recovery pickup, and A query all positive'))
    d['seeds']['technical'] = seed
    d['seed_derivation']['technical'] = dict(identity=identity, nonce=nonce)
    d['dagger_opponent_course']['targeted_fragment_sampling']['weak_episode_targets'] = {
        'A_LOOSE_RECOVERY': .70, 'B_CAGE_STOCK': .05,
        'CARRY_RETURN': .05, 'CAGE_DELIVERY': .20}
    d['parent_status'] = ('H6 v32 fixed32 completed; use its raw g32 Actor/Critic and full pool. '
                          'No later completed same-protocol point is established as superior.')
    text = json.dumps(d, indent=2) + '\n'
    path.write_text(text)
    Path('reports/je09_bun_20261015/stage_h8_v34_declaration_20261017.json').write_text(text)


def amend_attempt5():
    path = Path(NAMESPACE) / 'ROUND_DECLARATION.json'
    d = json.loads(path.read_text())
    used = set()
    for declaration in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior = json.loads(declaration.read_text())
        used.update(prior.get('seeds', {}).values())
        if 'seed' in prior:
            used.add(prior['seed'])
    identity = EXPERIMENT_ID + ':technical_smoke05'
    nonce = 0
    while True:
        seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
        if seed not in used:
            break
        nonce += 1
    d['smoke_attempts'].append(dict(attempt=5, seed=seed, seed_identity=identity,
        nonce=nonce, smoke_generations=2,
        target_weights={'A': .70, 'B': .05, 'return': .05, 'delivery': .20},
        acceptance='Across two fresh smoke generations, A-target native theft/custody, attributed kill/drop, own recovery pickup, and A query all positive'))
    d['seeds']['technical'] = seed
    d['seed_derivation']['technical'] = dict(identity=identity, nonce=nonce)
    d['smoke_policy'] = 'attempt05 is a two-generation eight-device production smoke; formal campaign remains fixed32.'
    d['parent_status'] = ('H6 v32 fixed32 completed; use its raw g32 Actor/Critic and full pool. '
                          'No later completed same-protocol point is established as superior.')
    text = json.dumps(d, indent=2) + '\n'
    path.write_text(text)
    Path('reports/je09_bun_20261015/stage_h8_v34_declaration_20261017.json').write_text(text)


def amend_attempt6():
    path = Path(NAMESPACE) / 'ROUND_DECLARATION.json'
    d = json.loads(path.read_text())
    used = set()
    for declaration in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior = json.loads(declaration.read_text())
        used.update(prior.get('seeds', {}).values())
        if 'seed' in prior:
            used.add(prior['seed'])
    identity = EXPERIMENT_ID + ':technical_smoke06'
    nonce = 0
    while True:
        seed = int.from_bytes(hashlib.sha256(f'{identity}:{nonce}'.encode()).digest()[:4], 'big')
        if seed not in used:
            break
        nonce += 1
    d['smoke_attempts'].append(dict(attempt=6, seed=seed, seed_identity=identity,
        nonce=nonce, smoke_generations=4,
        target_weights={'A': .70, 'B': .05, 'return': .05, 'delivery': .20},
        acceptance='Across four iterative smoke generations, A-target native theft/custody, attributed kill/drop, own recovery pickup, and A query all positive'))
    d['seeds']['technical'] = seed
    d['seed_derivation']['technical'] = dict(identity=identity, nonce=nonce)
    d['smoke_generations'] = 4
    d['smoke_policy'] = 'attempt06 is a four-generation eight-device iterative DAgger smoke; formal campaign remains fixed32.'
    d['parent_status'] = ('H6 v32 fixed32 completed; use its raw g32 Actor/Critic and full pool. '
                          'No later completed same-protocol point is established as superior.')
    text = json.dumps(d, indent=2) + '\n'
    path.write_text(text)
    Path('reports/je09_bun_20261015/stage_h8_v34_declaration_20261017.json').write_text(text)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'amend-attempt2':
        amend_attempt2()
    elif len(sys.argv) > 1 and sys.argv[1] == 'amend-attempt3':
        amend_attempt3()
    elif len(sys.argv) > 1 and sys.argv[1] == 'amend-attempt4':
        amend_attempt4()
    elif len(sys.argv) > 1 and sys.argv[1] == 'amend-attempt5':
        amend_attempt5()
    elif len(sys.argv) > 1 and sys.argv[1] == 'amend-attempt6':
        amend_attempt6()
    else:
        declare()
