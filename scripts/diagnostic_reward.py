"""Apply sealed H7 production reward ledgers to observed production Web transitions.

The Web world is never stepped here. Production safety functions only evaluate
the recorded state/actions; native custody and damage events come from Web.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  JAXBOMB_RULE='bun', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / os.environ.get('REPLAY_INPUT', 'runs/replay_review_20261017/inputs') / 'authority'
sys.path.insert(0, str(AUTHORITY))
import numpy as np
import jax
import jax.numpy as jnp
from jax_bomb import je07_bun_env as env, bun_safety as safety
from scripts import je09_bun_ops as ops, je09_stage_h_reward as reward, je09_stage_h_items as items

assert jax.default_backend() == 'cpu'
env.prepare()
env.configure_training('full=1', 1, reward_profile='legacy', retention_replay=0,
                       tactical_bomb_placement_reward=.1)
env._DANGER_TRACKING = True
TEMPLATE = jax.tree.map(np.asarray, env.init_batch(jax.random.PRNGKey(0), 1))
CHUNK = 32
COMPONENTS = ('lesson_task', 'global_safety', 'settled_kill', 'settled_shaping',
              'combat_clawback', 'blocked_route', 'transport_shape',
              'transport_clawback', 'transport_capture', 'drop_bonus', 'drop_clawback',
              'drop_approach_net', 'drop_approach_clawback', 'carry_home_net',
              'carry_home_clawback', 'bridge_reward', 'bridge_clawback',
              'item_positive', 'item_clawback')


def convert(frame, linger, scene=8):
    core = TEMPLATE.core
    mapping = {'pos': 'pos', 'fuse': 'fuse', 'owner': 'owner', 'bomb_blast': 'bombBlast',
               'blast_linger': 'blastLinger', 'brick_linger': 'brickLinger', 'wall': 'wall',
               'brick': 'brick', 'pushable': 'pushable', 'push_t': 'pushT', 'bush': 'bush',
               'crate': 'crate', 'rec_crate': 'recycle', 'alive': 'alive', 'hp': 'hp',
               'invuln': 'invuln', 'bombs_cap': 'bombsCap', 'blast_cap': 'blastCap',
               'spd_g': 'spdG', 't': 't'}
    fields = {k: np.asarray(frame[v], dtype=getattr(core, k).dtype).reshape(getattr(core, k).shape)
              for k, v in mapping.items()}
    core = core._replace(**fields)
    mapping = {'bun_stored': 'bunStored', 'bun_loose': 'bunLoose', 'bun_carried': 'bunCarried',
               'bun_respawn': 'bunRespawn', 'bun_spawn_pos': 'bunSpawnPos', 'held_item': 'heldItem',
               'field_item': 'fieldItem', 'field_owner': 'fieldOwner', 'field_armed': 'fieldArmed',
               'move_status': 'movementStatus', 'status_ticks': 'movementStatusTicks',
               'slide_dir': 'slideDir', 'last_move_dir': 'lastMoveDir', 'trapped': 'trapped'}
    fields = {k: np.asarray(frame[v], dtype=getattr(TEMPLATE, k).dtype).reshape(getattr(TEMPLATE, k).shape)
              for k, v in mapping.items()}
    fields.update(core=core, scene=np.array([scene], np.int32),
                  blast_owner_linger=linger[None],
                  item_slots=np.array([[([s['item'] for s in row]+[0]*7)[:7] for row in frame['itemSlots']]], np.int8),
                  item_counts=np.array([[([s['count'] for s in row]+[0]*7)[:7] for row in frame['itemSlots']]], np.int16))
    return TEMPLATE._replace(**fields)


def observed(before, entry, causes):
    after, raw = entry['state'], entry['info']
    bun = entry['bun_before']
    z = lambda: np.zeros((1, 2), bool)
    names = set(ops.legacy_ops.TASK_EVENTS) | {
        'death', 'own_bomb_defeat', 'avoidable_danger_death', 'drop', 'carrier_death',
        'surviving_kill', 'threat_bomb_placed', 'safe_bomb_placed', 'danger_safe_resolution',
        'pickup_loose', 'native_transfer', 'steal_window_trap', 'steal_window_dead',
        'trap_hit', 'blocked_own_route'}
    info = {name: z() for name in names}
    info.update(mutual_death=np.array([raw['mutualDeath']], bool),
                drop_origin=np.full((1, 2), -1, np.int32),
                drop_credit=np.full((1, 2), -1, np.int32),
                pickup_origin=np.full((1, 2), -1, np.int32),
                trap_owner=np.full((1, 2), -1, np.int32),
                drop_cells=np.zeros((1, 2, 2), np.int32),
                tactical_bomb_enemy_safe_moves_before=np.zeros((1, 2), np.int16),
                tactical_bomb_enemy_safe_moves_after=np.zeros((1, 2), np.int16),
                bomb_placed=np.array([raw['placed']], bool),
                h_joint_actions=np.array([np.array(entry['actions']).reshape(2, 2)], np.int32),
                web_legal=np.array([entry.get('prior_legal', {'mm': [[1]*5]*2})['mm']], bool))
    drops = {d['player']: d for d in entry['drops']}
    for p in range(2):
        enemy = 1-p
        info['death'][0, p] = raw['died'][p]
        if before['trapped'][p] == 0 and after['trapped'][p] > 0:
            causes[p] = [bool(v) for v in raw['causalDamageSource'][p]]
        if raw['died'][p]:
            pop = before['trapped'][p] > 0 and bun['alive'][enemy] and bun['trapped'][enemy] == 0
            pop &= abs(bun['pos'][p*2]-bun['pos'][enemy*2]) < 41/40
            pop &= abs(bun['pos'][p*2+1]-bun['pos'][enemy*2+1]) < 41/40
            source = [q == enemy for q in range(2)] if pop else (causes[p] if before['trapped'][p] else raw['causalDamageSource'][p])
            if not raw['mutualDeath'] and sum(source) == 1:
                info['surviving_kill'][0, enemy] = source[enemy] and bun['alive'][enemy]
                info['own_bomb_defeat'][0, p] = source[p]
            causes[p] = [False, False]
        if p in drops:
            d = drops[p]
            info['drop'][0, p] = True
            info['drop_origin'][0, p] = d['origin']
            info['drop_cells'][0, p] = d['cell']
            if info['surviving_kill'][0, enemy]:
                info['drop_credit'][0, p] = enemy
        info['carrier_death'][0, p] = raw['died'][p] and before['bunCarried'][p] >= 0
        info['native_transfer'][0, p] = (before['bunCarried'][p] < 0 <= bun['bunCarried'][p]
                                       and raw['died'][enemy] and enemy not in drops)
        r, c = np.clip(np.array(bun['pos'][p*2:p*2+2], int), [0, 0], [12, 14])
        i = r*15+c
        if bun['alive'][p] and not bun['trapped'][p]:
            origin = bun['bunCarried'][p]
            if origin >= 0 and bun['bunStored'][p][origin] < after['bunStored'][p][origin]:
                info['capture' if origin != p else 'return_bun'][0, p] = True
            if origin < 0 <= after['bunCarried'][p]:
                picked = after['bunCarried'][p]
                loose = bun['bunLoose'][i*2+picked] > after['bunLoose'][i*2+picked]
                info['pickup_loose'][0, p] = loose
                info['pickup_origin'][0, p] = picked if loose else -1
                info['recover'][0, p] = loose and picked == p
                info['steal'][0, p] = not loose
                info['steal_window_trap'][0, p] = not loose and bun['trapped'][enemy] > 0
                info['steal_window_dead'][0, p] = not loose and (not bun['alive'][enemy] or before['bunRespawn'][enemy] > 0)
        info['blocked_own_route'][0, p] = entry['actions'][p*2+1] == 2 and before['heldItem'][p] == 2 and before['bunCarried'][p] >= 0
        field = before['fieldItem'][i]
        hit = field > 0 and before['fieldArmed'][i] and after['fieldItem'][i] == 0
        hit &= after['movementStatus'][p] != before['movementStatus'][p] and after['movementStatus'][p] in (1, 2)
        info['trap_hit'][0, p] = hit
        info['trap_owner'][0, p] = before['fieldOwner'][i] if hit else -1
        target = env.ROUTE_TARGETS[p]
        k = int(target[0])*15+int(target[1])
        owner_cover = [any(s['owner'] == q and s['covered'][k] for s in entry['explosions']) for q in range(2)]
        info['route_success'][0, p] = before['brick'][k] and owner_cover[p] and not owner_cover[enemy]
    return info


@jax.jit
def evaluate_chunk(carry, data):
    def step(carry, data):
        aux, pending, avoidable, chain, moved = carry
        before, after, info = data
        joint = info['h_joint_actions']
        actions = joint.at[:, :, 1].set((joint[:, :, 1] == 1).astype(jnp.int32))
        actions = jnp.where((before.trapped > 0)[:, :, None], jnp.array([4, 0]), actions)
        actions = actions.at[:, :, 1].set(jnp.where(before.bun_carried >= 0, 0, actions[:, :, 1]))
        danger = jax.vmap(safety.analyze_selected_actions)(before, actions)
        tactical = jax.vmap(safety.analyze_tactical_bomb_placements)(before, actions, danger.selected_survivable)
        fresh = (pending <= 0) & ~chain & danger.exposed & ~danger.doomed
        ticks = jnp.maximum(danger.resolution_ticks, 1)
        pending = jnp.where(fresh, ticks, pending)
        refreshed = danger.exposed & (pending > 0) & (danger.resolution_ticks > 0)
        pending = jnp.where(refreshed, jnp.minimum(pending, ticks), pending)
        avoidable = jnp.where((fresh | refreshed) & danger.exposed & ~danger.doomed, danger.avoidable, avoidable)
        chain = chain | fresh
        moved = jnp.where(fresh, False, moved | ((joint[:, :, 0] < 4) & (pending > 0)))
        resolved = (pending > 0) & (pending-1 <= 0) & after.alive & moved
        pending = jnp.maximum(pending-1, 0)
        info = dict(info, danger_safe_resolution=resolved,
                    avoidable_danger_death=info['death'] & avoidable,
                    safe_bomb_placed=info['bomb_placed'] & (info['web_legal'][:, :, :4].sum(-1) >= 2),
                    threat_bomb_placed=info['bomb_placed'] & tactical.safe & tactical.newly_threatens_enemy,
                    tactical_bomb_enemy_safe_moves_before=tactical.enemy_safe_moves_before,
                    tactical_bomb_enemy_safe_moves_after=tactical.enemy_safe_moves_after)
        clear = resolved | info['death']
        avoidable = jnp.where(clear, False, avoidable)
        hazards = jnp.any(after.fuse > 0, (1, 2)) | jnp.any(after.core.blast_linger > 0, (1, 2))
        chain = jnp.where(clear & ~hazards[:, None], False, chain)
        moved = jnp.where(clear, False, moved)
        settled = ~hazards & ~jnp.any(after.trapped > 0, 1)
        total, aux, parts, metrics = reward.combined_reward(before, after, info, aux, settled,
                                                           bridge_config={'window': 120, 'scale': 2})
        terms = jnp.concatenate([parts[..., :6], parts[..., 7:12],
            jnp.stack([metrics[k] for k in COMPONENTS[11:]], -1)], -1)
        return (aux, pending, avoidable, chain, moved), (total, terms, info['own_bomb_defeat'])
    return jax.lax.scan(step, carry, data)


def run(file, verify=False):
    doc = json.loads(file.read_text())
    trace = file.with_suffix('.trace.jsonl')
    frames = [json.loads(line) for line in trace.open()]
    assert len(frames) == len(doc['actions'])+1
    aux = ops.reward_state(1)
    aux['carry_items'] = items.state(1)
    zeros = jnp.zeros((1, 2), bool)
    carry = (aux, jnp.zeros((1, 2), jnp.int16), zeros, zeros, zeros)
    linger = np.zeros((2, 2, 13, 15), np.int8)
    scene={'native':8,'open':0,'carry_home':2}.get(doc['meta'].get('scene'),8)
    converted = [convert(frames[0]['state'], linger,scene)]
    infos, causes = [], [[False, False], [False, False]]
    for i, entry in enumerate(frames[1:]):
        entry['prior_legal'] = frames[i]['legal']
        infos.append(observed(frames[i]['state'], entry, causes))
        linger = np.maximum(linger-1, 0)
        for source in entry['explosions']:
            if source['owner'] in (0, 1):
                for axis, key in enumerate(['horzCovered', 'vertCovered']):
                    mask = np.array(source[key], bool).reshape(13, 15)
                    linger[source['owner'], axis][mask] = 2
        converted.append(convert(entry['state'], linger,scene))
    sparse, totals, self_ticks = [], np.zeros(2), []
    for start in range(0, len(infos), CHUNK):
        n = min(CHUNK, len(infos)-start)
        indices = list(range(start, start+n)) + [start+n-1]*(CHUNK-n)
        before = jax.tree.map(lambda *x: np.stack(x), *[converted[i] for i in indices])
        after = jax.tree.map(lambda *x: np.stack(x), *[converted[i+1] for i in indices])
        info = jax.tree.map(lambda *x: np.stack(x), *[infos[i] for i in indices])
        carry, (reward_values, values, own) = evaluate_chunk(carry, (before, after, info))
        reward_values, values, own = map(np.asarray, (reward_values, values, own))
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values[:n].sum(-1), reward_values[:n], atol=2e-5,
                                   err_msg='production reward component sum')
        for j in range(n):
            tick = start+j+1
            row = [{k: float(v) for k, v in zip(COMPONENTS, values[j, 0, p]) if v != 0}
                   for p in range(2)]
            if row[0] or row[1]:
                sparse.append([tick, *row])
            totals += values[j, 0].sum(-1)
            if own[j, 0, 0]:
                self_ticks.append(tick)
    if verify:
        assert sparse == doc['rewards'], 'production reward rerun mismatch'
        np.testing.assert_allclose(totals, doc['summary']['reward'], atol=1e-5)
        print(file.stem, 'reward rerun exact', flush=True)
        return
    doc['rewards'] = sparse
    doc['summary']['reward'] = totals.tolist()
    for tick in self_ticks:
        if not any(e['kind'] == 'self_bomb' and e['tick'] == tick for e in doc['events']):
            doc['events'].append(dict(tick=tick, kind='self_bomb', player=0))
    if self_ticks and 'self_bomb' not in doc['summary']['failures']:
        doc['summary']['failures'].append('self_bomb')
    doc['events'].sort(key=lambda e: e['tick'])
    source_names = ['scripts/je09_stage_h_reward.py', 'scripts/je09_stage_h_items.py',
                    'scripts/je09_stage_f_reward.py', 'scripts/je09_bun_bridge.py',
                    'scripts/je04_bun_ops.py', 'scripts/je05_bun_transport.py',
                    'scripts/je07_bun_curriculum.py', 'scripts/je08_bun_potential.py',
                    'jax_bomb/bun_safety.py', 'jax_bomb/je07_bun_env.py']
    doc['meta']['reward_source'] = os.environ.get('REPLAY_STAGE', 'h7').upper() + ' sealed production ledger on observed Web transitions; CPU safety detectors; not training episode telemetry'
    doc['meta']['reward_hashes'] = {name: hashlib.sha256((AUTHORITY/name).read_bytes()).hexdigest() for name in source_names}
    doc['meta']['reward_hashes']['scripts/diagnostic_reward.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    doc['meta']['reward_bridge'] = {'window': 120, 'scale': 2}
    doc['meta']['reward_components'] = list(COMPONENTS)
    file.write_text(json.dumps(doc, separators=(',', ':'), ensure_ascii=False))
    print(file.stem, 'reward', totals.tolist(), 'nonzero ticks', len(sparse), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for file in args.files:
        run(file, args.verify)
