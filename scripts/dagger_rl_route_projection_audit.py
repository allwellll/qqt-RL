"""Bounded CPU mechanism audit on genuine, already-collected learner states.

No dispatch, optimizer, teacher modification, or new acquisition initialization.
One-step distance ranking is not action Q or a long-horizon safety judgment.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from scripts import combat_item_engine as T, carrier_mechanism_engine as E
from scripts import dagger_rl_oracle as O
from scripts.je10_route_oracle import labels, continuous_distance
from jax_bomb import bun_safety
from scripts.population_run_state import atomic_json, sha, lock
from scripts.terminal_checkpoint_restore import digest


@jax.jit
def evaluate(params, states, aux, keys):
    obs, _, mm, bm = E.inputs(states)
    seat = j.zeros(len(keys), j.int32)
    raw = labels(states, aux, mm, seat, obs[:, 5])
    route = T.route_teacher(states, aux)
    item, _ = O.teacher(states, aux)
    safe = jax.vmap(bun_safety.analyze_actions)(states).survivable[:, 0, :, 0]
    allowed = safe & mm & bm[:, 0, None]
    kind = raw[:, 6].astype(j.int32)
    channel = j.where((kind == 1) | (kind == 4), 0, j.where(kind == 2, 2, 4))
    field = aux['dagger']['distance'][j.arange(len(keys)), channel]
    blocked = aux['dagger']['blocked']
    before = continuous_distance(states.pos[:, 0], field, blocked)
    _, envkeys, factual, _ = E.choose(params, states, aux, keys)
    repeated = lambda tr: jax.tree.map(lambda a: j.repeat(a, 5, axis=0), tr)
    actions = j.repeat(factual, 5, axis=0).at[:, 0, 0].set(j.tile(j.arange(5), len(keys)))
    # Compare route directions at the common ability0 factor. The enemy and
    # environment keys are unchanged across each state's five branches.
    actions = actions.at[:, 0, 1].set(0)
    after, _ = jax.vmap(lambda s, a, k: E.env.step(s, a, k, auto_reset=False))(
        repeated(states), actions, j.repeat(envkeys, 5, axis=0))
    distances = continuous_distance(after.pos[:, 0], j.repeat(field, 5, axis=0),
                                    j.repeat(blocked, 5, axis=0)).reshape(-1, 5)
    return raw, route[:, :, 0], item.any((1, 2)), allowed, before, distances


def run(base):
    base = Path(base)
    unique = lock(base/'train_projection_audit.lock')
    protocol = json.loads((base/'TRAIN_PROJECTION_AUDIT_PROTOCOL.json').read_text())
    assert protocol['source_sha256'] == sha(Path(__file__))
    assert protocol['seed'] == 2026102661 and protocol['max_branch_steps'] == 2880
    T.configure()
    assert jax.default_backend() == 'cpu'
    output = base/'TRAIN_PROJECTION_AUDIT.json'
    result = dict(passed=False, protocol_sha256=sha(base/'TRAIN_PROJECTION_AUDIT_PROTOCOL.json'),
                  rows=[], snapshot_inputs=[], native_branch_steps=0,
                  padded_native_branch_steps=0, optimizer_steps=0, adopted_environment_steps=0,
                  source_scope='First paired TRAIN seed; original saved natural/real-TRAIN-Bank endpoints. '
                  'No synthetic custody, independent-episode claim or capability CI.',
                  scope='Actual native one-step target-distance ranking among the original temporal-safe '
                  'ability0 candidates, with other actions and RNG held fixed. Not Q/GAE or a full teacher proof.')
    for arm in protocol['arms']:
        for u in protocol['updates']:
            cp = base/'formal'/f'{arm}_s{protocol["seed"]}'/f'u{u:03d}'
            receipt = json.loads((cp/'COMMITTED.json').read_text())
            assert sha(cp/'state.pkl') == receipt['state_sha256']
            x = pickle.load((cp/'state.pkl').open('rb'))
            result['snapshot_inputs'].append(dict(checkpoint=str(cp.relative_to(base)), state_sha256=receipt['state_sha256']))
            for stream in ['PPO', 'auxiliary']:
                carry = x if stream == 'PPO' else x['auxiliary']
                flatten = lambda tr: jax.tree.map(lambda a: np.asarray(a).reshape((-1,)+a.shape[2:]), tr)
                states, aux, keys = map(flatten, [carry['states'], carry['aux'], carry['keys']])
                cargo = states.bun_carried[:, 0] == 1
                window = ~states.alive[:, 1] | (states.trapped[:, 1] > 0)
                eligible = states.alive[:, 0] & (states.trapped[:, 0] == 0) & ~aux['terminal'] & (cargo | window)
                source = aux['provenance_source']
                indexes = []
                for origin in [source < 0, source >= 0]:
                    for phase in [~cargo, cargo]:
                        indexes.extend(np.flatnonzero(eligible & origin & phase)[:4].tolist())
                if not indexes:
                    continue
                selected = np.asarray(indexes)
                padded = np.pad(selected, (0, 16-len(selected)), mode='edge')
                take = lambda tr: jax.tree.map(lambda a: a[padded], tr)
                ss, aa, kk = map(take, [states, aux, keys])
                raw, teacher, override, allowed, before, after = map(np.asarray, evaluate(x['params'], ss, aa, kk))
                assert np.isfinite(before).all() and np.isfinite(after).all()
                for slot, index in enumerate(selected):
                    valid = allowed[slot]
                    best = float(after[slot, valid].min()) if valid.any() else None
                    accepted = teacher[slot] & valid
                    active = bool(accepted.any() and not override[slot])
                    teacher_best = float(after[slot, accepted].min()) if active else None
                    result['rows'].append(dict(checkpoint=str(cp.relative_to(base)), stream=stream,
                        env_index=int(index), source=int(source[index]), carrying=bool(cargo[index]),
                        native_window=bool(window[index]), kind=int(raw[slot, 6]),
                        state_hash=digest(jax.tree.map(lambda a: a[index:index+1], states)),
                        aux_hash=digest(jax.tree.map(lambda a: a[index:index+1], aux)),
                        position=states.pos[index, 0].tolist(), movement_status=int(states.move_status[index, 0]),
                        inventory=int(states.held_item[index, 0]), raw_route=np.flatnonzero(raw[slot, :5] > 0).tolist(),
                        safe_ability0=np.flatnonzero(valid).tolist(), teacher_ability0=np.flatnonzero(accepted).tolist(),
                        item_overrides_route=bool(override[slot]), route_adoptable=active,
                        distance_before=float(before[slot]), native_distance_after=after[slot].tolist(),
                        best_safe_distance=best, teacher_best_distance=teacher_best,
                        teacher_one_step_regret=teacher_best-best if active else None))
                result['native_branch_steps'] += len(selected)*5
                result['padded_native_branch_steps'] += 80
                atomic_json(output, result)
    assert result['native_branch_steps'] <= protocol['max_branch_steps']
    result.update(passed=True, unix=time.time(), source_sha256=sha(Path(__file__)))
    atomic_json(output, result)
    active = [r for r in result['rows'] if r['route_adoptable']]
    print(json.dumps(dict(passed=True, rows=len(result['rows']), route_adoptable=len(active),
        positive_one_step_regret=sum(r['teacher_one_step_regret'] > 1e-5 for r in active),
        native_branch_steps=result['native_branch_steps'], padded_native_branch_steps=result['padded_native_branch_steps'])), flush=True)


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
