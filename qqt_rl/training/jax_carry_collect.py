"""Device-resident full Bun rollouts against fixed, independently trained actors."""
from functools import partial

import jax
import jax.numpy as j
import numpy as np

from scripts.r4_k4_observation import actor_critic_forward, forward
from qqt_rl.training.kill_ppo import masked_logits, gae
from qqt_rl.training import jax_carry_env as E
from qqt_rl.training.carry_league import opponent_counts


def allocation(count, stage, interceptor_fraction=.75, defenders=1):
    counts = opponent_counts(count, stage, interceptor_fraction, defenders)
    kinds = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    seats = np.arange(count, dtype=np.int32) % 2
    permissions = np.ones((count, 2), bool)
    if stage == 'interceptor':
        permissions[np.arange(count), seats] = False
    else:
        lanes = np.flatnonzero(kinds < defenders)
        permissions[lanes, 1 - seats[lanes]] = False
    return seats, kinds, permissions


def select(mask, new, old):
    return jax.tree.map(lambda a, b: j.where(mask.reshape(mask.shape +
                        (1,) * (a.ndim - mask.ndim)), a, b), new, old)


def reset_done(world, done, key):
    # Delivery can end each game independently. Reset on device and only select
    # finished lanes, retaining their fixed role and opponent for the next game.
    def reset(_):
        fresh = jax.vmap(E.fresh)(jax.random.split(key, done.size))
        return select(done, fresh, world)
    return jax.lax.cond(j.any(done), reset, lambda _: world, None)


def consistency_check(devices, minibatch):
    """Before the first PPO update, compare rollout and separate inference."""
    @partial(jax.pmap, devices=devices)
    def check(params, batch):
        order = j.arange(batch['value'].size).reshape(-1, minibatch)
        def body(_, indices):
            b = jax.tree.map(lambda x: x[indices], batch)
            mv, ab, value, _ = actor_critic_forward(params, b['obs'], b['state'])
            lm, la = masked_logits(mv, ab, b['mm'], b['bm'])
            rows = j.arange(minibatch)
            logp = lm[rows, b['action'][:, 0]] + la[rows, b['action'][:, 1]]
            return None, j.stack([j.max(j.abs(logp - b['logp'])),
                                  j.max(j.abs(value - b['value']))])
        _, errors = jax.lax.scan(body, None, order)
        return errors.max(0)
    return check


def build(devices, count, ticks, *, stage, interceptor_fraction=.75, defenders=1):
    """Separate inference and simulation executables; tensors remain on device.

    A Python loop dispatches ticks asynchronously. It never reads actions,
    states or rewards back to the host. Keeping native simulation out of the
    model executable allows inference to be verified independently.
    """
    E.prepare()
    seats, kinds, permissions = allocation(count, stage, interceptor_fraction, defenders)
    groups = [np.flatnonzero(kinds == i) for i in range(defenders + 1)]
    ids, seats, permissions = j.arange(count), j.asarray(seats), j.asarray(permissions)

    @partial(jax.pmap, devices=devices)
    def init(key):
        return jax.vmap(E.fresh)(jax.random.split(key, count))

    @partial(jax.pmap, devices=devices)
    def observe(world):
        return E.inputs(world, seats), E.inputs(world, 1-seats)

    @partial(jax.pmap, devices=devices)
    def predict(p, obs, gv):
        return actor_critic_forward(p, obs, gv)[:3]

    @partial(jax.pmap, devices=devices)
    def predict_enemy(actor, obs, gv):
        return forward(actor, obs, gv)[:2]

    @partial(jax.pmap, devices=devices)
    def sample(mv, ab, mm, bm, key):
        key, km, ka = jax.random.split(key, 3)
        lm, la = masked_logits(mv, ab, mm, bm)
        own = j.stack([jax.random.categorical(km, lm), jax.random.categorical(ka, la)], -1)
        ix = j.arange(len(mv))
        return own, lm[ix, own[:, 0]] + la[ix, own[:, 1]], key, ~(j.isfinite(mv).all() & j.isfinite(ab).all())

    @partial(jax.pmap, devices=devices)
    def join_actions(own, enemy):
        return j.zeros((count, 2, 2), j.int32).at[ids, seats].set(own).at[ids, 1-seats].set(enemy)

    @partial(jax.pmap, devices=devices)
    def empty_actions(own):
        return j.zeros_like(own)

    merge_bad_logits = jax.pmap(j.logical_or, devices=devices)

    @partial(jax.pmap, devices=devices)
    def stack_rows(*rows):
        # Stack INSIDE pmap. Global jnp.stack/gathers on PmapSharding arrays
        # could gather all cards onto the default card before re-sharding.
        return jax.tree.map(lambda *xs: j.stack(xs), *rows)

    def group_functions(kind, indices):
        ix = j.asarray(indices)
        @partial(jax.pmap, devices=devices)
        def gather(bank, enemy_input):
            obs, gv, mm, bm, _ = enemy_input
            return jax.tree.map(lambda x: x[kind], bank), obs[ix], gv[ix], mm[ix], bm[ix]
        @partial(jax.pmap, devices=devices)
        def scatter(enemy, action):
            return enemy.at[ix].set(action)
        return gather, scatter
    selectors = [group_functions(kind, indices) if len(indices) else None
                 for kind, indices in enumerate(groups)]

    @partial(jax.pmap, devices=devices)
    def evolve(world, actions, key):
        key, ke, kr = jax.random.split(key, 3)
        after, reward, done, metrics = jax.vmap(E.step)(
            world, actions, jax.random.split(ke, count), permissions)
        overflow = j.max(after.state.native_items.overflow)
        metric = []
        for indices in groups:
            ix = j.asarray(indices)
            metric.append(j.stack([metrics[ix, seats[ix]].sum(0),
                                   metrics[ix, 1 - seats[ix]].sum(0)]))
        after = reset_done(after, done, kr)
        data = dict(reward=reward[ids, seats], done=done,
                    metric=j.stack(metric), overflow=overflow)
        return after, data, key

    @partial(jax.pmap, devices=devices)
    def finish(data, bootstrap):
        adv, returns = gae(data['reward'], data['value'], data['done'], bootstrap)
        batch = {k: data[k] for k in ('obs', 'state', 'mm', 'bm', 'live', 'action', 'logp',
                                     'reward', 'value')}
        batch.update(adv=adv, returns=returns)
        batch = {k: v.reshape((-1,) + v.shape[2:]) for k, v in batch.items()}
        health = dict(item_overflow=j.max(data['overflow']), nonfinite=sum(
            (~j.isfinite(data[k])).sum() for k in ('value', 'logp', 'reward'))
            + (~j.isfinite(bootstrap)).sum() + data['bad_logits'].sum())
        return batch, data['metric'].sum(0), health

    def collect(p, world, bank, key):
        rows = []
        for _ in range(ticks):
            own_input, enemy_input = observe(world)
            obs, gv, mm, bm, live = own_input
            mv, ab, value = predict(p, obs, gv)
            own, logp, key, bad_logits = sample(mv, ab, mm, bm, key)
            enemy = empty_actions(own)
            for kind, indices in enumerate(groups):
                if not len(indices):
                    continue
                gather, scatter = selectors[kind]
                actor, eo, eg, emm, ebm = gather(bank, enemy_input)
                mv, ab = predict_enemy(actor, eo, eg)
                action, _, key, bad_enemy = sample(mv, ab, emm, ebm, key)
                bad_logits = merge_bad_logits(bad_logits, bad_enemy)
                enemy = scatter(enemy, action)
            actions = join_actions(own, enemy)
            world, transition, key = evolve(world, actions, key)
            inference = dict(obs=obs, state=gv, mm=mm, bm=bm, live=live,
                             action=own, logp=logp, value=value, bad_logits=bad_logits)
            rows.append(dict(inference, **transition))
        # Device arrays are stacked without materializing NumPy/host values.
        data = stack_rows(*rows)
        own_input, _ = observe(world)
        bootstrap = predict(p, own_input[0], own_input[1])[2]
        batch, metrics, health = finish(data, bootstrap)
        return world, key, batch, metrics, health
    return init, collect
