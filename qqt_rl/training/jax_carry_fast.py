"""Carry rollout experiments: fewer dispatches and batched deferred Critic.

Game rules, opponents, rewards and PPO sample counts are unchanged. Select
only after measured device inference/transition checks pass.
"""
from functools import partial

import jax
import jax.numpy as j
import numpy as np

from scripts.r4_k4_observation import forward
from qqt_rl.training import jax_carry_env as E
from qqt_rl.training.jax_carry_collect import allocation, select
from qqt_rl.training.kill_ppo import masked_logits, gae


def fresh_factory():
    """Exact full-map reset: all randomness left after full=1 is spawn choice.

    Keep the historical split/key indices, so resets match leaf-for-leaf, not
    merely in distribution. This opt-in factory rejects curriculum banks.
    """
    E.prepare()
    if E.E._START_STATE_BANK is not None or E.E._SPAWN_PAIRS is not None:
        raise ValueError('Fast reset requires plain full-map spawn sampling')
    template = E.fresh(jax.random.PRNGKey(0))
    spawns = E.E._BUN_SPAWNS

    def fresh(key):
        keys = jax.random.split(key, 13)
        i0 = jax.random.randint(keys[1], (), 0, spawns.shape[1])
        i1 = jax.random.randint(keys[2], (), 0, spawns.shape[1])
        pos = j.stack([spawns[0, i0], spawns[1, i1]]).astype(j.float32) + .5
        st = template.state._replace(core=template.state.core._replace(pos=pos),
                                     bun_spawn_pos=pos)
        return template._replace(state=st)
    return fresh


def build(devices, count, ticks, *, stage, interceptor_fraction=.75,
          fused=False, value_chunk=512, policy_chunk=512, defenders=1):
    E.prepare()
    seats, kinds, permissions = allocation(count, stage, interceptor_fraction, defenders)
    groups = [np.flatnonzero(kinds == i) for i in range(defenders + 1)]
    ids, seats, permissions = j.arange(count), j.asarray(seats), j.asarray(permissions)
    fresh = fresh_factory()
    total = count*ticks
    value_chunk = min(value_chunk, total)
    if total % value_chunk:
        raise ValueError('value_chunk must divide envs*ticks')
    policy_chunk = min(count, policy_chunk)
    if policy_chunk < 1 or count % policy_chunk:
        raise ValueError('policy_chunk must divide envs')

    @partial(jax.pmap, devices=devices)
    def init(key):
        return jax.vmap(fresh)(jax.random.split(key, count))

    def infer(actor, world, bank, key):
        key, km, ka, kb = jax.random.split(key, 4)
        obs, gv, mm, bm, live = E.inputs(world, seats)
        if policy_chunk == count:
            mv, ab, _, _ = forward(actor, obs, gv)
        else:
            # Match PPO's forward batch geometry: this BF16 backend changes
            # rare logits when the inference batch is increased independently.
            def policy(_, inputs):
                return None, forward(actor, inputs[0], inputs[1])[:2]
            _, (mv, ab) = jax.lax.scan(policy, None, (
                obs.reshape((-1, policy_chunk) + obs.shape[1:]),
                gv.reshape((-1, policy_chunk, gv.shape[-1]))))
            mv, ab = mv.reshape((count, -1)), ab.reshape((count, -1))
        bad_logits = ~(j.isfinite(mv).all() & j.isfinite(ab).all())
        lm, la = masked_logits(mv, ab, mm, bm)
        own = j.stack([jax.random.categorical(km, lm), jax.random.categorical(ka, la)], -1)
        logp = lm[ids, own[:, 0]] + la[ids, own[:, 1]]
        eo, eg, emm, ebm, _ = E.inputs(world, 1-seats)
        enemy = j.zeros_like(own)
        for kind, indices in enumerate(groups):
            if not len(indices):
                continue
            ix = j.asarray(indices)
            mv, ab, _, _ = forward(jax.tree.map(lambda x: x[kind], bank), eo[ix], eg[ix])
            bad_logits |= ~(j.isfinite(mv).all() & j.isfinite(ab).all())
            olm, ola = masked_logits(mv, ab, emm[ix], ebm[ix])
            k0, k1 = jax.random.split(jax.random.fold_in(kb, kind))
            enemy = enemy.at[ix].set(j.stack([jax.random.categorical(k0, olm),
                                             jax.random.categorical(k1, ola)], -1))
        actions = j.zeros((count, 2, 2), j.int32).at[ids, seats].set(own).at[ids, 1-seats].set(enemy)
        data = dict(obs=obs, state=gv, mm=mm, bm=bm, live=live,
                    action=own, logp=logp, bad_logits=bad_logits)
        return actions, data, key

    def evolve(world, actions, key):
        key, ke, kr = jax.random.split(key, 3)
        after, reward, done, metrics = jax.vmap(E.step)(
            world, actions, jax.random.split(ke, count), permissions)
        overflow = j.max(after.state.native_items.overflow)
        metric = []
        for indices in groups:
            ix = j.asarray(indices)
            metric.append(j.stack([metrics[ix, seats[ix]].sum(0),
                                   metrics[ix, 1-seats[ix]].sum(0)]))
        def reset(_):
            return select(done, jax.vmap(fresh)(jax.random.split(kr, count)), after)
        after = jax.lax.cond(j.any(done), reset, lambda _: after, None)
        return after, dict(reward=reward[ids, seats], done=done,
                           metric=j.stack(metric), overflow=overflow), key

    def finish(critic, world, data):
        # Critic never affects actions. Evaluate the SAME frozen critic once
        # over larger batches instead of one small forward on every tick.
        obs = data['obs'].reshape((-1, 25, 13, 15))
        gv = data['state'].reshape((-1, 38))
        def values(_, inputs):
            return None, forward(critic, inputs[0], inputs[1])[2]
        _, value = jax.lax.scan(values, None, (
            obs.reshape((-1, value_chunk, 25, 13, 15)),
            gv.reshape((-1, value_chunk, 38))))
        value = value.reshape((ticks, count))
        bo, bg, _, _, _ = E.inputs(world, seats)
        bootstrap = forward(critic, bo, bg)[2]
        adv, returns = gae(data['reward'], value, data['done'], bootstrap)
        batch = {k: data[k] for k in ('obs', 'state', 'mm', 'bm', 'live', 'action', 'logp', 'reward')}
        batch.update(value=value, adv=adv, returns=returns)
        batch = {k: v.reshape((-1,) + v.shape[2:]) for k, v in batch.items()}
        health = dict(item_overflow=j.max(data['overflow']), nonfinite=sum(
            (~j.isfinite(batch[k])).sum() for k in ('value', 'logp', 'reward', 'adv', 'returns'))
            + (~j.isfinite(bootstrap)).sum() + data['bad_logits'].sum())
        return batch, data['metric'].sum(0), health

    if fused:
        @partial(jax.pmap, devices=devices)
        def collect(p, world, bank, key):
            def body(carry, _):
                world, key = carry
                actions, inference, key = infer(p['actor'], world, bank, key)
                world, transition, key = evolve(world, actions, key)
                return (world, key), dict(inference, **transition)
            (world, key), data = jax.lax.scan(body, (world, key), None, length=ticks)
            batch, metrics, health = finish(p['critic'], world, data)
            return world, key, batch, metrics, health
    else:
        infer_device = jax.pmap(infer, devices=devices)
        evolve_device = jax.pmap(evolve, devices=devices)
        finish_device = jax.pmap(finish, devices=devices)
        @partial(jax.pmap, devices=devices)
        def stack_rows(*rows):
            return jax.tree.map(lambda *xs: j.stack(xs), *rows)
        def collect(p, world, bank, key):
            rows = []
            for _ in range(ticks):
                actions, inference, key = infer_device(p['actor'], world, bank, key)
                world, transition, key = evolve_device(world, actions, key)
                rows.append(dict(inference, **transition))
            batch, metrics, health = finish_device(p['critic'], world, stack_rows(*rows))
            return world, key, batch, metrics, health
    return init, collect
