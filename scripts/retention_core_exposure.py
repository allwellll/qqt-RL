"""R2 candidate: all existing history return slots use the same causal core.

Import alone has no effect. No additional rows, queries, labels or losses.
"""
import jax
import jax.numpy as jnp
from scripts import retention_core_replay as core

_HALF_AGGREGATE = core.aggregate


def replace_history(replay, bank, key, history_start):
    valid = core.eligible(bank)
    group = core.stratum(bank)
    counts = jnp.bincount(group, weights=valid.astype(jnp.float32), length=4)
    weights = valid / jnp.maximum(counts[group], 1)
    weights = jnp.where(weights.sum() > 0, weights / jnp.maximum(weights.sum(), 1),
                        jnp.ones_like(weights) / weights.size)
    ids = jax.random.choice(key, weights.size, (replay['valid'].size,), p=weights)
    slots = (replay['valid'] & (replay['kind'] == 3) & ~replay['preparation']
             & (jnp.arange(replay['valid'].size) >= history_start) & valid.any())
    result = {name: jnp.where(slots.reshape((-1,) + (1,) * (value.ndim-1)),
                             bank[name][ids], value) for name, value in replay.items()}
    return result, slots.sum()


def aggregate(old, batch, key, generation, actor, offset=48, joint_offset=95):
    updated, replay, stats = _HALF_AGGREGATE(old, batch, key, generation, actor, offset, joint_offset)
    # The shared implementation reports mass for alternating slots. In this
    # candidate every eligible history-return slot is core, so use the actual
    # final history-return mass, already computed by that implementation.
    stats = dict(stats, core_selected_joint_mass_sum=jnp.where(
        core.eligible(old['retention_core']).any(),
        stats['replay_history_return_target_mass_sum'], 0.))
    return updated, replay, stats


def install():
    core.install()
    core.replace_history = replace_history
    from scripts import return_error_replay as baseline
    baseline.aggregate = aggregate
