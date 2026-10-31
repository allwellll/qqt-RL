"""Candidate causal core sampler. Import alone never installs a recipe.

Only within-stratum sampling changes. Stored causal labels and all existing
history/current quotas remain authoritative; current fit is not a new label.
"""
import jax
import jax.numpy as jnp
from scripts import retention_core_replay as core
from scripts import return_error_replay as baseline

_UNIFORM_AGGREGATE = core.aggregate


def selection_weights(bank, target_mass):
    valid = core.eligible(bank)
    group = core.stratum(bank)
    score = jax.lax.stop_gradient(jnp.clip(target_mass, 0., 1.))
    error = jnp.where(valid, baseline.EPSILON + 1. - score, 0.)
    totals = jnp.bincount(group, weights=error, length=4)
    weights = error / jnp.maximum(totals[group], baseline.EPSILON)
    return weights / jnp.maximum(weights.sum(), 1.)


def replace_history(replay, bank, key, history_start, target_mass):
    """Replace the existing alternating history-return slots, never add rows."""
    original, count = core.replace_history(replay, bank, key, history_start)
    valid = core.eligible(bank)
    group = core.stratum(bank)
    score = jax.lax.stop_gradient(jnp.clip(target_mass, 0., 1.))
    low = jnp.full(4, jnp.inf).at[group].min(jnp.where(valid, score, jnp.inf))
    high = jnp.full(4, -jnp.inf).at[group].max(jnp.where(valid, score, -jnp.inf))
    varies = valid.any() & (high > low).any()

    def redraw(_):
        ids = jax.random.choice(key, valid.size, (replay['valid'].size,),
                                p=selection_weights(bank, score))
        returning = (replay['valid'] & (replay['kind'] == 3) & ~replay['preparation']
                     & (jnp.arange(replay['valid'].size) >= history_start))
        slots = returning & (jnp.cumsum(returning) % 2 == 1)
        return {name: jnp.where(slots.reshape((-1,) + (1,) * (value.ndim-1)),
                                bank[name][ids], value)
                for name, value in replay.items()}

    return jax.lax.cond(varies, redraw, lambda _: original, None), count


def aggregate(old, batch, key, generation, actor, offset=48, joint_offset=95):
    """Reuse the unchanged writer; recalculate telemetry for the final batch."""
    updated, replay, stats = _UNIFORM_AGGREGATE(
        old, batch, key, generation, actor, offset, joint_offset)
    bank = old['retention_core']
    replay, count = replace_history(replay, bank, jax.random.fold_in(key, 0xC0AE),
                                   replay['valid'].size//2, baseline.target_mass(actor, bank))
    mass = baseline.target_mass(actor, replay)
    history = (replay['valid'] & (replay['kind'] == 3) & ~replay['preparation']
               & (jnp.arange(replay['valid'].size) >= replay['valid'].size//2))
    slots = history & (jnp.cumsum(history) % 2 == 1) & core.eligible(bank).any()
    stats = dict(stats, core_replaced_history_return=count,
                 core_selected_joint_mass_sum=jnp.where(slots, mass, 0.).sum(),
                 replay_history_return_target_mass_sum=jnp.where(history, mass, 0.).sum(),
                 replay_history_return_error_sum=jnp.where(history, 1.-mass, 0.).sum(),
                 history_fraction=jnp.mean((replay['round'] < generation) & replay['valid']),
                 new_fraction=jnp.mean((replay['round'] == generation) & replay['valid']))
    return updated, replay, stats


def install():
    core.install()
    baseline.aggregate = aggregate
