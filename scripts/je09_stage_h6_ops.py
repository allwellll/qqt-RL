"""H6 sampler: more complete learner-run games with earliest-fragment labels."""
import os

import jax
import jax.numpy as jnp
from scripts import je09_stage_h3_ops as _h3
from scripts import je09_stage_h5c_ops as _parent
from scripts.je09_stage_h5c_ops import *
super_collect = _parent.collect
FRAGMENT_IDS = jnp.arange(1, 5, dtype=jnp.int32)


def configured_verifier_slots():
    return int(os.environ.get('JE09_H6_VERIFIER_SLOTS', '1'))


def select_verifier_slots(eligible, fragment, score, slots_per_fragment=1):
    """Select bounded strict-verifier candidates independently by fragment."""
    if slots_per_fragment < 1:
        raise ValueError('slots_per_fragment must be positive')
    scores = jnp.stack([
        jnp.where(eligible & (fragment == fragment_id), score, -1)
        for fragment_id in range(1, 5)
    ], axis=0)
    _, ids = jax.lax.top_k(scores, slots_per_fragment)
    ids = ids.reshape((-1,))
    selected_fragment = jnp.repeat(FRAGMENT_IDS, slots_per_fragment)
    selected = eligible[ids] & (fragment[ids] == selected_fragment)
    selected_rows = jnp.zeros_like(eligible).at[ids].max(selected)
    return ids, selected, selected_rows


def curriculum_weights(counts, combat_fraction=.40, bridge_fraction=.25):
    """Allocate 35% to native full games, preserving bridge and combat mass."""
    base = _h3._base_curriculum_weights(counts)
    natural = jnp.float32(.35)
    remainder = jnp.maximum(1. - combat_fraction - bridge_fraction, 0.) - natural
    non_full = base.at[0].set(0.).at[8].set(0.).at[9].set(0.)
    non_full = non_full / jnp.maximum(non_full.sum(), 1e-6) * jnp.maximum(remainder, 0.)
    return non_full.at[0].set(combat_fraction).at[8].set(natural).at[9].set(bridge_fraction)


def earliest_fragment(ctx, offset=48):
    """Classify learner-visited objective labels; combat stays PPO-only."""
    dagger = ctx[..., offset:offset + 12]
    active = dagger[..., 7] > 0
    kind = dagger[..., 6].astype(jnp.int32)
    scene = ctx[..., 0] == 8
    delivery = dagger[..., 9] > 0
    # 1=A loose recovery, 2=B stock cage, 3=carry return, 4=cage delivery.
    objective = jnp.where(delivery, 4, jnp.where(kind == 3, 3,
        jnp.where(kind == 2, 2, jnp.where((kind == 1) | (kind == 4), 1, 0))))
    return jnp.where(scene & active, objective, 0).astype(jnp.int32)


def first_fragment_deviations(mismatch, fragment, terminal):
    """Keep the first deviation independently for A, B, return, and delivery."""
    def scan_episode(m, frag, end_flags):
        def step(failed, values):
            is_mismatch, kind, end = values
            idx = jnp.clip(kind - 1, 0, 3)
            first = is_mismatch & (kind > 0) & ~failed[idx]
            failed = failed.at[idx].set(failed[idx] | is_mismatch)
            failed = jnp.where(end, jnp.zeros_like(failed), failed)
            return failed, first
        return jax.lax.scan(step, jnp.zeros((4,), jnp.bool_),
                            (m, frag, end_flags))[1]
    return jax.vmap(scan_episode, in_axes=(1, 1, 1), out_axes=1)(
        mismatch, fragment, terminal)


def collect(*args, **kwargs):
    from scripts import je09_stage_h5c_dagger as dagger
    from scripts import je09_stage_h5c_ops as base_ops
    original_labels = dagger.labels
    original_targeted = base_ops.TARGETED_FRAGMENT_SAMPLING
    base_ops.TARGETED_FRAGMENT_SAMPLING = True

    slots = configured_verifier_slots()

    def verified_labels(states, aux, legal, seat, danger):
        # Explicit bounded joint teacher.
        from scripts.prestock_verified_labels import verified_labels as verify
        return verify(states,aux,legal,seat,danger,original_labels,slots)

    dagger.labels = verified_labels
    try:
        result = super_collect(*args, **kwargs)
    finally:
        dagger.labels = original_labels
        base_ops.TARGETED_FRAGMENT_SAMPLING = original_targeted
    states, aux, key, batch, stats = result
    # Each objective fragment has its own per-episode first-deviation quota.
    # Combat remains on the existing PPO/safety-expert path and cannot starve
    # A, B, return, or delivery queries.
    obs, gv, actions, logp, adv, returns, masks, context = batch
    c = context
    d = c[..., 48:60]
    proposed = d[..., :5]
    eligible_count = d[..., 7] > 0
    proposed_count = d[..., 8] > 0
    accepted_count = d[..., 5] > 0
    move = actions[..., 0].astype(jnp.int32)
    objective_active = d[..., 7] > 0
    from scripts.prestock_fragment_queries import joint_mismatch,first_deviations
    joint=c[...,95:110].reshape(c.shape[:-1]+(5,3))>0
    fragment = earliest_fragment(c)
    mismatch=joint_mismatch(joint,actions,objective_active&(d[...,5]>0))&(fragment>0)
    done = c[..., 47] > 0

    first = first_deviations(mismatch,fragment,done,c[...,110]>0,c[...,111].astype(jnp.int32),c[...,113]>0)
    keep = first
    # The parent aggregation consumes label gate d[..., 5] as its replay-valid
    # bit. Gate it here so only verified first-deviation rows reach Actor loss.
    d = d.at[..., 5].set(d[..., 5] * keep.astype(d.dtype))
    c = c.at[..., 48:60].set(d)
    c = c.at[..., 53].set(keep.astype(c.dtype))
    audit = jnp.stack([eligible_count.astype(c.dtype), proposed_count.astype(c.dtype),
                       accepted_count.astype(c.dtype), keep.astype(c.dtype),
                       (proposed_count & ~accepted_count).astype(c.dtype)], -1)
    accepted_by_fragment = jnp.stack([
        accepted_count & (fragment == i) for i in range(1, 5)
    ], -1).astype(c.dtype)
    audit = jnp.concatenate([audit, accepted_by_fragment], -1)
    context = jnp.concatenate([c, audit], -1)
    batch = (obs, gv, actions, logp, adv, returns, masks, context)
    return states, aux, key, batch, stats
