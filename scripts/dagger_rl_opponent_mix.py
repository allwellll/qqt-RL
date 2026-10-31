"""Frozen historical-opponent exposure without touching learner sampling or RNG."""
import jax
import jax.numpy as j


def assign(keys):
    """One reproducible draw per reset; independent of environment split keys."""
    return jax.vmap(lambda key: jax.random.bernoulli(
        jax.random.fold_in(key, 0xE631), .5))(keys)


def enemy_inputs(train, states):
    n = states.t.size
    mm, bm = train.both_masks(states)
    return (train.both_perspectives(states)[n:],
            train.both_states(states)[n:], mm[n:], bm[n:])


def sample_parent(move, ability, mm, bm, keys, terminal):
    bm = bm.at[:, 1:].set(bm[:, 1:] & ~terminal[:, None])
    split = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
    movement = jax.vmap(jax.random.categorical)(
        split[:, 0], j.where(mm, move, -1e9))
    release = jax.vmap(jax.random.categorical)(
        split[:, 1], j.where(bm, ability, -1e9))
    return j.stack([movement, release], -1)


def replace_enemy(actions, parent_actions, selected):
    return actions.at[:, 1].set(j.where(selected[:, None], parent_actions,
                                         actions[:, 1]))
