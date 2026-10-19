"""Training reset probabilities: retired synthetic custody is never sampled."""
import jax.numpy as jnp

INVALID_SCENES = (2, 3, 4, 6)


def scene_weights(weights):
    weights = jnp.asarray(weights).at[jnp.array(INVALID_SCENES)].set(0.)
    total = weights.sum()
    fallback = jnp.zeros_like(weights).at[8].set(1.)
    return jnp.where(total > 0, weights / jnp.maximum(total, 1e-12), fallback)


def lesson_weights(weights):
    # Historical carry-home / carry-return lessons have no acquisition event.
    weights = jnp.asarray(weights).at[jnp.array([1, 11])].set(0.)
    total = weights.sum()
    fallback = jnp.zeros_like(weights).at[0].set(1.)
    return jnp.where(total > 0, weights / jnp.maximum(total, 1e-12), fallback)
