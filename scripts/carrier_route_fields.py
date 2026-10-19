"""Candidate home fields for carriers, which cannot place native bombs.

Only the two home channels change. The two empty-hand/drop planning channels
retain their weighted brick costs exactly. This is not a production hook.
"""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import shortest


def fields(wall,brick,goals):
    """Reuse a single four-channel Bellman search with different traversability.

    Present brick is impassable in home channels. Pushable crate movement and
    future enemy explosions require native checks; no future opening is assumed.
    """
    assert goals.shape[-3]==4,'expected drop0/drop1/home0/home1 channels'
    channel_shape=goals.shape
    home=jnp.arange(4).reshape((1,4,1,1))>=2
    blocked=jnp.broadcast_to(wall[:,None],channel_shape)|(home&brick[:,None])
    weighted_brick=~home&brick[:,None]
    return jax.lax.stop_gradient(shortest(blocked,weighted_brick,goals))
