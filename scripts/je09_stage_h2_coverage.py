"""Initial item opportunities on real-drop lessons; native play is unchanged."""
import jax
import jax.numpy as jnp
from scripts import je09_bun_scenes as scenes
from jax_bomb import je07_bun_env as env

ORIGINAL_INIT = scenes.init_batch


def cover(states, key, fraction=.75):
    rows = jnp.arange(states.pos.shape[0]); seat = rows % 2; enemy = 1-seat
    selection, choice = jax.random.split(key)
    enabled = (states.scene == 9) & jax.random.bernoulli(selection, fraction, rows.shape)
    # Inventory is granted at episode initialization, never at/after pickup.
    # FIFO uses native placement/consumption. Rewards still require real custody.
    slots, counts = states.item_slots, states.item_counts
    for index, item in enumerate([env.ITEM_SLOW_GLUE, env.ITEM_BANANA]):
        slots = slots.at[rows, seat, index].set(jnp.where(enabled, item, slots[rows, seat, index]))
        counts = counts.at[rows, seat, index].set(jnp.where(enabled, 1, counts[rows, seat, index]))
    target = states.pos[rows, enemy].astype(jnp.int32)
    neighbors = jnp.clip(target[:, None]+jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]]),
                         jnp.array([0, 0]), jnp.array([env.H-1, env.W-1]))
    distance = scenes.DISTANCE[1, seat[:, None], neighbors[..., 0], neighbors[..., 1]]
    initial = scenes.DISTANCE[1, seat, target[:, 0], target[:, 1]]
    learner = states.pos[rows, seat].astype(jnp.int32)
    valid = (distance == initial[:, None]-1) & (distance > 0)
    valid &= ~states.wall[rows[:, None], neighbors[..., 0], neighbors[..., 1]]
    valid &= ~states.brick[rows[:, None], neighbors[..., 0], neighbors[..., 1]]
    valid &= jnp.any(neighbors != learner[:, None], -1)
    valid &= states.field_item[rows[:, None], neighbors[..., 0], neighbors[..., 1]] == 0
    chosen = jax.random.categorical(choice, jnp.where(valid, 0., -1e9))
    cell = neighbors[rows, chosen]; active = enabled & valid.any(-1)
    at = (rows, cell[:, 0], cell[:, 1])
    field = states.field_item.at[at].set(jnp.where(active, env.ITEM_BANANA, states.field_item[at]))
    owner = states.field_owner.at[at].set(jnp.where(active, enemy, states.field_owner[at]))
    armed = states.field_armed.at[at].set(jnp.where(active, True, states.field_armed[at]))
    return states._replace(item_slots=slots, item_counts=counts, held_item=slots[:, :, 0],
                           field_item=field, field_owner=owner, field_armed=armed)


def install(fraction=.75):
    def initialize(key, n, course, distance_stage=None):
        original = ORIGINAL_INIT(key, n, course, distance_stage)
        return cover(original, jax.random.fold_in(key, 0x482), fraction)
    scenes.init_batch = initialize
