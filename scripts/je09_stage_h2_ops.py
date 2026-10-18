"""Reuse H PPO/reward exactly; add passive custody/availability counters."""
import jax.numpy as jnp
from scripts import je09_stage_h_ops as parent
from scripts.je09_stage_h_ops import *

DIAGNOSTICS = ('h2_raw_custody_ticks', 'h2_custody_item_available_ticks',
               'h2_custody_glue_free_enemy_ticks', 'h2_empty_combat_ticks', 'h2_banana_inflight_ticks')
COMPONENTS = parent.COMPONENTS[:-1]+DIAGNOSTICS+parent.COMPONENTS[-1:]
original_reward = parent.combined_reward


def observed_reward(states, after, info, aux, settled, coefficient=1., bridge_config=None):
    reward, new, parts, metrics = original_reward(states, after, info, aux, settled, coefficient, bridge_config)
    enemy = 1-jnp.arange(2)[None]
    ledger = aux['carry_items']['ledger']
    live = states.alive & ~aux['terminal'][:, None]
    custody = ledger['custody'] & live & (states.bun_carried == enemy)
    inventory = jnp.any((states.item_slots > 0) & (states.item_counts > 0), -1)
    glue = jnp.any((states.item_slots == 2) & (states.item_counts > 0), -1)
    free = states.alive[:, ::-1] & (states.trapped[:, ::-1] == 0)
    empty_combat = (states.scene[:, None] == 0) & live & (states.bun_carried < 0)
    passive = jnp.stack([custody, custody & inventory, custody & glue & free,
                         empty_combat, ledger['slide'] & custody], -1).astype(jnp.float32)
    return reward, new, jnp.concatenate([parts[..., :-1], passive, parts[..., -1:]], -1), metrics


parent.COMPONENTS = COMPONENTS
parent.combined_reward = observed_reward
