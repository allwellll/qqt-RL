"""Preserve G combat/return rewards and add only the H carry-item ledger."""
import jax.numpy as jnp
from scripts.je09_stage_f_reward import combined_reward as parent_reward
from scripts import je09_stage_h_items as items
from scripts import je09_bun_scenes as scenes


def combined_reward(states, next_states, info, aux, settled, coefficient=1., bridge_config=None):
    old = {k: v for k, v in aux.items() if k != 'carry_items'}
    reward, new, parts, metrics = parent_reward(states, next_states, info, old, settled, coefficient, bridge_config)
    extra, ledger, observed = items.transition(states, next_states, info, info['h_joint_actions'],
                                               aux['carry_items'], jnp.asarray(scenes.goals), aux['terminal'])
    extra = coefficient*extra
    new['carry_items'] = ledger
    metrics.update(observed)
    reward = reward+extra
    return reward, new, jnp.concatenate([parts[..., :-1],
        jnp.stack([observed[k] for k in items.METRICS], -1), reward[..., None]], -1), metrics
