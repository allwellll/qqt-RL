"""H8 v34 sampler with one strict route-potential reward addition."""
import jax
import jax.numpy as jnp

from scripts import je09_stage_h5c_ops as base_ops
from scripts import je09_stage_h8_ops as h8
from scripts import je09_route_pbrs as pbrs

for _name in dir(h8):
    if not _name.startswith('_'):
        globals()[_name] = getattr(h8, _name)

COMPONENTS = h8.COMPONENTS[:-1] + pbrs.METRICS + h8.COMPONENTS[-1:]
_original_reward = h8.combined_reward
_original_state = h8.reward_state


def reward_state(n):
    aux = _original_state(n)
    aux['route_pbrs'] = pbrs.state(n)
    return aux


def combined_reward(states, next_states, info, aux, settled, coefficient=1., bridge_config=None):
    reward, newaux, parts, metrics = _original_reward(
        states, next_states, info, aux, settled, coefficient, bridge_config)
    if coefficient == 0:
        zeros = jnp.zeros_like(reward)
        metrics = dict(metrics, **{name: zeros for name in pbrs.METRICS})
        parts = jnp.concatenate([parts[..., :-1],
            jnp.stack([metrics[name] for name in pbrs.METRICS], -1), parts[..., -1:]], -1)
        return reward, newaux, parts, metrics
    reward, newaux, parts, route_metrics = pbrs.apply_reward(
        states, next_states, info, aux, reward, newaux, parts, pbrs.LAMBDA)
    return reward, newaux, parts, dict(metrics, **route_metrics)


def collect(*args, **kwargs):
    old_reward = base_ops.combined_reward
    from jax_bomb import je07_bun_env as env
    old_step = env.step

    def step_with_terminal(*step_args, **step_kwargs):
        next_state, done, info = old_step(*step_args, **step_kwargs)
        from scripts import je09_bun_scenes as scenes
        horizon = scenes.HORIZONS[next_state.scene.astype(jnp.int32)]
        terminal = done | jnp.any(info['capture'], axis=-1) | (next_state.t >= horizon)
        return next_state, done, dict(info, h_route_terminal=terminal)

    base_ops.combined_reward = combined_reward
    env.step = step_with_terminal
    try:
        return h8.collect(*args, **kwargs)
    finally:
        env.step = old_step
        base_ops.combined_reward = old_reward
