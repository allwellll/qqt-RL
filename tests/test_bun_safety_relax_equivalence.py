"""The shortened escape relaxation must match the full H*W fixed point."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bomb import bun_env
from jax_bomb import bun_safety
from jax_bomb import levels


@pytest.fixture(autouse=True)
def danger_arena_level():
    bun_env.prepare()
    bun_env.configure_training(
        "danger_arena=1", 1, reward_profile="danger_arena",
        tactical_bomb_placement_reward=1.0,
        tactical_bomb_resolution_reward=1.0)
    yield
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


def _random_states(seed: int, envs: int = 64, ticks: int = 48):
    key = jax.random.PRNGKey(seed)
    states = bun_env.init_batch(key, envs)
    step = jax.jit(jax.vmap(lambda s, a, k: bun_env.step(s, a, k)))
    collected = []
    for tick in range(ticks):
        key, move_key, bomb_key, step_key = jax.random.split(key, 4)
        moves = jax.random.randint(move_key, (envs, 2), 0, bun_env.N_MOVES)
        bombs = (jax.random.uniform(bomb_key, (envs, 2)) < 0.3).astype(jnp.int32)
        actions = jnp.stack([moves, bombs], axis=-1)
        states, _ = step(states, actions, jax.random.split(step_key, envs))
        if tick % 6 == 5:
            collected.append((states, actions))
    return collected


def _analyze(states, actions):
    def one(state, action):
        selected = bun_safety.analyze_selected_actions(state, action)
        tactical = bun_safety.analyze_tactical_bomb_placements(
            state, action, selected.selected_survivable)
        full = bun_safety.analyze_actions(state)
        return selected, tactical, full.survivable
    return jax.vmap(one)(states, actions)


def test_short_relaxation_matches_full_fixed_point(monkeypatch):
    assert bun_safety._RELAX_STEPS < bun_env.H * bun_env.W
    checked_bomb_states = 0
    for states, actions in _random_states(20260930):
        probe = actions.at[:, :, 1].set(1)
        short = _analyze(states, probe)
        with monkeypatch.context() as patch:
            patch.setattr(bun_safety, "_RELAX_STEPS", bun_env.H * bun_env.W)
            full = _analyze(states, probe)
        for got, want in zip(jax.tree.leaves(short), jax.tree.leaves(full)):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        checked_bomb_states += int((np.asarray(states.core.fuse) > 0).any(
            axis=(1, 2)).sum())
    assert checked_bomb_states > 100
