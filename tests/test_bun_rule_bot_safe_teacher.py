"""End-to-end safe-teacher regression in the real Bun environment.

Runs a short, fixed-seed tactical-bot self-play in danger_arena and asserts the
teacher is simultaneously *safe* (never defeats itself with its own bomb) and
*active* (keeps placing bombs). This guards against two opposite regressions:
an optimistic survival planner that self-bombs, and an over-conservative one
that collapses into never bombing (the passivity failure mode we must avoid).
Small batch/horizon keeps it CPU-friendly.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jax_bomb import bun_env as env
from jax_bomb import levels
from jax_bomb.bun_frozen_opponents import (
    FrozenTacticalOpponent, clear_destructible_bricks)

GAMES = 12
TICKS = 120
SEED = 424242


@pytest.fixture(autouse=True)
def danger_arena_level():
    env.prepare()
    env.configure_training("danger_arena=1", 1, reward_profile="danger_arena")
    yield
    env.configure_training("full=1", env.MAX_HP)
    levels.clear()


@jax.jit
def _step_batch(states, actions, key):
    keys = jax.random.split(key, GAMES)
    return jax.vmap(lambda s, a, r: env.step(
        s, a, r, auto_reset=False, return_info=True))(states, actions, keys)


def _run_selfplay(clear_bricks: bool):
    bot0 = FrozenTacticalOpponent(player_id=0)
    bot1 = FrozenTacticalOpponent(player_id=1)
    key = jax.random.PRNGKey(SEED)
    states = env.init_batch(key, GAMES)
    if clear_bricks:
        states = clear_destructible_bricks(states)

    active = np.ones((GAMES,), bool)
    own_bomb_defeats = 0
    bombs_placed = 0
    for _ in range(TICKS):
        if not active.any():
            break
        a0 = np.asarray(bot0.decide_batch(states), np.int32)
        a1 = np.asarray(bot1.decide_batch(states), np.int32)
        bombs_placed += int(((a0[:, 1] == 1) & active).sum())
        bombs_placed += int(((a1[:, 1] == 1) & active).sum())
        actions = jnp.asarray(np.stack([a0, a1], axis=1), jnp.int32)
        cand, done, info = _step_batch(states, actions, key)
        key, _ = jax.random.split(key)
        obd = np.asarray(info["own_bomb_defeat"])
        newly = np.asarray(done) & active
        own_bomb_defeats += int(obd[newly].sum())
        active &= ~np.asarray(done)
        states = jax.tree.map(
            lambda o, n: jnp.where(
                jnp.asarray(active).reshape((-1,) + (1,) * (o.ndim - 1)), n, o),
            states, cand)
    return own_bomb_defeats, bombs_placed


def test_tactical_teacher_is_safe_and_active_in_combat_arena():
    own_bomb_defeats, bombs_placed = _run_selfplay(clear_bricks=True)
    assert own_bomb_defeats == 0, (
        f"safe-teacher regressed: {own_bomb_defeats} own-bomb defeats")
    assert bombs_placed > 0, (
        "passivity regression: teacher never placed a bomb in combat arena")


def test_tactical_teacher_places_bombs_with_bricks():
    _, bombs_placed = _run_selfplay(clear_bricks=False)
    assert bombs_placed > 0, "teacher must stay active with destructible bricks"
