"""TDD for reward-v2 safe-aggression shaping (danger_arena profile).

Covers the four behaviours the reward rework must guarantee:
  * every safe bomb placement earns a small base reward (activity lift);
  * self-detonation / trade zero-out the base reward (no suicide farming);
  * forced-kill creation (enemy left with zero safe first moves) is rewarded;
  * avoidable-danger-death attribution reflects the *lethal* hazard, not a
    stale avoidable exposure survived earlier in the same danger chain.
"""
import jax
import jax.numpy as jnp
import pytest

from jax_bomb import bun_env
from jax_bomb import bun_safety
from jax_bomb import levels


BASE_BOMB = 0.06
PLACEMENT = 0.6
RESOLUTION = 1.0
FORCED_KILL = 2.0
THREAT = 0.4


@pytest.fixture(autouse=True)
def danger_arena_level():
    bun_env.prepare()
    bun_env.configure_training(
        "danger_arena=1", 1, reward_profile="danger_arena",
        tactical_bomb_placement_reward=PLACEMENT,
        tactical_bomb_resolution_reward=RESOLUTION,
        base_bomb_reward=BASE_BOMB, forced_kill_reward=FORCED_KILL,
        enemy_threat_reward=THREAT)
    yield
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


def _idle():
    return jnp.asarray([[4, 0], [4, 0]], jnp.int32)


def _danger_info(**overrides):
    """A batched danger_arena info template with every event neutralised.

    Built from a real ``bun_env.step`` so every key the reward reads is
    present with the right shape; ``lesson`` is pinned to danger_arena so
    ``reward_from_events`` returns the danger_arena branch.
    """
    state = bun_env._fresh(jax.random.PRNGKey(0))
    _, _, info = bun_env.step(
        state, _idle(), jax.random.PRNGKey(1), auto_reset=False,
        return_info=True)
    batched = {key: value[None] for key, value in info.items()}
    zero2 = jnp.zeros((1, 2), jnp.bool_)
    for key in ("bomb_placed", "safe_bomb_placed", "safe_tactical_bomb_placed",
                "own_bomb_defeat", "surviving_kill", "avoidable_danger_death",
                "danger_safe_resolution", "own_detonation", "safe_bomb_escape",
                "forced_kill_created", "credited_kill", "causal_kill",
                "threat_bomb_placed"):
        if key in batched:
            batched[key] = zero2
    # Default: any bomb a test places had an escape at placement time.
    batched["survivable_bomb_placed"] = overrides.get(
        "bomb_placed", zero2)
    batched["death"] = jnp.zeros((1, 2), jnp.float32)
    batched["damage_source"] = jnp.zeros((1, 2, 2), jnp.bool_)
    batched["tactical_bomb_safe_resolution"] = jnp.zeros((1, 2), jnp.int16)
    batched["mutual_death"] = jnp.zeros((1,), jnp.bool_)
    batched["winner"] = jnp.full((1,), -1, jnp.int32)
    batched["lesson"] = jnp.full((1,), bun_env.LESSON_DANGER_ARENA, jnp.int32)
    batched.update(overrides)
    return batched


def _reward(info):
    zero2f = jnp.zeros((1, 2), jnp.float32)
    return bun_env.reward_from_events(
        zero2f, jnp.ones((1, 2), jnp.bool_), jnp.ones((1, 2), jnp.bool_),
        jnp.ones((1, 2), jnp.float32), jnp.zeros((1,), jnp.bool_), zero2f,
        zero2f, zero2f, 0.0, 0.0, 0.0, 1.0, rule_info=info)


def test_safe_placement_earns_base_bomb_reward():
    info = _danger_info(bomb_placed=jnp.asarray([[True, False]]))
    reward = _reward(info)
    assert float(reward[0, 0]) == pytest.approx(BASE_BOMB, abs=1e-5)
    assert float(reward[0, 1]) == pytest.approx(0.0, abs=1e-5)


def test_self_detonation_zeroes_base_bomb_reward():
    info = _danger_info(
        bomb_placed=jnp.asarray([[True, False]]),
        own_bomb_defeat=jnp.asarray([[True, False]]),
        death=jnp.asarray([[1.0, 0.0]], jnp.float32))
    reward = _reward(info)
    # -2 death -8 self_kill, and NO +base (blocked on own-bomb defeat).
    assert float(reward[0, 0]) == pytest.approx(-10.0, abs=1e-5)


def test_forced_kill_creation_is_rewarded():
    info = _danger_info(
        bomb_placed=jnp.asarray([[True, False]]),
        forced_kill_created=jnp.asarray([[True, False]]))
    reward = _reward(info)
    assert float(reward[0, 0]) == pytest.approx(BASE_BOMB + FORCED_KILL, abs=1e-5)


def test_unsurvivable_placement_earns_no_base_reward():
    # Self-kill lands ~FUSE ticks after placement, so the same-tick block
    # cannot claw back a suicide bomb's base reward; gate it at placement.
    info = _danger_info(
        bomb_placed=jnp.asarray([[True, False]]),
        survivable_bomb_placed=jnp.asarray([[False, False]]))
    reward = _reward(info)
    assert float(reward[0, 0]) == pytest.approx(0.0, abs=1e-5)


def test_enemy_threat_outweighs_base_bomb():
    info = _danger_info(
        bomb_placed=jnp.asarray([[True, False]]),
        safe_tactical_bomb_placed=jnp.asarray([[True, False]]),
        threat_bomb_placed=jnp.asarray([[True, False]]))
    reward = _reward(info)
    assert float(reward[0, 0]) == pytest.approx(
        BASE_BOMB + PLACEMENT + THREAT, abs=1e-5)
    assert THREAT > BASE_BOMB


def test_trade_overrides_all_positive_shaping():
    info = _danger_info(
        bomb_placed=jnp.asarray([[True, True]]),
        forced_kill_created=jnp.asarray([[True, False]]),
        mutual_death=jnp.asarray([True]))
    reward = _reward(info)
    assert float(reward[0, 0]) == pytest.approx(-6.0, abs=1e-5)
    assert float(reward[0, 1]) == pytest.approx(-6.0, abs=1e-5)


# --- event-level: forced-kill detection in the safety analyzer --------------

def _open_state(seed=0):
    state = bun_env._fresh(jax.random.PRNGKey(seed))
    core = state.core._replace(
        wall=jnp.zeros_like(state.core.wall),
        brick=jnp.zeros_like(state.core.brick),
        pushable=jnp.zeros_like(state.core.pushable),
        brick_linger=jnp.zeros_like(state.core.brick_linger),
        fuse=jnp.zeros_like(state.core.fuse),
        owner=jnp.full_like(state.core.owner, -1),
        bomb_blast=jnp.zeros_like(state.core.bomb_blast),
        blast_linger=jnp.zeros_like(state.core.blast_linger),
        pos=jnp.asarray([[6.5, 5.5], [6.5, 6.5]], jnp.float32),
        hp=jnp.ones_like(state.core.hp),
        alive=jnp.ones_like(state.core.alive),
        blast_cap=jnp.asarray([1.0, 1.0], jnp.float32),
        bombs_cap=jnp.asarray([2.0, 2.0], jnp.float32),
        spd_g=jnp.ones_like(state.core.spd_g),
    )
    return state._replace(
        core=core,
        blast_owner_linger=jnp.zeros_like(state.blast_owner_linger),
        move_status=jnp.zeros_like(state.move_status),
        status_ticks=jnp.zeros_like(state.status_ticks),
    )


def _wall(state, cells):
    wall = state.core.wall
    for row, column in cells:
        wall = wall.at[row, column].set(True)
    return state._replace(core=state.core._replace(wall=wall))


def test_boxed_enemy_bomb_creates_forced_kill():
    # Enemy at (6,6) walled on three sides; player0 at (6,5) bombs (blast 1),
    # covering the enemy cell and turning the bomb cell impassable → the enemy
    # has zero surviving first moves after the placement.
    state = _wall(_open_state(7), [(5, 6), (7, 6), (6, 7)])
    actions = jnp.asarray([[4, 1], [4, 0]], jnp.int32)
    analysis = bun_safety.analyze_tactical_bomb_placements(state, actions)
    assert int(analysis.enemy_safe_moves_after[0]) == 0
    assert int(analysis.enemy_safe_moves_before[0]) >= 1
    assert bool(analysis.forces_kill[0])
    assert not bool(analysis.forces_kill[1])


def test_forced_kill_not_credited_on_already_doomed_enemy():
    # Enemy already fully boxed by a live player0 bomb: it has zero safe moves
    # before the new placement, so a second bomb creates nothing new.
    state = _wall(_open_state(9), [(5, 6), (7, 6), (6, 7)])
    core = state.core._replace(
        fuse=state.core.fuse.at[6, 5].set(8),
        owner=state.core.owner.at[6, 5].set(0),
        bomb_blast=state.core.bomb_blast.at[6, 5].set(1),
        pos=jnp.asarray([[4.5, 5.5], [6.5, 6.5]], jnp.float32))
    state = state._replace(core=core)
    actions = jnp.asarray([[4, 1], [4, 0]], jnp.int32)
    analysis = bun_safety.analyze_tactical_bomb_placements(state, actions)
    assert int(analysis.enemy_safe_moves_before[0]) == 0
    assert not bool(analysis.forces_kill[0])


def test_forced_kill_requires_safe_placement():
    # Same enemy box, but the bomb cell is a dead-end for player0 too
    # (walls trap the placer) → placement is not safe → not a forced kill.
    state = _wall(_open_state(8),
                  [(5, 6), (7, 6), (6, 7), (5, 5), (7, 5), (6, 4)])
    actions = jnp.asarray([[4, 1], [4, 0]], jnp.int32)
    analysis = bun_safety.analyze_tactical_bomb_placements(state, actions)
    assert not bool(analysis.forces_kill[0])


# --- avoidable-danger-death attribution (de-sticky-OR) -----------------------

def test_stale_avoidable_flag_cleared_on_safe_choiceful_tick():
    # A prior danger chain left a stale avoidable flag on player0. This tick a
    # long-fuse opponent bomb re-exposes player0 while an escape is still
    # available; idling keeps that escape (current verdict: not avoidable), so
    # the stale flag must be overwritten to False, not sticky-OR'd to True.
    state = _open_state(30)
    core = state.core._replace(
        pos=jnp.asarray([[6.5, 6.5], [10.5, 10.5]], jnp.float32),
        hp=jnp.asarray([1, 1], jnp.int32),
        fuse=state.core.fuse.at[6, 10].set(6),
        owner=state.core.owner.at[6, 10].set(1),
        bomb_blast=state.core.bomb_blast.at[6, 10].set(5),
    )
    state = state._replace(
        core=core,
        danger_pending_avoidable=jnp.asarray([True, False]),
        danger_chain_active=jnp.asarray([True, False]),
        danger_pending_ticks=jnp.asarray([5, 0], jnp.int16),
    )
    next_state, _, info = bun_env.step(
        state, _idle(), jax.random.PRNGKey(31),
        auto_reset=False, return_info=True)
    assert not bool(info["death"][0])
    assert not bool(info["avoidable_danger_death"][0])
    assert not bool(next_state.danger_pending_avoidable[0])
