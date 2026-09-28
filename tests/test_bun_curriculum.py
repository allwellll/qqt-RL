from collections import deque

import jax
import jax.numpy as jnp

from jax_bomb import bun_env
from jax_bomb.bun_curriculum import check_stage_gate, combat_gate_metrics


def _idle_actions():
    return jnp.asarray([[4, 0], [4, 0]], jnp.int32)


def _reachable(state, player, target_team=None):
    blocked = state.core.wall | state.core.brick
    start = tuple(int(value) for value in state.core.pos[player])
    team = player if target_team is None else target_team
    anchor = tuple(int(value) for value in bun_env._BUN_BASES[team])
    goals = {
        (row, column)
        for row in range(anchor[0], anchor[0] + 3)
        for column in range(anchor[1], anchor[1] + 3)
        if not bool(blocked[row, column])
    }
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell in goals:
            return True
        for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = cell[0] + drow, cell[1] + dcolumn
            if not (0 <= nxt[0] < bun_env.H and 0 <= nxt[1] < bun_env.W):
                continue
            if nxt in seen or bool(blocked[nxt]):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def _players_connected(state):
    blocked = state.core.wall | state.core.brick
    start = tuple(int(value) for value in state.core.pos[0])
    goal = tuple(int(value) for value in state.core.pos[1])
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = cell[0] + drow, cell[1] + dcolumn
            if not (0 <= nxt[0] < bun_env.H and 0 <= nxt[1] < bun_env.W):
                continue
            if nxt in seen or bool(blocked[nxt]):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def _cell_reachable(state, player, goal):
    blocked = state.core.wall | state.core.brick
    start = tuple(int(value) for value in state.core.pos[player])
    goal = tuple(int(value) for value in goal)
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = cell[0] + drow, cell[1] + dcolumn
            if not (0 <= nxt[0] < bun_env.H and 0 <= nxt[1] < bun_env.W):
                continue
            if nxt in seen or bool(blocked[nxt]):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def _batched(info):
    return {
        key: value[None] if getattr(value, "ndim", 0) > 0 else value[None]
        for key, value in info.items()
    }


def _reward(state, done, info):
    return bun_env.reward_from_events(
        info["dmg"][None], state.alive[None], info["alive"][None],
        info["hp"][None], done[None], info["crate"][None],
        jnp.zeros((1, 2), jnp.bool_), info["walls"][None],
        0.0, 0.0, 0.0, 1.0, rule_info=_batched(info))


def setup_function():
    bun_env.prepare()
    bun_env.configure_training("full=1", 1)


def test_carry_home_reset_is_reachable_without_bombs():
    bun_env.configure_training("carry_home=1", 1)
    for seed in range(32):
        state = bun_env._fresh(jax.random.PRNGKey(seed))
        assert int(state.lesson) == bun_env.LESSON_CARRY_HOME
        assert state.bun_carried.tolist() == [1, 0]
        assert _reachable(state, 0)
        assert _reachable(state, 1)


def test_carry_return_starts_at_enemy_base_and_has_open_home_route():
    bun_env.configure_training("carry_return=1", 1)
    for seed in range(32):
        state = bun_env._fresh(jax.random.PRNGKey(seed))
        assert int(state.lesson) == bun_env.LESSON_CARRY_RETURN
        assert state.bun_carried.tolist() == [1, -1]
        assert int(bun_env._base_team(
            state.core.pos[0].astype(jnp.int32))) == 1
        assert _reachable(state, 0, 0)
        _, ability_mask = bun_env.legal_mask(state)
        assert not bool(ability_mask[:, 1].any())


def test_route_break_reset_has_target_brick_and_escape_route():
    bun_env.configure_training("route_break=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(100))
    assert int(state.lesson) == bun_env.LESSON_ROUTE_BREAK
    assert int(state.core.brick.sum()) == 2
    for player, target in enumerate(bun_env.ROUTE_TARGETS):
        cell = state.core.pos[player].astype(jnp.int32)
        assert int(jnp.abs(cell - target).sum()) == 1
        move_mask, ability_mask = bun_env.legal_mask(state)
        assert bool(ability_mask[player, 1])
        assert int(move_mask[player, :4].sum()) >= 2


def test_route_to_base_requires_same_episode_open_then_enemy_base_entry():
    bun_env.configure_training("route_to_base=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(120))
    assert int(state.lesson) == bun_env.LESSON_ROUTE_TO_BASE
    assert int(state.core.brick.sum()) == 1
    assert bool(state.core.brick[tuple(bun_env.BRIDGE_TARGET.tolist())])
    move_mask, ability_mask = bun_env.legal_mask(state)
    assert bool(ability_mask[0, 1])
    assert move_mask[1].tolist() == [False, False, False, False, True]
    assert ability_mask[1].tolist() == [True, False, False]

    opened = state._replace(core=state.core._replace(
        brick=state.core.brick.at[
            bun_env.BRIDGE_TARGET[0], bun_env.BRIDGE_TARGET[1]].set(False)))
    enemy_base = bun_env._BUN_BASES[1] + 1
    assert _cell_reachable(opened, 0, enemy_base)

    bomb_cell = state.core.pos[0].astype(jnp.int32)
    core = state.core._replace(
        pos=state.core.pos.at[0].set(jnp.asarray([10.5, 6.5])),
        fuse=state.core.fuse.at[bomb_cell[0], bomb_cell[1]].set(1),
        owner=state.core.owner.at[bomb_cell[0], bomb_cell[1]].set(0),
        bomb_blast=state.core.bomb_blast.at[bomb_cell[0], bomb_cell[1]].set(2))
    state = state._replace(core=core)
    state, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(121),
        auto_reset=False, return_info=True)
    assert not bool(done)
    assert bool(info["bridge_route_open"][0])
    _, ability_mask = bun_env.legal_mask(state)
    assert not bool(ability_mask[:, 1].any())

    enemy_base_pos = bun_env._BUN_BASES[1].astype(jnp.float32) + 1.5
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(enemy_base_pos)))
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(122),
        auto_reset=False, return_info=True)
    reward = _reward(state, done, info)
    assert bool(done)
    assert bool(info["bridge_success"][0])
    assert int(info["winner"]) == 0
    assert float(reward[0, 0]) > 0.0


def test_route_to_base_relaxed_unlocks_only_after_crossing_enemy_half():
    bun_env.configure_training("route_to_base_relaxed=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(130))
    assert int(state.lesson) == bun_env.LESSON_ROUTE_TO_BASE_RELAXED

    _, ability_mask = bun_env.legal_mask(state)
    assert bool(ability_mask[0, 1])

    opened = state._replace(
        core=state.core._replace(
            brick=state.core.brick.at[
                bun_env.BRIDGE_TARGET[0], bun_env.BRIDGE_TARGET[1]].set(False)),
        milestones=state.milestones.at[0].set(jnp.uint8(8)),
    )
    _, ability_mask = bun_env.legal_mask(opened)
    assert not bool(ability_mask[:, 1].any())

    crossed = opened._replace(
        core=opened.core._replace(
            pos=opened.core.pos.at[0].set(jnp.asarray([9.5, 9.5]))),
        milestones=opened.milestones.at[0].set(jnp.uint8(9)),
    )
    move_mask, ability_mask = bun_env.legal_mask(crossed)
    assert int(move_mask[0, :4].sum()) >= 2
    assert bool(ability_mask[0, 1])
    assert not bool(ability_mask[1, 1])

    active_bomb = crossed._replace(core=crossed.core._replace(
        fuse=crossed.core.fuse.at[9, 9].set(10),
        owner=crossed.core.owner.at[9, 9].set(0),
        bomb_blast=crossed.core.bomb_blast.at[9, 9].set(2)))
    _, ability_mask = bun_env.legal_mask(active_bomb)
    assert not bool(ability_mask[0, 1])


def test_route_capture_uses_full_context_and_finishes_only_after_delivery():
    bun_env.configure_training("route_capture=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(140))
    assert int(state.lesson) == bun_env.LESSON_ROUTE_CAPTURE
    assert int(state.core.brick.sum()) > 1
    assert bool(state.core.brick[tuple(bun_env.BRIDGE_TARGET.tolist())])
    for cell in bun_env.BRIDGE_CAPTURE_CLEAR_CELLS:
        if not bool(jnp.all(cell == bun_env.BRIDGE_TARGET)):
            assert not bool(state.core.brick[int(cell[0]), int(cell[1])])

    _, ability_mask = bun_env.legal_mask(state)
    assert not bool(ability_mask[0, 1])
    opened = state._replace(
        core=state.core._replace(
            brick=state.core.brick.at[
                bun_env.BRIDGE_TARGET[0], bun_env.BRIDGE_TARGET[1]].set(False)),
        milestones=state.milestones.at[0].set(jnp.uint8(8)),
    )
    assert _reachable(opened, 0, 1)
    enemy_center = bun_env._BUN_BASES[1].astype(jnp.float32) + 1.5
    returning = opened._replace(core=opened.core._replace(
        pos=opened.core.pos.at[0].set(enemy_center)))
    assert _reachable(returning, 0, 0)

    enemy_base_pos = bun_env._BUN_BASES[1].astype(jnp.float32) + 1.5
    at_enemy = opened._replace(core=opened.core._replace(
        pos=opened.core.pos.at[0].set(enemy_base_pos)))
    carrying, done, info = bun_env.step(
        at_enemy, _idle_actions(), jax.random.PRNGKey(141),
        auto_reset=False, return_info=True)
    assert not bool(done)
    assert bool(info["bridge_success"][0])
    assert bool(info["steal"][0])
    assert int(carrying.bun_carried[0]) == 1

    home_pos = bun_env._BUN_BASES[0].astype(jnp.float32) + 1.5
    at_home = carrying._replace(core=carrying.core._replace(
        pos=carrying.core.pos.at[0].set(home_pos)))
    _, done, info = bun_env.step(
        at_home, _idle_actions(), jax.random.PRNGKey(142),
        auto_reset=False, return_info=True)
    reward = _reward(at_home, done, info)
    assert bool(done)
    assert bool(info["capture"][0])
    assert int(info["winner"]) == 0
    assert float(reward[0, 0]) > 0.0


def test_near_steal_and_combat_resets_are_reachable():
    bun_env.configure_training("near_steal=1", 1)
    for seed in range(8):
        state = bun_env._fresh(jax.random.PRNGKey(seed))
        assert _reachable(state, 0, 1)
        assert _reachable(state, 1, 0)

    for curriculum, lesson in (
            ("combat_static=1", bun_env.LESSON_COMBAT_STATIC),
            ("combat_moving=1", bun_env.LESSON_COMBAT_MOVING),
            ("combat_kill=1", bun_env.LESSON_COMBAT_KILL),
            ("combat=1", bun_env.LESSON_COMBAT)):
        bun_env.configure_training(curriculum, 1)
        state = bun_env._fresh(jax.random.PRNGKey(100))
        assert int(state.lesson) == lesson
        assert _players_connected(state)
        move_mask, ability_mask = bun_env.legal_mask(state)
        assert bool(ability_mask[0, 1])
        assert int(move_mask[0, :4].sum()) >= 2

    bun_env.configure_training("combat_static=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(101))
    move_mask, ability_mask = bun_env.legal_mask(state)
    assert move_mask[1].tolist() == [False, False, False, False, True]
    assert ability_mask[1].tolist() == [True, False, False]

    bun_env.configure_training("combat_kill=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(103))
    move_mask, ability_mask = bun_env.legal_mask(state)
    assert int(move_mask[1].sum()) == 1
    assert not bool(move_mask[1, 4])
    assert ability_mask[1].tolist() == [True, False, False]
    assert state.hp.tolist() == [1, 1]

    bun_env.configure_training("combat_moving=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(102))
    move_mask, ability_mask = bun_env.legal_mask(state)
    assert int(move_mask[1].sum()) == 1
    assert not bool(move_mask[1, 4])
    assert ability_mask[1].tolist() == [True, False, False]


def test_objective_lessons_disable_bombs_without_changing_full_rule():
    for curriculum in ("near_steal=1", "carry_home=1"):
        bun_env.configure_training(curriculum, 1)
        state = bun_env._fresh(jax.random.PRNGKey(150))
        _, ability_mask = bun_env.legal_mask(state)
        assert not bool(ability_mask[:, 1].any())

    bun_env.configure_training("full=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(151))
    _, ability_mask = bun_env.legal_mask(state)
    assert ability_mask[:, 1].tolist() == [True, True]


def test_near_steal_requires_event_and_timeout_cannot_win_by_holding():
    bun_env.configure_training("near_steal=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(200))
    enemy_base = bun_env._BUN_BASES[1].astype(jnp.float32) + 1.5
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(enemy_base)))
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(201),
        auto_reset=False, return_info=True)
    reward = _reward(state, done, info)
    assert bool(done)
    assert bool(info["steal"][0])
    assert int(info["winner"]) == 0
    assert float(reward[0, 0]) > 0.0

    state = bun_env._fresh(jax.random.PRNGKey(202))
    state = state._replace(
        core=state.core._replace(t=jnp.asarray(
            bun_env.LESSON_MAX_STEPS[bun_env.LESSON_NEAR_STEAL] - 1,
            jnp.int32)),
        bun_carried=state.bun_carried.at[0].set(1),
        bun_stored=state.bun_stored.at[1, 1].set(0))
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(203),
        auto_reset=False, return_info=True)
    assert bool(done)
    assert int(info["winner"]) == -1


def test_carry_home_capture_ends_lesson_and_has_positive_reward():
    bun_env.configure_training("carry_home=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(250))
    own_base = bun_env._BUN_BASES[0].astype(jnp.float32) + 1.5
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(own_base)))
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(251),
        auto_reset=False, return_info=True)
    reward = _reward(state, done, info)
    assert bool(done)
    assert bool(info["capture"][0])
    assert int(info["winner"]) == 0
    assert float(reward[0, 0]) > 0.0


def test_route_reward_favors_owned_wall_break_and_penalizes_self_kill():
    bun_env.configure_training("route_break=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(300))
    player_cell = state.core.pos[0].astype(jnp.int32)
    core = state.core._replace(
        pos=state.core.pos.at[0].set(jnp.asarray([10.5, 1.5])),
        fuse=state.core.fuse.at[player_cell[0], player_cell[1]].set(1),
        owner=state.core.owner.at[player_cell[0], player_cell[1]].set(0),
        bomb_blast=state.core.bomb_blast.at[player_cell[0], player_cell[1]].set(2))
    state = state._replace(core=core)
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(301),
        auto_reset=False, return_info=True)
    reward = _reward(state, done, info)
    assert int(info["walls_by_owner"][0]) == 1
    assert float(reward[0, 0]) > 0.0
    assert float(reward[0, 1]) <= 0.0


def test_combat_winner_and_reward_use_real_bomb_owner():
    bun_env.configure_training("combat_kill=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(400))
    core = state.core._replace(
        pos=jnp.asarray([[10.5, 4.5], [9.5, 6.5]], jnp.float32),
        hp=jnp.ones((2,), jnp.int32),
        fuse=state.core.fuse.at[9, 5].set(1),
        owner=state.core.owner.at[9, 5].set(0),
        bomb_blast=state.core.bomb_blast.at[9, 5].set(2))
    state = state._replace(core=core)
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(401),
        auto_reset=False, return_info=True)
    reward = _reward(state, done, info)
    assert bool(done)
    assert int(info["winner"]) == 0
    assert bool(info["death_source"][1, 0])
    assert float(reward[0, 0]) > 0.0
    assert float(reward[0, 1]) < 0.0


def test_combat_hit_and_delayed_safe_escape_use_real_owner():
    bun_env.configure_training("combat_static=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(450))
    core = state.core._replace(
        pos=jnp.asarray([[10.5, 6.5], [9.5, 7.5]], jnp.float32),
        hp=jnp.asarray([1, 2], jnp.int32),
        fuse=state.core.fuse.at[9, 5].set(1),
        owner=state.core.owner.at[9, 5].set(0),
        bomb_blast=state.core.bomb_blast.at[9, 5].set(2))
    state = state._replace(core=core)
    _, done, info = bun_env.step(
        state, _idle_actions(), jax.random.PRNGKey(451),
        auto_reset=False, return_info=True)
    reward = _reward(state, done, info)
    no_escape = dict(info)
    no_escape["safe_bomb_escape"] = jnp.zeros((2,), jnp.bool_)
    reward_without_escape = _reward(state, done, no_escape)
    assert bool(done)
    assert bool(info["credited_hit"][0])
    assert not bool(info["credited_kill"][0])
    assert bool(info["damage_source"][1, 0])
    assert bool(info["safe_bomb_escape"][0])
    assert bool(info["own_detonation"][0])
    assert float(reward[0, 0]) > float(reward_without_escape[0, 0])


def test_kill_rush_reset_and_dynamic_ability_mask_follow_phase():
    bun_env.configure_training("kill_rush=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(500))

    assert int(state.lesson) == bun_env.LESSON_KILL_RUSH
    assert _players_connected(state)
    _, ability_mask = bun_env.legal_mask(state)
    assert bool(ability_mask[0, 1])
    assert ability_mask[1].tolist() == [True, False, False]

    enemy_down = state._replace(
        core=state.core._replace(alive=state.core.alive.at[1].set(False)),
        bun_respawn=state.bun_respawn.at[1].set(80))
    _, ability_mask = bun_env.legal_mask(enemy_down)
    assert not bool(ability_mask[0, 1])

    carrying = enemy_down._replace(
        bun_carried=enemy_down.bun_carried.at[0].set(1))
    _, ability_mask = bun_env.legal_mask(carrying)
    assert not bool(ability_mask[0, 1])


def test_kill_window_reward_only_values_objective_during_respawn_window():
    bun_env.configure_training("full=1", 1, kill_window_reward=True)
    base_state = bun_env._fresh(jax.random.PRNGKey(510))
    enemy_base = bun_env._BUN_BASES[1].astype(jnp.float32) + 1.5
    base_state = base_state._replace(core=base_state.core._replace(
        pos=base_state.core.pos.at[0].set(enemy_base),
        brick=jnp.zeros_like(base_state.core.brick)))

    _, done, info = bun_env.step(
        base_state, _idle_actions(), jax.random.PRNGKey(511),
        auto_reset=False, return_info=True)
    no_window_reward = _reward(base_state, done, info)
    assert bool(info["steal"][0])
    assert bool(info["combat_phase"][0])
    assert not bool(info["kill_window_active"][0])

    window_state = base_state._replace(
        core=base_state.core._replace(
            alive=base_state.core.alive.at[1].set(False)),
        bun_respawn=base_state.bun_respawn.at[1].set(60),
        kill_window_ticks=base_state.kill_window_ticks.at[0].set(60))
    _, done, info = bun_env.step(
        window_state, _idle_actions(), jax.random.PRNGKey(512),
        auto_reset=False, return_info=True)
    window_reward = _reward(window_state, done, info)
    assert bool(info["steal"][0])
    assert bool(info["kill_window_active"][0])
    assert int(info["kill_window_remaining"][0]) == 59
    assert float(window_reward[0, 0]) > float(no_window_reward[0, 0]) + 10.0


def test_full_contact_reset_preserves_full_rules_and_is_locally_solvable():
    bun_env.configure_training("full_contact=1", 1, kill_window_reward=True)
    for seed in range(16):
        state = bun_env._fresh(jax.random.PRNGKey(513 + seed))
        cells = jnp.asarray(state.core.pos, jnp.int32)
        assert int(state.lesson) == bun_env.LESSON_FULL_CONTACT
        assert not bool(state.core.wall[cells[:, 0], cells[:, 1]].any())
        assert not bool(state.core.brick[cells[:, 0], cells[:, 1]].any())
        assert bool((~state.core.wall).sum() > 0)
        _, ability_mask = bun_env.legal_mask(state)
        assert bool(ability_mask[0, 1])
        assert int(state.bun_respawn.max()) == 0
        assert state.bun_stored.tolist() == [[1, 0], [0, 1]]


def test_stage_promotion_gates_require_success_and_safety():
    passed, failures = check_stage_gate("route", {
        "target_wall_rate": 0.70, "safe_detonation_ratio": 0.90,
        "self_death_rate": 0.10})
    assert passed and not failures
    passed, failures = check_stage_gate("bridge", {
        "bridge_success_rate": 0.70, "safe_detonation_ratio": 0.90,
        "self_death_rate": 0.10})
    assert passed and not failures
    passed, failures = check_stage_gate("bridge_relaxed", {
        "bridge_success_rate": 0.70, "safe_detonation_ratio": 0.90,
        "self_death_rate": 0.10})
    assert passed and not failures
    passed, failures = check_stage_gate("bridge_capture", {
        "bridge_success_rate": 0.70, "capture_rate": 0.70,
        "safe_detonation_ratio": 0.90, "self_death_rate": 0.10})
    assert passed and not failures
    passed, failures = check_stage_gate("combat", {
        "static_hit_rate": 0.90, "moving_hit_rate": 0.70,
        "causal_kill_rate": 0.60, "safe_detonation_ratio": 0.90,
        "self_death_rate": 0.15})
    assert passed and not failures
    passed, failures = check_stage_gate("objective", {
        "near_steal_rate": 0.85, "carry_capture_rate": 0.70,
        "carry_return_rate": 0.75,
        "self_death_rate": 0.10})
    assert passed and not failures
    passed, failures = check_stage_gate("full", {
        "full_capture_rate": 0.20, "safe_detonation_ratio": 0.90,
        "self_death_rate": 0.15})
    assert passed and not failures

    passed, failures = check_stage_gate("objective", {
        "near_steal_rate": 1.0, "carry_capture_rate": 0.0,
        "carry_return_rate": 1.0,
        "self_death_rate": 0.0})
    assert not passed
    assert "carry_capture_rate" in failures

    passed, failures = check_stage_gate("objective", {
        "near_steal_rate": 1.0, "carry_capture_rate": 1.0,
        "carry_return_rate": 1.0,
        "self_death_rate": 0.25})
    assert not passed
    assert "self_death_rate" in failures

    passed, failures = check_stage_gate("combat", {
        "static_hit_rate": 1.0, "moving_hit_rate": 1.0,
        "causal_kill_rate": 0.0, "safe_detonation_ratio": 1.0,
        "self_death_rate": 0.0})
    assert not passed
    assert "causal_kill_rate" in failures


def test_combat_gate_uses_active_self_play_kill_and_self_kill_rates():
    metrics = combat_gate_metrics(
        {"p0_credited_hit_rate": 1.0, "p0_safe_escape_rate": 1.0,
         "p0_safe_detonation_ratio": 1.0, "p0_self_kill_rate": 0.0},
        {"p0_credited_hit_rate": 0.9, "p0_safe_escape_rate": 0.8,
         "p0_safe_detonation_ratio": 0.8, "p0_self_kill_rate": 0.1},
        {"p0_causal_kill_rate": 0.0, "p0_safe_escape_rate": 0.7,
         "p0_safe_detonation_ratio": 0.7, "p0_self_kill_rate": 0.25},
    )
    assert metrics == {
        "static_hit_rate": 1.0,
        "moving_hit_rate": 0.9,
        "causal_kill_rate": 0.0,
        "safe_detonation_ratio": 0.7,
        "self_death_rate": 0.25,
    }
    passed, failures = check_stage_gate("combat", metrics)
    assert not passed
    assert set(failures) == {
        "causal_kill_rate", "safe_detonation_ratio", "self_death_rate"}
