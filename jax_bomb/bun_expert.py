"""Deterministic rule expert for Bun behavior-cloning demonstrations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq

import jax
import jax.numpy as jnp
import numpy as np

from . import bun_env as env
from . import jax_env as base
from .bun_safety import analyze_actions


PHASE_OPEN_HOME = 0
PHASE_OPEN_ROUTE = 1
PHASE_ESCAPE = 2
PHASE_ENTER_BASE = 3
PHASE_RETURN_HOME = 4
PHASE_WAIT_BLAST = 5
PHASE_KILL_COMBAT = 6
PHASE_KILL_RUSH = 7
PHASE_KILL_RETURN = 8
PHASE_NAMES = (
    "open_home_gate", "open_route", "escape", "enter_enemy_base",
    "return_home", "wait_blast", "kill_combat", "kill_window_rush",
    "kill_window_return",
)

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOME_GATE = (2, 3)


_STEP_FN = jax.jit(lambda current, action, key: env.step(
    current, action, key, auto_reset=False, return_info=True))


def _observe(current):
    core = current.core
    blocked = (core.fuse > 0) | core.wall | core.brick
    speed = core.spd_g[0] * env._movement_scale(current)[0]
    candidate_positions = jnp.stack([
        base._steer(core.pos[0], direction, core.alive[0], blocked,
                    speed, core.pushable)
        for direction in range(5)
    ])
    return (
        env.make_obs(current, 0), env.global_vec(current, 0),
        *env.legal_mask(current), candidate_positions,
    )


_OBSERVE_FN = jax.jit(_observe)
_SAFETY_FN = jax.jit(analyze_actions)


@dataclass(frozen=True)
class ExpertFailure:
    reason: str
    tick: int
    cell: tuple[int, int]


def _cell(position) -> tuple[int, int]:
    values = np.floor(np.asarray(position)).astype(np.int32)
    return int(values[0]), int(values[1])


def _base_goals(team: int, wall: np.ndarray) -> set[tuple[int, int]]:
    anchor = np.asarray(env._BUN_BASES[team], dtype=np.int32)
    return {
        (row, column)
        for row in range(int(anchor[0]), int(anchor[0]) + 3)
        for column in range(int(anchor[1]), int(anchor[1]) + 3)
        if not wall[row, column]
    }


def _bfs(start: tuple[int, int], goals: set[tuple[int, int]],
         blocked: np.ndarray) -> list[tuple[int, int]] | None:
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        if current in goals:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]
        for drow, dcolumn in _DIRS:
            nxt = current[0] + drow, current[1] + dcolumn
            if not (0 <= nxt[0] < env.H and 0 <= nxt[1] < env.W):
                continue
            if nxt in previous or blocked[nxt]:
                continue
            previous[nxt] = current
            queue.append(nxt)
    return None


def _route_through_bricks(start: tuple[int, int], goals: set[tuple[int, int]],
                          wall: np.ndarray, brick: np.ndarray
                          ) -> list[tuple[int, int]] | None:
    queue = [(0, start)]
    distance = {start: 0}
    previous = {start: None}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distance[current]:
            continue
        if current in goals:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]
        for drow, dcolumn in _DIRS:
            nxt = current[0] + drow, current[1] + dcolumn
            if not (0 <= nxt[0] < env.H and 0 <= nxt[1] < env.W):
                continue
            if wall[nxt]:
                continue
            candidate = cost + 1 + 8 * int(brick[nxt])
            if candidate >= distance.get(nxt, 1 << 30):
                continue
            distance[nxt] = candidate
            previous[nxt] = current
            heapq.heappush(queue, (candidate, nxt))
    return None


def _blast_footprint(origin: tuple[int, int], radius: int,
                     wall: np.ndarray, brick: np.ndarray
                     ) -> set[tuple[int, int]]:
    covered = {origin}
    for drow, dcolumn in _DIRS:
        for distance in range(1, radius + 1):
            cell = origin[0] + drow * distance, origin[1] + dcolumn * distance
            if not (0 <= cell[0] < env.H and 0 <= cell[1] < env.W):
                break
            if wall[cell]:
                break
            covered.add(cell)
            if brick[cell]:
                break
    return covered


def _move_toward(target_cell: tuple[int, int], move_mask: np.ndarray,
                 candidate_positions: np.ndarray) -> int | None:
    target = np.asarray(target_cell, np.float32) + 0.5
    scores = np.abs(candidate_positions - target).sum(axis=-1)
    scores = np.where(move_mask, scores, np.inf)
    if not np.isfinite(scores).any():
        return None
    return int(np.argmin(scores))


def _bomb_plan(start: tuple[int, int], target: tuple[int, int],
               blocked: np.ndarray, wall: np.ndarray, brick: np.ndarray
               ) -> tuple[list[tuple[int, int]] | None, tuple[int, int] | None]:
    route = _route_through_bricks(start, {target}, wall, brick)
    if not route:
        return None, None
    brick_index = next((index for index, cell in enumerate(route)
                        if brick[cell]), None)
    if brick_index is None or brick_index == 0:
        return None, None
    bomb_cell = route[brick_index - 1]
    return _bfs(start, {bomb_cell}, blocked), bomb_cell


def _ambush_bomb_plan(start: tuple[int, int], enemy: tuple[int, int], radius: int,
                      blocked: np.ndarray, wall: np.ndarray,
                      brick: np.ndarray) -> list[tuple[int, int]] | None:
    candidates = set()
    for drow, dcolumn in _DIRS:
        for distance in range(1, radius + 1):
            cell = enemy[0] + drow * distance, enemy[1] + dcolumn * distance
            if not (0 <= cell[0] < env.H and 0 <= cell[1] < env.W):
                break
            if wall[cell] or brick[cell]:
                break
            if not blocked[cell] or cell == start:
                candidates.add(cell)
    best = None
    for candidate in sorted(candidates):
        path = _bfs(start, {candidate}, blocked)
        if not path:
            continue
        footprint = _blast_footprint(candidate, radius, wall, brick)
        escape_blocked = blocked.copy()
        escape_blocked[candidate] = False
        safe_goals = {
            (row, column)
            for row in range(env.H)
            for column in range(env.W)
            if not escape_blocked[row, column] and (row, column) not in footprint
        }
        if not _bfs(candidate, safe_goals, escape_blocked):
            continue
        if best is None or len(path) < len(best):
            best = path
    return best


def expert_action(state, move_mask: np.ndarray, ability_mask: np.ndarray,
                  candidate_positions: np.ndarray
                  ) -> tuple[np.ndarray, int, ExpertFailure | None]:
    """Return a deterministic P0 action and its high-level phase."""
    core = state.core
    wall = np.asarray(core.wall, dtype=np.bool_)
    brick = np.asarray(core.brick, dtype=np.bool_)
    fuse = np.asarray(core.fuse)
    owner = np.asarray(core.owner)
    blast = np.asarray(core.blast_linger)
    start = _cell(core.pos[0])
    lesson = int(np.asarray(state.lesson))
    carrying = int(np.asarray(state.bun_carried[0])) >= 0
    kill_rush_lesson = lesson == env.LESSON_KILL_RUSH
    enemy_down = (not bool(np.asarray(core.alive[1]))
                  or int(np.asarray(state.bun_respawn[1])) > 0)
    blocked = wall | brick | (fuse > 0) | (blast > 0)
    blocked[start] = False
    move = 4
    ability = 0
    move_target = None

    active_bombs = np.argwhere((fuse > 0) & (owner == 0))
    if len(active_bombs):
        phase = PHASE_KILL_COMBAT if kill_rush_lesson else PHASE_ESCAPE
        bomb_cell = tuple(int(value) for value in active_bombs[0])
        radius = int(np.asarray(core.bomb_blast)[bomb_cell])
        footprint = _blast_footprint(bomb_cell, radius, wall, brick)
        safe_goals = {
            (row, column)
            for row in range(env.H)
            for column in range(env.W)
            if not blocked[row, column] and (row, column) not in footprint
        }
        path = _bfs(start, safe_goals, blocked)
        if not path:
            return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                "no_safe_escape", int(np.asarray(core.t)), start)
        if len(path) > 1:
            move_target = path[1]
    elif blast.any():
        phase = (PHASE_KILL_RUSH if kill_rush_lesson and enemy_down
                 else PHASE_KILL_COMBAT if kill_rush_lesson
                 else PHASE_WAIT_BLAST)
        if blast[start] > 0:
            safe_goals = {
                (row, column)
                for row in range(env.H)
                for column in range(env.W)
                if not blocked[row, column]
            }
            path = _bfs(start, safe_goals, blocked)
            if not path:
                return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                    "trapped_in_blast", int(np.asarray(core.t)), start)
            if len(path) > 1:
                move_target = path[1]
    elif (lesson in (env.LESSON_COMBAT_STATIC, env.LESSON_COMBAT_MOVING,
                     env.LESSON_COMBAT_KILL, env.LESSON_FULL_AMBUSH)
          or (kill_rush_lesson and not enemy_down and not carrying)):
        phase = PHASE_KILL_COMBAT
        radius = int(np.asarray(core.blast_cap[0]))
        if lesson == env.LESSON_FULL_AMBUSH:
            enemy = _cell(core.pos[1])
            path = _ambush_bomb_plan(start, enemy, radius, blocked, wall, brick)
            if not path:
                return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                    "ambush_has_no_safe_attack", int(np.asarray(core.t)), start)
            if len(path) > 1:
                move_target = path[1]
                radius = None
        if radius is None:
            pass
        else:
            footprint = _blast_footprint(start, radius, wall, brick)
            escape_blocked = blocked.copy()
            escape_blocked[start] = False
            safe_goals = {
                (row, column)
                for row in range(env.H)
                for column in range(env.W)
                if not escape_blocked[row, column]
                and (row, column) not in footprint
            }
            if not _bfs(start, safe_goals, escape_blocked):
                return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                    "combat_bomb_has_no_escape", int(np.asarray(core.t)), start)
            ability = 1
    else:
        target_team = 0 if carrying else 1
        goals = _base_goals(target_team, wall)
        path = _bfs(start, goals, blocked)
        forced_target = None
        if lesson == env.LESSON_ROUTE_BREAK:
            route_target = tuple(int(value) for value in np.asarray(
                env.ROUTE_TARGETS[0]))
            if brick[route_target]:
                forced_target = route_target
        elif lesson in (env.LESSON_ROUTE_TO_BASE,
                         env.LESSON_ROUTE_TO_BASE_RELAXED,
                         env.LESSON_ROUTE_CAPTURE):
            bridge_target = tuple(int(value) for value in np.asarray(
                env.BRIDGE_TARGET))
            if brick[bridge_target]:
                forced_target = bridge_target
        elif lesson == env.LESSON_FULL and brick[_HOME_GATE]:
            forced_target = _HOME_GATE
        if path and forced_target is None:
            if kill_rush_lesson:
                phase = PHASE_KILL_RETURN if carrying else PHASE_KILL_RUSH
            else:
                phase = PHASE_RETURN_HOME if carrying else PHASE_ENTER_BASE
            if len(path) > 1:
                move_target = path[1]
        elif carrying:
            return np.asarray([4, 0], np.int32), PHASE_RETURN_HOME, ExpertFailure(
                "no_return_path", int(np.asarray(core.t)), start)
        else:
            phase = (PHASE_OPEN_HOME if forced_target == _HOME_GATE
                     else PHASE_OPEN_ROUTE)
            target = forced_target
            if target is None:
                route = _route_through_bricks(start, goals, wall, brick)
                if not route:
                    return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                        "no_route_to_enemy", int(np.asarray(core.t)), start)
                target = next((cell for cell in route if brick[cell]), None)
                if target is None:
                    return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                        "route_has_no_frontier", int(np.asarray(core.t)), start)
            path, bomb_cell = _bomb_plan(start, target, blocked, wall, brick)
            if not path or bomb_cell is None:
                return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                    "cannot_reach_bomb_cell", int(np.asarray(core.t)), start)
            if len(path) > 1:
                move_target = path[1]
            else:
                center = np.asarray(bomb_cell, np.float32) + 0.5
                center_distance = float(np.abs(
                    np.asarray(core.pos[0]) - center).sum())
                align_move = _move_toward(
                    bomb_cell, move_mask, candidate_positions)
                align_distance = (
                    float(np.abs(candidate_positions[align_move] - center).sum())
                    if align_move is not None else np.inf)
                if (center_distance > 0.18
                        and align_distance + 1e-4 < center_distance):
                    move_target = bomb_cell
                else:
                    radius = int(np.asarray(core.blast_cap[0]))
                    footprint = _blast_footprint(bomb_cell, radius, wall, brick)
                    escape_blocked = blocked.copy()
                    escape_blocked[bomb_cell] = False
                    safe_goals = {
                        (row, column)
                        for row in range(env.H)
                        for column in range(env.W)
                        if not escape_blocked[row, column]
                        and (row, column) not in footprint
                    }
                    if not _bfs(bomb_cell, safe_goals, escape_blocked):
                        return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                            "bomb_has_no_escape", int(np.asarray(core.t)), start)
                    ability = 1

    if move_target is not None:
        selected = _move_toward(move_target, move_mask, candidate_positions)
        if selected is None:
            return np.asarray([4, 0], np.int32), phase, ExpertFailure(
                "planned_move_illegal", int(np.asarray(core.t)), start)
        move = selected
    elif not move_mask[move]:
        return np.asarray([4, 0], np.int32), phase, ExpertFailure(
            "planned_move_illegal", int(np.asarray(core.t)), start)
    if not ability_mask[ability]:
        return np.asarray([move, 0], np.int32), phase, ExpertFailure(
            "planned_ability_illegal", int(np.asarray(core.t)), start)
    return np.asarray([move, ability], np.int32), phase, None


def _contract_satisfied(curriculum: str, metrics: dict[str, object]) -> bool:
    lesson = curriculum.split("=", 1)[0]
    safe = (metrics["own_detonations"] == metrics["safe_detonations"]
            and not metrics["self_death"])
    if lesson == "full":
        return (safe and metrics["owner_walls"] > 0
                and metrics["own_detonations"] > 0
                and metrics["entered_enemy_base"] and metrics["stole"]
                and metrics["crossed_home"] and metrics["captured"])
    if lesson == "route_break":
        return (safe and metrics["owner_walls"] > 0
                and metrics["route_success"])
    if lesson in ("route_to_base", "route_to_base_relaxed"):
        return (safe and metrics["owner_walls"] > 0
                and metrics["bridge_success"])
    if lesson == "route_capture":
        return (safe and metrics["owner_walls"] > 0
                and metrics["entered_enemy_base"] and metrics["stole"]
                and metrics["crossed_home"] and metrics["captured"])
    if lesson == "near_steal":
        return safe and metrics["stole"]
    if lesson in ("carry_home", "carry_return"):
        return safe and metrics["captured"]
    if lesson in ("combat_static", "combat_moving"):
        return (safe and metrics["own_detonations"] > 0
                and metrics["credited_hit"])
    if lesson == "combat_kill":
        return (safe and metrics["own_detonations"] > 0
                and metrics["causal_kill"] and metrics["credited_kill"]
                and not metrics["trade"] and not metrics["own_bomb_defeat"])
    if lesson == "full_ambush":
        return (safe and metrics["own_detonations"] > 0
                and metrics["causal_kill"] and metrics["credited_kill"]
                and not metrics["trade"] and not metrics["own_bomb_defeat"])
    if lesson == "kill_rush":
        return (safe and metrics["causal_kill"] and metrics["credited_kill"]
                and not metrics["trade"] and not metrics["own_bomb_defeat"]
                and metrics["entered_enemy_base"] and metrics["stole"]
                and metrics["crossed_home"] and metrics["captured"]
                and metrics["steal_respawn_remaining"] > 0)
    raise ValueError(f"unsupported rule-expert curriculum: {curriculum}")


def rollout_expert(seed: int, record: bool = True,
                   curriculum: str = "full=1",
                   record_snapshots: bool = False,
                   recovery_perturbations: int = 0) -> dict:
    """Run one scripted episode. Failed episodes never expose demo frames."""
    env.prepare()
    env.configure_training(curriculum, 1)
    state = env._fresh(jax.random.PRNGKey(seed))
    lesson = int(jax.device_get(state.lesson))
    frames = []
    snapshots = []
    owner_walls = 0
    own_detonations = 0
    safe_detonations = 0
    entered_enemy_base = False
    stole = False
    crossed_home = False
    captured = False
    route_success = False
    bridge_success = False
    credited_hit = False
    credited_kill = False
    causal_kill = False
    trigger_kill = False
    trade = False
    own_bomb_defeat = False
    first_kill_tick = None
    first_enemy_base_tick = None
    first_steal_tick = None
    first_cross_tick = None
    first_capture_tick = None
    steal_respawn_remaining = -1
    self_death = False
    failure = None
    perturbations_applied = 0
    key = jax.random.PRNGKey(seed ^ 0x5A17)

    for tick in range(env.MAX_STEPS):
        (obs, global_state, move_masks, ability_masks,
         candidate_positions) = _OBSERVE_FN(state)
        (host_state, obs, global_state, move_masks, ability_masks,
         candidate_positions) = jax.device_get(
            (state, obs, global_state, move_masks, ability_masks,
             candidate_positions))
        action, phase, failure = expert_action(
            host_state, move_masks[0], ability_masks[0], candidate_positions)
        if failure:
            break
        if (perturbations_applied < recovery_perturbations
                and tick == 3 + perturbations_applied * 7):
            analysis = jax.device_get(_SAFETY_FN(state))
            safe = np.argwhere(
                np.asarray(analysis.legal[0]) & ~np.asarray(analysis.avoidable[0]))
            alternatives = [candidate for candidate in safe
                            if tuple(candidate) != tuple(action)]
            if alternatives:
                preferred = [candidate for candidate in alternatives
                             if int(candidate[1]) == (1 if perturbations_applied % 2 else 0)]
                selected = (preferred or alternatives)[seed % len(preferred or alternatives)]
                actions = jnp.asarray([selected, [4, 0]], jnp.int32)
                key, step_key = jax.random.split(key)
                state, done, info = _STEP_FN(state, actions, step_key)
                perturbations_applied += 1
                if bool(jax.device_get(info["death"])[0]):
                    self_death = True
                    failure = ExpertFailure(
                        "recovery_perturbation_death", tick,
                        _cell(jax.device_get(state.core.pos[0])))
                    break
                if bool(done):
                    break
                continue
        if record:
            frames.append({
                "obs": np.rint(np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8),
                "state": np.rint(np.clip(global_state, 0.0, 1.0) * 255.0).astype(np.uint8),
                "move_mask": move_masks[0].astype(np.bool_),
                "ability_mask": ability_masks[0].astype(np.bool_),
                "move_action": np.uint8(action[0]),
                "ability_action": np.uint8(action[1]),
                "phase": np.uint8(phase),
                "lesson": np.uint8(lesson),
                "tick": np.int16(tick),
            })
        if record_snapshots:
            snapshots.append(host_state)
        actions = jnp.asarray([action, [4, 0]], jnp.int32)
        key, step_key = jax.random.split(key)
        state, done, info = _STEP_FN(state, actions, step_key)
        info = jax.device_get(info)
        owner_walls += int(info["walls_by_owner"][0])
        own_detonations += int(info["own_detonation"][0])
        safe_detonations += int(info["safe_bomb_escape"][0])
        entered_enemy_base |= bool(info["enter_enemy_base"][0])
        stole |= bool(info["steal"][0])
        crossed_home |= bool(info["carry_cross_home"][0])
        captured |= bool(info["capture"][0])
        route_success |= bool(info["route_success"][0])
        bridge_success |= bool(info["bridge_success"][0])
        credited_hit |= bool(info["credited_hit"][0])
        credited_kill |= bool(info["credited_kill"][0])
        causal_kill |= bool(info["causal_kill"][0])
        trigger_kill |= bool(info["trigger_kill"][0])
        trade |= bool(info["mutual_death"])
        own_bomb_defeat |= bool(info["own_bomb_defeat"][0])
        if bool(info["causal_kill"][0]) and first_kill_tick is None:
            first_kill_tick = tick
        if (first_kill_tick is not None and bool(info["enter_enemy_base"][0])
                and first_enemy_base_tick is None):
            first_enemy_base_tick = tick
        if bool(info["steal"][0]) and first_steal_tick is None:
            first_steal_tick = tick
            steal_respawn_remaining = int(state.bun_respawn[1])
        if bool(info["carry_cross_home"][0]) and first_cross_tick is None:
            first_cross_tick = tick
        if bool(info["capture"][0]) and first_capture_tick is None:
            first_capture_tick = tick
        if bool(info["death"][0]):
            self_death = True
            failure = ExpertFailure("self_death", tick, _cell(state.core.pos[0]))
            break
        if lesson == env.LESSON_FULL_AMBUSH and bool(info["surviving_kill"][0]):
            break
        if bool(done):
            break
    else:
        failure = ExpertFailure("timeout", env.MAX_STEPS, _cell(state.core.pos[0]))

    metrics = {
        "owner_walls": owner_walls,
        "own_detonations": own_detonations,
        "safe_detonations": safe_detonations,
        "entered_enemy_base": entered_enemy_base,
        "stole": stole,
        "crossed_home": crossed_home,
        "captured": captured,
        "route_success": route_success,
        "bridge_success": bridge_success,
        "credited_hit": credited_hit,
        "credited_kill": credited_kill,
        "causal_kill": causal_kill,
        "trigger_kill": trigger_kill,
        "trade": trade,
        "own_bomb_defeat": own_bomb_defeat,
        "first_kill_tick": first_kill_tick,
        "first_enemy_base_tick": first_enemy_base_tick,
        "first_steal_tick": first_steal_tick,
        "first_cross_tick": first_cross_tick,
        "first_capture_tick": first_capture_tick,
        "steal_respawn_remaining": steal_respawn_remaining,
        "self_death": self_death,
        "recovery_perturbations": perturbations_applied,
    }
    success = failure is None and _contract_satisfied(curriculum, metrics)
    if not success and failure is None:
        failure = ExpertFailure("incomplete_success_contract", int(state.core.t),
                                _cell(state.core.pos[0]))
    return {
        "seed": seed,
        "curriculum": curriculum,
        "success": success,
        "length": len(frames),
        "owner_walls": owner_walls,
        "own_detonations": own_detonations,
        "safe_detonations": safe_detonations,
        "entered_enemy_base": entered_enemy_base,
        "stole": stole,
        "crossed_home": crossed_home,
        "captured": captured,
        "route_success": route_success,
        "bridge_success": bridge_success,
        "credited_hit": credited_hit,
        "credited_kill": credited_kill,
        "causal_kill": causal_kill,
        "trigger_kill": trigger_kill,
        "trade": trade,
        "own_bomb_defeat": own_bomb_defeat,
        "first_kill_tick": first_kill_tick,
        "first_enemy_base_tick": first_enemy_base_tick,
        "first_steal_tick": first_steal_tick,
        "first_cross_tick": first_cross_tick,
        "first_capture_tick": first_capture_tick,
        "steal_respawn_remaining": steal_respawn_remaining,
        "self_death": self_death,
        "recovery_perturbations": perturbations_applied,
        "failure": None if failure is None else {
            "reason": failure.reason, "tick": failure.tick,
            "cell": list(failure.cell),
        },
        "frames": frames if success else [],
        "snapshots": snapshots if success and record_snapshots else [],
    }
