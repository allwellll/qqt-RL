"""JAX 抢包子训练环境。

爆炸、碰撞、炸砖和属性成长复用 :mod:`jax_bomb.jax_env`，抢包子状态机独立
维护。训练进程只加载官方地图 806，不与普通消灭规则混用。
"""

from __future__ import annotations

import json
import heapq
import os
import pickle
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_env as base
from . import levels

H, W = base.H, base.W
MAX_HP = base.MAX_HP
MAX_STEPS = 2400
N_MOVES = base.N_MOVES
N_BOMB = 3
N_OBS_CH = 24

BUN_RESPAWN_TICKS = 100
BUN_CARRY_SPEED_SCALE = 0.5
TACTICAL_ITEM_FRACTION = 0.30
ITEM_NONE = 0
ITEM_BANANA = 1
ITEM_SLOW_GLUE = 2
STATUS_NONE = 0
STATUS_SLOW = 1
STATUS_SLIDE = 2

LESSON_FULL = 0
LESSON_CARRY_HOME = 1
LESSON_NEAR_STEAL = 2
LESSON_ROUTE_BREAK = 3
LESSON_COMBAT = 4
LESSON_COMBAT_STATIC = 5
LESSON_COMBAT_MOVING = 6
LESSON_COMBAT_KILL = 7
LESSON_ROUTE_TO_BASE = 8
LESSON_ROUTE_TO_BASE_RELAXED = 9
LESSON_ROUTE_CAPTURE = 10
LESSON_CARRY_RETURN = 11
LESSON_KILL_RUSH = 12
LESSON_FULL_CONTACT = 13
LESSON_FULL_AMBUSH = 14
LESSON_DANGER_ARENA = 15
LESSON_NAMES = (
    "full", "carry_home", "near_steal", "route_break", "combat",
    "combat_static", "combat_moving", "combat_kill", "route_to_base",
    "route_to_base_relaxed", "route_capture", "carry_return", "kill_rush",
    "full_contact", "full_ambush", "danger_arena",
)

_CURRICULUM_WEIGHTS = jnp.asarray(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0],
    jnp.float32)
_TRAIN_HP = MAX_HP
_KILL_WINDOW_REWARD = False
_REWARD_PROFILE = "legacy"
_DANGER_ESCAPE_REWARD = 0.75
_AVOIDABLE_DANGER_DEATH_PENALTY = 4.0
_TACTICAL_BOMB_PLACEMENT_REWARD = 0.0
_TACTICAL_BOMB_RESOLUTION_REWARD = 0.0
_BASE_BOMB_REWARD = 0.0
_FORCED_KILL_REWARD = 0.0
_DANGER_TRACKING = False
_START_STATE_BANK = None
_START_STATE_BUCKETS = None
_START_STATE_WEIGHTS = None
START_STATE_BUCKET_NAMES = ("return", "steal", "post_kill", "contact", "full_start")

ROUTE_TARGETS = jnp.asarray([[9, 3], [9, 11]], jnp.int32)
ROUTE_SPAWNS = jnp.asarray([[9, 2], [9, 12]], jnp.int32)
COMBAT_SPAWNS = jnp.asarray([[9, 5], [9, 9]], jnp.int32)
COMBAT_HIT_SPAWNS = jnp.asarray([[9, 5], [9, 7]], jnp.int32)
AMBUSH_SPAWN_PAIRS = jnp.asarray([
    [[9, 4], [9, 8]],
    [[9, 5], [9, 8]],
    [[9, 5], [9, 10]],
    [[9, 6], [9, 10]],
], jnp.int32)
AMBUSH_CLEAR_CELLS = jnp.asarray(
    [[9, column] for column in range(3, 12)]
    + [[10, 4], [10, 5], [10, 6]], jnp.int32)
BRIDGE_TARGET = jnp.asarray([9, 7], jnp.int32)
BRIDGE_SPAWNS = jnp.asarray([[9, 5], [12, 13]], jnp.int32)
BRIDGE_CAPTURE_SPAWNS = jnp.asarray([[4, 0], [4, 14]], jnp.int32)
BRIDGE_CAPTURE_CLEAR_CELLS = jnp.asarray(
    [[9, column] for column in range(2, 13)] + [[2, 3], [2, 11]], jnp.int32)
BRIDGE_RETURN_SPAWNS = jnp.asarray([[2, 10], [4, 14]], jnp.int32)
LESSON_MAX_STEPS = jnp.asarray(
    [2400, 800, 400, 300, 600, 180, 300, 300, 500, 500, 1200, 1000, 600,
     1200, 1200, 300],
    jnp.int32)

_danger_map = base._danger_map


def _default_levels_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "web/assets/maps/levels.json")


def _read_bun_level(path: str) -> tuple[int, dict]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    for index, level in enumerate(data):
        if (level.get("qqt_id") == 806 or level.get("source") == "bun06_8.map"):
            if not level.get("bun") or level.get("native_rule") != 3:
                raise ValueError(f"{path}: 地图 806 缺少抢包子规则标记")
            return index, level
    raise ValueError(f"{path}: 找不到地图 806 / bun06_8.map")


_LEVELS_PATH = os.environ.get("JAXBOMB_BUN_LEVELS", _default_levels_path())
_BUN_INDEX, _BUN_LEVEL = _read_bun_level(_LEVELS_PATH)
_BUN_BASES = jnp.asarray(_BUN_LEVEL.get("bun_bases", [[1, 4], [1, 8]]), jnp.int32)
_BUN_SPAWNS = jnp.asarray(_BUN_LEVEL["bun_spawns"], jnp.int32)
_BUN_INITIAL_STATS = jnp.asarray([
    _BUN_LEVEL["initial_stats"]["bombs"],
    _BUN_LEVEL["initial_stats"]["blast"],
    _BUN_LEVEL["initial_stats"]["speed"],
], jnp.float32)


def _build_route_distance(level: dict) -> jnp.ndarray:
    wall = np.asarray(level["wall"], dtype=np.bool_).reshape(H, W)
    brick = np.asarray(level["brick"], dtype=np.bool_).reshape(H, W)
    result = []
    for anchor in level.get("bun_bases", [[1, 4], [1, 8]]):
        distance = np.full((H, W), np.inf, dtype=np.float32)
        queue = []
        for row in range(anchor[0], anchor[0] + 3):
            for column in range(anchor[1], anchor[1] + 3):
                if not wall[row, column]:
                    distance[row, column] = 0.0
                    heapq.heappush(queue, (0.0, row, column))
        while queue:
            current, row, column = heapq.heappop(queue)
            if current != float(distance[row, column]):
                continue
            enter_cost = 1.0 + 3.0 * float(brick[row, column])
            for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                prev_row, prev_column = row + drow, column + dcolumn
                if not (0 <= prev_row < H and 0 <= prev_column < W):
                    continue
                if wall[prev_row, prev_column]:
                    continue
                candidate = current + enter_cost
                if candidate < distance[prev_row, prev_column]:
                    distance[prev_row, prev_column] = candidate
                    heapq.heappush(queue, (candidate, prev_row, prev_column))
        distance[~np.isfinite(distance)] = 255.0
        result.append(distance)
    return jnp.asarray(np.stack(result), jnp.float32)


def _build_danger_arena_pairs(level: dict) -> jnp.ndarray:
    """Build deterministic close-range spawn pairs without changing map geometry."""
    wall = np.asarray(level["wall"], dtype=np.bool_).reshape(H, W)
    brick = np.asarray(level["brick"], dtype=np.bool_).reshape(H, W)
    opened = ~wall & ~brick
    bases = level.get("bun_bases", [[1, 4], [1, 8]])

    def degree(cell):
        row, column = cell
        return sum(
            0 <= row + drow < H and 0 <= column + dcolumn < W
            and opened[row + drow, column + dcolumn]
            for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    def base_distance(cell):
        row, column = cell
        return min(
            max(anchor[0] - row, 0, row - (anchor[0] + 2))
            + max(anchor[1] - column, 0, column - (anchor[1] + 2))
            for anchor in bases)

    cells = [tuple(map(int, cell)) for cell in np.argwhere(opened)]
    pairs = []
    for start in cells:
        if degree(start) < 2 or base_distance(start) > 8:
            continue
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                target = (current[0] + drow, current[1] + dcolumn)
                if (0 <= target[0] < H and 0 <= target[1] < W
                        and opened[target] and target not in distances):
                    distances[target] = distances[current] + 1
                    queue.append(target)
        for target, distance in distances.items():
            if (start < target and degree(target) >= 2
                    and base_distance(target) <= 8 and 3 <= distance <= 8):
                pairs.append((start, target))
    if not pairs:
        raise ValueError("map 806 has no valid danger-arena spawn pairs")
    return jnp.asarray(pairs, jnp.int32)


_BUN_ROUTE_DISTANCE = _build_route_distance(_BUN_LEVEL)
_DANGER_ARENA_SPAWN_PAIRS = _build_danger_arena_pairs(_BUN_LEVEL)


def configure_training(curriculum: str | None = None,
                       hit_points: int | None = None,
                       kill_window_reward: bool = False,
                       reward_profile: str = "legacy",
                       danger_escape_reward: float = 0.75,
                       avoidable_danger_death_penalty: float = 4.0,
                       tactical_bomb_placement_reward: float = 0.0,
                       tactical_bomb_resolution_reward: float = 0.0,
                       base_bomb_reward: float = 0.0,
                       forced_kill_reward: float = 0.0) -> None:
    """配置 Bun 训练 reset；评估/网页未调用时保持完整地图与原 HP。"""
    global _CURRICULUM_WEIGHTS, _TRAIN_HP, _KILL_WINDOW_REWARD, _REWARD_PROFILE
    global _DANGER_ESCAPE_REWARD, _AVOIDABLE_DANGER_DEATH_PENALTY
    global _TACTICAL_BOMB_PLACEMENT_REWARD, _TACTICAL_BOMB_RESOLUTION_REWARD
    global _BASE_BOMB_REWARD, _FORCED_KILL_REWARD
    global _DANGER_TRACKING
    if curriculum:
        parsed = {name: 0.0 for name in LESSON_NAMES}
        for part in curriculum.split(","):
            name, value = part.strip().split("=", 1)
            if name not in parsed:
                raise ValueError(f"未知 Bun 课程: {name}")
            parsed[name] = float(value)
        weights = np.asarray([parsed[name] for name in LESSON_NAMES], np.float32)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("Bun 课程权重必须非负且总和大于 0")
        _CURRICULUM_WEIGHTS = jnp.asarray(weights / weights.sum(), jnp.float32)
        _DANGER_TRACKING = parsed["danger_arena"] > 0.0
    if hit_points is not None:
        if not 1 <= hit_points <= MAX_HP:
            raise ValueError(f"Bun HP 必须在 1..{MAX_HP} 之间")
        _TRAIN_HP = int(hit_points)
    _KILL_WINDOW_REWARD = bool(kill_window_reward)
    if reward_profile not in (
            "legacy", "auto_sparse", "combat_evolution", "danger_arena"):
        raise ValueError(f"未知 Bun reward profile: {reward_profile}")
    _REWARD_PROFILE = reward_profile
    _DANGER_ESCAPE_REWARD = float(danger_escape_reward)
    _AVOIDABLE_DANGER_DEATH_PENALTY = float(
        avoidable_danger_death_penalty)
    _TACTICAL_BOMB_PLACEMENT_REWARD = float(tactical_bomb_placement_reward)
    _TACTICAL_BOMB_RESOLUTION_REWARD = float(tactical_bomb_resolution_reward)
    _BASE_BOMB_REWARD = float(base_bomb_reward)
    _FORCED_KILL_REWARD = float(forced_kill_reward)


def configure_start_state_curriculum(path: str | None = None,
                                     weights: str | None = None) -> None:
    global _START_STATE_BANK, _START_STATE_BUCKETS, _START_STATE_WEIGHTS
    if not path:
        _START_STATE_BANK = _START_STATE_BUCKETS = _START_STATE_WEIGHTS = None
        return
    with open(path, "rb") as file:
        payload = pickle.load(file)
    states = payload["states"]
    if getattr(states, "danger_pending_ticks", None) is None:
        batch_shape = np.asarray(states.lesson).shape + (2,)
        states = states._replace(
            danger_pending_ticks=np.zeros(batch_shape, np.int16),
            danger_pending_avoidable=np.zeros(batch_shape, np.bool_),
            danger_chain_active=np.zeros(batch_shape, np.bool_),
            danger_moved=np.zeros(batch_shape, np.bool_))
    if getattr(states, "tactical_bomb_marks", None) is None:
        batch_shape = np.asarray(states.lesson).shape
        states = states._replace(tactical_bomb_marks=np.zeros(
            batch_shape + (2, H, W), np.bool_))
    _START_STATE_BANK = jax.tree.map(jnp.asarray, states)
    _START_STATE_BUCKETS = jnp.asarray(payload["bucket_ids"], jnp.int8)
    parsed = {name: 0.0 for name in START_STATE_BUCKET_NAMES}
    for item in (weights or "").split(","):
        if item.strip():
            name, value = item.split("=", 1)
            if name not in parsed:
                raise ValueError(f"unknown start-state bucket: {name}")
            parsed[name] = float(value)
    values = np.asarray([parsed[name] for name in START_STATE_BUCKET_NAMES], np.float32)
    if float(values.sum()) <= 0:
        values[:] = 1.0
    _START_STATE_WEIGHTS = jnp.asarray(values / values.sum(), jnp.float32)


def prepare(path: str | None = None) -> str:
    """在首次 JIT 前激活且仅采样地图 806。"""
    global _LEVELS_PATH, _BUN_INDEX, _BUN_LEVEL
    global _BUN_BASES, _BUN_SPAWNS, _BUN_INITIAL_STATS, _BUN_ROUTE_DISTANCE
    global _DANGER_ARENA_SPAWN_PAIRS

    chosen = path or os.environ.get("JAXBOMB_BUN_LEVELS") or _default_levels_path()
    index, level = _read_bun_level(chosen)
    _LEVELS_PATH, _BUN_INDEX, _BUN_LEVEL = chosen, index, level
    _BUN_BASES = jnp.asarray(level.get("bun_bases", [[1, 4], [1, 8]]), jnp.int32)
    _BUN_SPAWNS = jnp.asarray(level["bun_spawns"], jnp.int32)
    _BUN_INITIAL_STATS = jnp.asarray([
        level["initial_stats"]["bombs"],
        level["initial_stats"]["blast"],
        level["initial_stats"]["speed"],
    ], jnp.float32)
    _BUN_ROUTE_DISTANCE = _build_route_distance(level)
    _DANGER_ARENA_SPAWN_PAIRS = _build_danger_arena_pairs(level)
    levels.set_active(chosen, {str(index): 1.0})
    return chosen


class BunState(NamedTuple):
    core: base.BombState
    bun_stored: jnp.ndarray
    bun_loose: jnp.ndarray
    bun_carried: jnp.ndarray
    bun_respawn: jnp.ndarray
    bun_spawn_pos: jnp.ndarray
    held_item: jnp.ndarray
    tactical_crate: jnp.ndarray
    field_item: jnp.ndarray
    field_owner: jnp.ndarray
    field_armed: jnp.ndarray
    move_status: jnp.ndarray
    status_ticks: jnp.ndarray
    slide_dir: jnp.ndarray
    last_move_dir: jnp.ndarray
    blast_owner_linger: jnp.ndarray
    blast_causal_linger: jnp.ndarray
    blast_trigger_linger: jnp.ndarray
    kill_window_ticks: jnp.ndarray
    milestones: jnp.ndarray
    lesson: jnp.ndarray
    danger_pending_ticks: jnp.ndarray = None
    danger_pending_avoidable: jnp.ndarray = None
    danger_chain_active: jnp.ndarray = None
    danger_moved: jnp.ndarray = None
    tactical_bomb_marks: jnp.ndarray = None

    @property
    def pos(self):
        return self.core.pos

    @property
    def fuse(self):
        return self.core.fuse

    @property
    def wall(self):
        return self.core.wall

    @property
    def brick(self):
        return self.core.brick

    @property
    def bomb_blast(self):
        return self.core.bomb_blast

    @property
    def alive(self):
        return self.core.alive

    @property
    def hp(self):
        return self.core.hp

    @property
    def t(self):
        return self.core.t


def _fresh(key) -> BunState:
    (k_core, k0, k1, k_lesson, k_course0, k_course1,
     k_contact, k_contact_extra, k_ambush, k_danger,
     k_danger_swap) = jax.random.split(key, 11)
    core = base._fresh(k_core)
    i0 = jax.random.randint(k0, (), 0, _BUN_SPAWNS.shape[1])
    i1 = jax.random.randint(k1, (), 0, _BUN_SPAWNS.shape[1])
    spawn_cells = jnp.stack([_BUN_SPAWNS[0, i0], _BUN_SPAWNS[1, i1]])
    spawn_pos = spawn_cells.astype(jnp.float32) + 0.5
    core = core._replace(
        pos=spawn_pos,
        alive=jnp.ones((2,), jnp.bool_),
        hp=jnp.full((2,), _TRAIN_HP, jnp.int32),
        invuln=jnp.zeros((2,), jnp.int32),
        bombs_cap=jnp.full((2,), _BUN_INITIAL_STATS[0], jnp.float32),
        blast_cap=jnp.full((2,), _BUN_INITIAL_STATS[1], jnp.float32),
        spd_g=jnp.full((2,), _BUN_INITIAL_STATS[2], jnp.float32),
        buffs=jnp.zeros((2,), jnp.int8),
        debuffs=jnp.zeros((2,), jnp.int8),
        items=jnp.zeros((2, 4), jnp.int8),
        gametype=jnp.asarray(3, jnp.int8),
        t=jnp.zeros((), jnp.int32),
        level_id=jnp.asarray(_BUN_INDEX, jnp.int32),
    )
    lesson = jax.random.choice(
        k_lesson, len(LESSON_NAMES), p=_CURRICULUM_WEIGHTS).astype(jnp.int8)
    rows = jnp.arange(H)[:, None]
    columns = jnp.arange(W)[None, :]
    near_bases = jnp.zeros((H, W), jnp.bool_)
    for team in range(2):
        center = _BUN_BASES[team] + 1
        near_bases = near_bases | (
            (jnp.abs(rows - center[0]) + jnp.abs(columns - center[1])) <= 4)
    near_brick = jnp.where(
        (lesson == LESSON_NEAR_STEAL) & near_bases & ~core.wall,
        False, core.brick)
    empty_brick = jnp.zeros_like(core.brick)
    route_brick = empty_brick
    for target in ROUTE_TARGETS:
        route_brick = route_brick.at[target[0], target[1]].set(True)
    bridge_brick = empty_brick.at[
        BRIDGE_TARGET[0], BRIDGE_TARGET[1]].set(True)
    bridge_capture_brick = core.brick
    for cell in BRIDGE_CAPTURE_CLEAR_CELLS:
        bridge_capture_brick = bridge_capture_brick.at[cell[0], cell[1]].set(False)
    bridge_capture_brick = bridge_capture_brick.at[
        BRIDGE_TARGET[0], BRIDGE_TARGET[1]].set(True)
    bridge_return_brick = bridge_capture_brick.at[
        BRIDGE_TARGET[0], BRIDGE_TARGET[1]].set(False)
    combat_lesson = ((lesson == LESSON_COMBAT)
                     | (lesson == LESSON_COMBAT_STATIC)
                     | (lesson == LESSON_COMBAT_MOVING)
                     | (lesson == LESSON_COMBAT_KILL))
    course_brick = jnp.where(
        lesson == LESSON_ROUTE_BREAK, route_brick,
        jnp.where(lesson == LESSON_CARRY_RETURN, bridge_return_brick,
                  jnp.where(lesson == LESSON_ROUTE_CAPTURE, bridge_capture_brick,
                  jnp.where((lesson == LESSON_ROUTE_TO_BASE)
                            | (lesson == LESSON_ROUTE_TO_BASE_RELAXED), bridge_brick,
                  jnp.where((lesson == LESSON_CARRY_HOME) | combat_lesson,
                            empty_brick, near_brick)))))
    course_brick = jnp.where(
        lesson == LESSON_KILL_RUSH, bridge_return_brick, course_brick)
    contact_depth = jax.random.randint(k_contact, (), 3, 5)
    contact_columns = jnp.arange(W)
    contact_open = (((contact_columns >= COMBAT_SPAWNS[0, 1] - 1)
                     & (contact_columns <= COMBAT_SPAWNS[0, 1] + contact_depth))
                    | ((contact_columns <= COMBAT_SPAWNS[1, 1] + 1)
                       & (contact_columns >= COMBAT_SPAWNS[1, 1] - contact_depth)))
    contact_brick = core.brick.at[COMBAT_SPAWNS[0, 0]].set(
        jnp.where(contact_open, False, core.brick[COMBAT_SPAWNS[0, 0]]))
    contact_extra_cells = jnp.asarray(
        [[9, 2], [9, 3], [9, 11], [9, 12], [2, 3], [2, 11]], jnp.int32)
    contact_extra_mask = jax.random.bernoulli(
        k_contact_extra, 0.5, (contact_extra_cells.shape[0],))
    contact_brick = contact_brick.at[
        contact_extra_cells[:, 0], contact_extra_cells[:, 1]].set(
            jnp.where(
                contact_extra_mask, False,
                contact_brick[contact_extra_cells[:, 0], contact_extra_cells[:, 1]]))
    course_brick = jnp.where(
        lesson == LESSON_FULL_CONTACT, contact_brick, course_brick)
    ambush_brick = core.brick.at[
        AMBUSH_CLEAR_CELLS[:, 0], AMBUSH_CLEAR_CELLS[:, 1]].set(False)
    course_brick = jnp.where(
        lesson == LESSON_FULL_AMBUSH, ambush_brick, course_brick)
    core = core._replace(brick=course_brick)

    def sample_course_cell(sample_key, team, target_team, near_target):
        target = _BUN_BASES[target_team].astype(jnp.int32) + 1
        distance = jnp.abs(rows - target[0]) + jnp.abs(columns - target[1])
        own_half = jnp.where(team == 0, columns <= W // 2, columns >= W // 2)
        distance_ok = jnp.where(near_target,
                                (distance >= 3) & (distance <= 5),
                                (distance >= 4) & (distance <= 9))
        side_ok = jnp.where(near_target, ~own_half, own_half)
        candidates = ~core.wall & ~course_brick & distance_ok & side_ok
        logits = jnp.where(candidates.reshape(-1), 0.0, -1e9)
        flat = jax.random.categorical(sample_key, logits)
        return jnp.stack([flat // W, flat % W]).astype(jnp.float32) + 0.5

    carry_spawn = jnp.stack([
        sample_course_cell(k_course0, 0, 0, False),
        sample_course_cell(k_course1, 1, 1, False),
    ])
    near_spawn = jnp.stack([
        sample_course_cell(k_course0, 0, 1, True),
        sample_course_cell(k_course1, 1, 0, True),
    ])
    route_spawn = ROUTE_SPAWNS.astype(jnp.float32) + 0.5
    combat_spawn = COMBAT_SPAWNS.astype(jnp.float32) + 0.5
    combat_hit_spawn = COMBAT_HIT_SPAWNS.astype(jnp.float32) + 0.5
    bridge_spawn = BRIDGE_SPAWNS.astype(jnp.float32) + 0.5
    bridge_capture_spawn = BRIDGE_CAPTURE_SPAWNS.astype(jnp.float32) + 0.5
    bridge_return_spawn = BRIDGE_RETURN_SPAWNS.astype(jnp.float32) + 0.5
    ambush_pair = AMBUSH_SPAWN_PAIRS[jax.random.randint(
        k_ambush, (), 0, AMBUSH_SPAWN_PAIRS.shape[0])]
    ambush_spawn = ambush_pair.astype(jnp.float32) + 0.5
    danger_pair = _DANGER_ARENA_SPAWN_PAIRS[jax.random.randint(
        k_danger, (), 0, _DANGER_ARENA_SPAWN_PAIRS.shape[0])]
    danger_pair = jnp.where(
        jax.random.bernoulli(k_danger_swap), danger_pair[::-1], danger_pair)
    danger_spawn = danger_pair.astype(jnp.float32) + 0.5
    combat_hit_lesson = ((lesson == LESSON_COMBAT_STATIC)
                         | (lesson == LESSON_COMBAT_MOVING))
    combat_target_lesson = combat_hit_lesson | (lesson == LESSON_COMBAT_KILL)
    spawn_pos = jnp.where(
        lesson == LESSON_CARRY_HOME, carry_spawn,
        jnp.where(lesson == LESSON_NEAR_STEAL, near_spawn,
                  jnp.where(lesson == LESSON_ROUTE_BREAK, route_spawn,
                            jnp.where(lesson == LESSON_COMBAT,
                                      combat_spawn,
                                      jnp.where(combat_target_lesson,
                                                combat_hit_spawn,
                                                jnp.where(
                                                    (lesson == LESSON_ROUTE_TO_BASE)
                                                    | (lesson == LESSON_ROUTE_TO_BASE_RELAXED),
                                                    bridge_spawn,
                                                    jnp.where(
                                                        lesson == LESSON_ROUTE_CAPTURE,
                                                        bridge_capture_spawn,
                                                        jnp.where(
                                                            lesson == LESSON_CARRY_RETURN,
                                                            bridge_return_spawn,
                                                            jnp.where(
                                                            lesson == LESSON_KILL_RUSH,
                                                            combat_hit_spawn,
                                                            jnp.where(
                                                                lesson == LESSON_FULL_CONTACT,
                                                                combat_spawn,
                                                                jnp.where(
                                                                    lesson == LESSON_FULL_AMBUSH,
                                                                    ambush_spawn,
                                                                    jnp.where(
                                                                        lesson == LESSON_DANGER_ARENA,
                                                                        danger_spawn,
                                                                        spawn_pos))))))))))))
    hit_hp = core.hp.at[1].set(MAX_HP)
    core = core._replace(
        pos=spawn_pos,
        hp=jnp.where(combat_hit_lesson, hit_hp, core.hp),
    )
    carry_lesson = ((lesson == LESSON_CARRY_HOME)
                    | (lesson == LESSON_CARRY_RETURN))
    initial_stored = jnp.where(
        carry_lesson, jnp.zeros((2, 2), jnp.int8),
        jnp.asarray([[1, 0], [0, 1]], jnp.int8))
    initial_carried = jnp.where(
        lesson == LESSON_CARRY_HOME, jnp.asarray([1, 0], jnp.int8),
        jnp.where(lesson == LESSON_CARRY_RETURN,
                  jnp.asarray([1, -1], jnp.int8),
                  jnp.full((2,), -1, jnp.int8)))
    fresh = BunState(
        core=core,
        bun_stored=initial_stored,
        bun_loose=jnp.zeros((H, W, 2), jnp.int8),
        bun_carried=initial_carried,
        bun_respawn=jnp.zeros((2,), jnp.int16),
        bun_spawn_pos=spawn_pos,
        held_item=jnp.zeros((2,), jnp.int8),
        tactical_crate=jnp.zeros((H, W), jnp.int8),
        field_item=jnp.zeros((H, W), jnp.int8),
        field_owner=jnp.full((H, W), -1, jnp.int8),
        field_armed=jnp.zeros((H, W), jnp.bool_),
        move_status=jnp.zeros((2,), jnp.int8),
        status_ticks=jnp.zeros((2,), jnp.int16),
        slide_dir=jnp.asarray([3, 2], jnp.int8),
        last_move_dir=jnp.asarray([3, 2], jnp.int8),
        blast_owner_linger=jnp.zeros((2, 2, H, W), jnp.int8),
        blast_causal_linger=jnp.zeros((2, 2, H, W), jnp.int8),
        blast_trigger_linger=jnp.zeros((2, 2, H, W), jnp.int8),
        kill_window_ticks=jnp.zeros((2,), jnp.int16),
        milestones=jnp.zeros((2,), jnp.uint8),
        lesson=lesson,
        danger_pending_ticks=jnp.zeros((2,), jnp.int16),
        danger_pending_avoidable=jnp.zeros((2,), jnp.bool_),
        danger_chain_active=jnp.zeros((2,), jnp.bool_),
        danger_moved=jnp.zeros((2,), jnp.bool_),
        tactical_bomb_marks=jnp.zeros((2, H, W), jnp.bool_),
    )
    if _START_STATE_BANK is None:
        return fresh
    k_bucket, k_index = jax.random.split(k_contact)
    bucket = jax.random.choice(
        k_bucket, len(START_STATE_BUCKET_NAMES), p=_START_STATE_WEIGHTS)
    eligible = _START_STATE_BUCKETS == bucket
    index = jax.random.categorical(k_index, jnp.where(eligible, 0.0, -1e9))
    sampled = jax.tree.map(lambda value: value[index], _START_STATE_BANK)
    return sampled._replace(
        core=sampled.core._replace(t=jnp.zeros((), jnp.int32)),
        lesson=jnp.asarray(LESSON_FULL, jnp.int8),
        danger_pending_ticks=jnp.zeros((2,), jnp.int16),
        danger_pending_avoidable=jnp.zeros((2,), jnp.bool_),
        danger_chain_active=jnp.zeros((2,), jnp.bool_),
        danger_moved=jnp.zeros((2,), jnp.bool_),
        tactical_bomb_marks=jnp.zeros((2, H, W), jnp.bool_))


def init_batch(key, n: int) -> BunState:
    return jax.vmap(_fresh)(jax.random.split(key, n))


def _base_team(cell: jnp.ndarray) -> jnp.ndarray:
    row, column = cell[0], cell[1]
    team = jnp.asarray(-1, jnp.int32)
    for index in range(2):
        anchor = _BUN_BASES[index]
        inside = ((row >= anchor[0]) & (row < anchor[0] + 3)
                  & (column >= anchor[1]) & (column < anchor[1] + 3))
        team = jnp.where(inside, index, team)
    return team


def _movement_scale(state: BunState) -> jnp.ndarray:
    carry = jnp.where(state.bun_carried >= 0, BUN_CARRY_SPEED_SCALE, 1.0)
    status = jnp.where(state.move_status == STATUS_SLOW, 0.5,
                       jnp.where(state.move_status == STATUS_SLIDE, 1.6, 1.0))
    return carry * status


def legal_mask(state: BunState) -> tuple[jnp.ndarray, jnp.ndarray]:
    move, bomb = base.legal_mask(state.core)
    kill_rush_objective = ((state.lesson == LESSON_KILL_RUSH)
                           & ((state.bun_respawn[::-1] > 0)
                              | (state.bun_carried >= 0)))
    objective_lesson = ((state.lesson == LESSON_NEAR_STEAL)
                        | (state.lesson == LESSON_CARRY_HOME)
                        | (state.lesson == LESSON_CARRY_RETURN)
                        | kill_rush_objective)
    bridge_masked_lesson = state.lesson == LESSON_ROUTE_TO_BASE
    bridge_relaxed_lesson = state.lesson == LESSON_ROUTE_TO_BASE_RELAXED
    bridge_capture_lesson = state.lesson == LESSON_ROUTE_CAPTURE
    bridge_lesson = (bridge_masked_lesson | bridge_relaxed_lesson
                     | bridge_capture_lesson)
    bridge_opened = (state.milestones & jnp.uint8(8)) != 0
    bridge_crossed = (state.milestones & jnp.uint8(1)) != 0
    cells = state.core.pos.astype(jnp.int32)
    bridge_distance = jnp.abs(cells - BRIDGE_TARGET).sum(axis=-1)
    bridge_opening_bomb = ~bridge_opened & (bridge_distance <= 2)
    own_active_bomb = jnp.stack([
        ((state.core.owner == player) & (state.core.fuse > 0)).any()
        for player in range(2)
    ])
    has_escape_choice = move[:, :4].sum(axis=-1) >= 2
    bridge_released_bomb = (bridge_relaxed_lesson & bridge_opened
                            & bridge_crossed & ~own_active_bomb
                            & has_escape_choice)
    bridge_can_bomb = ((jnp.arange(2) == 0)
                       & (bridge_opening_bomb | bridge_released_bomb))
    lesson_allows_bomb = jnp.where(bridge_lesson, bridge_can_bomb,
                                   ~objective_lesson)
    can_place_bomb = (bomb[:, 1] & (state.bun_carried < 0)
                      & lesson_allows_bomb)
    can_use_item = state.core.alive & (state.held_item > 0)
    ability = jnp.stack([
        jnp.ones((2,), jnp.bool_),
        can_place_bomb | ~state.core.alive,
        can_use_item | ~state.core.alive,
    ], axis=-1)
    scripted_target = ((state.lesson == LESSON_COMBAT_STATIC)
                       | (state.lesson == LESSON_COMBAT_MOVING)
                       | (state.lesson == LESSON_COMBAT_KILL)
                       | (state.lesson == LESSON_KILL_RUSH)
                       | bridge_lesson)
    moving_right = ((state.core.t // 8) % 2) == 0
    preferred = jnp.where(moving_right, 3, 2).astype(jnp.int32)
    moving_dir = jnp.where(move[1, preferred], preferred, 4)
    moving_target = ((state.lesson == LESSON_COMBAT_MOVING)
                     | (state.lesson == LESSON_COMBAT_KILL))
    target_dir = jnp.where(moving_target,
                           moving_dir, 4)
    target_move = jax.nn.one_hot(target_dir, N_MOVES, dtype=jnp.bool_)
    move = move.at[1].set(jnp.where(scripted_target, target_move, move[1]))
    target_ability = jnp.asarray([True, False, False], jnp.bool_)
    ability = ability.at[1].set(
        jnp.where(scripted_target, target_ability, ability[1]))
    return move, ability


def _set_cell_value(grid, cell, value, condition):
    row = jnp.clip(cell[0], 0, H - 1)
    column = jnp.clip(cell[1], 0, W - 1)
    return grid.at[row, column].set(jnp.where(condition, value, grid[row, column]))


def step(state: BunState, actions: jnp.ndarray, key, auto_reset: bool = True,
         return_info: bool = False):
    key_base, key_drop, key_kind, key_reset = jax.random.split(key, 4)
    pre_move_mask, _ = legal_mask(state)
    requested_dirs = actions[:, 0]
    ability = actions[:, 1]
    scripted_target = ((state.lesson == LESSON_COMBAT_STATIC)
                       | (state.lesson == LESSON_COMBAT_MOVING)
                       | (state.lesson == LESSON_COMBAT_KILL)
                       | (state.lesson == LESSON_ROUTE_TO_BASE)
                       | (state.lesson == LESSON_ROUTE_TO_BASE_RELAXED)
                       | (state.lesson == LESSON_ROUTE_CAPTURE)
                       | (state.lesson == LESSON_CARRY_RETURN))
    scripted_dir = jnp.argmax(pre_move_mask[1]).astype(jnp.int32)
    requested_dirs = requested_dirs.at[1].set(
        jnp.where(scripted_target, scripted_dir, requested_dirs[1]))
    ability = ability.at[1].set(jnp.where(scripted_target, 0, ability[1]))
    active_slide = state.move_status == STATUS_SLIDE
    dirs = jnp.where(active_slide, state.slide_dir, requested_dirs)
    last_move_dir = jnp.where((requested_dirs < 4) & ~active_slide,
                              requested_dirs, state.last_move_dir).astype(jnp.int8)
    base_actions = jnp.stack([dirs, (ability == 1).astype(jnp.int32)], axis=-1)
    base_actions = base_actions.at[:, 1].set(
        jnp.where(state.bun_carried >= 0, 0, base_actions[:, 1]))
    tactical_bomb_tracking = (
        _TACTICAL_BOMB_PLACEMENT_REWARD != 0.0
        or _TACTICAL_BOMB_RESOLUTION_REWARD != 0.0
        or _BASE_BOMB_REWARD != 0.0
        or _FORCED_KILL_REWARD != 0.0)
    if _DANGER_TRACKING or tactical_bomb_tracking:
        from . import bun_safety
        needs_analysis = ((state.core.fuse > 0).any()
                          | (base_actions[:, 1] == 1).any()
                          | (state.danger_pending_ticks > 0).any())

        def analyze_danger(_):
            return bun_safety.analyze_selected_actions(state, base_actions)

        def no_danger(_):
            zeros = jnp.zeros((2,), jnp.bool_)
            return bun_safety.SelectedSafety(
                exposed=zeros, escapable=zeros, doomed=zeros,
                selected_survivable=zeros, avoidable=zeros,
                resolution_ticks=jnp.zeros((2,), jnp.int16),
                uniquely_own_hazard=zeros)

        danger_analysis = jax.lax.cond(
            needs_analysis, analyze_danger, no_danger, operand=None)
        if tactical_bomb_tracking:
            tactical_analysis = jax.lax.cond(
                (base_actions[:, 1] == 1).any(),
                lambda _: bun_safety.analyze_tactical_bomb_placements(
                    state, base_actions, danger_analysis.selected_survivable),
                lambda _: bun_safety.TacticalBombAnalysis(
                    safe=jnp.zeros((2,), jnp.bool_),
                    tactical=jnp.zeros((2,), jnp.bool_),
                    forces_kill=jnp.zeros((2,), jnp.bool_),
                    newly_threatens_enemy=jnp.zeros((2,), jnp.bool_),
                    enemy_safe_moves_before=jnp.zeros((2,), jnp.int16),
                    enemy_safe_moves_after=jnp.zeros((2,), jnp.int16)),
                operand=None)
        else:
            tactical_analysis = None
        new_exposure = ((state.danger_pending_ticks <= 0)
                        & ~state.danger_chain_active
                        & danger_analysis.exposed
                        & ~danger_analysis.doomed)
        exposure_ticks = jnp.maximum(danger_analysis.resolution_ticks, 1)
        danger_pending_ticks = jnp.where(
            new_exposure, exposure_ticks, state.danger_pending_ticks)
        refreshed = (danger_analysis.exposed
                     & (danger_pending_ticks > 0)
                     & (danger_analysis.resolution_ticks > 0))
        danger_pending_ticks = jnp.where(
            refreshed,
            jnp.minimum(danger_pending_ticks, exposure_ticks),
            danger_pending_ticks)
        # Reflect the currently-live hazard rather than sticky-OR. Re-capture
        # avoidability whenever a danger is newly established or refreshed, but
        # only on ticks where the player still had a choice (exposed yet not
        # doomed). Freezing once doomed keeps the verdict from the last tick an
        # escape existed, so a later unavoidable hazard resets a stale
        # "avoidable" flag left by an earlier exposure the player survived,
        # without the death tick (always doomed) erasing a genuine verdict.
        choiceful = danger_analysis.exposed & ~danger_analysis.doomed
        danger_pending_avoidable = jnp.where(
            (new_exposure | refreshed) & choiceful,
            danger_analysis.avoidable,
            state.danger_pending_avoidable)
        danger_chain_active = state.danger_chain_active | new_exposure
        danger_moved = jnp.where(
            new_exposure, False,
            state.danger_moved | ((requested_dirs < 4) & (danger_pending_ticks > 0)))
    else:
        danger_analysis = None
        tactical_analysis = None
        danger_pending_ticks = state.danger_pending_ticks
        danger_pending_avoidable = state.danger_pending_avoidable
        danger_chain_active = state.danger_chain_active
        danger_moved = state.danger_moved
    core, _base_done, base_info = base.step(
        state.core, base_actions, key_base, auto_reset=False, return_info=True,
        move_scale=_movement_scale(state), return_source_info=True)
    bomb_placed = jnp.stack([
        ((state.core.owner != player) & (core.owner == player)
         & (core.fuse > 0)).any()
        for player in range(2)
    ])
    safe_bomb_placed = bomb_placed & (pre_move_mask[:, :4].sum(axis=-1) >= 2)
    safe_tactical_bomb_placed = (
        bomb_placed & tactical_analysis.tactical
        if tactical_analysis is not None else jnp.zeros((2,), jnp.bool_))
    forced_kill_created = (
        bomb_placed & tactical_analysis.forces_kill
        if tactical_analysis is not None else jnp.zeros((2,), jnp.bool_))
    tactical_bomb_marks = state.tactical_bomb_marks
    placement_cells = jnp.clip(
        state.core.pos.astype(jnp.int32), jnp.asarray([0, 0]),
        jnp.asarray([H - 1, W - 1]))
    for player in range(2):
        row, column = placement_cells[player]
        tactical_bomb_marks = tactical_bomb_marks.at[player, row, column].set(
            tactical_bomb_marks[player, row, column]
            | safe_tactical_bomb_placed[player])

    blast_owner_linger = jnp.maximum(
        state.blast_owner_linger.astype(jnp.int16) - 1, 0).astype(jnp.int8)
    current_sources = jnp.stack([
        base_info["blast_owner_h"], base_info["blast_owner_v"]], axis=1)
    blast_owner_linger = jnp.where(
        current_sources, jnp.int8(base.BLAST_LINGER_TICKS),
        blast_owner_linger)
    blast_causal_linger = jnp.maximum(
        state.blast_causal_linger.astype(jnp.int16) - 1, 0).astype(jnp.int8)
    current_causal_sources = jnp.stack([
        base_info["blast_causal_h"], base_info["blast_causal_v"]], axis=1)
    blast_causal_linger = jnp.where(
        current_causal_sources, jnp.int8(base.BLAST_LINGER_TICKS),
        blast_causal_linger)
    blast_trigger_linger = jnp.maximum(
        state.blast_trigger_linger.astype(jnp.int16) - 1, 0).astype(jnp.int8)
    current_trigger_sources = jnp.stack([
        base_info["blast_trigger_h"], base_info["blast_trigger_v"]], axis=1)
    blast_trigger_linger = jnp.where(
        current_trigger_sources, jnp.int8(base.BLAST_LINGER_TICKS),
        blast_trigger_linger)
    source_hits = []
    causal_source_hits = []
    trigger_source_hits = []
    for source in range(2):
        source_hits.append(base._is_hit_by_explosion(
            core.pos, blast_owner_linger[source, 0] > 0,
            blast_owner_linger[source, 1] > 0, base.RADIUS))
        causal_source_hits.append(base._is_hit_by_explosion(
            core.pos, blast_causal_linger[source, 0] > 0,
            blast_causal_linger[source, 1] > 0, base.RADIUS))
        trigger_source_hits.append(base._is_hit_by_explosion(
            core.pos, blast_trigger_linger[source, 0] > 0,
            blast_trigger_linger[source, 1] > 0, base.RADIUS))
    source_cover = jnp.stack(source_hits, axis=-1)
    causal_source_cover = jnp.stack(causal_source_hits, axis=-1)
    trigger_source_cover = jnp.stack(trigger_source_hits, axis=-1)
    damage_source = source_cover & base_info["dmg"].astype(jnp.bool_)[:, None]
    causal_damage_source = (
        causal_source_cover & base_info["dmg"].astype(jnp.bool_)[:, None])
    trigger_damage_source = (
        trigger_source_cover & base_info["dmg"].astype(jnp.bool_)[:, None])
    own_explosion = current_sources.any(axis=(1, 2, 3))
    own_cover = jnp.stack([source_cover[0, 0], source_cover[1, 1]])
    safe_bomb_escape = own_explosion & core.alive & ~own_cover
    if _DANGER_TRACKING:
        pending_before_resolution = danger_pending_ticks
        danger_pending_ticks = jnp.maximum(
            danger_pending_ticks.astype(jnp.int32) - 1, 0).astype(jnp.int16)
        danger_safe_resolution = (
            (pending_before_resolution > 0)
            & (danger_pending_ticks == 0)
            & core.alive & danger_moved)
        danger_to_death = death = state.core.alive & ~core.alive
        avoidable_danger_death = danger_to_death & danger_pending_avoidable
        clear_danger = danger_safe_resolution | danger_to_death
        danger_pending_avoidable = jnp.where(
            clear_danger, False, danger_pending_avoidable)
        chain_still_live = ((core.fuse > 0).any()
                            | (core.blast_linger > 0).any())
        danger_chain_active = jnp.where(
            clear_danger & ~chain_still_live, False, danger_chain_active)
        danger_moved = jnp.where(clear_danger, False, danger_moved)
    else:
        danger_safe_resolution = jnp.zeros((2,), jnp.bool_)
        danger_to_death = jnp.zeros((2,), jnp.bool_)
        avoidable_danger_death = jnp.zeros((2,), jnp.bool_)
        danger_pending_ticks = jnp.zeros((2,), jnp.int16)
        danger_pending_avoidable = jnp.zeros((2,), jnp.bool_)
        danger_chain_active = jnp.zeros((2,), jnp.bool_)
        danger_moved = jnp.zeros((2,), jnp.bool_)
    route_target_hit = []
    for player, target in enumerate(ROUTE_TARGETS):
        owner_cover = current_sources[:, :, target[0], target[1]].any(axis=1)
        route_target_hit.append(
            state.core.brick[target[0], target[1]]
            & owner_cover[player] & ~owner_cover[1 - player])
    route_target_hit = jnp.stack(route_target_hit)
    bridge_cover = current_sources[
        :, :, BRIDGE_TARGET[0], BRIDGE_TARGET[1]].any(axis=1)
    bridge_route_open = jnp.stack([
        state.core.brick[BRIDGE_TARGET[0], BRIDGE_TARGET[1]]
        & bridge_cover[0] & ~bridge_cover[1],
        jnp.asarray(False),
    ])

    cells = jnp.stack([
        jnp.clip(core.pos[:, 0].astype(jnp.int32), 0, H - 1),
        jnp.clip(core.pos[:, 1].astype(jnp.int32), 0, W - 1),
    ], axis=-1)
    death = state.core.alive & ~core.alive

    bun_stored = state.bun_stored
    bun_loose = state.bun_loose
    bun_carried = state.bun_carried
    drop_event = death & (bun_carried >= 0)
    for player in range(2):
        origin = jnp.maximum(bun_carried[player], 0).astype(jnp.int32)
        row, column = cells[player, 0], cells[player, 1]
        old_count = bun_loose[row, column, origin]
        bun_loose = bun_loose.at[row, column, origin].set(
            jnp.where(drop_event[player], old_count + 1, old_count))
        bun_carried = bun_carried.at[player].set(
            jnp.where(drop_event[player], -1, bun_carried[player]))

    waiting = state.bun_respawn > 0
    bun_respawn = jnp.maximum(state.bun_respawn - 1, 0).astype(jnp.int16)
    respawn_now = waiting & (bun_respawn == 0)
    bun_respawn = jnp.where(death, BUN_RESPAWN_TICKS, bun_respawn).astype(jnp.int16)
    core = core._replace(
        pos=jnp.where(respawn_now[:, None], state.bun_spawn_pos, core.pos),
        alive=core.alive | respawn_now,
        hp=jnp.where(respawn_now, _TRAIN_HP, core.hp),
        invuln=jnp.where(respawn_now, 10, core.invuln),
    )
    cells = jnp.stack([
        jnp.clip(core.pos[:, 0].astype(jnp.int32), 0, H - 1),
        jnp.clip(core.pos[:, 1].astype(jnp.int32), 0, W - 1),
    ], axis=-1)

    status_ticks = jnp.maximum(state.status_ticks - 1, 0).astype(jnp.int16)
    move_status = jnp.where(status_ticks > 0, state.move_status, STATUS_NONE).astype(jnp.int8)
    slide_dir = state.slide_dir
    moved = jnp.linalg.norm(core.pos - state.core.pos, axis=-1)
    expected = base.STEP * state.core.spd_g * _movement_scale(state)
    slide_blocked = active_slide & (moved < expected * 0.35)
    move_status = jnp.where(slide_blocked, STATUS_NONE, move_status).astype(jnp.int8)
    status_ticks = jnp.where(slide_blocked, 0, status_ticks).astype(jnp.int16)
    move_status = jnp.where(death | respawn_now, STATUS_NONE, move_status).astype(jnp.int8)
    status_ticks = jnp.where(death | respawn_now, 0, status_ticks).astype(jnp.int16)

    blast_now = core.blast_linger > 0
    tactical_crate = jnp.where(blast_now, ITEM_NONE, state.tactical_crate).astype(jnp.int8)
    field_item = jnp.where(blast_now, ITEM_NONE, state.field_item).astype(jnp.int8)
    field_owner = jnp.where(field_item > 0, state.field_owner, -1).astype(jnp.int8)
    field_armed = state.field_armed & (field_item > 0)

    new_crate = (core.crate > 0) & (state.core.crate == 0)
    tactical_drop = new_crate & (jax.random.uniform(key_drop, (H, W)) < TACTICAL_ITEM_FRACTION)
    tactical_kind = jnp.where(jax.random.uniform(key_kind, (H, W)) < 0.5,
                              ITEM_BANANA, ITEM_SLOW_GLUE).astype(jnp.int8)
    tactical_crate = jnp.where(tactical_drop, tactical_kind, tactical_crate)
    core = core._replace(crate=jnp.where(tactical_drop, 0, core.crate).astype(jnp.int8))

    held_item = jnp.where(death, ITEM_NONE, state.held_item).astype(jnp.int8)
    for player in range(2):
        row, column = cells[player, 0], cells[player, 1]
        item = tactical_crate[row, column]
        pickup = core.alive[player] & (held_item[player] == ITEM_NONE) & (item > ITEM_NONE)
        held_item = held_item.at[player].set(jnp.where(pickup, item, held_item[player]))
        tactical_crate = tactical_crate.at[row, column].set(
            jnp.where(pickup, ITEM_NONE, tactical_crate[row, column]))

    for player in range(2):
        row, column = cells[player, 0], cells[player, 1]
        use_item = (core.alive[player] & (ability[player] == 2)
                    & (held_item[player] > ITEM_NONE)
                    & (field_item[row, column] == ITEM_NONE))
        field_item = field_item.at[row, column].set(
            jnp.where(use_item, held_item[player], field_item[row, column]))
        field_owner = field_owner.at[row, column].set(
            jnp.where(use_item, player, field_owner[row, column]))
        field_armed = field_armed.at[row, column].set(
            jnp.where(use_item, False, field_armed[row, column]))
        held_item = held_item.at[player].set(
            jnp.where(use_item, ITEM_NONE, held_item[player]))

    rows = jnp.arange(H)[:, None]
    columns = jnp.arange(W)[None, :]
    for player in range(2):
        on_owner_cell = ((rows == cells[player, 0]) & (columns == cells[player, 1]))
        field_armed = field_armed | ((field_owner == player) & ~on_owner_cell)

    trap_hit = jnp.zeros((2,), jnp.bool_)
    trap_owner = jnp.full((2,), -1, jnp.int8)
    for player in range(2):
        row, column = cells[player, 0], cells[player, 1]
        item = field_item[row, column]
        owner = field_owner[row, column]
        trigger = core.alive[player] & field_armed[row, column] & (owner != player) & (item > 0)
        trap_hit = trap_hit.at[player].set(trigger)
        trap_owner = trap_owner.at[player].set(jnp.where(trigger, owner, -1))
        slow = trigger & (item == ITEM_SLOW_GLUE)
        slide = trigger & (item == ITEM_BANANA)
        move_status = move_status.at[player].set(
            jnp.where(slow, STATUS_SLOW,
                      jnp.where(slide, STATUS_SLIDE, move_status[player])))
        status_ticks = status_ticks.at[player].set(
            jnp.where(slow, 100, jnp.where(slide, 20, status_ticks[player])))
        fallback_dir = jnp.where(player == 0, 3, 2)
        slide_dir = slide_dir.at[player].set(
            jnp.where(slide, jnp.where(last_move_dir[player] < 4,
                                       last_move_dir[player], fallback_dir),
                      slide_dir[player]))
        field_item = field_item.at[row, column].set(
            jnp.where(trigger, ITEM_NONE, field_item[row, column]))
        field_owner = field_owner.at[row, column].set(
            jnp.where(trigger, -1, field_owner[row, column]))
        field_armed = field_armed.at[row, column].set(
            jnp.where(trigger, False, field_armed[row, column]))

    steal = jnp.zeros((2,), jnp.bool_)
    pickup_loose = jnp.zeros((2,), jnp.bool_)
    recover = jnp.zeros((2,), jnp.bool_)
    returned = jnp.zeros((2,), jnp.bool_)
    capture = jnp.zeros((2,), jnp.bool_)
    carry_before = state.bun_carried
    objective_progress = jnp.zeros((2,), jnp.float32)
    enter_enemy_half = jnp.zeros((2,), jnp.bool_)
    enter_enemy_base = jnp.zeros((2,), jnp.bool_)
    carry_cross_home = jnp.zeros((2,), jnp.bool_)
    bridge_success = jnp.zeros((2,), jnp.bool_)
    milestones = state.milestones
    before_cells = jnp.stack([
        jnp.clip(state.core.pos[:, 0].astype(jnp.int32), 0, H - 1),
        jnp.clip(state.core.pos[:, 1].astype(jnp.int32), 0, W - 1),
    ], axis=-1)
    for player in range(2):
        target_before = jnp.where(carry_before[player] >= 0,
                                  player, 1 - player)
        before_cell = before_cells[player]
        before_dist = _BUN_ROUTE_DISTANCE[
            target_before, before_cell[0], before_cell[1]]

        cell = cells[player]
        base_team = _base_team(cell)
        carrying = bun_carried[player]
        deposit = core.alive[player] & (base_team == player) & (carrying >= 0)
        origin = jnp.maximum(carrying, 0).astype(jnp.int32)
        old_count = bun_stored[player, origin]
        bun_stored = bun_stored.at[player, origin].set(
            jnp.where(deposit, old_count + 1, old_count))
        capture = capture.at[player].set(deposit & (origin != player))
        returned = returned.at[player].set(deposit & (origin == player))
        bun_carried = bun_carried.at[player].set(jnp.where(deposit, -1, carrying))

        for loose_origin in range(2):
            row, column = cell[0], cell[1]
            available = bun_loose[row, column, loose_origin] > 0
            take = core.alive[player] & (bun_carried[player] < 0) & available
            bun_loose = bun_loose.at[row, column, loose_origin].set(
                jnp.where(take, bun_loose[row, column, loose_origin] - 1,
                          bun_loose[row, column, loose_origin]))
            bun_carried = bun_carried.at[player].set(
                jnp.where(take, loose_origin, bun_carried[player]))
            pickup_loose = pickup_loose.at[player].set(pickup_loose[player] | take)
            recover = recover.at[player].set(recover[player] | (take & (loose_origin == player)))

        enemy = 1 - player
        can_steal = (core.alive[player] & (bun_carried[player] < 0)
                     & (base_team == enemy) & (bun_stored[enemy, enemy] > 0))
        bun_stored = bun_stored.at[enemy, enemy].set(
            jnp.where(can_steal, bun_stored[enemy, enemy] - 1,
                      bun_stored[enemy, enemy]))
        bun_carried = bun_carried.at[player].set(
            jnp.where(can_steal, enemy, bun_carried[player]))
        steal = steal.at[player].set(can_steal)

        target_after = jnp.where(bun_carried[player] >= 0,
                                 player, 1 - player)
        after_dist = _BUN_ROUTE_DISTANCE[
            target_after, cell[0], cell[1]]
        same_phase = ((carry_before[player] >= 0)
                      == (bun_carried[player] >= 0))
        objective_progress = objective_progress.at[player].set(
            jnp.where(core.alive[player] & same_phase,
                      before_dist - after_dist, 0.0))

        enemy_half = jnp.where(player == 0, cell[1] > W // 2,
                               cell[1] < W // 2)
        own_half = jnp.where(player == 0, cell[1] <= W // 2,
                             cell[1] >= W // 2)
        bit_half = jnp.uint8(1)
        bit_base = jnp.uint8(2)
        bit_return = jnp.uint8(4)
        bit_route = jnp.uint8(8)
        bridge_ready = (((milestones[player] & bit_route) != 0)
                        | bridge_route_open[player])
        bridge_lesson = ((state.lesson == LESSON_ROUTE_TO_BASE)
                         | (state.lesson == LESSON_ROUTE_TO_BASE_RELAXED)
                         | (state.lesson == LESSON_ROUTE_CAPTURE))
        objective_ready = (~bridge_lesson
                           | bridge_ready)
        enter_enemy_half = enter_enemy_half.at[player].set(
            core.alive[player] & enemy_half & objective_ready
            & ((milestones[player] & bit_half) == 0))
        enter_enemy_base = enter_enemy_base.at[player].set(
            core.alive[player] & (base_team == enemy) & objective_ready
            & ((milestones[player] & bit_base) == 0))
        bridge_success = bridge_success.at[player].set(
            bridge_lesson
            & enter_enemy_base[player])
        carrying_enemy = bun_carried[player] == enemy
        carry_cross_home = carry_cross_home.at[player].set(
            core.alive[player] & carrying_enemy & own_half
            & ((milestones[player] & bit_return) == 0))
        new_bits = (enter_enemy_half[player].astype(jnp.uint8) * bit_half
                    | enter_enemy_base[player].astype(jnp.uint8) * bit_base
                    | carry_cross_home[player].astype(jnp.uint8) * bit_return
                    | bridge_route_open[player].astype(jnp.uint8) * bit_route)
        milestones = milestones.at[player].set(milestones[player] | new_bits)

    base_totals = bun_stored.sum(axis=-1).astype(jnp.int8)
    source_count = damage_source.sum(axis=-1)
    causal_source_count = causal_damage_source.sum(axis=-1)
    mutual_death = death.all()
    credited_hit = jnp.stack([
        damage_source[1, 0] & (source_count[1] == 1),
        damage_source[0, 1] & (source_count[0] == 1),
    ])
    credited_kill = jnp.stack([
        death[1] & damage_source[1, 0] & (source_count[1] == 1),
        death[0] & damage_source[0, 1] & (source_count[0] == 1),
    ]) & ~mutual_death
    causal_hit = jnp.stack([
        causal_damage_source[1, 0] & (causal_source_count[1] == 1),
        causal_damage_source[0, 1] & (causal_source_count[0] == 1),
    ])
    causal_kill = jnp.stack([
        death[1] & causal_damage_source[1, 0]
        & (causal_source_count[1] == 1),
        death[0] & causal_damage_source[0, 1]
        & (causal_source_count[0] == 1),
    ]) & ~mutual_death
    trigger_kill = causal_kill & jnp.stack([
        trigger_damage_source[1, 0], trigger_damage_source[0, 1],
    ])
    own_bomb_defeat = jnp.stack([
        death[0] & damage_source[0, 0] & (source_count[0] == 1)
        & ~causal_damage_source[0, 1],
        death[1] & damage_source[1, 1] & (source_count[1] == 1)
        & ~causal_damage_source[1, 0],
    ]) & ~mutual_death
    active_tactical_marks = jnp.stack([
        tactical_bomb_marks[player]
        & (core.owner == player) & (core.fuse > 0)
        for player in range(2)
    ])
    tactical_bomb_resolved_count = (
        tactical_bomb_marks & ~active_tactical_marks).sum(axis=(1, 2)).astype(
            jnp.int16)
    tactical_bomb_safe_resolution = jnp.where(
        core.alive & ~own_bomb_defeat & ~mutual_death,
        tactical_bomb_resolved_count, 0).astype(jnp.int16)
    # 放泡奖励采用两阶段归因：这里只保留“放置时有逃生路径且形成有效威胁”的
    # marker；较大的 resolution 奖励需等该泡真正结算且放泡者未自爆/换命。
    tactical_bomb_marks = jnp.where(
        death[:, None, None], False, active_tactical_marks)
    opponent_physical_defeat = jnp.stack([
        death[0] & damage_source[0, 1] & (source_count[0] == 1),
        death[1] & damage_source[1, 0] & (source_count[1] == 1),
    ]) & ~mutual_death
    opponent_causal_defeat = jnp.stack([
        death[0] & causal_damage_source[0, 1]
        & (causal_source_count[0] == 1),
        death[1] & causal_damage_source[1, 0]
        & (causal_source_count[1] == 1),
    ]) & ~mutual_death
    route_success = route_target_hit
    route_done = (state.lesson == LESSON_ROUTE_BREAK) & route_success.any()
    bridge_done = (((state.lesson == LESSON_ROUTE_TO_BASE)
                    | (state.lesson == LESSON_ROUTE_TO_BASE_RELAXED))
                   & bridge_success.any())
    combat_done = (state.lesson == LESSON_COMBAT) & causal_kill.any()
    combat_kill_done = (state.lesson == LESSON_COMBAT_KILL) & causal_kill.any()
    combat_hit_lesson = ((state.lesson == LESSON_COMBAT_STATIC)
                         | (state.lesson == LESSON_COMBAT_MOVING))
    combat_hit_done = combat_hit_lesson & credited_hit.any()
    steal_done = (state.lesson == LESSON_NEAR_STEAL) & steal.any()
    capture_done = ((state.lesson == LESSON_CARRY_HOME)
                    | (state.lesson == LESSON_ROUTE_CAPTURE)
                    | (state.lesson == LESSON_CARRY_RETURN)
                    | (state.lesson == LESSON_KILL_RUSH)
                    | (state.lesson == LESSON_FULL)
                    | (state.lesson == LESSON_FULL_CONTACT)
                    | (state.lesson == LESSON_FULL_AMBUSH)) & capture.any()
    timeout = core.t >= LESSON_MAX_STEPS[state.lesson]
    done = (route_done | bridge_done | combat_done | combat_kill_done
            | combat_hit_done | steal_done | capture_done | timeout)
    capture_winner = jnp.where(capture[0] & ~capture[1], 0,
                               jnp.where(capture[1] & ~capture[0], 1, -1))
    route_winner = jnp.where(route_success[0] & ~route_success[1], 0,
                             jnp.where(route_success[1] & ~route_success[0], 1, -1))
    bridge_winner = jnp.where(bridge_success[0], 0, -1)
    surviving_causal_kill = causal_kill & core.alive
    surviving_physical_kill = credited_kill & core.alive
    surviving_kill = surviving_causal_kill | surviving_physical_kill
    combat_winner = jnp.where(surviving_kill[0] & ~surviving_kill[1], 0,
                              jnp.where(surviving_kill[1] & ~surviving_kill[0], 1, -1))
    combat_hit_winner = jnp.where(credited_hit[0] & ~credited_hit[1], 0,
                                  jnp.where(credited_hit[1] & ~credited_hit[0], 1, -1))
    steal_winner = jnp.where(steal[0] & ~steal[1], 0,
                             jnp.where(steal[1] & ~steal[0], 1, -1))
    timeout_winner = jnp.where(base_totals[0] > base_totals[1], 0,
                               jnp.where(base_totals[1] > base_totals[0], 1, -1))
    event_winner = jnp.where(route_done, route_winner,
                             jnp.where(bridge_done, bridge_winner,
                                       jnp.where(combat_done | combat_kill_done,
                                                 combat_winner,
                                                 jnp.where(combat_hit_done,
                                                           combat_hit_winner,
                                                           jnp.where(steal_done,
                                                                     steal_winner,
                                                                     capture_winner)))))
    course_timeout_winner = jnp.where(
        state.lesson == LESSON_FULL, timeout_winner, -1)
    winner = jnp.where(route_done | bridge_done | combat_done | combat_kill_done
                       | combat_hit_done | steal_done | capture_done,
                       event_winner,
                       jnp.where(timeout, course_timeout_winner, -2)).astype(jnp.int8)

    items = core.items.at[:, 0].set(held_item)
    debuffs = move_status.astype(jnp.int8)
    core = core._replace(items=items, debuffs=debuffs)
    enemy_respawn = bun_respawn[::-1]
    kill_window_ticks = jnp.maximum(
        state.kill_window_ticks - 1, 0).astype(jnp.int16)
    kill_window_ticks = jnp.minimum(kill_window_ticks, enemy_respawn)
    kill_window_ticks = jnp.where(
        causal_kill, enemy_respawn, kill_window_ticks).astype(jnp.int16)
    return_phase = carry_before >= 0
    kill_window_active = (kill_window_ticks > 0) & ~return_phase
    combat_phase = ~kill_window_active & ~return_phase
    new_state = BunState(
        core, bun_stored, bun_loose, bun_carried, bun_respawn,
        state.bun_spawn_pos, held_item, tactical_crate, field_item,
        field_owner, field_armed, move_status, status_ticks, slide_dir,
        last_move_dir, blast_owner_linger, blast_causal_linger,
        blast_trigger_linger, kill_window_ticks, milestones, state.lesson,
        danger_pending_ticks, danger_pending_avoidable,
        danger_chain_active, danger_moved, tactical_bomb_marks)
    out = jax.lax.cond(done, lambda _: _fresh(key_reset), lambda _: new_state,
                       operand=None) if auto_reset else new_state

    info = {
        **base_info,
        "alive": core.alive,
        "hp": core.hp,
        "cell": cells,
        "death": death,
        "bomb_placed": bomb_placed,
        "safe_bomb_placed": safe_bomb_placed,
        "safe_tactical_bomb_placed": safe_tactical_bomb_placed,
        "forced_kill_created": forced_kill_created,
        "tactical_bomb_safe_resolution": tactical_bomb_safe_resolution,
        "tactical_bomb_newly_threatens_enemy": (
            tactical_analysis.newly_threatens_enemy
            if tactical_analysis is not None else jnp.zeros((2,), jnp.bool_)),
        "tactical_bomb_enemy_safe_moves_before": (
            tactical_analysis.enemy_safe_moves_before
            if tactical_analysis is not None else jnp.zeros((2,), jnp.int16)),
        "tactical_bomb_enemy_safe_moves_after": (
            tactical_analysis.enemy_safe_moves_after
            if tactical_analysis is not None else jnp.zeros((2,), jnp.int16)),
        "safe_bomb_escape": safe_bomb_escape,
        "danger_exposure": (
            danger_analysis.exposed if danger_analysis is not None
            else jnp.zeros((2,), jnp.bool_)),
        "escapable_danger": (
            (danger_analysis.exposed & danger_analysis.escapable)
            if danger_analysis is not None else jnp.zeros((2,), jnp.bool_)),
        "doomed_danger": (
            danger_analysis.doomed if danger_analysis is not None
            else jnp.zeros((2,), jnp.bool_)),
        "avoidable_danger_action": (
            danger_analysis.avoidable if danger_analysis is not None
            else jnp.zeros((2,), jnp.bool_)),
        "danger_safe_resolution": danger_safe_resolution,
        "danger_safe_resolution_reward": (
            danger_safe_resolution.astype(jnp.float32) * _DANGER_ESCAPE_REWARD),
        "danger_to_death": danger_to_death,
        "avoidable_danger_death": avoidable_danger_death,
        "uniquely_own_hazard": (
            danger_analysis.uniquely_own_hazard
            if danger_analysis is not None else jnp.zeros((2,), jnp.bool_)),
        "own_detonation": own_explosion,
        "damage_source": damage_source,
        "death_source": damage_source & death[:, None],
        "causal_damage_source": causal_damage_source,
        "causal_death_source": causal_damage_source & death[:, None],
        "trigger_damage_source": trigger_damage_source,
        "carrier_death": death & (carry_before >= 0),
        "drop": drop_event,
        "steal": steal,
        "pickup_loose": pickup_loose,
        "recover": recover,
        "return_bun": returned,
        "capture": capture,
        "objective_progress": objective_progress,
        "enter_enemy_half": enter_enemy_half,
        "enter_enemy_base": enter_enemy_base,
        "carry_cross_home": carry_cross_home,
        "trap_hit": trap_hit,
        "trap_owner": trap_owner,
        "walls_by_owner": base_info["walls_by_owner"],
        "route_success": route_success,
        "bridge_route_open": bridge_route_open,
        "bridge_success": bridge_success,
        "credited_hit": credited_hit,
        "credited_kill": credited_kill,
        "causal_hit": causal_hit,
        "causal_kill": causal_kill,
        "trigger_kill": trigger_kill,
        "surviving_causal_kill": surviving_causal_kill,
        "surviving_physical_kill": surviving_physical_kill,
        "surviving_kill": surviving_kill,
        "own_bomb_defeat": own_bomb_defeat,
        "opponent_physical_defeat": opponent_physical_defeat,
        "opponent_causal_defeat": opponent_causal_defeat,
        "mutual_death": mutual_death,
        "combat_phase": combat_phase,
        "kill_window_active": kill_window_active,
        "return_phase": return_phase,
        "kill_window_remaining": kill_window_ticks,
        "base_total": base_totals,
        "winner": winner,
        "lesson": state.lesson,
    }
    blocked_shaping = info["own_bomb_defeat"] | info["mutual_death"]
    info["tactical_bomb_placement_reward"] = jnp.where(
        blocked_shaping, 0.0,
        info["safe_tactical_bomb_placed"].astype(jnp.float32)
        * _TACTICAL_BOMB_PLACEMENT_REWARD)
    info["tactical_bomb_resolution_reward"] = jnp.where(
        blocked_shaping, 0.0,
        jnp.minimum(
            info["tactical_bomb_safe_resolution"].astype(jnp.float32), 1.0)
        * _TACTICAL_BOMB_RESOLUTION_REWARD)
    info["base_bomb_reward"] = jnp.where(
        blocked_shaping, 0.0,
        info["bomb_placed"].astype(jnp.float32) * _BASE_BOMB_REWARD)
    info["forced_kill_reward"] = jnp.where(
        blocked_shaping, 0.0,
        info["forced_kill_created"].astype(jnp.float32) * _FORCED_KILL_REWARD)
    if return_info:
        return out, done, info
    return out, done


def _base_mask(team: int) -> jnp.ndarray:
    anchor = _BUN_BASES[team]
    rows = jnp.arange(H)[:, None]
    columns = jnp.arange(W)[None, :]
    return ((rows >= anchor[0]) & (rows < anchor[0] + 3)
            & (columns >= anchor[1]) & (columns < anchor[1] + 3))


def make_obs(state: BunState, pid: int, danger=None,
             channels: int = N_OBS_CH) -> jnp.ndarray:
    core_obs = base.make_obs(state.core, pid, danger, channels=14)
    episode_steps = LESSON_MAX_STEPS[state.lesson]
    core_obs = core_obs.at[6].set(jnp.full(
        (H, W), state.core.t.astype(jnp.float32)
        / episode_steps.astype(jnp.float32)))
    own, enemy = pid, 1 - pid
    carried_origin = []
    for origin in range(2):
        plane = jnp.zeros((H, W), jnp.float32)
        for player in range(2):
            plane = plane + base._splat(
                state.core.pos[player],
                state.core.alive[player] & (state.bun_carried[player] == origin),
                H, W)
        carried_origin.append(jnp.clip(plane, 0.0, 1.0))
    banana = ((state.tactical_crate == ITEM_BANANA)
              | (state.field_item == ITEM_BANANA)).astype(jnp.float32)
    slow = ((state.tactical_crate == ITEM_SLOW_GLUE)
            | (state.field_item == ITEM_SLOW_GLUE)).astype(jnp.float32)
    extra = jnp.stack([
        _base_mask(own).astype(jnp.float32),
        _base_mask(enemy).astype(jnp.float32),
        state.bun_loose[:, :, own].astype(jnp.float32),
        state.bun_loose[:, :, enemy].astype(jnp.float32),
        carried_origin[own],
        carried_origin[enemy],
        banana,
        slow,
        (state.field_armed & (state.field_owner == enemy)).astype(jnp.float32),
        (state.field_armed & (state.field_owner == own)).astype(jnp.float32),
    ])
    return jnp.concatenate([core_obs, extra], axis=0)[:channels]


def global_vec(state: BunState, pid: int) -> jnp.ndarray:
    enemy = 1 - pid
    hp = state.core.hp.astype(jnp.float32) / MAX_HP
    bombs = state.core.bombs_cap / float(base.GROWTH_BOMBS_MAX)
    blast = state.core.blast_cap / float(base.GROWTH_BLAST_MAX)
    speed = state.core.spd_g / float(base.GROWTH_SPEED_MAX)
    carried = state.bun_carried
    stored = state.bun_stored.astype(jnp.float32)
    return jnp.stack([
        state.core.t.astype(jnp.float32)
        / LESSON_MAX_STEPS[state.lesson].astype(jnp.float32),
        hp[pid], hp[enemy],
        bombs[pid], blast[pid], speed[pid],
        bombs[enemy], blast[enemy], speed[enemy],
        state.core.alive[pid].astype(jnp.float32),
        state.core.alive[enemy].astype(jnp.float32),
        state.bun_respawn[pid].astype(jnp.float32) / BUN_RESPAWN_TICKS,
        state.bun_respawn[enemy].astype(jnp.float32) / BUN_RESPAWN_TICKS,
        (carried[pid] >= 0).astype(jnp.float32),
        (carried[pid] == enemy).astype(jnp.float32),
        (carried[enemy] >= 0).astype(jnp.float32),
        (carried[enemy] == pid).astype(jnp.float32),
        stored[pid, pid], stored[pid, enemy],
        stored[enemy, pid], stored[enemy, enemy],
        (state.held_item[pid] == ITEM_BANANA).astype(jnp.float32),
        (state.held_item[pid] == ITEM_SLOW_GLUE).astype(jnp.float32),
        (state.move_status[pid] != STATUS_NONE).astype(jnp.float32),
    ])


def reward_from_events(dmg, alive_before, alive_after, hp_after, done,
                       crate_grew, newly, walls_destroyed, crate_coef,
                       explore_coef, brick_coef, timeout_alpha,
                       win_bonus=10.0, lose_bonus=6.0,
                       timeout_lead_bonus=2.0,
                       timeout_trail_penalty=1.0,
                       timeout_draw_bonus=0.0,
                       mutual_hit_penalty=0.0,
                       double_death_penalty=5.0,
                       win_hp_bonus=0.0,
                       trade_win_bonus=3.5,
                       moves=None, bombs=None, idle_penalty=0.005,
                       rule_info=None):
    """分阶段抢包子奖励；physical owner 与 causal trigger 独立归因。"""
    del hp_after, timeout_alpha, win_bonus, lose_bonus
    del timeout_lead_bonus, timeout_trail_penalty, timeout_draw_bonus
    del mutual_hit_penalty, double_death_penalty, win_hp_bonus, trade_win_bonus
    del alive_after, done, moves, bombs, idle_penalty, walls_destroyed
    info = rule_info
    source = info["damage_source"]
    unique_source = source.sum(axis=-1) == 1
    own_hit = jnp.stack([
        source[:, 0, 0] & unique_source[:, 0],
        source[:, 1, 1] & unique_source[:, 1],
    ], axis=-1)
    enemy_hit = jnp.stack([
        source[:, 1, 0] & unique_source[:, 1],
        source[:, 0, 1] & unique_source[:, 0],
    ], axis=-1)
    death = info["death"].astype(jnp.float32)
    credited_kill = info["credited_kill"]
    causal_kill = info["causal_kill"]
    self_kill = info["own_bomb_defeat"]
    carrier_kill = causal_kill & jnp.stack([
        info["carrier_death"][:, 1], info["carrier_death"][:, 0]], axis=-1)
    trap_credit = jnp.stack([
        (info["trap_hit"] & (info["trap_owner"] == 0)).sum(axis=-1),
        (info["trap_hit"] & (info["trap_owner"] == 1)).sum(axis=-1),
    ], axis=-1)
    winner = info["winner"]
    win = jnp.stack([winner == 0, winner == 1], axis=-1)
    loss = jnp.stack([winner == 1, winner == 0], axis=-1)
    progress = jnp.clip(info["objective_progress"], -1.0, 1.0)
    safety = (-1.0 * own_hit.astype(jnp.float32)
              - 4.0 * death
              - 4.0 * self_kill.astype(jnp.float32))

    unsafe_bomb = info["bomb_placed"] & ~info["safe_bomb_placed"]
    unsafe_detonation = info["own_detonation"] & ~info["safe_bomb_escape"]
    route_reward = safety
    route_reward += 0.15 * info["safe_bomb_placed"].astype(jnp.float32)
    route_reward += 0.75 * info["safe_bomb_escape"].astype(jnp.float32)
    route_reward -= 1.0 * unsafe_detonation.astype(jnp.float32)
    route_reward -= 0.25 * unsafe_bomb.astype(jnp.float32)
    route_reward += (3.0 + brick_coef) * info["route_success"].astype(jnp.float32)
    route_reward += 5.0 * info["route_success"].astype(jnp.float32)

    bridge_reward = safety
    bridge_reward += 1.0 * info["safe_bomb_escape"].astype(jnp.float32)
    bridge_reward -= 1.5 * unsafe_detonation.astype(jnp.float32)
    bridge_reward += 6.0 * info["bridge_route_open"].astype(jnp.float32)
    bridge_reward += 8.0 * info["enter_enemy_half"].astype(jnp.float32)
    bridge_reward += 16.0 * info["bridge_success"].astype(jnp.float32)
    bridge_reward += 12.0 * win.astype(jnp.float32)
    bridge_reward *= jnp.asarray([1.0, 0.0], jnp.float32)

    bridge_capture_reward = safety
    bridge_capture_reward += 1.0 * info["safe_bomb_escape"].astype(jnp.float32)
    bridge_capture_reward -= 1.5 * unsafe_detonation.astype(jnp.float32)
    bridge_capture_reward -= 8.0 * info["drop"].astype(jnp.float32)
    bridge_capture_reward += 6.0 * info["bridge_route_open"].astype(jnp.float32)
    bridge_capture_reward += 5.0 * info["enter_enemy_half"].astype(jnp.float32)
    bridge_capture_reward += 8.0 * info["enter_enemy_base"].astype(jnp.float32)
    bridge_capture_reward += 14.0 * info["steal"].astype(jnp.float32)
    bridge_capture_reward += 12.0 * info["carry_cross_home"].astype(jnp.float32)
    bridge_capture_reward += 40.0 * info["capture"].astype(jnp.float32)
    bridge_capture_reward += 0.20 * progress
    bridge_capture_reward += 20.0 * win.astype(jnp.float32)
    bridge_capture_reward *= jnp.asarray([1.0, 0.0], jnp.float32)

    combat_reward = safety
    combat_reward += 1.0 * info["safe_bomb_escape"].astype(jnp.float32)
    combat_reward -= 1.5 * unsafe_detonation.astype(jnp.float32)
    combat_reward += 0.75 * enemy_hit.astype(jnp.float32)
    combat_reward += 6.0 * causal_kill.astype(jnp.float32)
    combat_reward += 10.0 * win.astype(jnp.float32)
    combat_reward -= 10.0 * loss.astype(jnp.float32)

    combat_hit_reward = safety
    combat_hit_reward += 1.0 * info["safe_bomb_escape"].astype(jnp.float32)
    combat_hit_reward -= 1.5 * unsafe_detonation.astype(jnp.float32)
    combat_hit_reward += 6.0 * enemy_hit.astype(jnp.float32)
    combat_hit_reward += 6.0 * win.astype(jnp.float32)
    combat_hit_reward *= jnp.asarray([1.0, 0.0], jnp.float32)

    near_reward = safety
    near_reward += 0.15 * progress
    near_reward += 2.0 * info["enter_enemy_half"].astype(jnp.float32)
    near_reward += 5.0 * info["enter_enemy_base"].astype(jnp.float32)
    near_reward += 10.0 * info["steal"].astype(jnp.float32)
    near_reward += 10.0 * win.astype(jnp.float32)

    carry_reward = safety
    carry_reward -= 6.0 * info["drop"].astype(jnp.float32)
    carry_reward += 0.20 * progress
    carry_reward += 10.0 * info["carry_cross_home"].astype(jnp.float32)
    carry_reward += 35.0 * info["capture"].astype(jnp.float32)
    carry_reward += 20.0 * win.astype(jnp.float32)
    carry_reward -= 20.0 * loss.astype(jnp.float32)

    full_reward = safety
    full_reward += 0.25 * enemy_hit.astype(jnp.float32)
    full_reward += 0.50 * info["safe_bomb_escape"].astype(jnp.float32)
    full_reward -= 2.0 * unsafe_detonation.astype(jnp.float32)
    full_reward += 1.0 * causal_kill.astype(jnp.float32)
    full_reward += 3.0 * carrier_kill.astype(jnp.float32)
    full_reward -= 5.0 * info["drop"].astype(jnp.float32)
    full_reward += 4.0 * info["steal"].astype(jnp.float32)
    full_reward += 2.0 * info["recover"].astype(jnp.float32)
    full_reward += 4.0 * info["return_bun"].astype(jnp.float32)
    full_reward += 30.0 * info["capture"].astype(jnp.float32)
    full_reward += 1.0 * info["enter_enemy_half"].astype(jnp.float32)
    full_reward += 3.0 * info["enter_enemy_base"].astype(jnp.float32)
    full_reward += 8.0 * info["carry_cross_home"].astype(jnp.float32)
    full_reward += 0.12 * progress
    full_reward += 0.75 * trap_credit.astype(jnp.float32)
    full_reward += (0.02 + crate_coef) * crate_grew.astype(jnp.float32)
    full_reward += explore_coef * newly.astype(jnp.float32)
    full_reward += (0.30 + brick_coef) * info["walls_by_owner"].astype(jnp.float32)
    full_reward += 20.0 * win.astype(jnp.float32)
    full_reward -= 20.0 * loss.astype(jnp.float32)

    auto_sparse_reward = safety
    auto_sparse_reward += 0.25 * info["safe_bomb_escape"].astype(jnp.float32)
    auto_sparse_reward += 0.25 * enemy_hit.astype(jnp.float32)
    auto_sparse_reward += 0.50 * causal_kill.astype(jnp.float32)
    auto_sparse_reward -= 1.0 * info["drop"].astype(jnp.float32)
    auto_sparse_reward += 1.0 * info["steal"].astype(jnp.float32)
    auto_sparse_reward += 1.0 * info["recover"].astype(jnp.float32)
    auto_sparse_reward += 1.0 * info["return_bun"].astype(jnp.float32)
    auto_sparse_reward += 2.0 * info["carry_cross_home"].astype(jnp.float32)
    auto_sparse_reward += 12.0 * info["capture"].astype(jnp.float32)
    auto_sparse_reward += 20.0 * win.astype(jnp.float32)
    auto_sparse_reward -= 20.0 * loss.astype(jnp.float32)
    if _REWARD_PROFILE == "auto_sparse":
        full_reward = auto_sparse_reward

    any_kill = causal_kill | credited_kill
    surviving_kill = info["surviving_kill"]
    combat_evolution_reward = safety
    combat_evolution_reward += 0.50 * info["safe_bomb_escape"].astype(jnp.float32)
    combat_evolution_reward -= 1.0 * unsafe_detonation.astype(jnp.float32)
    combat_evolution_reward += 0.20 * enemy_hit.astype(jnp.float32)
    combat_evolution_reward += 12.0 * surviving_kill.astype(jnp.float32)
    combat_evolution_reward += 2.0 * (
        any_kill & ~surviving_kill).astype(jnp.float32)
    combat_evolution_reward -= 6.0 * info["mutual_death"].astype(jnp.float32)[:, None]
    if _REWARD_PROFILE == "combat_evolution":
        full_reward = combat_evolution_reward

    danger_arena_reward = -2.0 * death
    danger_arena_reward += 0.25 * enemy_hit.astype(jnp.float32)
    danger_arena_reward += 12.0 * surviving_kill.astype(jnp.float32)
    danger_arena_reward -= 6.0 * info["mutual_death"].astype(jnp.float32)[:, None]
    danger_arena_reward -= 8.0 * self_kill.astype(jnp.float32)
    danger_arena_reward -= _AVOIDABLE_DANGER_DEATH_PENALTY * info[
        "avoidable_danger_death"].astype(jnp.float32)
    danger_arena_reward += _DANGER_ESCAPE_REWARD * info[
        "danger_safe_resolution"].astype(jnp.float32)
    danger_arena_reward += tactical_bomb_shaping_from_events(
        info, _TACTICAL_BOMB_PLACEMENT_REWARD,
        _TACTICAL_BOMB_RESOLUTION_REWARD,
        _BASE_BOMB_REWARD, _FORCED_KILL_REWARD)
    danger_arena_reward = jnp.where(
        info["mutual_death"][:, None], -6.0, danger_arena_reward)

    combat_phase = info["combat_phase"].astype(jnp.float32)
    rush_phase = info["kill_window_active"].astype(jnp.float32)
    return_phase = info["return_phase"].astype(jnp.float32)
    window_fraction = (info["kill_window_remaining"].astype(jnp.float32)
                       / float(BUN_RESPAWN_TICKS))
    kill_window_reward = safety
    kill_window_reward += 0.75 * info["safe_bomb_escape"].astype(jnp.float32)
    kill_window_reward -= 2.0 * unsafe_detonation.astype(jnp.float32)
    kill_window_reward += 0.50 * combat_phase * enemy_hit.astype(jnp.float32)
    kill_window_reward += 8.0 * causal_kill.astype(jnp.float32)
    kill_window_reward += rush_phase * (
        0.30 * progress
        + 2.0 * info["enter_enemy_half"].astype(jnp.float32)
        + 8.0 * info["enter_enemy_base"].astype(jnp.float32)
        + (16.0 + 4.0 * window_fraction)
        * info["steal"].astype(jnp.float32))
    kill_window_reward += return_phase * (
        0.25 * progress
        - 10.0 * info["drop"].astype(jnp.float32)
        + 12.0 * info["carry_cross_home"].astype(jnp.float32)
        + 40.0 * info["capture"].astype(jnp.float32))
    kill_window_reward += 20.0 * win.astype(jnp.float32)
    kill_window_reward -= 20.0 * loss.astype(jnp.float32)
    lesson_kill_window_reward = kill_window_reward * jnp.asarray(
        [1.0, 0.0], jnp.float32)
    if _KILL_WINDOW_REWARD:
        full_reward = kill_window_reward

    lesson = info["lesson"][:, None]
    reward = jnp.where(lesson == LESSON_ROUTE_BREAK, route_reward,
                       jnp.where(lesson == LESSON_ROUTE_CAPTURE,
                                 bridge_capture_reward,
                                 jnp.where((lesson == LESSON_ROUTE_TO_BASE)
                                           | (lesson == LESSON_ROUTE_TO_BASE_RELAXED),
                                           bridge_reward,
                                 jnp.where((lesson == LESSON_COMBAT)
                                           | (lesson == LESSON_COMBAT_KILL),
                                           combat_reward,
                                           jnp.where((lesson == LESSON_COMBAT_STATIC)
                                                     | (lesson == LESSON_COMBAT_MOVING),
                                                     combat_hit_reward,
                                                     jnp.where(lesson == LESSON_KILL_RUSH,
                                                               lesson_kill_window_reward,
                                                     jnp.where(lesson == LESSON_NEAR_STEAL,
                                                               near_reward,
                                                               jnp.where((lesson == LESSON_CARRY_HOME)
                                                                         | (lesson == LESSON_CARRY_RETURN),
                                                                         carry_reward,
                                                                         full_reward))))))))
    reward = jnp.where(lesson == LESSON_FULL_CONTACT, full_reward, reward)
    reward = jnp.where(lesson == LESSON_FULL_AMBUSH, full_reward, reward)
    reward = jnp.where(lesson == LESSON_DANGER_ARENA,
                       danger_arena_reward, reward)
    return reward


def tactical_bomb_shaping_from_events(
        info, placement_reward: float, resolution_reward: float,
        base_bomb_reward: float = 0.0, forced_kill_reward: float = 0.0):
    """Return capped one-shot safe tactical bomb shaping by player.

    Every newly placed bomb earns ``base_bomb_reward`` (activity lift), a
    pressure placement adds ``placement_reward``, a forced-kill placement adds
    ``forced_kill_reward``, and a survived resolution adds ``resolution_reward``.
    All positive shaping is zeroed on self-detonation or a trade so suicide /
    trade bombs cannot farm reward.
    """
    bomb = info["bomb_placed"].astype(jnp.float32)
    placed = info["safe_tactical_bomb_placed"].astype(jnp.float32)
    forced = info["forced_kill_created"].astype(jnp.float32)
    resolved = info["tactical_bomb_safe_resolution"].astype(jnp.float32)
    trade = jnp.asarray(info["mutual_death"])
    while trade.ndim < placed.ndim:
        trade = trade[..., None]
    blocked = info["own_bomb_defeat"] | trade
    reward = (base_bomb_reward * bomb
              + placement_reward * placed
              + forced_kill_reward * forced
              + resolution_reward * resolved)
    return jnp.where(blocked, 0.0, reward)
