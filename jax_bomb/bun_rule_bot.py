"""Lightweight deterministic tactical rule bot for Bun mode.

The planner is deliberately bounded to 40 ticks (4 seconds at 10 Hz).  It uses
only public state, builds a time-expanded blast map with chain reactions, and
runs small breadth-first searches over grid cells.  It is a frozen code policy,
not a learned model and not a JAX-jittable function.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TICK_HZ = 10
HORIZON_STEPS = 40
DEFAULT_FUSE = 30
BLAST_LINGER_STEPS = 2
HEIGHT = 13
WIDTH = 15
MOVE_IDLE = 4
ABILITY_NONE = 0
ABILITY_BOMB = 1
ABILITY_ITEM = 2
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
COMPLEXITY = "O(HORIZON_STEPS * height * width + bombs^2 * blast) time; O(HORIZON_STEPS * height * width) space"
PHASE_NAMES = (
    "COMBAT", "KILL_CONFIRMED", "OBJECTIVE_RUSH",
    "CARRY_RETURN", "DELIVER", "RECOVER",
)


@dataclass(frozen=True)
class PredictedBomb:
    row: int
    col: int
    explode_step: int
    blast: int
    physical_owner: int
    causal_owner: int


@dataclass(frozen=True)
class Decision:
    action: np.ndarray
    reason: str
    doomed: bool
    claimed_escape: bool
    safe_escape_after_bomb: bool
    predicted_bombs: tuple[PredictedBomb, ...]
    survival_steps: int
    phase: str = "COMBAT"


@dataclass(frozen=True)
class _Player:
    row: int
    col: int
    alive: bool
    bombs: int
    blast: int
    speed: float


@dataclass(frozen=True)
class _Bomb:
    row: int
    col: int
    fuse: int
    blast: int
    physical_owner: int
    causal_owner: int


@dataclass
class _ParsedState:
    height: int
    width: int
    wall: np.ndarray
    brick: np.ndarray
    blast_linger: np.ndarray
    players: tuple[_Player, _Player]
    bombs: list[_Bomb]
    bun_bases: tuple[tuple[int, int], tuple[int, int]]
    bun_carried: tuple[int, int]
    bun_respawn: tuple[int, int]
    bun_loose: np.ndarray


@dataclass(frozen=True)
class _Plan:
    survived: bool
    first_action: int
    survival_steps: int
    end_cell: int
    safe_actions: tuple[int, ...] = ()


def _cells_to_grid(cells: Iterable[Sequence[int]], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.bool_)
    for row, col in cells:
        if 0 <= int(row) < height and 0 <= int(col) < width:
            grid[int(row), int(col)] = True
    return grid


def _grid(value: Any, cells: Iterable[Sequence[int]], height: int, width: int,
          dtype: np.dtype[Any] = np.bool_) -> np.ndarray:
    if value is None:
        return _cells_to_grid(cells, height, width).astype(dtype, copy=False)
    array = np.asarray(value, dtype=dtype)
    if array.size != height * width:
        raise ValueError(f"grid has {array.size} values, expected {height * width}")
    return array.reshape(height, width).copy()


def _parse_state(state: Mapping[str, Any]) -> _ParsedState:
    height = int(state.get("height", HEIGHT))
    width = int(state.get("width", WIDTH))
    players_raw = state.get("players")
    if not isinstance(players_raw, Sequence) or len(players_raw) != 2:
        raise ValueError("state.players must contain exactly two players")
    players = tuple(
        _Player(
            row=int(player["row"]),
            col=int(player["col"]),
            alive=bool(player.get("alive", True)),
            bombs=max(0, int(player.get("bombs", 0))),
            blast=max(1, int(player.get("blast", 2))),
            speed=max(0.1, float(player.get("speed", 1.3))),
        )
        for player in players_raw
    )
    bombs = [
        _Bomb(
            row=int(bomb["row"]),
            col=int(bomb["col"]),
            fuse=max(1, min(HORIZON_STEPS + 1, int(bomb.get("fuse", DEFAULT_FUSE)))),
            blast=max(1, int(bomb.get("blast", 2))),
            physical_owner=int(bomb.get("physical_owner", bomb.get("owner", -1))),
            causal_owner=int(bomb.get("causal_owner", bomb.get("physical_owner", bomb.get("owner", -1)))),
        )
        for bomb in state.get("bombs", ())
    ]
    bases_raw = state.get("bun_bases", ((1, 4), (1, 8)))
    bun_bases = tuple((int(base[0]), int(base[1])) for base in bases_raw)
    carried_raw = state.get("bun_carried", (-1, -1))
    respawn_raw = state.get("bun_respawn", (0, 0))
    loose = np.zeros((height, width, 2), dtype=np.int16)
    if state.get("bun_loose") is not None:
        raw = np.asarray(state["bun_loose"], dtype=np.int16)
        if raw.size != height * width * 2:
            raise ValueError("bun_loose must have height*width*2 values")
        loose = raw.reshape(height, width, 2).copy()
    for item in state.get("bun_loose_cells", ()):
        loose[int(item["row"]), int(item["col"]), int(item["origin"])] = int(item.get("count", 1))
    return _ParsedState(
        height=height,
        width=width,
        wall=_grid(state.get("wall"), state.get("wall_cells", ()), height, width),
        brick=_grid(state.get("brick"), state.get("brick_cells", ()), height, width),
        blast_linger=_grid(
            state.get("blast_linger"), state.get("blast_cells", ()),
            height, width, np.int16),
        players=players,  # type: ignore[arg-type]
        bombs=bombs,
        bun_bases=bun_bases,  # type: ignore[arg-type]
        bun_carried=(int(carried_raw[0]), int(carried_raw[1])),
        bun_respawn=(int(respawn_raw[0]), int(respawn_raw[1])),
        bun_loose=loose,
    )


class BunRuleBot:
    """Bounded Bun tactical planner with deterministic tie-breaking."""

    def __init__(self, horizon_steps: int = HORIZON_STEPS):
        if horizon_steps != HORIZON_STEPS:
            raise ValueError("BunRuleBot uses the fixed 40-step/4-second contract")
        self.horizon_steps = horizon_steps
        self._topology_key: tuple[Any, ...] | None = None
        self._neighbors: tuple[tuple[int, ...], ...] = ()
        self._route_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._phases = ["COMBAT", "COMBAT"]

    def reset(self) -> None:
        self._phases = ["COMBAT", "COMBAT"]

    def phase(self, player_id: int = 0) -> str:
        return self._phases[player_id]

    def restore_phase(self, player_id: int, phase: str) -> None:
        if phase not in PHASE_NAMES:
            raise ValueError(f"unknown Bun rule-bot phase: {phase}")
        self._phases[player_id] = phase

    @staticmethod
    def _event_flag(info: Mapping[str, Any], name: str, player_id: int) -> bool:
        aliases = {
            "causal_kill": ("causal_kill", "causalKill"),
            "credited_kill": ("credited_kill", "creditedKill"),
            "mutual_death": ("mutual_death", "mutualDeath"),
            "own_bomb_defeat": ("own_bomb_defeat", "ownBombDefeat"),
            "death": ("death", "died"),
            "drop": ("drop",),
            "steal": ("steal",),
            "capture": ("capture",),
        }
        for key in aliases.get(name, (name,)):
            if key not in info:
                continue
            value = info[key]
            if name == "mutual_death":
                return bool(np.asarray(value))
            array = np.asarray(value)
            return bool(array[player_id]) if array.ndim else bool(array)
        return False

    def observe_transition(self, info: Mapping[str, Any], next_state: Mapping[str, Any],
                           player_id: int = 0) -> str:
        parsed = _parse_state(next_state)
        player = parsed.players[player_id]
        enemy = 1 - player_id
        valid_kill = (
            self._event_flag(info, "causal_kill", player_id)
            and self._event_flag(info, "credited_kill", player_id)
            and not self._event_flag(info, "mutual_death", player_id)
            and player.alive
        )
        if self._event_flag(info, "capture", player_id):
            self._phases[player_id] = "DELIVER"
        elif (self._event_flag(info, "death", player_id)
              or self._event_flag(info, "drop", player_id)
              or not player.alive):
            self._phases[player_id] = "RECOVER"
        elif parsed.bun_carried[player_id] == enemy:
            self._phases[player_id] = "CARRY_RETURN"
        elif valid_kill:
            self._phases[player_id] = "KILL_CONFIRMED"
        elif self._phases[player_id] == "KILL_CONFIRMED":
            self._phases[player_id] = "OBJECTIVE_RUSH"
        else:
            self.sync_phase(next_state, player_id)
        return self._phases[player_id]

    def sync_phase(self, state: Mapping[str, Any], player_id: int = 0) -> str:
        parsed = _parse_state(state)
        player = parsed.players[player_id]
        enemy = 1 - player_id
        current = self._phases[player_id]
        if current == "DELIVER":
            return current
        if not player.alive:
            self._phases[player_id] = "RECOVER"
        elif parsed.bun_carried[player_id] == enemy:
            self._phases[player_id] = "CARRY_RETURN"
        elif current == "CARRY_RETURN":
            self._phases[player_id] = "RECOVER"
        elif current == "RECOVER":
            has_loose = bool(np.any(parsed.bun_loose[:, :, enemy] > 0))
            if not has_loose:
                self._phases[player_id] = (
                    "COMBAT" if parsed.players[enemy].alive else "OBJECTIVE_RUSH")
        elif current == "OBJECTIVE_RUSH" and parsed.players[enemy].alive:
            self._phases[player_id] = "COMBAT"
        return self._phases[player_id]

    def decide(self, state: Mapping[str, Any], player_id: int = 0) -> np.ndarray:
        return self.analyze(state, player_id).action

    def analyze(self, state: Mapping[str, Any], player_id: int = 0) -> Decision:
        self.sync_phase(state, player_id)
        parsed = _parse_state(state)
        if player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        player = parsed.players[player_id]
        if not player.alive:
            return Decision(np.asarray([MOVE_IDLE, ABILITY_NONE], np.int32), "dead", False,
                            False, False, (), 0, self.phase(player_id))

        neighbors = self._get_neighbors(parsed)
        danger, predicted, bomb_until = self._predict(parsed)
        base_plan = self._survival_plan(parsed, player_id, danger, bomb_until, neighbors)
        exposed = self._future_hit(danger, player.row, player.col)
        if exposed:
            reason = "escape_immediate" if base_plan.survived else "doomed_max_survival"
            return Decision(
                np.asarray([base_plan.first_action, ABILITY_NONE], np.int32),
                reason,
                not base_plan.survived,
                base_plan.survived,
                False,
                predicted,
                base_plan.survival_steps,
                self.phase(player_id),
            )

        phase = self.phase(player_id)
        if phase in ("KILL_CONFIRMED", "OBJECTIVE_RUSH", "CARRY_RETURN", "RECOVER", "DELIVER"):
            if phase == "DELIVER":
                return Decision(np.asarray([MOVE_IDLE, ABILITY_NONE], np.int32), "delivered",
                                False, True, False, predicted, base_plan.survival_steps, phase)
            goals = self._phase_goals(parsed, player_id, phase)
            if goals:
                move = self._goal_move(parsed, player_id, goals, base_plan, neighbors)
                return Decision(
                    np.asarray([move, ABILITY_NONE], np.int32),
                    "objective_route" if move != MOVE_IDLE else "objective_wait",
                    False, base_plan.survived, False, predicted,
                    base_plan.survival_steps, phase)

        if not parsed.players[1 - player_id].alive:
            return Decision(np.asarray([MOVE_IDLE, ABILITY_NONE], np.int32),
                            "unattributed_enemy_down", False, True, False,
                            predicted, base_plan.survival_steps, self.phase(player_id))

        attack = self._safe_attack(parsed, player_id, neighbors)
        if attack is not None:
            plan, attack_predicted, verified_kill = attack
            return Decision(
                np.asarray([plan.first_action, ABILITY_BOMB], np.int32),
                "safe_non_trade_attack" if verified_kill else "safe_pressure_attack",
                False,
                True,
                True,
                attack_predicted,
                plan.survival_steps,
                self.phase(player_id),
            )

        move = self._control_move(parsed, player_id, danger, bomb_until, neighbors)
        return Decision(
            np.asarray([move, ABILITY_NONE], np.int32),
            "control_space" if move != MOVE_IDLE else "wait_no_safe_attack",
            False,
            base_plan.survived,
            False,
            predicted,
            base_plan.survival_steps,
            self.phase(player_id),
        )

    def _get_neighbors(self, state: _ParsedState) -> tuple[tuple[int, ...], ...]:
        static = state.wall | state.brick
        key = (state.height, state.width, static.tobytes())
        if key == self._topology_key:
            return self._neighbors
        neighbors: list[tuple[int, ...]] = []
        for row in range(state.height):
            for col in range(state.width):
                values = []
                for drow, dcol in DIRS:
                    next_row, next_col = row + drow, col + dcol
                    if (0 <= next_row < state.height and 0 <= next_col < state.width
                            and not static[next_row, next_col]):
                        values.append(next_row * state.width + next_col)
                    else:
                        values.append(-1)
                neighbors.append(tuple(values))
        self._topology_key = key
        self._neighbors = tuple(neighbors)
        return self._neighbors

    def _blast_cells(self, state: _ParsedState, row: int, col: int, blast: int,
                     live_bombs: set[tuple[int, int]]) -> list[tuple[int, int]]:
        cells = [(row, col)]
        for drow, dcol in DIRS:
            for distance in range(1, blast + 1):
                next_row, next_col = row + drow * distance, col + dcol * distance
                if not (0 <= next_row < state.height and 0 <= next_col < state.width):
                    break
                if state.wall[next_row, next_col]:
                    break
                cells.append((next_row, next_col))
                if state.brick[next_row, next_col] or (next_row, next_col) in live_bombs:
                    break
        return cells

    def _predict(self, state: _ParsedState) -> tuple[np.ndarray, tuple[PredictedBomb, ...], np.ndarray]:
        danger = np.zeros((self.horizon_steps + 1, state.height, state.width), dtype=np.bool_)
        linger_rows, linger_cols = np.nonzero(state.blast_linger > 0)
        for row, col in zip(linger_rows.tolist(), linger_cols.tolist()):
            linger = min(self.horizon_steps + 1, max(1, int(state.blast_linger[row, col])))
            danger[:linger, row, col] = True

        bombs = [
            {"row": bomb.row, "col": bomb.col, "step": bomb.fuse, "blast": bomb.blast,
             "physical": bomb.physical_owner, "causal": bomb.causal_owner, "done": False}
            for bomb in state.bombs
        ]
        live_cells = {(bomb["row"], bomb["col"]) for bomb in bombs}
        while True:
            pending = [bomb for bomb in bombs if not bomb["done"]]
            if not pending:
                break
            step = min(int(bomb["step"]) for bomb in pending)
            same_step = deque(bomb for bomb in pending if int(bomb["step"]) == step)
            while same_step:
                bomb = same_step.popleft()
                if bomb["done"]:
                    continue
                bomb["done"] = True
                live_cells.discard((int(bomb["row"]), int(bomb["col"])))
                footprint = self._blast_cells(
                    state, int(bomb["row"]), int(bomb["col"]), int(bomb["blast"]), live_cells)
                if step <= self.horizon_steps:
                    end = min(self.horizon_steps + 1, step + BLAST_LINGER_STEPS)
                    for row, col in footprint:
                        danger[step:end, row, col] = True
                footprint_set = set(footprint)
                for other in bombs:
                    if (not other["done"] and int(other["step"]) > step
                            and (int(other["row"]), int(other["col"])) in footprint_set):
                        other["step"] = step
                        other["causal"] = int(bomb["causal"])
                        same_step.append(other)

        bomb_until = np.zeros((state.height, state.width), dtype=np.int16)
        predicted = []
        for bomb in bombs:
            explode_step = int(bomb["step"])
            bomb_until[int(bomb["row"]), int(bomb["col"])] = min(
                self.horizon_steps + 1, explode_step)
            predicted.append(PredictedBomb(
                int(bomb["row"]), int(bomb["col"]), explode_step, int(bomb["blast"]),
                int(bomb["physical"]), int(bomb["causal"])))
        predicted.sort(key=lambda bomb: (bomb.explode_step, bomb.row, bomb.col))
        return danger, tuple(predicted), bomb_until

    def _future_hit(self, danger: np.ndarray, row: int, col: int) -> bool:
        return bool(np.any(danger[:, row, col]))

    def _move_ticks(self, player: _Player) -> int:
        return max(1, int(ceil(1.0 / (0.3 * player.speed))))

    def _transition_safe(self, danger: np.ndarray, bomb_until: np.ndarray,
                         source: int, target: int, start: int, end: int,
                         width: int, allow_leave_bomb: bool) -> bool:
        source_row, source_col = divmod(source, width)
        target_row, target_col = divmod(target, width)
        if target != source and bomb_until[target_row, target_col] >= end:
            return False
        if target == source and bomb_until[source_row, source_col] >= end and not allow_leave_bomb:
            return False
        for tick in range(start + 1, min(end, self.horizon_steps) + 1):
            if danger[tick, source_row, source_col] or danger[tick, target_row, target_col]:
                return False
        return True

    def _survival_plan(self, state: _ParsedState, player_id: int, danger: np.ndarray,
                       bomb_until: np.ndarray, neighbors: tuple[tuple[int, ...], ...],
                       start_on_new_bomb: bool = False) -> _Plan:
        player = state.players[player_id]
        start = player.row * state.width + player.col
        move_ticks = self._move_ticks(player)
        count = state.height * state.width
        all_mask = (1 << count) - 1
        static_flat = (state.wall | state.brick).reshape(-1)
        open_mask = all_mask ^ int.from_bytes(
            np.packbits(static_flat, bitorder="little").tobytes(), "little")
        danger_masks = [
            int.from_bytes(np.packbits(layer.reshape(-1), bitorder="little").tobytes(), "little")
            for layer in danger
        ]
        bomb_cells = [(int(index), int(value)) for index, value in enumerate(bomb_until.reshape(-1)) if value > 0]
        bomb_masks = [0] * (self.horizon_steps + 1)
        for tick in range(self.horizon_steps + 1):
            mask = 0
            for cell, until in bomb_cells:
                if until >= tick:
                    mask |= 1 << cell
            bomb_masks[tick] = mask
        left_column = sum(1 << (row * state.width) for row in range(state.height))
        right_column = sum(1 << (row * state.width + state.width - 1) for row in range(state.height))

        def shift(mask: int, action: int) -> int:
            if action == 0:
                return mask >> state.width
            if action == 1:
                return (mask << state.width) & all_mask
            if action == 2:
                return (mask & ~left_column) >> 1
            if action == 3:
                return ((mask & ~right_column) << 1) & all_mask
            return mask

        safe_between: dict[tuple[int, int], int] = {}

        def interval_safe(start_tick: int, end_tick: int) -> int:
            key = (start_tick, end_tick)
            cached = safe_between.get(key)
            if cached is not None:
                return cached
            blocked = 0
            for tick in range(start_tick + 1, end_tick + 1):
                blocked |= danger_masks[tick]
            safe = all_mask ^ blocked
            safe_between[key] = safe
            return safe

        reached = [[0] * 5 for _ in range(self.horizon_steps + 1)]
        start_mask = 1 << start
        for action in (0, 1, 2, 3, MOVE_IDLE):
            duration = 1 if action == MOVE_IDLE else move_ticks
            if duration > self.horizon_steps:
                continue
            safe = interval_safe(0, duration)
            source = start_mask & safe
            target = shift(source, action) & safe & open_mask
            if action != MOVE_IDLE:
                target &= ~bomb_masks[duration]
            elif start_on_new_bomb:
                target = 0
            reached[duration][action] |= target

        best_tick = 0
        best_actions: list[int] = []
        for tick in range(1, self.horizon_steps + 1):
            active_actions = [action for action in range(5) if reached[tick][action]]
            if active_actions:
                best_tick = tick
                best_actions = active_actions
            if tick == self.horizon_steps and active_actions:
                first = self._rank_first_action(
                    state, player_id, active_actions, start, neighbors)
                mask = reached[tick][first]
                return _Plan(True, first, self.horizon_steps,
                             (mask & -mask).bit_length() - 1,
                             tuple(active_actions))
            for first in active_actions:
                current = reached[tick][first]
                if tick + 1 <= self.horizon_steps:
                    safe = interval_safe(tick, tick + 1)
                    reached[tick + 1][first] |= current & safe
                end = tick + move_ticks
                if end > self.horizon_steps:
                    continue
                safe = interval_safe(tick, end)
                source = current & safe
                for action in range(4):
                    target = shift(source, action) & safe & open_mask & ~bomb_masks[end]
                    reached[end][first] |= target
        first = min(best_actions) if best_actions else MOVE_IDLE
        mask = reached[best_tick][first] if best_actions else start_mask
        return _Plan(False, first, best_tick, (mask & -mask).bit_length() - 1,
                     tuple(best_actions))

    def _base_cells(self, state: _ParsedState, team: int) -> set[int]:
        anchor = state.bun_bases[team]
        cells = set()
        for row in range(anchor[0], anchor[0] + 3):
            for col in range(anchor[1], anchor[1] + 3):
                if (0 <= row < state.height and 0 <= col < state.width
                        and not state.wall[row, col] and not state.brick[row, col]):
                    cells.add(row * state.width + col)
        return cells

    def _phase_goals(self, state: _ParsedState, player_id: int,
                     phase: str) -> set[int]:
        enemy = 1 - player_id
        if phase in ("KILL_CONFIRMED", "OBJECTIVE_RUSH"):
            return self._base_cells(state, enemy)
        if phase == "CARRY_RETURN":
            return self._base_cells(state, player_id)
        if phase == "RECOVER":
            cells = np.argwhere(state.bun_loose[:, :, enemy] > 0)
            return {int(row) * state.width + int(col) for row, col in cells}
        return set()

    def _distance_to_goals(self, state: _ParsedState, goals: set[int],
                           neighbors: tuple[tuple[int, ...], ...]) -> np.ndarray:
        static = state.wall | state.brick
        key = (state.height, state.width, static.tobytes(), tuple(sorted(goals)))
        cached = self._route_cache.get(key)
        if cached is not None:
            return cached
        distance = np.full(state.height * state.width, 1 << 14, dtype=np.int16)
        queue = deque()
        for goal in goals:
            distance[goal] = 0
            queue.append(goal)
        while queue:
            cell = queue.popleft()
            candidate = int(distance[cell]) + 1
            for neighbor in neighbors[cell]:
                if neighbor >= 0 and candidate < int(distance[neighbor]):
                    distance[neighbor] = candidate
                    queue.append(neighbor)
        self._route_cache[key] = distance
        return distance

    def _goal_move(self, state: _ParsedState, player_id: int, goals: set[int],
                   plan: _Plan,
                   neighbors: tuple[tuple[int, ...], ...]) -> int:
        player = state.players[player_id]
        start = player.row * state.width + player.col
        if start in goals:
            return MOVE_IDLE
        safe_actions = plan.safe_actions if plan.survived else (plan.first_action,)
        distance = self._distance_to_goals(state, goals, neighbors)
        best_action = MOVE_IDLE
        best_score = 1 << 30
        for action in safe_actions:
            target = start if action == MOVE_IDLE else neighbors[start][action]
            if target < 0:
                continue
            score = int(distance[target]) * 10 + (2 if action == MOVE_IDLE else 0) + action
            if score < best_score:
                best_score = score
                best_action = action
        return best_action

    def _rank_first_action(self, state: _ParsedState, player_id: int,
                           actions: Sequence[int], start: int,
                           neighbors: tuple[tuple[int, ...], ...]) -> int:
        player = state.players[player_id]
        opponent = state.players[1 - player_id]
        bomb_cells = [(bomb.row, bomb.col) for bomb in state.bombs]
        best_action = actions[0]
        best_score = -1e9
        for action in actions:
            target = start if action == MOVE_IDLE else neighbors[start][action]
            if target < 0:
                continue
            row, col = divmod(target, state.width)
            bomb_distance = min(
                (abs(row - bomb_row) + abs(col - bomb_col) for bomb_row, bomb_col in bomb_cells),
                default=0)
            away_alignment = 0
            if bomb_cells and action != MOVE_IDLE:
                nearest_row, nearest_col = min(
                    bomb_cells,
                    key=lambda cell: abs(player.row - cell[0]) + abs(player.col - cell[1]))
                away_alignment = ((row - player.row) * (player.row - nearest_row)
                                  + (col - player.col) * (player.col - nearest_col))
            opponent_distance = abs(row - opponent.row) + abs(col - opponent.col)
            opponent_alignment = ((row - player.row) * (player.row - opponent.row)
                                  + (col - player.col) * (player.col - opponent.col))
            degree = sum(1 for value in neighbors[target] if value >= 0)
            score = (3.0 * bomb_distance + 2.0 * away_alignment + opponent_distance
                     + opponent_alignment + 0.1 * degree)
            if action == MOVE_IDLE:
                score -= 0.5
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _safe_attack(self, state: _ParsedState, player_id: int,
                     neighbors: tuple[tuple[int, ...], ...]) -> tuple[_Plan, tuple[PredictedBomb, ...], bool] | None:
        player = state.players[player_id]
        opponent_id = 1 - player_id
        opponent = state.players[opponent_id]
        if player.bombs <= 0 or not opponent.alive:
            return None
        if any(bomb.row == player.row and bomb.col == player.col for bomb in state.bombs):
            return None
        if not self._line_of_blast(state, player, opponent):
            return None
        attack_state = _ParsedState(
            state.height, state.width, state.wall, state.brick, state.blast_linger,
            state.players,
            state.bombs + [_Bomb(player.row, player.col, DEFAULT_FUSE, player.blast,
                                 player_id, player_id)],
            state.bun_bases, state.bun_carried, state.bun_respawn,
            state.bun_loose,
        )
        danger, predicted, bomb_until = self._predict(attack_state)
        own_plan = self._survival_plan(
            attack_state, player_id, danger, bomb_until, neighbors, start_on_new_bomb=True)
        if not own_plan.survived or own_plan.first_action == MOVE_IDLE:
            return None
        opponent_plan = self._survival_plan(
            attack_state, opponent_id, danger, bomb_until, neighbors)
        return own_plan, predicted, not opponent_plan.survived

    def _line_of_blast(self, state: _ParsedState, player: _Player, opponent: _Player) -> bool:
        if player.row != opponent.row and player.col != opponent.col:
            return False
        distance = abs(player.row - opponent.row) + abs(player.col - opponent.col)
        if distance > player.blast:
            return False
        drow = 0 if player.row == opponent.row else (1 if opponent.row > player.row else -1)
        dcol = 0 if player.col == opponent.col else (1 if opponent.col > player.col else -1)
        for step in range(1, distance + 1):
            row, col = player.row + drow * step, player.col + dcol * step
            if state.wall[row, col] or state.brick[row, col]:
                return False
            if step < distance and any(bomb.row == row and bomb.col == col for bomb in state.bombs):
                return False
        return True

    def _control_move(self, state: _ParsedState, player_id: int, danger: np.ndarray,
                      bomb_until: np.ndarray, neighbors: tuple[tuple[int, ...], ...]) -> int:
        player = state.players[player_id]
        opponent = state.players[1 - player_id]
        start = player.row * state.width + player.col
        move_ticks = self._move_ticks(player)
        best_action = MOVE_IDLE
        best_score = -1e9
        for action in (0, 1, 2, 3, MOVE_IDLE):
            target = start if action == MOVE_IDLE else neighbors[start][action]
            if target < 0:
                continue
            end = 1 if action == MOVE_IDLE else move_ticks
            if not self._transition_safe(danger, bomb_until, start, target, 0, end,
                                         state.width, False):
                continue
            row, col = divmod(target, state.width)
            distance = abs(row - opponent.row) + abs(col - opponent.col)
            degree = sum(1 for value in neighbors[target] if value >= 0)
            idle_penalty = 0.25 if action == MOVE_IDLE else 0.0
            score = -distance + 0.08 * degree - idle_penalty
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


def decide_batch(states: Sequence[Mapping[str, Any]],
                 player_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
    """Precompute-friendly CPU batch wrapper; loops in Python, no Node bridge."""
    if player_ids is None:
        player_ids = np.zeros(len(states), dtype=np.int32)
    if len(player_ids) != len(states):
        raise ValueError("player_ids length must match states")
    bot = BunRuleBot()
    return np.stack([
        bot.decide(state, int(player_id))
        for state, player_id in zip(states, player_ids)
    ]).astype(np.int32, copy=False)


def state_from_bun_state(state: Any) -> dict[str, Any]:
    """Convert a public ``bun_env.BunState`` into the stable rule-bot mapping.

    This adapter intentionally imports no JAX modules.  Callers may convert a
    device state outside a jitted region, then use :func:`decide_batch` for CPU
    opponent precomputation.
    """
    core = state.core
    fuse = np.asarray(core.fuse)
    owner = np.asarray(core.owner)
    blast = np.asarray(core.bomb_blast)
    positions = np.asarray(core.pos)
    alive = np.asarray(core.alive)
    bombs_cap = np.asarray(core.bombs_cap)
    blast_cap = np.asarray(core.blast_cap)
    speed = np.asarray(core.spd_g)
    bombs = []
    for row, col in np.argwhere(fuse > 0):
        physical = int(owner[row, col])
        bombs.append({
            "row": int(row), "col": int(col), "fuse": int(fuse[row, col]),
            "blast": int(blast[row, col]), "physical_owner": physical,
            "causal_owner": physical,
        })
    live_counts = [sum(1 for bomb in bombs if bomb["physical_owner"] == pid) for pid in range(2)]
    return {
        "height": int(np.asarray(core.wall).shape[0]),
        "width": int(np.asarray(core.wall).shape[1]),
        "wall": np.asarray(core.wall, dtype=np.bool_),
        "brick": np.asarray(core.brick, dtype=np.bool_),
        "blast_linger": np.asarray(core.blast_linger, dtype=np.int16),
        "players": [
            {
                "row": int(np.floor(positions[pid, 0])),
                "col": int(np.floor(positions[pid, 1])),
                "alive": bool(alive[pid]),
                "bombs": max(0, int(bombs_cap[pid]) - live_counts[pid]),
                "blast": int(blast_cap[pid]),
                "speed": float(speed[pid]),
            }
            for pid in range(2)
        ],
        "bombs": bombs,
        "bun_bases": [
            [int(value) for value in np.asarray(base)]
            for base in getattr(state, "bun_bases", ((1, 4), (1, 8)))
        ] if hasattr(state, "bun_bases") else [[1, 4], [1, 8]],
        "bun_carried": np.asarray(state.bun_carried, dtype=np.int8).tolist(),
        "bun_respawn": np.asarray(state.bun_respawn, dtype=np.int16).tolist(),
        "bun_loose": np.asarray(state.bun_loose, dtype=np.int16),
    }


__all__ = [
    "ABILITY_BOMB", "ABILITY_ITEM", "ABILITY_NONE", "BunRuleBot", "COMPLEXITY",
    "Decision", "HORIZON_STEPS", "PHASE_NAMES", "PredictedBomb", "TICK_HZ", "decide_batch",
    "state_from_bun_state",
]
