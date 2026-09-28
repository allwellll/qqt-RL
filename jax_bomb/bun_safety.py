"""Owner-aware counterfactual safety analysis for Bun joint actions."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import bun_env as env
from . import jax_env as base


_INF_TICK = jnp.int32(1 << 14)
_NEG_LOGIT = -1e9
_ALL_MOVES = jnp.arange(env.N_MOVES, dtype=jnp.int32)
_ALL_ABILITIES = jnp.arange(env.N_BOMB, dtype=jnp.int32)


class SafetyAnalysis(NamedTuple):
    legal: jnp.ndarray
    survivable: jnp.ndarray
    avoidable: jnp.ndarray
    doomed: jnp.ndarray
    own_deadline: jnp.ndarray


class SelectedSafety(NamedTuple):
    exposed: jnp.ndarray
    escapable: jnp.ndarray
    doomed: jnp.ndarray
    selected_survivable: jnp.ndarray
    avoidable: jnp.ndarray
    resolution_ticks: jnp.ndarray
    uniquely_own_hazard: jnp.ndarray


class TacticalBombAnalysis(NamedTuple):
    safe: jnp.ndarray
    tactical: jnp.ndarray
    newly_threatens_enemy: jnp.ndarray
    enemy_safe_moves_before: jnp.ndarray
    enemy_safe_moves_after: jnp.ndarray


def adjusted_joint_logits(move_logits, ability_logits, move_mask,
                          ability_mask, avoidable, mode: str,
                          penalty: float):
    """Build a legal joint-action distribution with an optional shield."""
    legal = move_mask[..., :, None] & ability_mask[..., None, :]
    logits = move_logits[..., :, None] + ability_logits[..., None, :]
    if mode == "hard":
        allowed = legal & ~avoidable
        has_allowed = allowed.any(axis=(-2, -1), keepdims=True)
        allowed = jnp.where(has_allowed, allowed, legal)
        logits = jnp.where(allowed, logits, _NEG_LOGIT)
    elif mode == "soft":
        logits = jnp.where(
            legal, logits - penalty * avoidable.astype(logits.dtype),
            _NEG_LOGIT)
    elif mode == "off":
        logits = jnp.where(legal, logits, _NEG_LOGIT)
    else:
        raise ValueError(f"unknown safety mode: {mode}")
    return logits.reshape(logits.shape[:-2] + (env.N_MOVES * env.N_BOMB,))


def _place_candidate(core, player: int):
    fuse = jnp.where(core.fuse > 0, core.fuse - 1, core.fuse)
    owner = core.owner
    bomb_blast = core.bomb_blast
    cell = jnp.clip(core.pos[player].astype(jnp.int32),
                    jnp.asarray([0, 0]),
                    jnp.asarray([env.H - 1, env.W - 1]))
    row, column = cell[0], cell[1]
    live = ((owner == player) & (fuse > 0)).sum()
    can_place = (core.alive[player]
                 & (fuse[row, column] <= 0)
                 & ~core.wall[row, column]
                 & ~core.brick[row, column]
                 & (live < core.bombs_cap[player]))
    fuse = fuse.at[row, column].set(
        jnp.where(can_place, base.FUSE, fuse[row, column]))
    owner = owner.at[row, column].set(
        jnp.where(can_place, player, owner[row, column]))
    bomb_blast = bomb_blast.at[row, column].set(jnp.where(
        can_place, core.blast_cap[player].astype(jnp.int32),
        bomb_blast[row, column]))
    return fuse, owner, bomb_blast


def _candidate_bomb_field(core, place_player: int | None):
    fuse = jnp.where(core.fuse > 0, core.fuse - 1, core.fuse)
    owner = core.owner
    bomb_blast = core.bomb_blast
    if place_player is not None:
        fuse, owner, bomb_blast = _place_candidate(core, place_player)
    return fuse, owner, bomb_blast


def _source_deadlines(core, blast_owner_linger, place_player: int | None):
    fuse, owner, bomb_blast = _candidate_bomb_field(core, place_player)
    bombed = owner >= 0
    blast = jnp.where(bomb_blast > 0, bomb_blast, base.BLAST)
    max_time = float(base.FUSE + 1)
    detonation = jnp.where(bombed, jnp.maximum(fuse, 0), _INF_TICK)
    score = jnp.where(
        bombed,
        (max_time - detonation.astype(jnp.float32)) / max_time,
        0.0)
    propagated = base._stage_a_matrix(
        score, blast.astype(jnp.float32), bombed, core.wall, core.brick,
        base.MAX_CHAIN)
    passable = (~core.wall).astype(jnp.float32)
    not_solid = (~bombed & ~core.brick).astype(jnp.float32)
    deadlines = []
    for source in range(2):
        source_seed = jnp.where(owner == source, propagated, 0.0)
        source_blast = jnp.where(owner == source, blast, 0)
        danger_score = jnp.maximum(
            source_seed,
            base._spread_all(source_seed, source_blast, passable, not_solid))
        deadline = jnp.where(
            danger_score > 0,
            jnp.rint(max_time * (1.0 - danger_score)).astype(jnp.int32),
            _INF_TICK)
        lingering = (blast_owner_linger[source] > 1).any(axis=0)
        deadlines.append(jnp.where(lingering, 0, deadline))
    return jnp.stack(deadlines), fuse, owner


def owned_hazard_deadlines(state: env.BunState,
                           place_player: int | None = None):
    """Return uniquely attributable own-bomb hazard deadlines for both players.

    A cell is not attributed to a player when the opponent's blast reaches it
    first or the two source windows overlap. This mirrors the environment's
    unique-owner kill credit and deliberately prefers false negatives over
    blaming an enemy or ambiguous explosion on the acting player.
    """
    deadlines, fuse, owner = _source_deadlines(
        state.core, state.blast_owner_linger, place_player)
    result = []
    for player in range(2):
        own = deadlines[player]
        other = deadlines[1 - player]
        uniquely_first = ((own < _INF_TICK)
                          & ((other >= _INF_TICK)
                             | (own + base.BLAST_LINGER_TICKS < other)))
        result.append(jnp.where(uniquely_first, own, _INF_TICK))
    return jnp.stack(result), fuse, owner


def _candidate_position(state: env.BunState, player: int, move, ability):
    base_fuse, _base_owner, _base_blast = _candidate_bomb_field(
        state.core, None)
    placed_fuse, _placed_owner, _placed_blast = _candidate_bomb_field(
        state.core, player)
    fuse = jnp.where(ability == 1, placed_fuse, base_fuse)
    blocked = (fuse > 0) | state.core.wall | state.core.brick
    requested = jnp.where(
        state.move_status[player] == env.STATUS_SLIDE,
        state.slide_dir[player], move)
    return base._steer(
        state.core.pos[player], requested, state.core.alive[player], blocked,
        _movement_scale_for_player(state, player), state.core.pushable)


def _movement_scale_for_player(state: env.BunState, player: int):
    return env._movement_scale(state)[player]


def _shift_arrivals(arrival):
    inf_row = jnp.full(arrival.shape[:-2] + (1, env.W), _INF_TICK)
    inf_column = jnp.full(arrival.shape[:-2] + (env.H, 1), _INF_TICK)
    from_up = jnp.concatenate([inf_row, arrival[..., :-1, :]], axis=-2)
    from_down = jnp.concatenate([arrival[..., 1:, :], inf_row], axis=-2)
    from_left = jnp.concatenate([inf_column, arrival[..., :, :-1]], axis=-1)
    from_right = jnp.concatenate([arrival[..., :, 1:], inf_column], axis=-1)
    return jnp.minimum(
        jnp.minimum(from_up, from_down),
        jnp.minimum(from_left, from_right))


def _can_escape(position, passable, deadline, speed_scale, margin_ticks):
    cell = jnp.clip(position.astype(jnp.int32),
                    jnp.asarray([0, 0]),
                    jnp.asarray([env.H - 1, env.W - 1]))
    start = jnp.zeros((env.H, env.W), jnp.bool_).at[
        cell[0], cell[1]].set(True)
    allowed = passable & ~start
    arrival = jnp.full((env.H, env.W), _INF_TICK)
    start_permanently_safe = deadline[cell[0], cell[1]] >= _INF_TICK

    step_distance = jnp.maximum(
        base.STEP * speed_scale.astype(jnp.float32), 1e-3)
    edge_ticks = jnp.ceil(1.0 / step_distance).astype(jnp.int32)
    for direction, (drow, dcolumn) in enumerate(base._DIRS):
        target = cell + jnp.asarray([drow, dcolumn], jnp.int32)
        in_bounds = ((target[0] >= 0) & (target[0] < env.H)
                     & (target[1] >= 0) & (target[1] < env.W))
        target = jnp.clip(target, jnp.asarray([0, 0]),
                          jnp.asarray([env.H - 1, env.W - 1]))
        center = target.astype(jnp.float32) + 0.5
        direct_ticks = jnp.ceil(
            jnp.abs(center - position).sum() / step_distance).astype(jnp.int32)
        direct_ok = (in_bounds & passable[target[0], target[1]]
                     & (direct_ticks + margin_ticks
                        < deadline[target[0], target[1]]))
        arrival = arrival.at[target[0], target[1]].min(
            jnp.where(direct_ok, direct_ticks, _INF_TICK))

    def relax(current, _):
        candidate = _shift_arrivals(current) + edge_ticks
        candidate = jnp.where(
            allowed & (candidate + margin_ticks < deadline),
            candidate, _INF_TICK)
        return jnp.minimum(current, candidate), None

    arrival, _ = jax.lax.scan(relax, arrival, None, length=env.H * env.W)
    permanent_safe = deadline >= _INF_TICK
    escaped = ((arrival < _INF_TICK) & permanent_safe).any()
    return start_permanently_safe | escaped


def analyze_actions(state: env.BunState, margin_ticks: int = 0) -> SafetyAnalysis:
    """枚举移动×能力联合动作，判断放泡后是否仍有可达逃生分支。

    安全战术放泡不能只看当前位置是否着火：这里同时模拟自己的泡、连锁触发、
    移动耗时和余焰；只有至少一条后续路径能活过爆炸窗口才标记为可生存。
    """
    move_mask, ability_mask = env.legal_mask(state)
    legal = move_mask[:, :, None] & ability_mask[:, None, :]
    no_bomb_deadlines, no_bomb_fuse, no_bomb_owner = owned_hazard_deadlines(
        state, None)
    placed_deadlines = []
    placed_fields = []
    for player in range(2):
        deadlines, fuse, owner = owned_hazard_deadlines(state, player)
        placed_deadlines.append(deadlines)
        placed_fields.append((fuse, owner))

    survivable_players = []
    for player in range(2):
        ability_survival = []
        for ability in range(env.N_BOMB):
            if ability == 1:
                deadlines = placed_deadlines[player][player]
                fuse, owner = placed_fields[player]
            else:
                deadlines = no_bomb_deadlines[player]
                fuse, owner = no_bomb_fuse, no_bomb_owner
            bombed = (owner >= 0) & (fuse > 0)
            passable = ~state.core.wall & ~state.core.brick & ~bombed
            move_survival = []
            for move in range(env.N_MOVES):
                position = _candidate_position(state, player, move, ability)
                move_survival.append(_can_escape(
                    position, passable, deadlines,
                    _movement_scale_for_player(state, player), margin_ticks))
            ability_survival.append(jnp.stack(move_survival))
        survivable_players.append(jnp.stack(ability_survival, axis=-1))
    survivable = jnp.stack(survivable_players)
    has_survival = (survivable & legal).any(axis=(1, 2))
    doomed = state.core.alive & ~has_survival
    avoidable = legal & ~survivable & ~doomed[:, None, None]
    avoidable = avoidable & state.core.alive[:, None, None]
    return SafetyAnalysis(
        legal=legal,
        survivable=survivable,
        avoidable=avoidable,
        doomed=doomed,
        own_deadline=no_bomb_deadlines)


def analyze_selected_actions(state: env.BunState, actions: jnp.ndarray,
                             margin_ticks: int = 0) -> SelectedSafety:
    """Evaluate the chosen joint action without masking or changing it.

    The baseline asks whether the current state has any escape under the real
    all-owner hazard field. The selected action is avoidable only when that
    baseline is survivable but committing the selected first action removes
    every escape. This preserves the doomed-state exemption.
    """
    move_mask, ability_mask = env.legal_mask(state)
    source_deadlines, base_fuse, base_owner = _source_deadlines(
        state.core, state.blast_owner_linger, None)
    all_deadline = source_deadlines.min(axis=0)
    bombed = (base_owner >= 0) & (base_fuse > 0)
    passable = ~state.core.wall & ~state.core.brick & ~bombed

    exposed = []
    escapable = []
    selected_survivable = []
    avoidable = []
    resolution_ticks = []
    uniquely_own_hazard = []
    for player in range(2):
        cell = jnp.clip(
            state.core.pos[player].astype(jnp.int32),
            jnp.asarray([0, 0]), jnp.asarray([env.H - 1, env.W - 1]))
        current_deadline = all_deadline[cell[0], cell[1]]
        current_exposed = current_deadline < _INF_TICK
        baseline_survivable = _can_escape(
            state.core.pos[player], passable, all_deadline,
            _movement_scale_for_player(state, player), margin_ticks)

        move = actions[player, 0]
        ability = actions[player, 1]
        legal_action = (move_mask[player, move]
                        & ability_mask[player, ability])
        placed_deadlines, placed_fuse, placed_owner = _source_deadlines(
            state.core, state.blast_owner_linger, player)
        selected_sources = jnp.where(
            ability == 1, placed_deadlines, source_deadlines)
        selected_fuse = jnp.where(ability == 1, placed_fuse, base_fuse)
        selected_owner = jnp.where(ability == 1, placed_owner, base_owner)
        selected_deadline = selected_sources.min(axis=0)
        selected_bombed = (selected_owner >= 0) & (selected_fuse > 0)
        selected_passable = (
            ~state.core.wall & ~state.core.brick & ~selected_bombed)
        position = _candidate_position(state, player, move, ability)
        action_survivable = _can_escape(
            position, selected_passable, selected_deadline,
            _movement_scale_for_player(state, player), margin_ticks)
        own_deadline = selected_sources[player, cell[0], cell[1]]
        other_deadline = selected_sources[1 - player, cell[0], cell[1]]

        exposed.append(current_exposed)
        escapable.append(baseline_survivable)
        selected_survivable.append(action_survivable)
        avoidable.append(
            legal_action & baseline_survivable & ~action_survivable)
        resolution_ticks.append(jnp.where(
            current_exposed,
            jnp.minimum(current_deadline + base.BLAST_LINGER_TICKS,
                        _INF_TICK - 1),
            0))
        uniquely_own_hazard.append(
            (own_deadline < _INF_TICK)
            & ((other_deadline >= _INF_TICK)
               | (own_deadline + base.BLAST_LINGER_TICKS < other_deadline)))

    exposed = jnp.stack(exposed) & state.core.alive
    escapable = jnp.stack(escapable) & state.core.alive
    return SelectedSafety(
        exposed=exposed,
        escapable=escapable,
        doomed=exposed & ~escapable,
        selected_survivable=jnp.stack(selected_survivable),
        avoidable=jnp.stack(avoidable) & state.core.alive,
        resolution_ticks=jnp.stack(resolution_ticks).astype(jnp.int16),
        uniquely_own_hazard=jnp.stack(uniquely_own_hazard) & exposed)


def _safe_move_count(state: env.BunState, player: int, deadlines,
                     fuse, owner, margin_ticks: int = 0):
    move_mask, _ = env.legal_mask(state)
    bombed = (owner >= 0) & (fuse > 0)
    passable = ~state.core.wall & ~state.core.brick & ~bombed
    survivable = []
    for move in range(env.N_MOVES):
        position = _candidate_position(state, player, move, 0)
        survivable.append(
            move_mask[player, move]
            & _can_escape(position, passable, deadlines,
                          _movement_scale_for_player(state, player),
                          margin_ticks))
    return jnp.stack(survivable).sum().astype(jnp.int16)


def analyze_tactical_bomb_placements(
        state: env.BunState, actions: jnp.ndarray,
        selected_survivable: jnp.ndarray | None = None,
        margin_ticks: int = 0) -> TacticalBombAnalysis:
    """Classify selected bomb placements using simulator-grounded reachability.

    A placement is tactical only when the new bomb either newly covers the
    opponent's current cell or strictly reduces the opponent's count of legal
    first moves that retain an escape continuation through the blast horizon.
    """
    if selected_survivable is None:
        selected_survivable = analyze_selected_actions(
            state, actions, margin_ticks).selected_survivable
    move_mask, ability_mask = env.legal_mask(state)
    base_deadlines, base_fuse, base_owner = _source_deadlines(
        state.core, state.blast_owner_linger, None)
    base_all_deadline = base_deadlines.min(axis=0)
    safe = []
    tactical = []
    threatens = []
    before_counts = []
    after_counts = []
    for player in range(2):
        enemy = 1 - player
        placed_deadlines, placed_fuse, placed_owner = _source_deadlines(
            state.core, state.blast_owner_linger, player)
        placed_all_deadline = placed_deadlines.min(axis=0)
        enemy_cell = jnp.clip(
            state.core.pos[enemy].astype(jnp.int32),
            jnp.asarray([0, 0]), jnp.asarray([env.H - 1, env.W - 1]))
        newly_threatens = (
            (placed_deadlines[player, enemy_cell[0], enemy_cell[1]] < _INF_TICK)
            & (placed_deadlines[player, enemy_cell[0], enemy_cell[1]]
               < base_deadlines[player, enemy_cell[0], enemy_cell[1]]))
        before = _safe_move_count(
            state, enemy, base_all_deadline, base_fuse, base_owner,
            margin_ticks)
        after = _safe_move_count(
            state, enemy, placed_all_deadline, placed_fuse, placed_owner,
            margin_ticks)
        move, ability = actions[player]
        legal_bomb = move_mask[player, move] & ability_mask[player, ability]
        is_safe = (state.core.alive[player] & (ability == 1)
                   & legal_bomb & selected_survivable[player])
        is_tactical = newly_threatens | (after < before)
        safe.append(is_safe)
        tactical.append(is_safe & is_tactical)
        threatens.append(newly_threatens)
        before_counts.append(before)
        after_counts.append(after)
    return TacticalBombAnalysis(
        safe=jnp.stack(safe), tactical=jnp.stack(tactical),
        newly_threatens_enemy=jnp.stack(threatens),
        enemy_safe_moves_before=jnp.stack(before_counts),
        enemy_safe_moves_after=jnp.stack(after_counts))


def both_avoidable_masks(states, margin_ticks: int = 0):
    analysis = jax.vmap(
        lambda state: analyze_actions(state, margin_ticks))(states)
    return jnp.concatenate([
        analysis.avoidable[:, 0], analysis.avoidable[:, 1]], axis=0)
