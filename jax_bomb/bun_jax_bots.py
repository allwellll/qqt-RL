"""Device-side graded Bun rule bots (pure JAX, vmappable, no host callbacks).

One parameterised policy covers every tier; tiers differ only in a row of
`TierParams`, so a batch mixing tiers runs as a single fused computation.
Per tick each bot builds lethal-deadline fields (base / +own bomb / +enemy
hypothetical bomb / both), runs a capped escape BFS for each first move, and
scores the enemy's escape space before/after its own bomb to decide attacks.
The difficulty axes mirror web/bun_hunter_bot.js: perception delay, chain
awareness, robustness to an enemy counter-bomb, mistake rate, bomb hesitation
and attack selectivity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import bun_env as env
from . import jax_env as base

H, W = env.H, env.W
INF = 1 << 14
BIG = 1e4
FUSE = base.FUSE
ESCAPE_STEPS = 16
FIELD_STEPS = 30
ROAM_WINDOW = 40
STYLE_ROAM, STYLE_FLEE, STYLE_HUNT = 0, 1, 2
BOMB_NONE, BOMB_LINE, BOMB_PRESSURE = 0, 1, 2
_DIRS = jnp.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
_HI = jnp.asarray([H - 1, W - 1], jnp.int32)


class TierParams(NamedTuple):
    reaction_delay: jnp.ndarray
    mistake_rate: jnp.ndarray
    chain_aware: jnp.ndarray
    style: jnp.ndarray
    bomb_mode: jnp.ndarray
    pressure_ratio: jnp.ndarray
    hesitation: jnp.ndarray
    max_live: jnp.ndarray
    robust: jnp.ndarray
    attack_radius: jnp.ndarray
    dig: jnp.ndarray
    margin: jnp.ndarray
    flee_radius: jnp.ndarray
    bomb_slack: jnp.ndarray


# Ordered easy -> hard. dodge_* never bomb (offense practice for the learner);
# bomber/hunter tiers bomb back (danger-avoidance practice).
TIERS = {
    "dodge_easy": dict(reaction_delay=10, mistake_rate=0.12, chain_aware=0,
                       style=STYLE_ROAM, bomb_mode=BOMB_NONE, pressure_ratio=0.0,
                       hesitation=1.0, max_live=0, robust=0, attack_radius=0,
                       dig=0, margin=0, flee_radius=0, bomb_slack=0),
    "dodge": dict(reaction_delay=2, mistake_rate=0.03, chain_aware=1,
                  style=STYLE_FLEE, bomb_mode=BOMB_NONE, pressure_ratio=0.0,
                  hesitation=1.0, max_live=0, robust=0, attack_radius=0,
                  dig=0, margin=1, flee_radius=5, bomb_slack=0),
    "bomber_easy": dict(reaction_delay=6, mistake_rate=0.08, chain_aware=0,
                        style=STYLE_HUNT, bomb_mode=BOMB_LINE, pressure_ratio=0.0,
                        hesitation=0.5, max_live=1, robust=0, attack_radius=3,
                        dig=1, margin=0, flee_radius=0, bomb_slack=2),
    "hunter": dict(reaction_delay=1, mistake_rate=0.02, chain_aware=1,
                   style=STYLE_HUNT, bomb_mode=BOMB_PRESSURE, pressure_ratio=0.3,
                   hesitation=0.05, max_live=2, robust=1, attack_radius=5,
                   dig=1, margin=1, flee_radius=0, bomb_slack=3),
    "hunter_hard": dict(reaction_delay=0, mistake_rate=0.0, chain_aware=1,
                        style=STYLE_HUNT, bomb_mode=BOMB_PRESSURE,
                        pressure_ratio=0.45, hesitation=0.0, max_live=10,
                        robust=1, attack_radius=7, dig=1, margin=1,
                        flee_radius=0, bomb_slack=3),
}
TIER_NAMES = tuple(TIERS)

_INT_FIELDS = {"reaction_delay", "style", "bomb_mode", "max_live",
               "attack_radius", "margin", "flee_radius", "bomb_slack"}
_BOOL_FIELDS = {"chain_aware", "robust", "dig"}


def tier_table(tiers=TIERS) -> TierParams:
    """Stack tier rows into `(T,)` arrays in `TIER_NAMES` order."""
    columns = {}
    for field in TierParams._fields:
        values = [tiers[name][field] for name in tiers]
        if field in _INT_FIELDS:
            columns[field] = jnp.asarray(values, jnp.int32)
        elif field in _BOOL_FIELDS:
            columns[field] = jnp.asarray(values, jnp.bool_)
        else:
            columns[field] = jnp.asarray(values, jnp.float32)
    return TierParams(**columns)


def _shift_min(x, fill):
    """Min over the 4-neighbour values of every cell (out of bounds = fill)."""
    pad = [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)]
    p = jnp.pad(x, pad, constant_values=fill)
    return jnp.minimum(
        jnp.minimum(p[..., :-2, 1:-1], p[..., 2:, 1:-1]),
        jnp.minimum(p[..., 1:-1, :-2], p[..., 1:-1, 2:]))


def deadline_field(fuse_after, bombed, blast, wall, brick, linger, chain_aware):
    """Tick at which each cell becomes lethal (INF = never within the fuse).

    Same timing model as bun_safety._source_deadlines but over all owners at
    once; `chain_aware=False` drops chain propagation (easy-tier blind spot).
    """
    max_time = float(FUSE + 1)
    detonation = jnp.where(bombed, jnp.maximum(fuse_after, 0), INF)
    score = jnp.where(bombed, (max_time - detonation.astype(jnp.float32)) / max_time, 0.0)
    blast_f = jnp.where(blast > 0, blast, base.BLAST).astype(jnp.float32)
    chained = base._stage_a_matrix(score, blast_f, bombed, wall, brick, base.MAX_CHAIN)
    score = jnp.where(chain_aware, chained, score)
    passable = (~wall).astype(jnp.float32)
    not_solid = (~bombed & ~brick).astype(jnp.float32)
    reach = jnp.where(bombed, blast_f, 0.0)
    danger = jnp.maximum(score, base._spread_all(score, reach, passable, not_solid))
    deadline = jnp.where(
        danger > 0, jnp.rint(max_time * (1.0 - danger)).astype(jnp.int32), INF)
    return jnp.where(linger, 0, deadline)


def edge_entry_blocked(blocked):
    """(4,H,W): entering a cell moving UP/DOWN/LEFT/RIGHT is rejected by
    jax_env._move_player's centre-path check even though the cell is open.

    That check reads a MAX_SWEEP-wide dynamic_slice, which JAX clamps at the
    bottom/right border, so the checked cells shift by one there: entering
    row H-1 / col W-1 fails if the source cell is blocked (e.g. own bomb), and
    entering row H-2 / col W-2 from the border fails if the cell beyond is.
    """
    down = jnp.zeros((H, W), jnp.bool_).at[H - 1].set(blocked[H - 2])
    up = jnp.zeros((H, W), jnp.bool_).at[H - 2].set(blocked[H - 3])
    right = jnp.zeros((H, W), jnp.bool_).at[:, W - 1].set(blocked[:, W - 2])
    left = jnp.zeros((H, W), jnp.bool_).at[:, W - 2].set(blocked[:, W - 3])
    return jnp.stack([up, down, left, right])


def escape_bfs(position, start_t, valid, passable, deadline, step, margin,
               steps: int = ESCAPE_STEPS):
    """Return (survivable, #reachable safe cells, earliest safe arrival).

    `position` is the continuous body centre at tick `start_t`. The first hop
    uses the exact remaining distance to each neighbour centre; later hops
    cost a full cell. A hop c->n needs n safe on arrival and c safe until the
    body has left it (conservatively: until arrival).
    """
    edge = jnp.ceil(1.0 / step).astype(jnp.int32)
    start = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, _HI)
    start_dl = deadline[start[0], start[1]]
    ok = valid & ((start_t + margin < start_dl) | (start_dl >= INF))
    arrival = jnp.full((H, W), INF, jnp.int32).at[start[0], start[1]].set(
        jnp.where(ok, start_t, INF))
    nbr = start[None, :] + _DIRS
    inb = ((nbr >= 0) & (nbr <= _HI)).all(axis=-1)
    nbr = jnp.clip(nbr, 0, _HI)
    direct = start_t + jnp.ceil(
        jnp.abs(nbr.astype(jnp.float32) + 0.5 - position).sum(-1) / step
    ).astype(jnp.int32)
    nbr_dl = deadline[nbr[:, 0], nbr[:, 1]]
    edge_block = edge_entry_blocked(~passable)
    first_block = edge_block[jnp.arange(4), nbr[:, 0], nbr[:, 1]]
    hop_ok = (valid & inb & passable[nbr[:, 0], nbr[:, 1]] & ~first_block
              & (direct + margin < nbr_dl)
              & ((direct + margin < start_dl) | (start_dl >= INF)))
    arrival = arrival.at[nbr[:, 0], nbr[:, 1]].min(jnp.where(hop_ok, direct, INF))

    # The start cell only departs via the exact first hops above: the body is
    # generally off-centre there, so a full-cell hop would underestimate time.
    is_start = jnp.zeros((H, W), jnp.bool_).at[start[0], start[1]].set(True)
    inf_row = jnp.full((1, W), INF, jnp.int32)
    inf_col = jnp.full((H, 1), INF, jnp.int32)

    def relax(a, _):
        d = jnp.where(~is_start & (a + edge + margin < deadline), a, INF)
        entries = jnp.stack([
            jnp.concatenate([d[1:], inf_row], axis=0),       # moving up
            jnp.concatenate([inf_row, d[:-1]], axis=0),      # moving down
            jnp.concatenate([d[:, 1:], inf_col], axis=1),    # moving left
            jnp.concatenate([inf_col, d[:, :-1]], axis=1),   # moving right
        ])
        cand = jnp.where(edge_block, INF, entries).min(axis=0) + edge
        cand = jnp.where(passable & (cand + margin < deadline), cand, INF)
        return jnp.minimum(a, cand), None

    arrival, _ = jax.lax.scan(relax, arrival, None, length=steps)
    safe = (arrival < INF) & (deadline >= INF)
    t_safe = jnp.where(safe, arrival, INF).min()
    return safe.any(), safe.sum().astype(jnp.int32), t_safe


def goal_distance(seed, cost, steps: int = FIELD_STEPS):
    """dist[c] = min over paths c->goal of the summed entry costs."""
    dist = jnp.where(seed, 0.0, BIG)

    def relax(d, _):
        return jnp.minimum(d, _shift_min(jnp.minimum(d + cost, BIG), BIG)), None

    dist, _ = jax.lax.scan(relax, dist, None, length=steps)
    return dist


def _add_bomb(fuse_after, bombed, blast, cell, blast_value, cond):
    r, c = cell[0], cell[1]
    return (fuse_after.at[r, c].set(jnp.where(cond, FUSE, fuse_after[r, c])),
            bombed.at[r, c].set(bombed[r, c] | cond),
            blast.at[r, c].set(jnp.where(cond, blast_value, blast[r, c])))


def _pick(mask, score):
    return jnp.argmax(jnp.where(mask, score, -jnp.inf)).astype(jnp.int32)


def bot_action(state: env.BunState, me, move_mask, ability_mask, key,
               prm: TierParams, debug: bool = False):
    """One env, one bot seat `me` (traced int). Returns int32 `[move, ability]`."""
    core = state.core
    foe = 1 - me
    wall, brick, fuse, owner = core.wall, core.brick, core.fuse, core.owner
    k_noise, k_hes, k_mis, k_mis_move = jax.random.split(key, 4)

    fuse_after = jnp.where(fuse > 0, fuse - 1, fuse)
    bombed_all = (owner >= 0) & (fuse > 0)
    perceived = bombed_all & ((owner == me) | (fuse <= FUSE - prm.reaction_delay))
    blast = core.bomb_blast.astype(jnp.int32)
    linger = core.blast_linger != 0
    cells = jnp.clip(jnp.floor(core.pos).astype(jnp.int32), 0, _HI)
    my_cell, foe_cell = cells[me], cells[foe]
    scale = env._movement_scale(state)
    step_me = jnp.maximum(base.STEP * core.spd_g[me] * scale[me], 1e-3)
    step_foe = jnp.maximum(base.STEP * core.spd_g[foe] * scale[foe], 1e-3)
    my_blast = core.blast_cap[me].astype(jnp.int32)
    live = jnp.stack([((owner == p) & (fuse > 0)).sum() for p in range(2)])
    alive_me, alive_foe = core.alive[me], core.alive[foe]

    can_bomb = (ability_mask[1] & alive_me & (prm.bomb_mode > 0)
                & (live[me] < prm.max_live) & (state.bun_carried[me] < 0))
    foe_blast_i = core.blast_cap[foe].astype(jnp.int32)
    foe_can_bomb = (alive_foe & (live[foe] < core.bombs_cap[foe])
                    & (jnp.abs(my_cell - foe_cell).sum() <= foe_blast_i + 4)
                    & (state.bun_carried[foe] < 0)
                    & (fuse[foe_cell[0], foe_cell[1]] <= 0)
                    & ~brick[foe_cell[0], foe_cell[1]])
    foe_blast = foe_blast_i

    # Variants: 0 base, 1 +mine, 2 +enemy hypothetical, 3 +both.
    f1, p1, b1 = _add_bomb(fuse_after, perceived, blast, my_cell, my_blast, can_bomb)
    f2, p2, b2 = _add_bomb(fuse_after, perceived, blast, foe_cell, foe_blast, foe_can_bomb)
    f3, p3, b3 = _add_bomb(f1, p1, b1, foe_cell, foe_blast, foe_can_bomb)
    fv = jnp.stack([fuse_after, f1, f2, f3])
    pv = jnp.stack([perceived, p1, p2, p3])
    bv = jnp.stack([blast, b1, b2, b3])
    deadlines = jax.vmap(deadline_field, in_axes=(0, 0, 0, None, None, None, None))(
        fv, pv, bv, wall, brick, linger, prm.chain_aware)
    mine_cell = jnp.zeros((H, W), jnp.bool_).at[my_cell[0], my_cell[1]].set(can_bomb)
    foe_hypo = jnp.zeros((H, W), jnp.bool_).at[foe_cell[0], foe_cell[1]].set(foe_can_bomb)
    solid_v = jnp.stack([bombed_all, bombed_all | mine_cell,
                         bombed_all | foe_hypo, bombed_all | mine_cell | foe_hypo])
    passable_v = ~wall[None] & ~brick[None] & ~solid_v

    # First move is simulated with the env's own steering (a blocked move can
    # slide sideways), then the BFS continues from the resulting cell at t=1.
    nbr = my_cell[None, :] + _DIRS
    inb = ((nbr >= 0) & (nbr <= _HI)).all(axis=-1)
    nbr = jnp.clip(nbr, 0, _HI)
    sliding = state.move_status[me] == env.STATUS_SLIDE
    move_speed = core.spd_g[me] * scale[me]

    def next_positions(blocked):
        def one(d):
            d = jnp.where(sliding, state.slide_dir[me].astype(jnp.int32), d)
            return base._steer(core.pos[me], d, alive_me, blocked, move_speed,
                               core.pushable)
        return jax.vmap(one)(jnp.arange(5, dtype=jnp.int32))

    blocked0 = (fuse_after > 0) | wall | brick
    pos_plain = next_positions(blocked0)
    pos_bomb = next_positions(blocked0 | mine_cell)
    starts_plain = jnp.clip(jnp.floor(pos_plain).astype(jnp.int32), 0, _HI)
    starts_bomb = jnp.clip(jnp.floor(pos_bomb).astype(jnp.int32), 0, _HI)
    pos_v = jnp.stack([pos_plain, pos_bomb, pos_plain, pos_bomb])
    own_deadline = deadlines[:, my_cell[0], my_cell[1]]
    # Placing a bomb commits to an escape; slack absorbs steering/corner error.
    margins = prm.margin + jnp.asarray([0, 1, 0, 1], jnp.int32) * prm.bomb_slack
    leave_ok = own_deadline > 1 + margins
    start_t = jnp.ones((5,), jnp.int32)

    def run_me(dl, pas, positions, ok, margin):
        return jax.vmap(lambda p, t: escape_bfs(
            p, t, ok, pas, dl, step_me, margin))(positions, start_t)

    surv, count, t_safe = jax.vmap(run_me)(
        deadlines, passable_v, pos_v, leave_ok, margins)
    starts = starts_plain

    def run_foe(dl, pas):
        return escape_bfs(core.pos[foe], jnp.int32(0), alive_foe, pas, dl, step_foe, 0)

    _, foe_count, _ = jax.vmap(run_foe)(deadlines[:2], passable_v[:2])

    # Goal fields.
    free = ~wall & ~brick & ~bombed_all
    danger0 = deadlines[0] < INF
    cost = jnp.where(wall | bombed_all, BIG,
                     jnp.where(brick, jnp.where(prm.dig, 5.0, BIG),
                               1.0 + 3.0 * danger0.astype(jnp.float32)))
    foe_onehot = jnp.zeros((H, W), jnp.bool_).at[foe_cell[0], foe_cell[1]].set(True)
    rays = base._rays(foe_onehot, bombed_all, jnp.full((H, W), my_blast, jnp.int32),
                      wall, brick)
    hunt_goal = (rays & free) | foe_onehot
    hunt_valid = alive_foe & hunt_goal.any()
    spawn_hash = (state.bun_spawn_pos[me] * jnp.asarray([17.0, 131.0])).sum()
    roam_key = jax.random.fold_in(
        jax.random.fold_in(jax.random.PRNGKey(0x5EED), core.t // ROAM_WINDOW),
        spawn_hash.astype(jnp.int32) + me * 7919)
    roam_flat = jax.random.categorical(
        roam_key, jnp.where(free.reshape(-1), 0.0, -1e9))
    roam_goal = jnp.zeros((H * W,), jnp.bool_).at[roam_flat].set(True).reshape(H, W)
    seed = jnp.where((prm.style == STYLE_HUNT) & hunt_valid, hunt_goal, roam_goal)
    goal_dist = goal_distance(seed, cost)
    foe_dist = goal_distance(foe_onehot, jnp.where(free, 1.0, BIG))

    nbr_cost = cost[nbr[:, 0], nbr[:, 1]]
    move_value = jnp.where(inb, nbr_cost + goal_dist[nbr[:, 0], nbr[:, 1]], BIG)
    value = jnp.concatenate([move_value, goal_dist[my_cell[0], my_cell[1]][None]])
    foe_near = alive_foe & (foe_dist[my_cell[0], my_cell[1]] <= prm.flee_radius)
    flee_value = -jnp.minimum(foe_dist[starts[:, 0], starts[:, 1]], 20.0)
    value = jnp.where((prm.style == STYLE_FLEE) & foe_near, flee_value, value)

    best_dir = jnp.argmin(move_value)
    best_cell = nbr[best_dir]
    dig = (prm.dig & (move_value[best_dir] < BIG / 2)
           & brick[best_cell[0], best_cell[1]]
           & (deadlines[0][best_cell[0], best_cell[1]] >= INF))

    # Attack decision.
    manhattan = jnp.abs(my_cell - foe_cell).sum()
    before, after = foe_count[0], foe_count[1]
    in_foot = (deadlines[1][foe_cell[0], foe_cell[1]]
               < deadlines[0][foe_cell[0], foe_cell[1]])
    foe_target = (alive_foe & (core.invuln[foe] < FUSE)
                  & (manhattan <= prm.attack_radius))
    kill = (after == 0) & (before > 0)
    line_ok = jnp.where(prm.bomb_mode == BOMB_LINE,
                        in_foot & (manhattan <= my_blast), True)
    pressure = ((prm.bomb_mode == BOMB_PRESSURE) & (manhattan <= 4)
                & (after.astype(jnp.float32)
                   <= prm.pressure_ratio * jnp.maximum(before, 1)))
    attack = foe_target & line_ok & (kill | in_foot | pressure)
    safe_after = move_mask & surv[1]
    robust_after = safe_after & surv[3]
    bomb_moves = jnp.where(prm.robust, robust_after, safe_after)
    want_bomb = (can_bomb & (attack | dig) & bomb_moves.any()
                 & (jax.random.uniform(k_hes) >= prm.hesitation))

    noise = jax.random.uniform(k_noise, (5,), maxval=0.01)
    is_idle = jnp.arange(5) == 4

    def score(variant):
        # Threatened: shortest time to a permanently safe cell wins, goal
        # progress only breaks ties (greedy progress made bots dither in fire).
        threatened = own_deadline[variant] < INF
        dest = jnp.where(variant == 1, starts_bomb, starts)
        danger = deadlines[variant][dest[:, 0], dest[:, 1]] < INF
        calm = (-jnp.clip(value, -50.0, 999.0)
                + 0.2 * jnp.log2(1.0 + count[variant].astype(jnp.float32))
                - 8.0 * danger - 0.3 * is_idle)
        urgent = (-3.0 * jnp.minimum(t_safe[variant], 99).astype(jnp.float32)
                  + 0.5 * jnp.log2(1.0 + count[variant].astype(jnp.float32))
                  - 0.05 * jnp.clip(value, -50.0, 999.0) - 2.0 * is_idle)
        return jnp.where(threatened, urgent, calm) + noise

    safe0 = move_mask & surv[0]
    robust0 = safe0 & surv[2]
    acts = jnp.where(prm.robust & robust0.any(), robust0, safe0)
    bomb_move = _pick(bomb_moves, score(1))
    safe_move = _pick(acts, score(0))
    fallback = _pick(move_mask, deadlines[0][starts[:, 0], starts[:, 1]].astype(jnp.float32)
                     + noise)
    move = jnp.where(want_bomb, bomb_move,
                     jnp.where(acts.any(), safe_move, fallback))
    ability = want_bomb.astype(jnp.int32)

    mistake = jax.random.uniform(k_mis) < prm.mistake_rate
    random_move = _pick(move_mask, jax.random.uniform(k_mis_move, (5,)))
    move = jnp.where(mistake, random_move, move)
    ability = jnp.where(mistake, 0, ability)
    move = jnp.where(alive_me, move, 4)
    ability = jnp.where(alive_me, ability, 0)
    action = jnp.stack([move, ability]).astype(jnp.int32)
    if debug:
        return action, dict(surv=surv, count=count, t_safe=t_safe,
                            deadlines=deadlines, starts_plain=starts_plain,
                            starts_bomb=starts_bomb, can_bomb=can_bomb,
                            pos_plain=pos_plain, pos_bomb=pos_bomb,
                            want_bomb=want_bomb, foe_count=foe_count)
    return action


def rule_bot_actions(states, players, move_mask, ability_mask, key, tiers,
                     table: TierParams | None = None):
    """Batched `(K,2)` actions. `players` (K,) seat each bot plays, `tiers`
    (K,) index into `TIER_NAMES`, masks are the bot seat's legal masks."""
    table = tier_table() if table is None else table
    prm = jax.tree_util.tree_map(lambda column: column[tiers], table)
    keys = jax.random.split(key, players.shape[0])
    return jax.vmap(bot_action)(states, players.astype(jnp.int32), move_mask,
                                ability_mask, keys, prm)


def parse_tier_names(spec: str, names) -> np.ndarray:
    values = {name: 0.0 for name in names}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, raw = item.split("=", 1)
        if name not in values:
            raise ValueError(f"unknown jax bot: {name} (known: {', '.join(names)})")
        values[name] = float(raw)
    weights = np.asarray([values[name] for name in names], np.float32)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("jax bot weights must be non-negative and sum > 0")
    return weights / weights.sum()
