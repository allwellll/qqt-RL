"""Training-only JAX hunter with one-step counter-bomb anticipation.

Derived from bun_jax_bots.bot_action; existing bot/evaluator sources stay fixed.
No trap-search, policy reward, physical transition, or training data changes.
This approximates the observed JS robust=2 mechanism, not the whole JS hunter.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_bomb import bun_jax_bots as B
from jax_bomb.bun_jax_bots import (
    env, base, H, W, INF, BIG, FUSE, ROAM_WINDOW, STYLE_HUNT, STYLE_FLEE,
    BOMB_LINE, BOMB_PRESSURE, TierParams, deadline_field, escape_bfs,
    goal_distance, _add_bomb, _pick, _DIRS, _HI,
)


def approaching_cell(my_cell, foe_cell, wall, brick, bombed):
    """Nearest open neighbor strictly closer to hunter; UP/DOWN/LEFT/RIGHT ties."""
    raw = foe_cell[None, :] + _DIRS
    in_bounds = ((raw >= 0) & (raw <= _HI)).all(-1)
    cells = jnp.clip(raw, 0, _HI)
    dist = jnp.abs(cells - my_cell).sum(-1)
    valid = (in_bounds & ~wall[cells[:, 0], cells[:, 1]]
             & ~brick[cells[:, 0], cells[:, 1]] & ~bombed[cells[:, 0], cells[:, 1]]
             & (dist < jnp.abs(my_cell - foe_cell).sum()))
    index = jnp.argmin(jnp.where(valid, dist, INF))
    return cells[index], valid.any()


def delayed_deadline_field(fuse_after, bombed, blast, wall, brick, linger, chain_aware):
    """Extend original normalized deadline math to a future placed bomb.

    FUSE+arrival can exceed the original FUSE+1 normalization horizon.
    Increasing that horizon retains positive scores and chained detonation times.
    Existing four deadline variants still call the unchanged original function.
    """
    max_time = jnp.maximum(float(FUSE + 1), jnp.max(jnp.where(bombed, fuse_after + 1, 0)).astype(jnp.float32))
    detonation = jnp.where(bombed, jnp.maximum(fuse_after, 0), INF)
    score = jnp.where(bombed, (max_time - detonation.astype(jnp.float32)) / max_time, 0.)
    blast_f = jnp.where(blast > 0, blast, base.BLAST).astype(jnp.float32)
    chained = base._stage_a_matrix(score, blast_f, bombed, wall, brick, base.MAX_CHAIN)
    score = jnp.where(chain_aware, chained, score)
    danger = jnp.maximum(score, base._spread_all(score, jnp.where(bombed, blast_f, 0.),
        (~wall).astype(jnp.float32), (~bombed & ~brick).astype(jnp.float32)))
    deadlines = jnp.where(danger > 0, jnp.rint(max_time * (1. - danger)).astype(jnp.int32), INF)
    return jnp.where(linger, 0, deadlines)


def neighbor_bot_action(state: env.BunState, me, move_mask, ability_mask, key,
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

    # Keep the original four variants; add neighbor counter-bombs separately.
    neighbor_cell, neighbor_valid = approaching_cell(my_cell, foe_cell, wall, brick, bombed_all)
    neighbor_can_bomb = (alive_foe & (live[foe] < core.bombs_cap[foe])
                         & (jnp.abs(my_cell - foe_cell).sum() <= foe_blast_i + 4)
                         & (state.bun_carried[foe] < 0) & neighbor_valid)
    arrival = jnp.maximum(1, jnp.ceil(jnp.abs(
        neighbor_cell.astype(jnp.float32) + .5 - core.pos[foe]).sum() / step_foe).astype(jnp.int32))

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
    f4, p4, b4 = _add_bomb(fuse_after, perceived, blast, neighbor_cell, foe_blast, neighbor_can_bomb)
    f5, p5, b5 = _add_bomb(f1, p1, b1, neighbor_cell, foe_blast, neighbor_can_bomb)
    nr, nc = neighbor_cell
    f4 = f4.at[nr, nc].set(jnp.where(neighbor_can_bomb, FUSE + arrival, f4[nr, nc]))
    f5 = f5.at[nr, nc].set(jnp.where(neighbor_can_bomb, FUSE + arrival, f5[nr, nc]))
    neighbor_deadlines = jax.vmap(delayed_deadline_field, in_axes=(0, 0, 0, None, None, None, None))(
        jnp.stack([f4, f5]), jnp.stack([p4, p5]), jnp.stack([b4, b5]),
        wall, brick, linger, prm.chain_aware)
    neighbor_solid = jnp.zeros((H, W), jnp.bool_).at[nr, nc].set(neighbor_can_bomb)
    extra_solid = jnp.stack([bombed_all | neighbor_solid, bombed_all | mine_cell | neighbor_solid])
    deadlines = jnp.concatenate([deadlines, neighbor_deadlines], axis=0)
    passable_v = jnp.concatenate([passable_v, ~wall[None] & ~brick[None] & ~extra_solid], axis=0)

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
    pos_v = jnp.stack([pos_plain, pos_bomb, pos_plain, pos_bomb, pos_plain, pos_bomb])
    own_deadline = deadlines[:, my_cell[0], my_cell[1]]
    # Placing a bomb commits to an escape; slack absorbs steering/corner error.
    margins = prm.margin + jnp.asarray([0, 1, 0, 1, 0, 1], jnp.int32) * prm.bomb_slack
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
    robust_after = safe_after & surv[3] & surv[5]
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
    robust0 = safe0 & surv[2] & surv[4]
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
                            want_bomb=want_bomb, foe_count=foe_count,
                            neighbor_cell=neighbor_cell, neighbor_valid=neighbor_can_bomb,
                            neighbor_arrival=arrival)
    return action

def rule_bot_actions(states, players, move_mask, ability_mask, key, tiers, table=None):
    table = B.tier_table() if table is None else table
    params = jax.tree.map(lambda column: column[tiers], table)
    keys = jax.random.split(key, players.shape[0])
    return jax.vmap(neighbor_bot_action)(states, players.astype(jnp.int32), move_mask,
                                        ability_mask, keys, params)
