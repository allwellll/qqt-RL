"""Strict discounted route potential ledger for JE09 H8.

This module contains pure PBRS accounting plus the production state adapter.
The potential is internal reward state and never enters Actor observations.
"""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import shortest

GAMMA = 0.995
LAMBDA = 0.05
INF = 32767
PHASE_NONE, PHASE_STOCK, PHASE_DROP, PHASE_CARRY, PHASE_DELIVERY = range(5)
METRICS = (
    'route_phi_before', 'route_phi_after', 'route_potential_delta',
    'route_shaped_reward', 'route_positive', 'route_negative',
    'route_topology_gain', 'route_phase_transition', 'route_invalidated',
    'route_terminal_correction', 'route_distance_before',
    'route_distance_after', 'route_inf_to_finite', 'route_safety_rejected',
    'route_phase', 'route_target_identity', 'route_target_provenance',
)


def state(n):
    shape = (n, 2)
    return dict(phi=jnp.zeros(shape, jnp.float32), offset=jnp.zeros(shape, jnp.float32),
                phase=jnp.zeros(shape, jnp.int8), identity=jnp.full(shape, -1, jnp.int32),
                retired=jnp.full(shape, -1, jnp.int32),
                stock_ready=jnp.zeros(n, jnp.bool_),
                stock_wall=jnp.zeros((n, 13, 15), jnp.bool_),
                stock_brick=jnp.zeros((n, 13, 15), jnp.bool_),
                stock_distance=jnp.full((n, 2, 13, 15), INF, jnp.int32))


def discounted_delta(phi_now, phi_next, terminal, gamma=GAMMA):
    """PBRS delta; the terminal potential is exactly zero."""
    phi_now = jnp.asarray(phi_now, jnp.float32)
    terminal = jnp.asarray(terminal, jnp.bool_)
    next_value = jnp.where(terminal, 0., jnp.asarray(phi_next, jnp.float32))
    return gamma * next_value - phi_now


def transition(ledger, phase, identity, base_now, next_phase, next_identity,
               base_next, terminal, invalidated=False, gamma=GAMMA):
    """Continue potential across phase/target changes without boundary credit.

    A new phase/identity receives an offset making its potential equal to the
    previous potential at the boundary. Inactive/invalid targets hold the
    previous potential until a new real objective appears or the episode ends.
    """
    phase = jnp.asarray(phase, jnp.int8)
    identity = jnp.asarray(identity, jnp.int32)
    active = phase != PHASE_NONE
    same = active & (phase == ledger['phase']) & (identity == ledger['identity'])
    offset = jnp.where(same, ledger['offset'], ledger['phi'] - base_now)
    phi_now = jnp.where(active, base_now + offset, ledger['phi'])
    next_phase = jnp.asarray(next_phase, jnp.int8)
    next_identity = jnp.asarray(next_identity, jnp.int32)
    next_active = next_phase != PHASE_NONE
    same_next = next_active & (next_phase == phase) & (next_identity == identity)
    phi_next_candidate = jnp.where(same_next, base_next + offset, phi_now)
    phi_next = jnp.where(terminal, 0., phi_next_candidate)
    delta = discounted_delta(phi_now, phi_next, terminal, gamma)
    next_ledger = dict(phi=jnp.where(terminal, 0., phi_next),
        offset=jnp.where(terminal, 0., offset),
        phase=jnp.where(terminal, PHASE_NONE, phase),
        identity=jnp.where(terminal, -1, identity),
        retired=jnp.where(invalidated, identity, ledger['retired']))
    return next_ledger, delta, phi_now, phi_next


def _cells(pos, h, w):
    return jnp.clip(pos.astype(jnp.int32), jnp.array([0, 0]), jnp.array([h-1, w-1]))


def _lookup(distance, cells):
    rows = jnp.arange(cells.shape[0])[:, None]
    teams = jnp.arange(2)[None, :]
    return distance[rows, teams, cells[..., 0], cells[..., 1]]


def continuous_distance(states,route,phase,_target_identity=None):
    # Shared topology-continuous field, successor-only recipe.
    from scripts.continuous_route_metric import sampled_distance
    n=states.pos.shape[0]
    teams=jnp.broadcast_to(jnp.arange(2)[None],(n,2))
    channels=jnp.where(phase>=PHASE_CARRY,teams+2,teams)
    pos=states.pos.reshape(n,2,2)
    main=sampled_distance(pos,route['distance'],states.wall,channels)
    stock=sampled_distance(pos,route['stock_distance'],states.wall,teams)
    return jnp.where(phase==PHASE_STOCK,stock,main)


def _phase_values_before_recovery_alignment(states, route, drop_state):
    """Return observed objective phase, identity, distance and provenance.

    Stock access is enabled only for native death/respawn or native trap
    control. Loose-drop access requires a learner-attributed origin token.
    """
    from scripts import je09_bun_scenes as scenes
    n = states.pos.shape[0]
    team = jnp.broadcast_to(jnp.arange(2)[None], (n, 2))
    enemy = 1-team
    empty = states.bun_carried < 0
    window = (states.bun_respawn[:, ::-1] > 0) | (states.trapped[:, ::-1] > 0)
    stock_count = states.bun_stored[jnp.arange(n)[:, None], enemy, enemy]
    stock = empty & window & (stock_count > 0) & states.alive

    drop_origin = enemy
    nonce = drop_state['nonce'][jnp.arange(n)[:, None], drop_origin]
    credit = drop_state['credit'][jnp.arange(n)[:, None], drop_origin]
    target = route['target'][:, ::-1]
    target_cell = _cells(target, states.wall.shape[-2], states.wall.shape[-1])
    loose = states.bun_loose[jnp.arange(n)[:, None], target_cell[..., 0],
                             target_cell[..., 1], drop_origin] > 0
    drop = empty & states.alive & loose & (nonce > 0) & (credit == team)
    carry = states.alive & (states.bun_carried == enemy)
    cell = _cells(states.pos, states.wall.shape[-2], states.wall.shape[-1])
    at_home = scenes.HOME[scenes.OPEN_SCENE[states.scene].astype(jnp.int32)[:, None],
                          team, cell[..., 0], cell[..., 1]]
    delivery = carry & at_home

    phase = jnp.where(carry, PHASE_CARRY,
        jnp.where(drop, PHASE_DROP, jnp.where(stock, PHASE_STOCK, PHASE_NONE))).astype(jnp.int8)
    phase = jnp.where(delivery, PHASE_DELIVERY, phase).astype(jnp.int8)
    identity = jnp.where(phase == PHASE_DROP, nonce,
        jnp.where(phase == PHASE_STOCK, stock_count,
            jnp.where(phase >= PHASE_CARRY, enemy + 1, -1))).astype(jnp.int32)
    identity = identity + phase.astype(jnp.int32) * 100000
    variant = scenes.OPEN_SCENE[states.scene].astype(jnp.int32)
    cell = _cells(states.pos, states.wall.shape[-2], states.wall.shape[-1])
    maps = route['distance']
    d_drop = maps[jnp.arange(n)[:, None], team,
                  cell[..., 0], cell[..., 1]]
    d_home = maps[jnp.arange(n)[:, None], team + 2,
                  cell[..., 0], cell[..., 1]]
    d_stock = route['stock_distance'][jnp.arange(n)[:, None], team,
        cell[..., 0], cell[..., 1]]
    distance = jnp.where(phase == PHASE_STOCK, d_stock,
        jnp.where(phase == PHASE_DROP, d_drop, d_home))
    # The route planner's brick weight is 1+3 per entered brick. Distances are
    # already topology-weighted; normalization keeps the feature bounded.
    continuous = continuous_distance(states, route, phase)
    clipped = jnp.minimum(continuous, 64).astype(jnp.float32)
    base = jnp.where(phase == PHASE_NONE, 0.,
                     jnp.where(distance < INF, -clipped / 64., -1.))
    provenance = jnp.where(phase == PHASE_DROP, nonce,
        jnp.where(phase == PHASE_STOCK, 2, jnp.where(phase >= PHASE_CARRY, 3, 0)))
    return phase, identity, base, distance, provenance


def phase_values(states, route, drop_state):
    # Attributed own-recovery target alignment, successor-only recipe.
    from scripts.route_objective_consistency import align_recovery
    values = _phase_values_before_recovery_alignment(states, route, drop_state)
    return align_recovery(states, route, drop_state, values)


def apply_reward(states, next_states, info, aux, reward, newaux, parts, coefficient):
    """Attach a single PBRS delta after the unchanged H8 reward composition."""
    from scripts import je09_bun_scenes as scenes
    from scripts.je08_bun_potential import INF as DIST_INF
    route_state = aux['route_pbrs']
    old_route = aux['routing']
    new_route = newaux['routing']
    from scripts.je08_bun_potential import INF as DIST_INF
    stock_goals = jnp.broadcast_to(jnp.asarray(scenes.goals)[None, ::-1],
                                   (states.pos.shape[0], 2, 13, 15))
    dirty_now = (~route_state['stock_ready'] |
        jnp.any(states.wall != route_state['stock_wall'], (-2, -1)) |
        jnp.any(states.brick != route_state['stock_brick'], (-2, -1)))
    stock_now = jax.lax.cond(jnp.any(dirty_now),
        lambda _: jnp.where(dirty_now[:, None, None, None],
            shortest(states.wall[:, None], states.brick[:, None], stock_goals),
            route_state['stock_distance']),
        lambda _: route_state['stock_distance'], None)
    now_route = dict(old_route, stock_distance=stock_now)
    phase, identity, base_now, dist_now, provenance = phase_values(states, now_route, aux['drop'])
    # Shared native preparation potential.
    from scripts.prestock_training_state import potential as prestock_potential,update as prestock_update
    phase,identity,base_now,dist_now,provenance=prestock_potential(states,aux['prestock'],
        (phase,identity,base_now,dist_now,provenance))
    dirty_next = (jnp.any(next_states.wall != states.wall, (-2, -1)) |
        jnp.any(next_states.brick != states.brick, (-2, -1)))
    stock_next = jax.lax.cond(jnp.any(dirty_next),
        lambda _: jnp.where(dirty_next[:, None, None, None],
            shortest(next_states.wall[:, None], next_states.brick[:, None], stock_goals),
            stock_now),
        lambda _: stock_now, None)
    phase_next, identity_next, base_next, dist_next, _ = phase_values(
        next_states, dict(new_route, stock_distance=stock_next), newaux['drop'])
    next_selection_aux=dict(newaux,terminal=aux['terminal']|info['h_route_terminal'])
    newaux['prestock']=prestock_update(next_states,next_selection_aux,aux['prestock'])
    phase_next,identity_next,base_next,dist_next,_=prestock_potential(next_states,newaux['prestock'],
        (phase_next,identity_next,base_next,dist_next,jnp.zeros_like(identity_next)))
    # Preserve the active objective at event edges; phase/identity transitions
    # are offset-continuous. The next state's native observable classification
    # is used only to select its potential, never as an Actor input.
    terminal = jnp.broadcast_to(info['h_route_terminal'][:, None], phase.shape)
    invalidated = (phase > 0) & (phase_next == 0) & ~terminal
    # Topology changes count only where the same currently active target got
    # strictly closer (or became reachable), and both teammates remain alive,
    # untrapped, and the learner remains alive. No placement-only payment.
    changed = jnp.any((states.wall != next_states.wall) |
                      (states.brick != next_states.brick), (-2, -1))
    safety = states.alive & next_states.alive & (states.trapped == 0) & (next_states.trapped == 0)
    safety &= jnp.broadcast_to(~info['mutual_death'][:, None], safety.shape)
    safety &= ~info['death']
    safety &= jnp.broadcast_to(~info['danger_exposure'], safety.shape)
    teams = jnp.broadcast_to(jnp.arange(2)[None], safety.shape)
    next_cell = _cells(next_states.pos, states.wall.shape[-2], states.wall.shape[-1])
    rows = jnp.arange(states.pos.shape[0])[:, None]
    ally_home_d = new_route['distance'][rows, teams + 2,
        next_cell[..., 0], next_cell[..., 1]]
    safety &= ally_home_d < DIST_INF
    def time_expanded_safe(_):
        from jax_bomb import bun_safety
        certificate = jax.vmap(bun_safety.analyze_actions)(states)
        actions = info['h_joint_actions']
        survivable = certificate.survivable[rows, teams,
            jnp.clip(actions[..., 0], 0, 4), jnp.clip(actions[..., 1], 0, 2)]
        return survivable
    temporal_safe = jax.lax.cond(jnp.any(changed), time_expanded_safe,
        lambda _: jnp.ones_like(safety), operand=None)
    safety &= temporal_safe
    same_objective = (phase == phase_next) & (identity == identity_next)
    topology_gain = changed[:, None] & (phase > 0) & same_objective & safety & (
        ((dist_now >= DIST_INF) & (dist_next < DIST_INF)) |
        ((dist_now < DIST_INF) & (dist_next < dist_now)))
    topology_gain = topology_gain.astype(jnp.float32)
    safety_rejected = changed[:, None] & (phase > 0) & ~safety
    unsafe_map_change = changed[:, None] & ~safety
    # Rebase unsafe topology changes so a broken wall cannot pay route credit.
    base_next = jnp.where(unsafe_map_change & (phase_next == phase) &
                          (identity_next == identity), base_now, base_next)
    new_route_state, delta, phi_before, phi_after = transition(
        route_state, phase, identity, base_now, phase_next, identity_next,
        base_next, terminal, invalidated=invalidated)
    shaped = jnp.asarray(coefficient, jnp.float32) * delta
    new_route_state.update(stock_ready=jnp.ones_like(route_state['stock_ready']),
        stock_wall=next_states.wall, stock_brick=next_states.brick,
        stock_distance=stock_next)
    metrics = dict(
        route_phi_before=phi_before, route_phi_after=phi_after,
        route_potential_delta=delta, route_shaped_reward=shaped,
        route_positive=jnp.maximum(shaped, 0.), route_negative=jnp.minimum(shaped, 0.),
        route_topology_gain=topology_gain, route_phase_transition=(phase != phase_next).astype(jnp.float32),
        route_invalidated=invalidated.astype(jnp.float32),
        route_terminal_correction=(terminal & (phi_before != 0)).astype(jnp.float32),
        route_distance_before=dist_now.astype(jnp.float32),
        route_distance_after=dist_next.astype(jnp.float32),
        route_inf_to_finite=((dist_now >= DIST_INF) & (dist_next < DIST_INF)).astype(jnp.float32),
        route_safety_rejected=safety_rejected.astype(jnp.float32),
        route_phase=phase.astype(jnp.float32), route_target_identity=identity.astype(jnp.float32),
        route_target_provenance=provenance.astype(jnp.float32))
    newaux['route_pbrs'] = new_route_state
    reward = reward + shaped
    parts = jnp.concatenate([parts[..., :-1], jnp.stack([metrics[k] for k in METRICS], -1), parts[..., -1:]], -1)
    return reward, newaux, parts, metrics
