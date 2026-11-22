"""Full map806 Bun games with an explicit, non-transporting interceptor role.

The role changes custody permissions, never actions or observations. The
reviewed native physics are shared with the kill adapter; its cleared map,
kill respawns and empty Bun observations are deliberately not used.
"""
import inspect
from types import FunctionType
from typing import NamedTuple

import jax
import jax.numpy as j

from qqt_rl.training import jax_kill_env as native

_READY = False
METRICS = ('kills', 'deaths', 'self_deaths', 'carrier_kills', 'steals',
           'captures', 'drops', 'bombs', 'games', 'wins', 'draws', 'return_buns')


class World(NamedTuple):
    state: object
    batches: object
    counters: object
    last_placed: object


def prepare():
    global _READY, E, STEP_CODE, STEP_GLOBALS, MAKE_OBS
    if _READY:
        return
    native.prepare()
    E = native.E
    # Full stock, bricks, ordinary spawn sampling and Bun death/respawn.
    E.configure_training('full=1', 1, reward_profile='legacy', retention_replay=0)
    source = inspect.getsource(E.step)
    # Rebuild the private audited native function from its existing globals.
    # The four custody checks act BEFORE mutation, preserving dropped buns.
    for kind in ('owner', 'causal', 'trigger'):
        prefix = '' if kind == 'owner' else kind + '_'
        for axis in (0, 1):
            source = source.replace(f'blast_{kind}_linger[source, {axis}] > 0',
                                    f'current_{prefix}sources[source, {axis}]')
    source = source.replace('base._is_hit_by_explosion(', 'base._native_contact_any(')
    replacements = {
        '(requested_dirs < 4) & ~active_slide,\n                              requested_dirs, state.last_move_dir':
        '(dirs < 4) & state.alive & (state.trapped == 0),\n                              dirs, state.last_move_dir',
        "transfer=trap['death'][victim]&":
        "transfer=_CAN_TRANSPORT[popper]&trap['death'][victim]&",
        'deposit = core.alive[player] &':
        'deposit = _CAN_TRANSPORT[player] & core.alive[player] &',
        'take = core.alive[player] &':
        'take = _CAN_TRANSPORT[player] & core.alive[player] &',
        'can_steal = (core.alive[player] &':
        'can_steal = (_CAN_TRANSPORT[player] & core.alive[player] &',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Native carry adapter anchor changed: ' + old)
        source = source.replace(old, new)
    scope = dict(native.STEP_GLOBALS, _movement_scale=movement_scale)
    exec(compile(source, __file__ + ':private-carry-step', 'exec'), scope)
    STEP_CODE, STEP_GLOBALS = scope['step'].__code__, scope
    # The historical observation function indexes a Python list by pid.
    # Convert that one list to a device array for dynamically balanced seats.
    obs_source = inspect.getsource(E.make_obs)
    anchor = '    banana = '
    if obs_source.count(anchor) != 1:
        raise RuntimeError('Native observation anchor changed')
    obs_source = obs_source.replace(anchor, '    carried_origin = jnp.stack(carried_origin)\n' + anchor)
    obs_scope = dict(E.make_obs.__globals__)
    exec(compile(obs_source, __file__ + ':private-carry-obs', 'exec'), obs_scope)
    MAKE_OBS = obs_scope['make_obs']
    _READY = True


def movement_scale(st):
    # Native absolute movement speeds: cargo/glue 72, banana 480, shoes 268
    # px/s. The kill adapter deliberately ignores cargo and cannot be reused.
    absolute = j.where(st.move_status == 2, 480., j.where(st.move_status == 3, 268.,
                       j.where((st.move_status == 1) | (st.bun_carried >= 0), 72., 0.)))
    return j.where(absolute > 0, absolute / (40 * native.C.TICK_HZ * native.C.STEP *
                                            st.core.spd_g), 1.)


def fresh(key):
    prepare()
    st = E._fresh(key, jax.nn.one_hot(E.LESSON_FULL, len(E.LESSON_NAMES)))
    return World(st, j.full((13, 15), -1, j.int32), j.zeros(2, j.int32),
                 j.full(2, -1000, j.int32))


def events_reward(info):
    """Legacy full-game transport reward / 5; keep categorical values in range.

    No exploration bonus or action teacher. Delivery remains the main payoff.
    The separate interceptor objective has NO transport or terminal reward.
    """
    rows = j.arange(2)
    source = info['damage_source']
    unique = source.sum(-1) == 1
    own_hit = source[rows, rows] & unique
    enemy_hit = (source[rows, 1 - rows] & unique)[::-1]
    kills = info['credited_kill'] | info['causal_kill']
    carrier_kills = kills & info['carrier_death'][::-1]
    death = info['death'].astype(j.float32)
    own = info['own_bomb_defeat'].astype(j.float32)
    # Positive reward in this objective comes exclusively from credited kills.
    intercept = 5. * kills - 5. * death - 2. * own
    trap_credit = j.stack([(info['trap_hit'] & (info['trap_owner'] == p)).sum()
                           for p in range(2)])
    courier = (-own_hit.astype(j.float32) - 4. * death - 4. * own
               + .25 * enemy_hit + .5 * info['safe_bomb_escape']
               - 2. * (info['own_detonation'] & ~info['safe_bomb_escape'])
               + info['causal_kill'] + 3. * carrier_kills - 5. * info['drop']
               + 4. * info['steal'] + 2. * info['recover']
               + 4. * info['return_bun'] + 30. * info['capture']
               + info['enter_enemy_half'] + 3. * info['enter_enemy_base']
               + 8. * info['carry_cross_home']
               + .12 * j.clip(info['objective_progress'], -1., 1.)
               + .75 * trap_credit + .3 * info['walls_by_owner']
               + 20. * (info['winner'] == rows)
               - 20. * (info['winner'] == 1 - rows)) / 5.
    return intercept, courier, kills, carrier_kills


def advance(world, actions, key, can_transport):
    """Return raw native events as well, for custody/termination verification."""
    prepare()
    # A separate globals dict per traced call: roles are explicit device values,
    # never Python mutable global state shared across games/seats.
    fn = FunctionType(STEP_CODE, dict(STEP_GLOBALS, _CAN_TRANSPORT=can_transport))
    before = world.state
    after, done, info = fn(before, actions, key, auto_reset=False, return_info=True,
                           require_stock_window=False)
    added = (after.fuse > 0) & ~((before.fuse > 0) &
                               (before.core.owner == after.core.owner))
    batches, counters, last = world.batches, world.counters, world.last_placed
    for p in range(2):
        placed = info['bomb_placed'][p]
        counter = counters[p] + (placed & (before.t - last[p] >= 8)).astype(j.int32)
        batches = j.where(added & (after.core.owner == p), counter, batches)
        counters = counters.at[p].set(counter)
        last = last.at[p].set(j.where(placed, before.t, last[p]))
    batches = j.where(after.fuse > 0, batches, -1)
    intercept, courier, kills, carrier_kills = events_reward(info)
    # Reward for every interceptor's transport objective is exactly zero.
    reward = j.where(can_transport, courier, intercept)
    rows = j.arange(2)
    metrics = j.stack([kills, info['death'], info['own_bomb_defeat'], carrier_kills,
                       info['steal'], info['capture'], info['drop'], info['bomb_placed'],
                       j.full(2, done), done & (info['winner'] == rows),
                       j.full(2, done & (info['winner'] == -1)), info['return_bun']], -1)
    return World(after, batches, counters, last), reward, done, metrics.astype(j.float32), info


def step(world, actions, key, can_transport):
    return advance(world, actions, key, can_transport)[:4]


def inputs(worlds, seats):
    prepare()
    st, rows = worlds.state, j.arange(len(seats))
    danger = jax.vmap(lambda c: native.C._danger_map(c.fuse, c.wall, c.bomb_blast, c.brick))(st.core)
    obs = jax.vmap(MAKE_OBS)(st, seats, danger)
    gv = jax.vmap(E.global_vec)(st, seats)
    from scripts.r3_items_observation import expand_inputs
    obs, gv = expand_inputs(obs, gv, st, seats, dict(bomb_batch=worlds.batches))
    # Normal Bun mode, including for an interceptor acting on the same map.
    gv = j.concatenate([gv, j.zeros((len(seats), 1), j.float32)], -1)
    mm, bm = jax.vmap(E.legal_mask)(st)
    live = st.alive[rows, seats] & (st.trapped[rows, seats] == 0)
    mm = j.where(live[:, None], mm[rows, seats], j.array([False] * 4 + [True]))
    bm = j.where(live[:, None], bm[rows, seats], j.array([True, False, False]))
    return obs, gv, mm, bm, live
