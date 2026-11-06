"""Reset-only balanced combat course with passive continuous-attack telemetry.

No bubble-count reward: every inherited transport/combat reward is retained.
Regional candidates skip A3SR map edits and genuine Bank selection. New
telemetry is checkpointed and cannot pay reward or alter policy inputs.
"""
import inspect
import json
from pathlib import Path
import jax
import jax.numpy as j
from scripts import combat_specialist_g_mixed as base
from scripts.multi_region_combat_v1 import REGIONS

B = Path('runs/combat_regions_independent_20261104')
WEIGHTS, NAMES, updater = base.WEIGHTS, base.NAMES, base.updater


def select_course(keys):
    return jax.vmap(lambda k: jax.random.bernoulli(jax.random.fold_in(k, 0x6D5), .30))(keys)


def passive_state(n):
    return dict(mr_region=j.zeros(n, j.int32), mr_requested=j.zeros(n, bool),
        mr_distance=j.zeros(n, j.int32), mr_heading=j.zeros(n, j.int32),
        ca_last_tick=j.full(n, -1, j.int32), ca_eligible_age=j.zeros(n, j.int32))


def migrate_aux(aux):
    meta = dict(passive_state(aux['terminal'].shape[0]), **aux['a3sr'])
    return dict(aux, a3sr=meta)


def apply_course(st, ax, keys, seats):
    manifest = json.loads((B / 'train_starts.json').read_text())
    pairs = j.asarray([r['cells'] for r in manifest['rows']], j.int32)
    roles = j.asarray([r['learner_seat'] for r in manifest['rows']])
    labels = j.asarray([REGIONS.index(r['region']) + 1 for r in manifest['rows']], j.int32)
    distance = j.asarray([r['distance'] for r in manifest['rows']], j.int32)
    heading = j.asarray([('north', 'south', 'west', 'east').index(r['relative_heading']) for r in manifest['rows']], j.int32)
    flat = pairs[:, :, 0] * 15 + pairs[:, :, 1]
    blocked = (st.wall | st.brick | st.core.pushable).reshape(len(keys), 195)
    legal = ~blocked[:, flat].any(-1) & (seats[:, None] == roles[None, :])
    drawkeys = jax.vmap(lambda k: jax.random.fold_in(k, 0x6D6))(keys)
    choice = jax.vmap(lambda k, mask: jax.random.categorical(k, j.where(mask, 0., -1e9)))(drawkeys, legal)
    requested = select_course(keys)
    selected = requested & legal.any(-1)
    pos = pairs[choice].astype(st.pos.dtype) + .5
    st = st._replace(core=st.core._replace(pos=j.where(selected[:, None, None], pos, st.pos)))
    meta = dict(ax['a3sr'], **passive_state(len(keys)))
    meta.update(mr_region=j.where(selected, labels[choice], 0), mr_requested=requested,
        mr_distance=j.where(selected, distance[choice], 0), mr_heading=j.where(selected, heading[choice], 0))
    return st, dict(ax, a3sr=meta)


def attack_observation(before, after, ax, info, seats):
    """Actual native placements and strict causal events; no shaping payment."""
    rows = j.arange(len(seats)); enemy = 1 - seats
    pick = lambda x: x[rows, seats]
    live = pick(before.alive) & (pick(before.trapped) == 0) & ~ax['terminal']
    # The native physics decrements fuses before placement; inspect its actual
    # pre-movement cell and capacity, including bombs expiring this step.
    cells = j.floor(before.pos[rows, seats]).astype(j.int32)
    cell_fuse = before.fuse[rows, cells[:, 0], cells[:, 1]]
    live_bombs = ((before.core.owner == seats[:, None, None]) & (before.fuse > 1)).sum((-2, -1))
    legal = live & (pick(before.bun_carried) < 0) & ~before.brick[rows, cells[:, 0], cells[:, 1]] & (cell_fuse <= 1) & (live_bombs < pick(before.core.bombs_cap))
    distance = j.abs(before.pos[:, 0] - before.pos[:, 1]).sum(-1)
    enemy_live = before.alive[rows, enemy] & (before.trapped[rows, enemy] == 0)
    contact = live & enemy_live & (distance <= 6)
    last = ax['a3sr']['ca_last_tick']; age = ax['a3sr']['ca_eligible_age']
    opportunity = legal & (last >= 0)
    age = age + opportunity.astype(j.int32)
    placed = pick(info['bomb_placed'])
    repeat = placed & (last >= 0)
    gap = after.t - last
    died = pick(info['death'])
    meta = dict(ax['a3sr'], ca_last_tick=j.where(died, -1, j.where(placed, after.t, last)),
                ca_eligible_age=j.where(placed | died, 0, age))
    counts = dict(ca_legal_capacity=legal, ca_repeat_opportunities=opportunity,
        ca_placed=placed, ca_contact_placed=placed & contact,
        ca_empty_placed=placed & ~contact, ca_repeats=repeat,
        ca_contact_repeat30=repeat & contact & (gap <= 30),
        ca_gap_sum=j.where(repeat, gap, 0), ca_eligible_gap_sum=j.where(repeat, age, 0),
        ca_causal_control=info['native_trap_enter'][rows, enemy] & pick(info['causal_hit']), ca_causal_kill=pick(info['causal_kill']),
        ca_self_defeat=pick(info['own_bomb_defeat']))
    return meta, counts


def bun_backend():
    source = inspect.getsource(base.bun_backend)
    anchor = "    fresh_source = fresh_source.replace('.25/.9', '.25')"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\n    fresh_source=fresh_source.replace('use=bank_selection(split[:,2],seats,.25)&~designated', 'use=bank_selection(split[:,2],seats,.25)&~designated&~select_course(keys)')\n    fresh_source=fresh_source.replace('reset_mask=reset_mask,enabled=True', 'reset_mask=(~select_course(keys)) if reset_mask is None else reset_mask&~select_course(keys),enabled=True')")
    source = source.replace('    scope = dict(raw.__globals__)', '    scope = dict(raw.__globals__, select_course=select_course)')
    anchor = '        return st._replace(bun_spawn_pos=j.where(bank[:,None,None], homes, st.bun_spawn_pos)), ax'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '        st=st._replace(bun_spawn_pos=j.where(bank[:,None,None], homes, st.bun_spawn_pos))\n        return apply_course(st, ax, keys, args[4])')
    source = source.replace('    return impl', '    impl._league_source=source\n    return impl')
    scope = dict(base.__dict__, select_course=select_course, apply_course=apply_course)
    exec(compile(source, __file__, 'exec'), scope)
    impl = scope['bun_backend']()
    transition = impl.transition
    def measured_transition(st, ax, actions, keys, coefficient):
        after, na, reward, done, info, parts, sm = transition(st, ax, actions, keys, coefficient)
        meta, counters = attack_observation(st, after, ax, info, ax['combat_mix']['learner_seat'])
        return after, dict(na, a3sr=meta), reward, done, info, parts, dict(sm, **counters)
    impl.transition = measured_transition
    code = impl._league_source
    anchor = '            diagnostics.update(policy_diag)'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor + '''
            region=jax.nn.one_hot(ax['a3sr']['mr_region'],9)
            reset_region=jax.nn.one_hot(fa['a3sr']['mr_region'],9)*done[:,None]
            diagnostics.update(multi_region_ticks=region.sum(0),
                multi_region_resets=reset_region.sum(0),
                multi_region_requested_resets=(done&fa['a3sr']['mr_requested']).sum(),
                multi_region_legal_fallbacks=(done&fa['a3sr']['mr_requested']&(fa['a3sr']['mr_region']==0)).sum(),
                multi_region_seat_resets=j.einsum('ni,nj->ij',reset_region,onehot),
                multi_region_distance_resets=j.einsum('ni,nj->ij',reset_region,jax.nn.one_hot(fa['a3sr']['mr_distance']//2,4)),
                multi_region_heading_resets=j.einsum('ni,nj->ij',reset_region,jax.nn.one_hot(fa['a3sr']['mr_heading'],4)))
            diagnostics.update({k:(region*v[:,None]).sum(0) for k,v in sm.items() if k.startswith('ca_')})
''')
    scope = dict(impl.collector.__globals__)
    exec(compile(code, __file__, 'exec'), scope)
    for name in ('fresh', 'transition', 'metadata', 'inputs', 'prediction', 'E'):
        scope[name] = getattr(impl, name)
    scope.update(fixed_allocation=base.allocation, LEAGUE_NAMES=NAMES, ATTACK=True)
    impl.collector = scope['collector']
    return impl


def collector(steps, count, seed, spawn_pairs):
    from scripts import r4_k4_collector as common
    bun = bun_backend()
    collect_bun = bun.collector(steps, count, seed, 0., 0., spawn_pairs, WEIGHTS)
    def collect(params, states, aux, keys, bs, ba, bank, generation):
        end, batch, exposure = collect_bun(params, states, aux, keys, bs, ba, None, None, bank, generation)
        modes = j.zeros_like(batch[4], j.int8); live = batch[8]
        _, normalization = common.normalize_by_mode(batch[4], modes, live)
        return end, batch[:8] + (modes, live), dict(bun=exposure, **normalization)
    return collect
