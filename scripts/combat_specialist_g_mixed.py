"""G-only natural bun collection with episode-bound C/G snapshot opponents."""
from types import ModuleType
import ast
import inspect
import jax
import jax.numpy as j
import numpy as np
from scripts import r4_k4_mixed as base, r4_k4_collector as common
from scripts.combat_specialist_episode_bun import build

WEIGHTS = (.05, .20, .15, .15, .10, .05, .15, .10, .05)
NAMES = ('rule', 'hunter', 'hard', 'Gcurrent', 'Gring', 'Gfixed',
         'Ccurrent', 'Cring', 'Cfixed')


def allocation(count, seed, weights=WEIGHTS):
    weights = np.asarray(weights, float)
    assert weights.shape == (9,) and np.isclose(weights.sum(), 1.)
    return np.random.default_rng(seed).choice(9, count, p=weights).astype(np.int8)


def bun_backend():
    from scripts import r3_items_collector as parent
    accepted = base.bun_backend()
    template = build(parent, accepted)
    source = template._league_source
    anchor = "groups={name:np.flatnonzero(kinds==i) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]}"
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        "groups={name:np.flatnonzero(kinds==i) for i,name in enumerate(LEAGUE_NAMES) if i>=3}")
    anchor = "for name,bank in [('current',current),('ring',ring),('fixed',fixed)]:"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "for name,bank in current.items():")
    source = source.replace("one_hot(ax['combat_mix']['kind'],6)",
                            "one_hot(ax['combat_mix']['kind'],9)")
    impl = ModuleType('league_g_private_bun')
    exec(compile(source, __file__, 'exec'), impl.__dict__)
    for name in ('fresh', 'transition', 'metadata', 'inputs', 'prediction', 'E'):
        setattr(impl, name, getattr(accepted, name))
    # All G episodes use real bun rules, so the historical .25/.9
    # compensation for a10% lesson bucket would oversample the Bank.
    raw = accepted.fresh
    while 'original' in inspect.getclosurevars(raw).nonlocals:
        raw = inspect.getclosurevars(raw).nonlocals['original']
    from scripts import bun_native_v2_custody_collector as custody
    tree = ast.parse(custody.source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fresh')
    fresh_source = ast.get_source_segment(custody.source, node)
    assert fresh_source.count('.25/.9') == 1
    fresh_source = fresh_source.replace('.25/.9', '.25')
    scope = dict(raw.__globals__)
    exec(compile(fresh_source, __file__, 'exec'), scope)
    natural_fresh = scope['fresh']
    def fresh(keys, *args, **kwargs):
        from jax_bomb import je07_bun_env_r1_items as env
        st, ax = natural_fresh(keys, *args, **kwargs)
        homes = jax.vmap(env._own_side_native)(keys)
        bank = ax['provenance_source'] >= 0
        return st._replace(bun_spawn_pos=j.where(bank[:,None,None], homes, st.bun_spawn_pos)), ax
    impl.fresh = fresh
    impl.fixed_allocation = allocation
    impl.LEAGUE_NAMES = NAMES
    impl.ATTACK = True
    return impl


def initialize(keys, bs, ba, spawn_pairs, seed):
    from scripts import r3_items_collector as parent
    bun = bun_backend(); n = len(keys)
    kinds = allocation(n, seed)
    # Original genuine Bank budget is retained; natural games keep the real
    # rules and initial inventory. No arena lesson and no fabricated carrier.
    designated = j.zeros(n, bool)
    seats = j.asarray(parent.seat_allocation(n, seed, .5))
    states, aux = bun.fresh(keys, bs, ba, designated, spawn_pairs, seats)
    meta = bun.metadata(states, kinds, designated, seats, aux['r1_attribute_highwater'])
    aux = dict(aux, combat_mix=dict(meta, opponent_generation=j.zeros(n, j.int32)))
    return states, aux, keys


def collector(steps, count, seed, spawn_pairs):
    bun = bun_backend()
    collect_bun = bun.collector(steps, count, seed, 0., 0., spawn_pairs, WEIGHTS)
    def collect(params, states, aux, keys, bs, ba, bank, generation):
        end, batch, exposure = collect_bun(params, states, aux, keys, bs, ba,
                                          None, None, bank, generation)
        modes = j.zeros_like(batch[4], j.int8)
        live = batch[8]
        _, normalization = common.normalize_by_mode(batch[4], modes, live)
        return end, batch[:8] + (modes, live), dict(bun=exposure, **normalization)
    return collect


updater = base.updater
