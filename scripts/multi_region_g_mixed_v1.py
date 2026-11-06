"""Independent G reset curriculum; inherited rewards, opponents and PPO unchanged."""
import inspect
import json
from pathlib import Path
from types import ModuleType
import jax
import jax.numpy as j
from scripts import combat_specialist_g_mixed as base
from scripts.multi_region_combat_v1 import REGIONS

BASE = Path('runs/multi_region_combat_20261104')
WEIGHTS, NAMES, updater = base.WEIGHTS, base.NAMES, base.updater


def select_course(keys):
    return jax.vmap(lambda k:jax.random.bernoulli(jax.random.fold_in(k,0x6D5),.30))(keys)


def apply_course(st, ax, keys, seats):
    """Use unchanged-map legal pairs; clean native state, no fabricated carrier/bomb."""
    manifest=json.loads((BASE/'dev_starts.json').read_text())
    pairs=j.asarray([r['cells'] for r in manifest['rows']],j.int32)
    roles=j.asarray([r['learner_seat'] for r in manifest['rows']])
    labels=j.asarray([REGIONS.index(r['region'])+1 for r in manifest['rows']],j.int32)
    rows=j.arange(len(keys));flat=pairs[:,:,0]*15+pairs[:,:,1]
    blocked=(st.wall|st.brick).reshape(len(keys),195)
    legal=~blocked[:,flat].any(-1)&(seats[:,None]==roles[None,:])
    drawkeys=jax.vmap(lambda k:jax.random.fold_in(k,0x6D6))(keys)
    choice=jax.vmap(lambda k,mask:jax.random.categorical(k,j.where(mask,0.,-1e9)))(drawkeys,legal)
    selected=select_course(keys)&legal.any(-1)
    pos=pairs[choice].astype(st.pos.dtype)+.5
    # Respawn homes retain the native own-side contract; only first positions vary.
    st=st._replace(core=st.core._replace(pos=j.where(selected[:,None,None],pos,st.pos)))
    meta=dict(ax['a3sr'],mr_region=j.where(selected,labels[choice],0),mr_requested=select_course(keys))
    return st,dict(ax,a3sr=meta)


def migrate_aux(aux):
    """Zero metadata for continuing old episodes; no state or policy mutation."""
    if 'mr_region' in aux['a3sr']:
        return aux
    n=aux['terminal'].shape[0]
    return dict(aux,a3sr=dict(aux['a3sr'],mr_region=j.zeros(n,j.int32),mr_requested=j.zeros(n,bool)))


def bun_backend():
    source=inspect.getsource(base.bun_backend)
    anchor="    fresh_source = fresh_source.replace('.25/.9', '.25')"
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n    fresh_source=fresh_source.replace('use=bank_selection(split[:,2],seats,.25)&~designated', 'use=bank_selection(split[:,2],seats,.25)&~designated&~select_course(keys)')")
    source=source.replace('    scope = dict(raw.__globals__)','    scope = dict(raw.__globals__, select_course=select_course)')
    anchor='        return st._replace(bun_spawn_pos=j.where(bank[:,None,None], homes, st.bun_spawn_pos)), ax'
    assert source.count(anchor)==1
    source=source.replace(anchor,'        st=st._replace(bun_spawn_pos=j.where(bank[:,None,None], homes, st.bun_spawn_pos))\n        return apply_course(st, ax, keys, args[4])')
    source=source.replace('    return impl','    impl._league_source=source\n    return impl')
    scope=dict(base.__dict__,select_course=select_course,apply_course=apply_course)
    exec(compile(source,__file__,'exec'),scope)
    impl=scope['bun_backend']()
    src=impl.collector.__globals__
    # Add passive course/exposure counters to the same private scan.
    code=impl._league_source
    anchor='            diagnostics.update(policy_diag)'
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+'''
            region=jax.nn.one_hot(ax['a3sr']['mr_region'],9)
            course=ax['a3sr']['mr_region']>0
            diagnostics.update(multi_region_ticks=region.sum(0),
                multi_region_live_ticks=(region*live[:,None]).sum(0),
                multi_region_seat_ticks=j.einsum('ni,nj->ij',region,onehot),
                multi_region_resets=(jax.nn.one_hot(fa['a3sr']['mr_region'],9)*done[:,None]).sum(0),
                multi_region_requested_resets=(done&fa['a3sr']['mr_requested']).sum(),
                multi_region_legal_fallbacks=(done&fa['a3sr']['mr_requested']&(fa['a3sr']['mr_region']==0)).sum(),
                multi_region_kills=(region*select_player(info['causal_kill'],seat)[:,None]).sum(0),
                multi_region_deaths=(region*select_player(info['death'],seat)[:,None]).sum(0),
                multi_region_self_defeat=(region*select_player(info['own_bomb_defeat'],seat)[:,None]).sum(0))
''')
    # Preserve the already patched fresh/metadata/physics functions.
    scope=dict(src)
    exec(compile(code,__file__,'exec'),scope)
    for name in ('fresh','transition','metadata','inputs','prediction','E'):
        scope[name]=getattr(impl,name)
    scope.update(fixed_allocation=base.allocation, LEAGUE_NAMES=NAMES, ATTACK=True)
    impl.collector=scope['collector']
    return impl


def collector(steps,count,seed,spawn_pairs):
    from scripts import r4_k4_collector as common
    bun=bun_backend();collect_bun=bun.collector(steps,count,seed,0.,0.,spawn_pairs,WEIGHTS)
    def collect(params,states,aux,keys,bs,ba,bank,generation):
        end,batch,exposure=collect_bun(params,states,aux,keys,bs,ba,None,None,bank,generation)
        modes=j.zeros_like(batch[4],j.int8);live=batch[8]
        _,normalization=common.normalize_by_mode(batch[4],modes,live)
        return end,batch[:8]+(modes,live),dict(bun=exposure,**normalization)
    return collect


def initialize(*args,**kwargs):
    raise RuntimeError('This phase requires the complete existing G league state; no fresh model initialization')
