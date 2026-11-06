"""Explicit K4 two-backend collection, no modifications to existing workers."""
from types import ModuleType,SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from scripts import r4_k4_collector as common,r4_k4_observation as observation
from scripts import r4_k4_arena_collector as arena,r4_k4_runtime as runtime
from scripts.combat_mix_opponents import fixed_allocation,combat_scene_mask

BUN_WEIGHTS=(.05,.35,.30,.15,.10,.05)


def bun_backend():
    from scripts import r3_items_collector as parent
    source=parent.src
    old='    return jax.pmap(collect,in_axes=(0,0,0,0,None,None,0,0))'
    assert source.count(old)==1
    source=source.replace(old,'    return collect')
    old='output=(o,g,own,lp,v,nv,reward,done,mm,bm,diagnostics)'
    assert source.count(old)==1
    source=source.replace(old,'output=(o,g,own,lp,v,nv,reward,done,mm,bm,live&st.alive[j.arange(count),seat],diagnostics)')
    source=source.replace('o,g,act,lp,v,nv,r,done,mm,bm,diag=d','o,g,act,lp,v,nv,r,done,mm,bm,live,diag=d')
    source=source.replace('(o,g,act,lp,adv,adv+v,mm,bm),','(o,g,act,lp,adv,adv+v,mm,bm,live),')
    impl=ModuleType('k4_private_bun_collector');exec(compile(source,__file__,'exec'),impl.__dict__)
    for name in ('fresh','transition','metadata'):setattr(impl,name,getattr(parent._impl,name))
    def inputs(st,seats,ledger=None):
        obs,global_state,mm,bm=parent.inputs(st,seats,ledger)
        return obs,observation.append_mode(global_state,0),mm,bm
    def prediction(p,st,seats,ledger=None):
        o,g,mm,bm=inputs(st,seats,ledger)
        mv,ab,v,vl=observation.actor_critic_forward(p,o,g)
        return mv,ab,v,vl,o,g,mm,bm
    impl.inputs=inputs;impl.prediction=prediction;impl.ATTACK=True
    engine=parent.E
    train=SimpleNamespace(**{k:v for k,v in vars(engine.train).items() if not k.startswith('__')})
    train.net_forward=lambda p,a,o,g:observation.actor_critic_forward(p,o,g)
    impl.E=SimpleNamespace(**{k:v for k,v in vars(engine).items() if not k.startswith('__')})
    impl.E.train=train
    return impl


def initialize(keys,bs,ba,spawn_pairs,seed,*,new_update=1):
    """New mode starts fresh episodes; full policy/Adam/ring migrate exactly."""
    from scripts import r3_items_collector as parent
    nb,na=common.mix_counts(len(keys));bun=bun_backend()
    kinds=fixed_allocation(nb,seed,BUN_WEIGHTS)
    designated=j.asarray(combat_scene_mask(nb,seed^0x806,.10))
    seats=j.asarray(parent.seat_allocation(nb,seed,.5))
    st,ax=bun.fresh(keys[:nb],bs,ba,designated,spawn_pairs,seats)
    ax=dict(ax,combat_mix=bun.metadata(st,kinds,designated,seats,ax['r1_attribute_highwater']))
    arenas=runtime.batch_fresh(keys[nb:],new_update=new_update)
    aa=jax.vmap(lambda _:arena.auxiliary())(keys[nb:])
    return dict(bun=st,arena=arenas),dict(bun=ax,arena=aa),keys


def collector(steps,count,seed,spawn_pairs):
    nb,na=common.mix_counts(count);bun=bun_backend()
    bcollect=bun.collector(steps,nb,seed,.10,0.,spawn_pairs,BUN_WEIGHTS)
    acollect=arena.collector(steps,na,seed^0xA4)
    def collect(params,states,aux,keys,bs,ba,ring,fixed,new_update):
        bend,bb,bd=bcollect(params,states['bun'],aux['bun'],keys[:nb],bs,ba,ring,fixed)
        aend,ab,ad=acollect(params,states['arena'],aux['arena'],keys[nb:],ring,fixed,new_update)
        combined=tuple(j.concatenate([b,a],1) for b,a in zip(bb[:8],ab))
        modes=j.concatenate([j.zeros_like(bb[4],j.int8),j.ones_like(ab[4],j.int8)],1)
        live=j.concatenate([bb[8],ad['live']],1)
        _,normalization=common.normalize_by_mode(combined[4],modes,live)
        # The frozen amendment specifies train-minibatch normalization. Keep
        # raw GAE and return targets here; the updater normalizes each actual
        # shuffled minibatch by mode, excluding dead/post-terminal actor rows.
        batch=combined+(modes,live)
        metrics=dict(bun=bd,arena=jax.tree.map(lambda a:a.sum((0,1)),
            {k:v for k,v in ad.items() if k not in ('map_id','mode','live')}),**normalization,
            raw_adv_abs=j.stack([j.where(live&(modes==mode),j.abs(combined[4]),0.).sum()/j.maximum(1,(live&(modes==mode)).sum()) for mode in (0,1)]))
        return (dict(bun=bend[0],arena=aend[0]),dict(bun=bend[1],arena=aend[1]),j.concatenate([bend[2],aend[2]],0)),batch,metrics
    return collect


def updater(minibatch,optimizer):
    """Retained separate Adam; normalize each true on-policy train minibatch."""
    import optax
    def update(params,adam,batch,key):
        batch=tuple(v.reshape((-1,)+v.shape[2:]) for v in batch)
        order=jax.random.permutation(key,len(batch[4])).reshape(-1,minibatch)
        def body(carry,indices):
            p,a=carry;mb=tuple(v[indices] for v in batch)
            advantage,normalization=common.normalize_by_mode(mb[4],mb[8],mb[9])
            mb=mb[:4]+(advantage,)+mb[5:8]
            (loss,metrics),gradient=jax.value_and_grad(common.ppo_objective,has_aux=True)(p,mb)
            changes={};next_adam={}
            for branch in ('actor','critic'):
                changes[branch],next_adam[branch]=optimizer.update(gradient[branch],a[branch],p[branch])
            next_params=optax.apply_updates(p,changes)
            values=j.concatenate([loss[None],metrics,optax.global_norm(gradient)[None],
                optax.global_norm(gradient['actor']['heads']['wb'])[None],
                normalization['mode_samples'],normalization['mode_mean_abs_adv']])
            return (next_params,next_adam),values
        return jax.lax.scan(body,(params,adam),order)
    return jax.jit(update)
