"""Adapters around the existing JE10 collector, league and scene curriculum.

The vmap unit is a complete one-learner native environment (seat P0). This keeps
the new experiment's learner seat while reusing JE10's actual opponent dispatch,
PFSP, weak roles, scene reset, native reward and event bookkeeping.
"""
import jax
import jax.numpy as j
import numpy as np
from scripts import je10_ops as ops, je09_bun_scenes as scenes
from scripts.je05_bun_recipe import POOL
from scripts import carrier_mechanism_engine as E

EXTRA=('tactics','provenance_source','combat_mix')


def course():
    return ops.curriculum_weights(j.zeros((10,len(ops.FUNNEL))),.4,.25)


def base_aux(ax):
    return {k:v for k,v in ax.items() if k not in EXTRA}


def raw(steps,params,states,aux,keys,pool,frozen,pfsp,generation):
    """Call the historical implementation; no rewritten opponent/course rules."""
    weights=course()
    def episode(st,ax,key):
        ss,aa=jax.tree.map(lambda a:a[None],(st,base_aux(ax)))
        # ctx layout belongs to the historical return/chase collector ABI.
        end,na,nk,batch,stats=ops.collect(params,ss,aa,key,pool,frozen,weights,pfsp,steps,
            assist_probability=.1,expert_hazard_mode='local_safe',
            bridge_config=dict(window=120,scale=2.),
            return_config=dict(window=120),chase_stage=j.int32(0),
            chase_supervision=False,dagger_round=generation)
        return jax.tree.map(lambda a:a[0],end),jax.tree.map(lambda a:a[0],na),nk,batch,stats
    states,ax,keys,batch,stats=jax.vmap(episode)(states,aux,keys)
    # [env,tick,1,...] -> [tick,env,...]. Only actual learner rows enter PPO.
    o,g,a,lp,adv,ret,(mm,bm),ctx=batch
    convert=lambda a:j.swapaxes(a[:,:,0],0,1)
    data=tuple(convert(a) for a in (o,g,a,lp,adv,ret,mm,bm))
    counts,poolcounts,parts,means=stats
    # Historical ctx columns 92:95 hold last-override source, member, alive.
    # Nominal dispatch counts alone overstate league exposure during assist.
    provenance=ctx[:,:,0,92:95]
    live=provenance[...,2]>0
    source=provenance[...,0].astype(j.int32)
    member=provenance[...,1].astype(j.int32)
    diagnostics=dict(scene_opponent_counts=counts.sum(0),league_counts=poolcounts.sum(0),
        native_parts=parts.sum(0),mean_reward=means[:,0].mean(),
        mean_terminal=means[:,1].mean(),course=weights,
        executed_source_live_ticks=j.stack([((source==i)&live).sum() for i in range(7)]),
        executed_league_member_live_ticks=j.stack([((source==3)&live&(member==i)).sum() for i in range(11)]))
    return (states,ax,keys),data,diagnostics


def collector(steps):
    return jax.pmap(lambda p,s,a,k,unused1,unused2,pool,extras:
        raw(steps,p,s,a,k,pool,extras['frozen'],extras['pfsp'],extras['generation']),
        in_axes=(0,0,0,0,None,None,0,0))


def advance_pool(pool,actor,index):
    """Same current slot and ring replacement order as je10_worker."""
    pool=jax.tree.map(lambda a,b:j.asarray(a).at[0].set(b),pool,actor)
    if index%16==0:
        slot=7+(index//16-1)%4
        pool=jax.tree.map(lambda a,b:a.at[slot].set(b),pool,actor)
    return pool


def initial_pool(historical,parent_actor):
    # Keep the six immutable historical policies; current/ring restart from the
    # matched common father, as in JE10's first-run initialization.
    result=jax.tree.map(j.asarray,historical)
    for i in [0,7,8,9,10]:
        result=jax.tree.map(lambda a,b:a.at[i].set(b),result,parent_actor)
    return result
