"""Direct original JE10 batched collector, including historical seat and RNG ABI.

No per-environment vmap surrounds ops.collect: its rows%2 and rows%3 rules and
single league member per device rollout must see the original full batch.
"""
import jax
import jax.numpy as j
from scripts import combat_mix_legacy as shared
ops=shared.ops
course=shared.course
base_aux=shared.base_aux
advance_pool=shared.advance_pool
initial_pool=shared.initial_pool

def adapt_result(result,count):
    states,aux,key,batch,stats=result
    o,g,a,lp,adv,ret,(mm,bm),ctx=batch
    counts,poolcounts,parts,means=stats
    provenance=ctx[...,92:95];live=provenance[...,2]>0
    source=provenance[...,0].astype(j.int32);member=provenance[...,1].astype(j.int32)
    diag=dict(scene_opponent_counts=counts,league_counts=poolcounts,native_parts=parts,
        mean_reward=means[0],mean_terminal=means[1],course=course(),
        executed_source_live_ticks=j.stack([((source==i)&live).sum() for i in range(7)]),
        executed_league_member_live_ticks=j.stack([((source==3)&live&(member==i)).sum() for i in range(11)]))
    # This shape adapter carries the original device key unchanged into the next
    # call. It does not split or resample it. All rows are a storage ABI only.
    keys=j.broadcast_to(key,(count,)+key.shape)
    return (states,aux,keys),(o,g,a,lp,adv,ret,mm,bm),diag

def raw(steps,params,states,aux,keys,pool,frozen,pfsp,generation):
    result=ops.collect(params,states,base_aux(aux),keys[0],pool,frozen,course(),pfsp,steps,
        assist_probability=.1,expert_hazard_mode='local_safe',
        bridge_config=dict(window=120,scale=2.),return_config=dict(window=120),
        chase_stage=j.int32(0),chase_supervision=False,dagger_round=generation)
    return adapt_result(result,len(keys))

def collector(steps):
    return jax.pmap(lambda p,s,a,k,unused1,unused2,pool,extras:
        raw(steps,p,s,a,k,pool,extras['frozen'],extras['pfsp'],extras['generation']),
        in_axes=(0,0,0,0,None,None,0,0))
