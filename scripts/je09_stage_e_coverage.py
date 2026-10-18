"""Process-local Stage E coverage adapter; Stage C/D source files stay sealed.

Only legal episode initialization is reweighted. Native drop, actions, rewards,
observations and first pickup remain untouched; labels use current states only.
"""
from functools import lru_cache
import types
import numpy as np
import jax
import jax.numpy as jnp
from scripts import je09_bun_scenes as scenes,je09_return_expert as teacher

ORIGINAL_INIT=scenes.init_batch

def coverage_choice(rank,fraction):
    return rank<jnp.int32(rank.shape[0]*fraction)

@lru_cache(None)
def tables():
    # [return group, chase bucket(short + four D buckets), learner seat]
    bounds=((2,8),)+scenes.DISTANCE_BUCKETS
    groups=[[[[] for _ in range(2)] for _ in bounds] for _ in range(2)]
    distance=np.asarray(scenes.TABLES[1]['distance'])
    candidates=np.asarray(scenes.BRIDGE_PAIRS)
    longs=np.asarray(scenes.LONG_PAIRS);counts=np.asarray(scenes.LONG_COUNTS)
    for bi in range(5):
        pairs=candidates if bi==0 else longs[bi-1,:counts[bi-1]]
        for team in range(2):
            d=distance[team,pairs[:,0],pairs[:,1]]
            for ri,mask in enumerate([(d>=1)&(d<24),(d>=24)&(d<=36)]):
                groups[ri][bi][team]=pairs[mask]
                assert mask.any(),(ri,bi,team)
    count=np.array([[[len(p) for p in seats] for seats in group] for group in groups],np.int32)
    maximum=int(count.max())
    packed=np.array([[[np.concatenate([p,np.repeat(p[:1],maximum-len(p),axis=0)]) for p in seats] for seats in group] for group in groups],np.int32)
    return jnp.asarray(packed[0]),jnp.asarray(packed[1]),jnp.asarray(count)

def init_batch(key,n,course,distance_stage=None,fraction=.5,force_long_chase=False):
    states=ORIGINAL_INIT(key,n,course,distance_stage)
    kr,kp,kc=jax.random.split(jax.random.fold_in(key,0xE10),3)
    rows=jnp.arange(n);seat=rows%2;enemy=1-seat
    short,long,counts=tables();packed=jnp.stack([short,long])
    group=coverage_choice(jax.random.permutation(kr,n),fraction).astype(jnp.int32)
    stage=jnp.int32(0) if distance_stage is None else distance_stage
    chase=jnp.full(n,stage+1) if force_long_chase else jnp.where(jax.random.bernoulli(kc,.5,(n,)),stage+1,0)
    selected=jax.random.randint(kp,(n,),0,counts[group,chase,seat]);pair=packed[group,chase,seat,selected]
    chain=states.scene==9
    pos=states.pos.at[rows,seat].set(jnp.where(chain[:,None],pair[:,2:]+.5,states.pos[rows,seat]))
    pos=pos.at[rows,enemy].set(jnp.where(chain[:,None],pair[:,:2]+.5,pos[rows,enemy]))
    return states._replace(core=states.core._replace(pos=pos),bun_spawn_pos=pos)

def gate(states,aux,seat,window=120):
    rows=jnp.arange(len(seat));enemy=1-seat;b=aux['bridge'];drop=aux['drop'];chase=aux['chase']
    start=b['start_distance'][rows,seat]
    nonce=drop['nonce'][rows,enemy]
    return (teacher.gate(states,aux,seat,window)&(start>0)&(start<=36)
        &(nonce>0)&(drop['credit'][rows,enemy]==seat)&(chase['nonce'][rows,seat]==nonce)
        &(chase['initial'][rows,seat]>=2)&(chase['initial'][rows,seat]<=32))

def labels(states,aux,legal,seat,danger,window=120):
    target,active=teacher.labels(states,aux,legal,seat,danger,window)
    return target,jax.lax.stop_gradient(active&gate(states,aux,seat,window))

def install(fraction):
    from scripts import je09_bun_ops as ops
    # Reweighted initialization is dataset-only; formal course remains Stage D.
    ops.return_expert=types.SimpleNamespace(**{k:getattr(teacher,k) for k in ['state','refresh','supervised_loss']},gate=gate,labels=labels)


def bind_ops():
    from scripts import je09_stage_e_ops as ops
    ops.return_expert=types.SimpleNamespace(**{k:getattr(teacher,k) for k in ['state','refresh','supervised_loss']},gate=gate,labels=labels)
