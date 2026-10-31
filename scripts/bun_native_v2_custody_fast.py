"""Exact sparse recomputation of custody-v2 fields; identical cache ABI."""
import inspect
import jax
import jax.numpy as j
from scripts import bun_native_v2_custody_reward as reference

REFERENCE=reference.potential
BLOCK=64


def sparse_fields(blocked,goals,previous,dirty):
    """Compact dirty rows; never run a whole2048-env BFS for one changed row."""
    from scripts.je08_bun_potential import shortest
    n=len(dirty)
    indices=j.nonzero(dirty,size=n,fill_value=n)[0]
    count=j.sum(dirty.astype(j.int32))
    # A fixed small kernel amortizes launch cost while a dynamic loop visits
    # only the needed blocks. Out-of-range padded writes are dropped.
    padded=j.pad(indices,(0,BLOCK),constant_values=n)
    def body(i,fields):
        ids=jax.lax.dynamic_slice(padded,(i*BLOCK,),(BLOCK,))
        safe=j.minimum(ids,n-1)
        result=shortest(blocked[safe,None],j.zeros_like(blocked[safe,None]),goals[safe])
        return fields.at[ids].set(result,mode='drop')
    return jax.lax.fori_loop(0,(count+BLOCK-1)//BLOCK,body,previous)


def fields(blocked,target,allgoals,cache):
    topology=~cache['ready']|j.any(blocked!=cache['blocked'],(-2,-1))
    object_changed=topology|j.any(target!=cache['target'],(-2,-1))
    cages=sparse_fields(blocked,allgoals[:,:2],cache['fields'][:,:2],topology)
    # Absent objects have exactly INF everywhere; no BFS is needed.
    from scripts.je08_bun_potential import INF
    present=(target>=0).all(-1).any(-1)
    objects=j.where((object_changed&~present)[:,None,None,None],INF,cache['fields'][:,2:])
    objects=sparse_fields(blocked,allgoals[:,2:],objects,object_changed&present)
    return j.concatenate([cages,objects],1)


def build():
    source=inspect.getsource(REFERENCE).replace('def potential(states,cache):','def potential_fast(states,cache):')
    begin=source.index("    dirty=~cache['ready']")
    end=source.index('    # Minimum actual cage-to-cage',begin)
    source=source[:begin]+'    fields=fast_fields(blocked,target,allgoals,cache)\n'+source[end:]
    scope=dict(REFERENCE.__globals__,fast_fields=fields)
    exec(compile(source,__file__,'exec'),scope)
    return scope['potential_fast']


potential=build()


def install():
    reference.potential=potential
