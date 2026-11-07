"""Opt-in implementation of the web owner's bun-native-v3-items contract.

Pure functions; importing this module never changes an existing environment.
PRNG keys belong to the saved collector stream. A uniform tape is accepted for
exact comparisons against the authoritative JS implementation.
"""
import jax
import jax.numpy as j
from jax_bomb import je07_native_items as old

VERSION = 'bun-native-v3-items'


def available(core, tactical, field, native, loose=None, pending_crate=None):
    rr, cc = j.indices((13, 15))
    # Match JS _cellOccupiedByPlayer: CFG.radius=.36, EPS=1e-4,
    # including living sugar cages and every AABB-overlapped cell.
    y, x = core.pos[:, 0, None, None], core.pos[:, 1, None, None]
    occupied = (core.alive[:, None, None] & (y+.36 > rr+1e-4)
        & (y-.36 < rr+1-1e-4) & (x+.36 > cc+1e-4)
        & (x-.36 < cc+1-1e-4)).any(0)
    free = (~core.wall & ~core.brick & ~core.pushable & ~native.cover
        & (core.crate == 0) & (tactical == 0) & (field == 0)
        & (core.fuse == 0) & (core.blast_linger == 0)
        & (core.brick_linger == 0) & ~occupied & (native.fall_type.reshape(13,15) < 0))
    if loose is not None:
        free &= ~(loose > 0).any(-1)
    if pending_crate is not None:
        free &= pending_crate < 0
    # Native JAX retains an expiring brick in core.brick until its pending
    # crate is exposed, so core.brick also excludes its deferred crate cell.
    return free


def choose(free, uniform):
    flat=free.reshape(-1); count=flat.sum()
    rank=j.minimum(j.floor(uniform*j.maximum(count,1)).astype(j.int32),j.maximum(count-1,0))
    return j.where(count>0,j.argmax(flat & (j.cumsum(flat)-1 == rank)),-1).astype(j.int32)


def release_deaths(core,tactical,field,native,key,loose=None,draws=None,pending_crate=None):
    pending,size=native.pending,native.pending_size
    native=native._replace(pending=j.zeros_like(pending),pending_size=j.int32(0))
    free=available(core,tactical,field,native,loose,pending_crate)
    def body(i,carry):
        core,tactical,native,free=carry
        drop=pending[i]
        u=jax.random.uniform(jax.random.fold_in(key,i)) if draws is None else draws[i]
        cell=choose(free,u); ok=cell>=0; index=j.maximum(cell,0)
        core,tactical,native=old.spawn(core,tactical,native,index,drop[0],drop[1].astype(bool),drop[2],ok)
        free=free.reshape(-1).at[index].set(j.where(ok,False,free.reshape(-1)[index])).reshape(13,15)
        q,n,over=old.append(native.pending,native.pending_size,drop,~ok,native.overflow)
        return core,tactical,native._replace(pending=q,pending_size=n,overflow=over),free
    core,tactical,native,_=jax.lax.fori_loop(0,size,body,(core,tactical,native,free))
    return core,tactical,native


def initial_supply(state,key,rate,super_fraction,tactical_fraction=.30,enabled=True,draws=None):
    core=state.core; tactical=state.tactical_crate; native=state.native_items
    initial=((core.crate>0)|(tactical>0)).sum()
    bricks=(core.brick & ~core.wall).sum()
    rate=j.where(enabled,j.clip(rate,0.,1.),0.)
    baseline=initial+bricks*rate
    requested=j.where(enabled,j.maximum(0,j.ceil(j.where(baseline>0,baseline,10.)*1.5-baseline)),0).astype(j.int32)
    free=available(core,tactical,state.field_item,native,state.bun_loose)
    count=j.minimum(requested,free.sum())
    # Constant-size tape enables vmap of dynamic loop bounds without drawing
    # from any host RNG or expanding the checkpoint state schema.
    draws=jax.random.uniform(key,(195,4)) if draws is None else draws
    def body(i,carry):
        core,tactical,native,free=carry; d=draws[i]
        cell=choose(free,d[0])
        is_tactical=d[1]<tactical_fraction
        typ=j.where(is_tactical,3+j.floor(d[2]*2),j.floor(d[2]*3)).astype(j.int16)
        sup=~is_tactical & (d[3]<super_fraction)
        # count=0 means native implicit quantity: growth/banana=1, glue=3.
        core,tactical,native=old.spawn(core,tactical,native,cell,typ,sup,j.int16(0),j.bool_(True))
        free=free.reshape(-1).at[cell].set(False).reshape(13,15)
        return core,tactical,native,free
    core,tactical,native,_=jax.lax.fori_loop(0,count,body,(core,tactical,native,free))
    summary=j.stack([initial,bricks,rate,baseline,requested,count,requested-count,initial+count,baseline+count])
    return state._replace(core=core,tactical_crate=tactical,native_items=native),summary


def supply_batch(st,keys,duel=False,eligible=None):
    from jax_bomb import levels,je07_bun_env as env
    table=levels.active()
    def one(state,key):
        index=j.maximum(state.core.level_id,0)
        rate=0. if duel else table.rate[index]
        return initial_supply(state,jax.random.fold_in(key,0x17E3),rate,
            table.super_f[index],env.TACTICAL_ITEM_FRACTION)
    newer,summary=jax.vmap(one)(st,keys)
    if eligible is not None:
        newer=jax.tree.map(lambda a,b:j.where(eligible.reshape(eligible.shape+(1,)*(a.ndim-eligible.ndim)),a,b),newer,st)
        summary=j.where(eligible[:,None],summary,0.)
    return newer,summary
