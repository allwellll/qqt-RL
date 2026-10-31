"""Typed native inventory/death-drop/airdrop transitions for the JE07 candidate.

Queues are bounded explicitly; overflow is an observable error, never silently
counted as a successful parity result. This candidate requires new state fields.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

H, W, N = 13, 15, 195
QUEUE_CAPACITY = 1024


class NativeItems(NamedTuple):
    cover: jax.Array
    counts: jax.Array
    grave: jax.Array
    grave_size: jax.Array
    flight: jax.Array
    flight_size: jax.Array
    flight_total: jax.Array
    flight_dropped: jax.Array
    pending: jax.Array
    pending_size: jax.Array
    fall_type: jax.Array
    fall_super: jax.Array
    fall_count: jax.Array
    fall_tick: jax.Array
    overflow: jax.Array


def fresh(cover=None):
    cover=jnp.zeros((H,W),bool) if cover is None else cover
    return NativeItems(cover,jnp.zeros((H,W),jnp.int16),
        jnp.zeros((QUEUE_CAPACITY,3),jnp.int16),jnp.int32(0),
        jnp.zeros((QUEUE_CAPACITY,3),jnp.int16),jnp.int32(0),jnp.int32(0),jnp.int32(0),
        jnp.zeros((QUEUE_CAPACITY,5),jnp.int16),jnp.int32(0),
        jnp.full(N,-1,jnp.int16),jnp.zeros(N,bool),jnp.zeros(N,jnp.int16),
        jnp.zeros(N,jnp.int32),jnp.int32(0))


def append(queue, size, item, valid, overflow):
    queue=queue.at[jnp.minimum(size,QUEUE_CAPACITY-1)].set(
        jnp.where(valid & (size<QUEUE_CAPACITY),item,queue[jnp.minimum(size,QUEUE_CAPACITY-1)]))
    overflow=overflow+(valid & (size>=QUEUE_CAPACITY)).astype(jnp.int32)
    return queue,jnp.minimum(QUEUE_CAPACITY,size+valid.astype(jnp.int32)),overflow


def bury(s, types, supers, counts, mask):
    def body(i,carry):
        q,n,over=carry
        return append(q,n,jnp.array([types[i],supers[i],counts[i]],jnp.int16),mask[i],over)
    q,n,o=jax.lax.fori_loop(0,types.size,body,(s.grave,s.grave_size,s.overflow))
    return s._replace(grave=q,grave_size=n,overflow=o)


def available(core, tactical, field, s):
    rr,cc=jnp.indices((H,W)); cells=jnp.clip(core.pos.astype(jnp.int32),jnp.array([0,0]),jnp.array([H-1,W-1]))
    occupied=jnp.any(core.alive[:,None,None] & (rr[None]==cells[:,0,None,None]) & (cc[None]==cells[:,1,None,None]),axis=0)
    return (~core.wall & ~core.brick & ~core.pushable & ~s.cover
        & (core.crate==0) & (tactical==0) & (field==0) & (core.fuse==0)
        & (core.blast_linger==0) & ~occupied & (s.fall_type.reshape(H,W)<0))


def death_inventory(s, core, dead, slots, counts, lows, speed_step):
    # The producer preserves each attribute's quantity, and each inventory stack.
    for p in range(2):
        quantities=jnp.concatenate([jnp.array([
            jnp.maximum(0,core.bombs_cap[p]-lows[0]),
            jnp.maximum(0,core.blast_cap[p]-lows[1]),
            jnp.maximum(0,jnp.floor((core.spd_g[p]-lows[2])/speed_step+0.5))],jnp.int16),counts[p]])
        types=jnp.concatenate([jnp.arange(3,dtype=jnp.int16),slots[p].astype(jnp.int16)+2])
        cell=jnp.floor(core.pos[p]).astype(jnp.int16)
        for i in range(10):
            q,n,o=append(s.pending,s.pending_size,jnp.array([types[i],0,quantities[i],cell[0],cell[1]],jnp.int16),dead[p]&(quantities[i]>0),s.overflow)
            s=s._replace(pending=q,pending_size=n,overflow=o)
    core=core._replace(bombs_cap=jnp.where(dead,lows[0],core.bombs_cap),blast_cap=jnp.where(dead,lows[1],core.blast_cap),spd_g=jnp.where(dead,lows[2],core.spd_g))
    return s,core,jnp.where(dead[:,None],0,slots),jnp.where(dead[:,None],0,counts)


def spawn(core,tactical,s,cell,typ,sup,count,valid):
    r,c=cell//W,cell%W
    core=core._replace(crate=core.crate.at[r,c].set(jnp.where(valid,jnp.where(typ<3,1+typ+sup.astype(jnp.int16)*3,0),core.crate[r,c]).astype(jnp.int8)),rec_crate=core.rec_crate.at[r,c].set(jnp.where(valid,False,core.rec_crate[r,c])))
    tactical=tactical.at[r,c].set(jnp.where(valid,jnp.where(typ>=3,typ-2,0),tactical[r,c]).astype(jnp.int8))
    s=s._replace(counts=s.counts.at[r,c].set(jnp.where(valid,count,s.counts[r,c])))
    return core,tactical,s


def release_deaths(core,tactical,field,s):
    rr,cc=jnp.indices((H,W))
    pending=s.pending; size=s.pending_size
    s=s._replace(pending=jnp.zeros_like(pending),pending_size=jnp.int32(0))
    def body(i,carry):
        core,tactical,s=carry; drop=pending[i]
        free=available(core,tactical,field,s)
        distance=jnp.abs(rr-drop[3])+jnp.abs(cc-drop[4])
        cell=jnp.argmin(jnp.where(free,distance*N+jnp.arange(N).reshape(H,W),999999)).astype(jnp.int32)
        ok=free.any()
        core,tactical,s=spawn(core,tactical,s,cell,drop[0],drop[1].astype(bool),drop[2],ok)
        q,n,o=append(s.pending,s.pending_size,drop,~ok,s.overflow)
        return core,tactical,s._replace(pending=q,pending_size=n,overflow=o)
    return jax.lax.fori_loop(0,size,body,(core,tactical,s))


def airdrop(core,tactical,field,s,key):
    """JS native schedule; only the chosen launch cell is stochastic."""
    t=core.t
    def land(i,carry):
        core,tactical,s=carry
        due=(s.fall_type[i]>=0)&(t-s.fall_tick[i]>=3)
        r,c=i//W,i%W
        free=~core.wall[r,c]&~core.brick[r,c]&~core.pushable[r,c]&(core.crate[r,c]==0)&(tactical[r,c]==0)&(field[r,c]==0)&(core.fuse[r,c]==0)
        q,n,o=append(s.grave,s.grave_size,jnp.array([s.fall_type[i],s.fall_super[i],s.fall_count[i]],jnp.int16),due&~free,s.overflow)
        s=s._replace(grave=q,grave_size=n,overflow=o)
        core,tactical,s=spawn(core,tactical,s,i,s.fall_type[i],s.fall_super[i],s.fall_count[i],due&free)
        s=s._replace(fall_type=s.fall_type.at[i].set(jnp.where(due,-1,s.fall_type[i])))
        return core,tactical,s
    core,tactical,s=jax.lax.fori_loop(0,N,land,(core,tactical,s))
    cycle=t%300
    start=cycle==270
    s=s._replace(flight=jnp.where(start,s.grave,s.flight),flight_size=jnp.where(start,s.grave_size,s.flight_size),flight_total=jnp.where(start,s.grave_size,s.flight_total),flight_dropped=jnp.where(start,0,s.flight_dropped),grave=jnp.where(start,jnp.zeros_like(s.grave),s.grave),grave_size=jnp.where(start,0,s.grave_size))
    col=jnp.clip(jnp.floor(W+1.5-(W+3)*(cycle-270)/30).astype(jnp.int32),0,W-1)
    target=jnp.floor((cycle-272)*s.flight_total/24+0.5).astype(jnp.int32)
    enabled=(cycle>=273)&(cycle<=296)
    draws=jax.random.uniform(key,(H,))
    launched=jnp.full(H,-1,jnp.int16)
    def launch(i,carry):
        s,launched=carry
        free=available(core,tactical,field,s)&(jnp.arange(W)[None,:]==col)
        n=free.sum(); choice=jnp.floor(draws[i]*n).astype(jnp.int32)
        cell=jnp.argmax((jnp.cumsum(free.reshape(-1))-1==choice)&free.reshape(-1))
        ok=enabled&(s.flight_size>0)&(s.flight_dropped<target)&(n>0)
        drop=s.flight[0]
        flight=jnp.concatenate([s.flight[1:],jnp.zeros((1,3),jnp.int16)],axis=0)
        s=s._replace(flight=jnp.where(ok,flight,s.flight),flight_size=s.flight_size-ok.astype(jnp.int32),flight_dropped=s.flight_dropped+ok.astype(jnp.int32),
            fall_type=s.fall_type.at[cell].set(jnp.where(ok,drop[0],s.fall_type[cell])),fall_super=s.fall_super.at[cell].set(jnp.where(ok,drop[1].astype(bool),s.fall_super[cell])),fall_count=s.fall_count.at[cell].set(jnp.where(ok,drop[2],s.fall_count[cell])),fall_tick=s.fall_tick.at[cell].set(jnp.where(ok,t,s.fall_tick[cell])))
        return s,launched.at[i].set(jnp.where(ok,cell,-1).astype(jnp.int16))
    s,launched=jax.lax.fori_loop(0,H,launch,(s,launched))
    def restore(s):
        def body(i,s):
            q,n,o=append(s.grave,s.grave_size,s.flight[i],jnp.bool_(True),s.overflow)
            return s._replace(grave=q,grave_size=n,overflow=o)
        s=jax.lax.fori_loop(0,s.flight_size,body,s)
        return s._replace(flight=jnp.zeros_like(s.flight),flight_size=jnp.int32(0))
    s=jax.lax.cond(cycle==299,restore,lambda s:s,s)
    s=s._replace(flight_total=jnp.where(cycle==0,0,s.flight_total),flight_dropped=jnp.where(cycle==0,0,s.flight_dropped))
    return core,tactical,s,launched


def pickup(core,tactical,trapped,slots,counts,s,key,caps,speed_step):
    """Stable actor order, typed stacks, capped overflow remains typed."""
    unknown=(jax.random.uniform(key,(2,))*3).astype(jnp.int32)
    picked=jnp.zeros(2,bool); grew=jnp.zeros(2,bool)
    for p in range(2):
        r,c=jnp.clip(core.pos[p].astype(jnp.int32),jnp.array([0,0]),jnp.array([H-1,W-1]))
        code=core.crate[r,c]; item=tactical[r,c]; qty=jnp.maximum(1,s.counts[r,c])
        ok=core.alive[p]&(trapped[p]==0)
        # Tactical stacks use their explicit death-drop quantity (default glue=3).
        same=(slots[p]==item)&(counts[p]>0); empty=counts[p]==0
        slot=jnp.where(same.any(),jnp.argmax(same),jnp.argmax(empty))
        take=ok&(item>0)&(same.any()|empty.any())
        gain=jnp.where(s.counts[r,c]>0,s.counts[r,c],jnp.where(item==2,3,1))
        slots=slots.at[p,slot].set(jnp.where(take,item,slots[p,slot]))
        counts=counts.at[p,slot].set(jnp.where(take,counts[p,slot]+gain,counts[p,slot]))
        attr=jnp.where(code==7,unknown[p],(code-1)%3)
        sup=(code>=4)&(code<=6)
        take_attr=ok&(code>0)
        before=jnp.array([core.bombs_cap[p],core.blast_cap[p],core.spd_g[p]])
        def grow(i,carry):
            stats,s=carry
            at_cap=stats[attr]>=caps[attr]-jnp.where(attr==2,1e-4,0.)
            q,n,o=append(s.grave,s.grave_size,jnp.array([attr,sup,0],jnp.int16),at_cap,s.overflow)
            s=s._replace(grave=q,grave_size=n,overflow=o)
            add=jnp.where(sup,5.,1.)*jnp.where(attr==2,speed_step,1.)
            stats=stats.at[attr].set(jnp.where(at_cap,stats[attr],jnp.minimum(caps[attr],stats[attr]+add)))
            return stats,s
        stats,s=jax.lax.fori_loop(0,jnp.where(take_attr,qty,0),grow,(before,s))
        core=core._replace(bombs_cap=core.bombs_cap.at[p].set(stats[0]),blast_cap=core.blast_cap.at[p].set(stats[1]),spd_g=core.spd_g.at[p].set(stats[2]),
            crate=core.crate.at[r,c].set(jnp.where(take_attr,0,core.crate[r,c])),rec_crate=core.rec_crate.at[r,c].set(jnp.where(take|take_attr,False,core.rec_crate[r,c])))
        tactical=tactical.at[r,c].set(jnp.where(take,0,tactical[r,c]))
        s=s._replace(counts=s.counts.at[r,c].set(jnp.where(take|take_attr,0,s.counts[r,c])))
        picked=picked.at[p].set(take|take_attr); grew=grew.at[p].set((stats>before).any())
    return core,tactical,slots,counts,s,picked,grew
