"""Isolated JE05 sugar-trap semantics and bounded transport reward ledger."""
import jax
import jax.numpy as jnp

UNREACHABLE=32767
SHAPING_CAP=4.
CAPTURE_REWARD=24.
ENCODING='bun24-trap-timer-v2'


def trap_tick(ticks,hit,alive,pos,teams=None):
    teams=jnp.arange(2) if teams is None else teams
    newly=hit & alive & (ticks==0)
    current=jnp.where(newly,60,ticks)
    delta=jnp.abs(pos[:,None]-pos[None,:])
    close=jnp.all(delta<jnp.float32(41/40),-1)
    contact=close & ~jnp.eye(2,dtype=bool) & alive[None,:] & (current[None,:]==0)
    pop=contact & (teams[:,None]!=teams[None,:])
    rescue=contact & (teams[:,None]==teams[None,:])
    old=alive & (ticks>0) & ~newly
    killed=old & (pop.any(-1)|((ticks<=1)&~rescue.any(-1)))
    rescued=old & rescue.any(-1) & ~pop.any(-1)
    next_ticks=jnp.where(old,jnp.maximum(current-1,0),current)
    next_ticks=jnp.where(killed|rescued|~alive,0,next_ticks)
    return dict(ticks=next_ticks.astype(jnp.int16),alive=alive&~killed,death=killed,
                newly=newly,rescue=rescued,popper=jnp.where(old&pop.any(-1),jnp.argmax(pop,-1),-1).astype(jnp.int8))


def eligible_steal(steal,origin,trapped,alive,respawn):
    enemy=1-jnp.arange(2)
    vulnerable=(trapped[enemy]>0)|~alive[enemy]|(respawn[enemy]>0)
    return steal & (origin==enemy) & vulnerable


def precompute_regions(wall, brick, goals, bands=32):
    """CPU reverse Dijkstra per team; no route search inside rollout.

    Walls are impassable; breakable brick entry costs four (movement plus
    three route-breaking units). This frozen planning metric never claims a
    brick is presently walkable: only actual legal transitions earn progress.
    """
    import heapq
    import numpy as np
    wall=np.asarray(wall,bool);brick=np.asarray(brick,bool);goals=np.asarray(goals,bool)
    h,w=wall.shape;dist=np.full((2,h,w),UNREACHABLE,np.int32)
    for team in range(2):
        queue=[]
        for r,c in np.argwhere(goals[team]&~wall):
            dist[team,r,c]=0;heapq.heappush(queue,(0,int(r),int(c)))
        while queue:
            d,r,c=heapq.heappop(queue)
            if d!=dist[team,r,c]:continue
            cost=1+3*int(brick[r,c])
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w and not wall[rr,cc] and d+cost<dist[team,rr,cc]:
                    dist[team,rr,cc]=d+cost;heapq.heappush(queue,(d+cost,rr,cc))
    region=np.full_like(dist,-1);home=np.zeros_like(dist,bool);scale=[]
    for team in range(2):
        span=int(dist[team][goals[1-team]&~wall].max())
        if not 0<span<UNREACHABLE:raise ValueError('cages disconnected')
        scale.append(span);valid=dist[team]<UNREACHABLE
        region[team][valid]=np.clip((span-dist[team][valid])*bands//span,0,bands)
        home[team]=valid&(dist[team]<=dist[1-team])
    return dict(distance=dist,regions=region.astype(np.int16),home=home,bands=bands,cage_distance=np.asarray(scale,np.int32))


def transport_state(n):
    return dict(stage_id=jnp.zeros(n,jnp.int32),token=jnp.zeros(n,jnp.int32),active=jnp.zeros(n,bool),
        seen=jnp.zeros(n,jnp.uint32),highwater=jnp.full((n,2),-1,jnp.int16),home_seen=jnp.zeros((n,2),bool),
        balance=jnp.zeros(n),pending_capture=jnp.zeros(n),episode_gross=jnp.zeros(n),captured=jnp.zeros(n,jnp.uint32))


def transport_reward(e,s,coefficient=1.,legacy=None):
    # Map806 has one bun per origin. Origin is conserved across drop/repickup.
    # Custody loss never resets highwater; only a full episode reset does.
    rows=jnp.arange(s['active'].size);token=jnp.clip(e['token'],0,1)
    bit=jnp.left_shift(jnp.uint32(1),token.astype(jnp.uint32))
    newly=(e['carried']|e['capture'])&(~s['active']|(token!=s['token']))
    seen=(s['seen']&bit)!=0;steal=e['steal']&e['eligible']&~seen
    old=s['highwater'][rows,token]
    # Curriculum carry starts at its declared band without free progress.
    baseline=jnp.maximum(old,jnp.where(newly,e['before_region'],old))
    valid=(e['before_region']>=0)&(e['after_region']>=0)
    moving=e['carried']|e['capture']
    advance=moving&valid&(e['after_region']>baseline)
    delta=jnp.where(advance,e['after_region']-baseline,0)
    thresholds=jnp.array([8,16,24],jnp.int16)
    crossed=advance[:,None]&(baseline[:,None]<thresholds)&(e['after_region'][:,None]>=thresholds)
    marks=jnp.sum(crossed.astype(jnp.uint8)*jnp.array([1,2,4],jnp.uint8),-1).astype(jnp.uint8)
    home_old=s['home_seen'][rows,token]|(newly&e['home_half']&(e['before_region']==e['after_region']))
    home=advance&e['home_half']&~home_old
    proposed=.75*steal+.0625*delta+.25*crossed.sum(-1)+.25*home
    positive=jnp.minimum(proposed,jnp.maximum(0,SHAPING_CAP-s['episode_gross']))
    bad=e['death']|e['drop']|e['trade']
    balance=jnp.where(newly,0,s['balance'])+positive
    clawback=jnp.where(bad,-balance,0.)
    capture=e['capture']&~bad&((s['captured']&bit)==0)
    settled=e.get('settled',jnp.ones_like(capture))
    pending=jnp.where(bad,0.,s['pending_capture']+CAPTURE_REWARD*capture)
    paid_capture=jnp.where(settled&~bad,pending,0.)
    total=positive+clawback+paid_capture
    reward=legacy if coefficient==0 else coefficient*total
    water=jnp.where(moving&valid,jnp.maximum(baseline,e['after_region']),old)
    new=dict(s,stage_id=s['stage_id']+newly.astype(jnp.int32),token=jnp.where(moving,token,s['token']),
        active=e['carried']&~bad&~e['capture'],seen=s['seen']|jnp.where(e['steal']|newly,bit,jnp.uint32(0)),
        highwater=s['highwater'].at[rows,token].set(water.astype(jnp.int16)),home_seen=s['home_seen'].at[rows,token].set(home_old|home),
        balance=jnp.where(bad|settled&((s['pending_capture']>0)|capture),0,balance),pending_capture=jnp.where(settled,0.,pending),episode_gross=s['episode_gross']+positive,
        captured=s['captured']|jnp.where(capture,bit,jnp.uint32(0)))
    metrics=dict(eligible_steal=steal,unconditional_steal=e['steal']&~e['eligible'],repeat_pickup=e['pickup']&seen,
        path_delta=delta,reverse=moving&valid&(e['after_region']<e['before_region']),shaping=positive,
        clawback=clawback,capture=paid_capture,progress_marks=marks,home=home,region_advance=advance)
    return reward,new,metrics


def inventory_pickup(slots,counts,item):
    same=(slots==item)&(item>0);empty=slots==0
    accepted=(item>0)&(same.any()|empty.any())
    index=jnp.where(same.any(),jnp.argmax(same),jnp.argmax(empty))
    slots=slots.at[index].set(jnp.where(accepted,item,slots[index]).astype(slots.dtype))
    counts=counts.at[index].add(jnp.where(accepted,jnp.where(item==2,3,1),0).astype(counts.dtype))
    return slots,counts,accepted


def inventory_use(slots,counts):
    counts=counts.at[0].set(jnp.maximum(0,counts[0]-1))
    empty=counts[0]==0
    slots=jnp.where(empty,jnp.concatenate([slots[1:],jnp.zeros(1,slots.dtype)]),slots)
    counts=jnp.where(empty,jnp.concatenate([counts[1:],jnp.zeros(1,counts.dtype)]),counts)
    return slots,counts
