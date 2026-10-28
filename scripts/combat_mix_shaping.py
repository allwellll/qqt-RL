"""Bounded, delayed combat-chain credit. No payment for live inventory."""
import jax.numpy as j

CELLS=13*15
CAP=5


def state(n):
    return dict(pending=j.zeros(n,j.int32),gross=j.zeros(n,j.int32),
        enemy_seen=j.zeros((n,CELLS),bool),old_seen=j.full((n,CELLS),-1,j.int32))


def settle(e,s,coefficient):
    """One unique own old-bomb event + newly covered enemy cell; settle safely."""
    n=s['pending'].shape[0];rows=j.arange(n)
    cell=j.clip(e['enemy_cell'],0,CELLS-1)
    old_cell=j.clip(e.get('old_cell',j.zeros(n,j.int32)),0,CELLS-1)
    old_id=e.get('old_id',j.zeros(n,j.int32))
    novel=(~s['enemy_seen'][rows,cell])&(old_id>s['old_seen'][rows,old_cell])
    candidate=(e['placed']&e['connected_uncredited']&e['newly_threatens']&
        e['safe_placement']&e['active']&~e['bad']&~e['terminal']&novel&
        (s['gross']<CAP)&(s['pending']<CAP))
    pending=s['pending']+candidate.astype(j.int32)
    pending=j.where(e['bad']|e['terminal'],0,pending)
    count=j.where(e['settled']&e['active']&~e['bad']&~e['terminal'],
        j.minimum(pending,CAP-s['gross']),0)
    reward=count.astype(j.float32)*coefficient
    new=dict(pending=j.where(e['settled'],0,pending),gross=s['gross']+count,
        enemy_seen=s['enemy_seen'].at[rows,cell].set(s['enemy_seen'][rows,cell]|candidate),
        old_seen=s['old_seen'].at[rows,old_cell].set(j.where(candidate,old_id,s['old_seen'][rows,old_cell])))
    return new,reward,dict(candidate=candidate,settled_count=count,pay=reward)


def chain_event(before,after,info,aux,metadata):
    """Direct real-time old->new ray, own attribution, never geometric proximity."""
    n=len(before.t);rows=j.arange(n)
    pos=j.floor(before.pos[:,0]).astype(j.int32)
    r=j.clip(pos[:,0],0,12);c=j.clip(pos[:,1],0,14)
    rr,cc=j.indices((13,15))
    dr=j.abs(rr[None]-r[:,None,None]);dc=j.abs(cc[None]-c[:,None,None])
    solid=before.wall|before.brick|before.core.pushable
    # Prefix counts exclude both endpoints; bricks stop the ray.
    row_prefix=j.cumsum(solid.astype(j.int32),axis=2)
    col_prefix=j.cumsum(solid.astype(j.int32),axis=1)
    row_hi=j.maximum(cc[None],c[:,None,None]);row_lo=j.minimum(cc[None],c[:,None,None])
    col_hi=j.maximum(rr[None],r[:,None,None]);col_lo=j.minimum(rr[None],r[:,None,None])
    row_between=j.take_along_axis(row_prefix,j.maximum(row_hi-1,0),axis=2)-j.take_along_axis(row_prefix,row_lo,axis=2)
    col_between=j.take_along_axis(col_prefix,j.maximum(col_hi-1,0),axis=1)-j.take_along_axis(col_prefix,col_lo,axis=1)
    aligned=((dr==0)&(row_between<=0))|((dc==0)&(col_between<=0))
    old=(before.fuse>0)&(before.core.owner==0)&(dr+dc>0)&aligned
    old&=(dr+dc<=before.core.bomb_blast)&(before.fuse<after.fuse[rows,r,c][:,None,None])
    # Deterministic earliest old event, tie by cell. Every later reuse is denied.
    score=j.where(old,before.fuse.astype(j.int32)*CELLS+j.arange(CELLS).reshape(1,13,15),1<<28)
    idx=score.reshape(n,-1).argmin(-1)
    birth=metadata['birth'].reshape(n,-1)[rows,idx]
    enemy=j.clip(j.floor(before.pos[:,1]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    active=~aux['terminal']
    event=dict(placed=info['bomb_placed'][:,0],connected_uncredited=old.any((1,2))&(birth>=0),
        old_cell=idx,old_id=birth,enemy_cell=enemy[:,0]*15+enemy[:,1],
        newly_threatens=info['tactical_bomb_newly_threatens_enemy'][:,0],
        safe_placement=info['survivable_bomb_placed'][:,0],active=active,
        settled=~((after.fuse>0).any((1,2))|(after.core.blast_linger>0).any((1,2))|(after.trapped>0).any(1)),
        bad=info['death'][:,0]|(after.trapped[:,0]>0)|~after.alive[:,0],
        terminal=aux['terminal'])
    added=(after.fuse>0)&~((before.fuse>0)&(before.core.owner==after.core.owner))
    birth=j.where(added,before.t[:,None,None],j.where(after.fuse>0,metadata['birth'],-1))
    return event,birth
