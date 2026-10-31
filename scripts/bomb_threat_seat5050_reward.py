"""A3S physical-seat projection of frozen A3 threat events; no new reward."""
import jax
import jax.numpy as j
from scripts.bomb_threat_reward import distance
from scripts.bomb_threat_iter2_recipe import settlement
CELLS=195
settle=settlement(5.,.6)

def chain_event(before,after,info,aux,metadata,seats):
    """Direct real-time old->new ray, own attribution, never geometric proximity."""
    n=len(before.t);rows=j.arange(n)
    pos=j.floor(before.pos[rows,seats]).astype(j.int32)
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
    old=(before.fuse>0)&(before.core.owner==seats[:,None,None])&(dr+dc>0)&aligned
    old&=(dr+dc<=before.core.bomb_blast)&(before.fuse<after.fuse[rows,r,c][:,None,None])
    # Deterministic earliest old event, tie by cell. Every later reuse is denied.
    score=j.where(old,before.fuse.astype(j.int32)*CELLS+j.arange(CELLS).reshape(1,13,15),1<<28)
    idx=score.reshape(n,-1).argmin(-1)
    birth=metadata['birth'].reshape(n,-1)[rows,idx]
    enemy=j.clip(j.floor(before.pos[rows,1-seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    active=~aux['terminal']
    event=dict(placed=info['bomb_placed'][rows,seats],connected_uncredited=old.any((1,2))&(birth>=0),
        old_cell=idx,old_id=birth,enemy_cell=enemy[:,0]*15+enemy[:,1],
        newly_threatens=info['tactical_bomb_newly_threatens_enemy'][rows,seats],
        safe_placement=info['survivable_bomb_placed'][rows,seats],active=active,
        settled=~((after.fuse>0).any((1,2))|(after.core.blast_linger>0).any((1,2))|(after.trapped>0).any(1)),
        bad=info['death'][rows,seats]|(after.trapped[rows,seats]>0)|~after.alive[rows,seats],
        terminal=aux['terminal'])
    added=(after.fuse>0)&~((before.fuse>0)&(before.core.owner==after.core.owner))
    birth=j.where(added,before.t[:,None,None],j.where(after.fuse>0,metadata['birth'],-1))
    return event,birth


def transition(before,after,info,aux,metadata,seats):
    e,birth=chain_event(before,after,info,aux,metadata,seats)
    s=metadata['bomb_threat'];rows=j.arange(len(before.t))
    # At blast contact retain unique OWN source until delayed brick clearance.
    cover=info['blast_owner_h']|info['blast_owner_v']
    owned=j.where(cover[rows,seats]|cover[rows,1-seats],cover[rows,seats]&~cover[rows,1-seats],s['brick_owned'])&before.brick
    removed=before.brick&~after.brick&owned
    blocked=after.wall|after.brick|after.core.pushable
    start=j.clip(j.floor(before.pos[rows,seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    enemy=j.clip(j.floor(after.pos[rows,1-seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    # Stock is a target only in the current native window. Carrier keeps home.
    window=~after.alive[rows,1-seats]|(after.trapped[rows,1-seats]>0)
    base=j.asarray([[[6,1],[6,13]]],j.int32)
    # Use engine bases, not hard-coded map coordinates.
    from scripts.carrier_mechanism_engine import env
    cages=j.asarray(env._BUN_BASES,j.int32)+1
    goal=j.where((after.bun_carried[rows,seats]>=0)[:,None],cages[seats],j.where(window[:,None],cages[1-seats],enemy))
    def compare(_):
        old=distance(blocked|removed,goal,start);new=distance(blocked,goal,start)
        return (new<old)&(new<1000)
    improved=jax.lax.cond(removed.any(),compare,lambda _:j.zeros(len(rows),bool),None)
    e=dict(e,safe=e['safe_placement'],threat=info['tactical_bomb_newly_threatens_enemy'][rows,seats]|
        (info['tactical_bomb_enemy_safe_moves_after'][rows,seats]<info['tactical_bomb_enemy_safe_moves_before'][rows,seats]),
        chain=e['connected_uncredited'],owned_removed=removed,improved=improved)
    n,r,m=settle(e,s);n['brick_owned']=owned&after.brick
    return n,r,m,birth
