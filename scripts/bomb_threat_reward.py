"""BOMB-THREAT-V1: event-only, owned topology and first-cover credit.

Episode gross cap .20 is below the existing .25 kill residual and 4 control
credit. Pending credit dies on self harm, terminal, or loss of control; gross
and cell dedup never refund. No inventory/tick/request reward.
"""
import jax
import jax.numpy as j
CAP=.20

def state(n):
    return dict(pending=j.zeros(n),gross=j.zeros(n),enemy_seen=j.zeros((n,195),bool),
        old_seen=j.full((n,195),-1,j.int32),brick_seen=j.zeros((n,13,15),bool),
        brick_owned=j.zeros((n,13,15),bool),brick_pending=j.zeros(n,j.int32))

def distance(blocked,goal,start):
    """Exact four-neighbor walking distance, full grid convergence; no dig cost."""
    n=blocked.shape[0];rows=j.arange(n)
    d=j.full(blocked.shape,1000,j.int32).at[rows,goal[:,0],goal[:,1]].set(0)
    def body(c):
        k,old,_=c
        z=j.pad(old,((0,0),(1,1),(1,1)),constant_values=1000)
        nxt=j.minimum(old,1+j.minimum(j.minimum(z[:,:-2,1:-1],z[:,2:,1:-1]),j.minimum(z[:,1:-1,:-2],z[:,1:-1,2:])))
        nxt=j.where(blocked,1000,nxt)
        return k+1,nxt,j.any(nxt!=old)
    _,d,_=jax.lax.while_loop(lambda c:(c[0]<195)&c[2],body,(0,d,j.array(True)))
    return d[rows,start[:,0],start[:,1]]

def settle(e,s):
    rows=j.arange(len(s['gross']));cell=j.clip(e['enemy_cell'],0,194);old=j.clip(e['old_cell'],0,194)
    valid=e['active']&~e['bad']&~e['terminal']
    threat=e['placed']&e['safe']&e['threat']&~s['enemy_seen'][rows,cell]&valid
    chain=e['placed']&e['safe']&e['chain']&(e['old_id']>s['old_seen'][rows,old])&valid
    bricks=e['owned_removed']&~s['brick_seen']
    topology=e['improved']&bricks.any((1,2))&valid
    amount=.02*threat+.01*chain+.04*topology
    pending=j.minimum(s['pending']+amount,j.maximum(0.,CAP-s['gross']))
    pending=j.where(valid,pending,0.)
    pay=j.where(e['settled']&valid,pending,0.)
    n=dict(s,pending=j.where(e['settled'],0.,pending),gross=s['gross']+pay,
        enemy_seen=s['enemy_seen'].at[rows,cell].set(s['enemy_seen'][rows,cell]|threat),
        old_seen=s['old_seen'].at[rows,old].set(j.where(chain,e['old_id'],s['old_seen'][rows,old])),
        brick_seen=s['brick_seen']|bricks,
        brick_pending=j.where(valid,s['brick_pending']+topology.astype(j.int32),0))
    paid_bricks=j.where(e['settled']&valid,n['brick_pending'],0)
    n['brick_pending']=j.where(e['settled'],0,n['brick_pending'])
    return n,pay,dict(candidate=threat|chain|topology,settled_count=(pay>0).astype(j.int32),pay=pay,
        threat_events=threat.astype(j.int32),chain_events=chain.astype(j.int32),brick_events=topology.astype(j.int32),
        brick_paid_events=paid_bricks,threat_pending=pending)

def transition(before,after,info,aux,metadata):
    from scripts.combat_mix_shaping import chain_event
    e,birth=chain_event(before,after,info,aux,metadata)
    s=metadata['bomb_threat'];rows=j.arange(len(before.t))
    # At blast contact retain unique OWN source until delayed brick clearance.
    cover=info['blast_owner_h']|info['blast_owner_v']
    owned=j.where(cover[:,0]|cover[:,1],cover[:,0]&~cover[:,1],s['brick_owned'])&before.brick
    removed=before.brick&~after.brick&owned
    blocked=after.wall|after.brick|after.core.pushable
    start=j.clip(j.floor(before.pos[:,0]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    enemy=j.clip(j.floor(after.pos[:,1]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    # Stock is a target only in the current native window. Carrier keeps home.
    window=~after.alive[:,1]|(after.trapped[:,1]>0)
    base=j.asarray([[[6,1],[6,13]]],j.int32)
    # Use engine bases, not hard-coded map coordinates.
    from scripts.carrier_mechanism_engine import env
    cages=j.asarray(env._BUN_BASES,j.int32)+1
    goal=j.where((after.bun_carried[:,0]>=0)[:,None],cages[0],j.where(window[:,None],cages[1],enemy))
    def compare(_):
        old=distance(blocked|removed,goal,start);new=distance(blocked,goal,start)
        return (new<old)&(new<1000)
    improved=jax.lax.cond(removed.any(),compare,lambda _:j.zeros(len(rows),bool),None)
    e=dict(e,safe=e['safe_placement'],threat=info['tactical_bomb_newly_threatens_enemy'][:,0]|
        (info['tactical_bomb_enemy_safe_moves_after'][:,0]<info['tactical_bomb_enemy_safe_moves_before'][:,0]),
        chain=e['connected_uncredited'],owned_removed=removed,improved=improved)
    n,r,m=settle(e,s);n['brick_owned']=owned&after.brick
    return n,r,m,birth
