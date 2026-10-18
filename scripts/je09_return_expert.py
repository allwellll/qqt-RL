"""Training-only action-set teacher on actual post-pickup learner states.

All equally short safe edges contribute; no actions or reward are replaced.
The ledger is native attributed drop -> first loose pickup, never carry init.
"""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import shortest, INF
DELTA=jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])

def state(n,h=13,w=15):
    return dict(distance=jnp.full((n,2,h,w),INF,jnp.int32))

def refresh(states,old,dirty,goals):
    def build(_):
        distance=shortest((states.wall|states.brick)[:,None],jnp.zeros_like(states.brick[:,None]),goals[None])
        return dict(distance=jnp.where(dirty[:,None,None,None],distance,old['distance']))
    return jax.lax.cond(jnp.any(dirty),build,lambda _:old,None)

def gate(states,aux,seat,window=120):
    rows=jnp.arange(seat.shape[0]);b=aux['bridge']
    # ticks=0 is the state AFTER the first pickup. Each action supervises
    # resulting ages1..120; pre-state age120 would produce age121, excluded.
    return (b['active'][rows,seat]&b['paid'][rows,seat,0]&b['paid'][rows,seat,1]
        &(b['ticks'][rows,seat]<window)&(states.bun_carried[rows,seat]==1-seat)
        &states.alive[rows,seat]&(states.trapped[rows,seat]==0)&~aux['terminal']&(states.scene!=0))

def action_set(pos,distance,legal,danger,blocked,fuse):
    n,h,w=distance.shape;rows=jnp.arange(n);cell=jnp.clip(pos.astype(jnp.int32),jnp.array([0,0]),jnp.array([h-1,w-1]))
    neighbor=jnp.clip(cell[:,None]+DELTA,jnp.array([0,0]),jnp.array([h-1,w-1]))
    d=distance[rows[:,None],neighbor[...,0],neighbor[...,1]];current=distance[rows,cell[:,0],cell[:,1]]
    safe=(danger[rows[:,None],neighbor[...,0],neighbor[...,1]]==0)&(fuse[rows[:,None],neighbor[...,0],neighbor[...,1]]<=0)&~blocked[rows[:,None],neighbor[...,0],neighbor[...,1]]
    edges=(d==current[:,None]-1)&(d<INF)&safe
    offset=neighbor.astype(jnp.float32)+.5-pos[:,None]
    gain=jnp.einsum('nad,bd->nab',offset,DELTA)
    # For each equivalent shortest edge choose all tied legal positive
    # waypoint gains. A subcell centring action is allowed even if it is not
    # itself the edge direction; the native legal mask decides collision.
    allowed=legal[:,None,:] & (gain>1e-4)
    best=jnp.max(jnp.where(allowed,gain,-1e9),-1,keepdims=True)
    choices=allowed & (gain>=best-1e-6) & edges[:,:,None]
    # Actual movement neighbor must also be safe. Wall-adjacent centring can
    # stay in the current cell; legal mask still supplies collision safety.
    actionsafe=(danger[rows[:,None],neighbor[...,0],neighbor[...,1]]==0)&(fuse[rows[:,None],neighbor[...,0],neighbor[...,1]]<=0)
    choices &= actionsafe[:,None,:]
    peredge=choices/jnp.maximum(choices.sum(-1,keepdims=True),1)
    target=peredge.sum(1);total=target.sum(-1,keepdims=True);target=target/jnp.maximum(total,1)
    valid=(current>0)&(current<INF)&(total[:,0]>0)&(danger[rows,cell[:,0],cell[:,1]]==0)
    return jax.lax.stop_gradient(target),jax.lax.stop_gradient(valid)

def labels(states,aux,legal,seat,danger,window=120):
    rows=jnp.arange(seat.shape[0]);distance=aux['return_expert']['distance'][rows,seat]
    danger=jnp.maximum(danger,(states.core.blast_linger>0).astype(jnp.float32))
    target,reachable=action_set(states.pos[rows,seat],distance,legal,danger,states.wall|states.brick,states.fuse)
    return target,jax.lax.stop_gradient(gate(states,aux,seat,window)&reachable)

def supervised_loss(move,legal,target,active,coefficient=0.,temperature=1.):
    target=jax.lax.stop_gradient(jnp.where(legal,target,0.));mass=target.sum(-1)
    target=target/jnp.maximum(mass[:,None],1e-8)
    lp=jax.nn.log_softmax(jnp.where(legal,move/temperature,-1e9))
    ce=-jnp.sum(target*lp,-1);valid=active&(mass>0)
    mean=jnp.sum(jnp.where(valid,ce,0.))/jnp.maximum(1,valid.sum())
    return coefficient*mean,jnp.array([mean,valid.mean()])
