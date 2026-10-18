"""Training-only legal walking teacher, gated by native drop provenance.

The actor receives ordinary observations only. This module emits auxiliary
labels; it never changes actions, rewards, observations or evaluation policy.
"""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import shortest, INF


def state(n,h=13,w=15):
    return dict(distance=jnp.full((n,2,h,w),INF,jnp.int32))


def refresh(states,routing,old,dirty):
    n=states.pos.shape[0];h,w=states.wall.shape[-2:]
    target=routing['target'][:,::-1];safe=jnp.clip(target,jnp.array([0,0]),jnp.array([h-1,w-1]))
    goals=jnp.zeros((n,2,h,w),bool).at[jnp.arange(n)[:,None],jnp.arange(2)[None],safe[...,0],safe[...,1]].set(jnp.all(target>=0,-1))
    def build(_):
        # Bricks are impassable: reward's weighted brick planning is separate.
        distance=shortest((states.wall|states.brick)[:,None],jnp.zeros_like(states.brick[:,None]),goals)
        return dict(distance=jnp.where(dirty[:,None,None,None],distance,old['distance']))
    return jax.lax.cond(jnp.any(dirty),build,lambda _:old,None)


def direction(pos,distance,legal):
    """Follow a descending grid edge via its centre, including subcell turns."""
    n,h,w=distance.shape;rows=jnp.arange(n)
    cell=jnp.clip(pos.astype(jnp.int32),jnp.array([0,0]),jnp.array([h-1,w-1]))
    delta=jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])
    neighbors=jnp.clip(cell[:,None]+delta,jnp.array([0,0]),jnp.array([h-1,w-1]))
    d=distance[rows[:,None],neighbors[...,0],neighbors[...,1]]
    current=distance[rows,cell[:,0],cell[:,1]]
    downhill=(d<current[:,None])&(d<INF)
    # Stable shortest-edge waypoint; choose a different axis only to centre
    # for a legal corner. No Manhattan shortcut replaces the grid search.
    edge=jnp.argmin(jnp.where(downhill,d,INF),-1)
    waypoint=neighbors[rows,edge].astype(jnp.float32)+.5
    offset=waypoint-pos
    gain=jnp.sum(delta*offset[:,None],-1)
    allowed=legal & (gain>1e-4)
    move=jnp.argmax(jnp.where(allowed,gain,-1e9),-1)
    valid=(current>0)&(current<INF)&downhill.any(-1)&allowed.any(-1)
    return move,valid


def labels(states,aux,legal,seat,danger=None,hazard_mode="global"):
    n=states.pos.shape[0];rows=jnp.arange(n);enemy=1-seat
    target=aux['routing']['target'][rows,enemy]
    safe=jnp.clip(target,jnp.array([0,0]),jnp.array([states.wall.shape[-2]-1,states.wall.shape[-1]-1]))
    loose=states.bun_loose[rows,safe[:,0],safe[:,1],enemy]>0
    drop=aux['drop'];nonce=drop['nonce'][rows,enemy]
    active=(nonce>0)&(drop['credit'][rows,enemy]==seat)&~drop['cycle_paid'][rows,seat,enemy]
    active &= aux['potential']['drop']['active'][rows,seat]
    active &= aux['potential']['drop']['retired'][rows,seat]!=nonce
    active &= (aux['routing']['nonce'][rows,enemy]==nonce)&jnp.all(target>=0,-1)&loose
    active &= states.alive[rows,seat]&(states.bun_carried[rows,seat]<0)&(states.trapped[rows,seat]==0)&~aux['terminal']
    move,reachable=direction(states.pos[rows,seat],aux['expert']['distance'][rows,seat],legal)
    if hazard_mode=='global':
        active &= ~jnp.any((states.fuse>0)|(states.core.blast_linger>0),axis=(1,2))
    elif hazard_mode=='local_safe':
        if danger is None:
            from jax_bomb import jax_env as base
            danger=jax.vmap(lambda s:base._danger_map(s.fuse,s.wall,s.core.bomb_blast,brick=s.brick))(states)
        danger=jnp.maximum(danger,(states.core.blast_linger>0).astype(jnp.float32))
        cell=states.pos[rows,seat].astype(jnp.int32)
        delta=jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])
        neighbor=jnp.clip(cell+delta[move],jnp.array([0,0]),jnp.array([states.wall.shape[-2]-1,states.wall.shape[-1]-1]))
        active &= (danger[rows,cell[:,0],cell[:,1]]==0)&(danger[rows,neighbor[:,0],neighbor[:,1]]==0)
        active &= states.fuse[rows,neighbor[:,0],neighbor[:,1]]<=0
    else:raise ValueError('unknown expert hazard mode')
    return jax.lax.stop_gradient(move),jax.lax.stop_gradient(active&reachable)


def supervised_loss(move,legal,target,active,coefficient=.2,temperature=1.):
    lp=jax.nn.log_softmax(jnp.where(legal,move/temperature,-1e9))
    ce=-lp[jnp.arange(move.shape[0]),target]
    valid=active&legal[jnp.arange(move.shape[0]),target]
    mean=jnp.sum(jnp.where(valid,ce,0.))/jnp.maximum(1,valid.sum())
    return coefficient*mean,jnp.array([mean,valid.mean()])
