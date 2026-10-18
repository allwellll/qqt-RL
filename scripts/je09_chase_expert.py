"""Native attributed loose chase action sets; no actor inputs or action takeover."""
import jax
import jax.numpy as jnp
from scripts.je09_return_expert import action_set,supervised_loss
from scripts.je08_bun_potential import INF

def state(n):
    return dict(nonce=jnp.zeros((n,2),jnp.int32),age=jnp.zeros((n,2),jnp.int32),
                initial=jnp.full((n,2),INF,jnp.int32),admitted=jnp.zeros((n,2),bool))

def admit(distance,bounds):
    return ((distance>=2)&(distance<=8))|((distance>=bounds[0])&(distance<=bounds[1])&(distance<INF))

def update(states,aux,old,bounds):
    n=states.pos.shape[0];rows=jnp.arange(n)[:,None];teams=jnp.arange(2)[None]
    nonce=aux['drop']['nonce'][:,::-1]
    fresh=(nonce>0)&(nonce!=old['nonce'])
    cell=states.pos.astype(jnp.int32)
    d=aux['expert']['distance'][rows,teams,cell[...,0],cell[...,1]]
    return dict(nonce=nonce,age=jnp.where(fresh,0,old['age']+1),
        initial=jnp.where(fresh,d,old['initial']),admitted=jnp.where(fresh,admit(d,bounds),old['admitted']))

def gate(states,aux,seat):
    rows=jnp.arange(seat.shape[0]);enemy=1-seat;b=aux['chase'];drop=aux['drop']
    nonce=drop['nonce'][rows,enemy];target=aux['routing']['target'][rows,enemy]
    cell=jnp.clip(target,jnp.array([0,0]),jnp.array([states.bun_loose.shape[1]-1,states.bun_loose.shape[2]-1]))
    loose=states.bun_loose[rows,cell[:,0],cell[:,1],enemy]>0
    return ((nonce>0)&(drop['credit'][rows,enemy]==seat)&~drop['cycle_paid'][rows,seat,enemy]
        &(aux['routing']['nonce'][rows,enemy]==nonce)&jnp.all(target>=0,-1)&loose
        &aux['potential']['drop']['active'][rows,seat]&(aux['potential']['drop']['retired'][rows,seat]!=nonce)
        &(b['nonce'][rows,seat]==nonce)&b['admitted'][rows,seat]&(b['age'][rows,seat]<100)
        &(states.bun_respawn[rows,enemy]>0)&(states.bun_carried[rows,seat]<0)
        &states.alive[rows,seat]&(states.trapped[rows,seat]==0)&~aux['terminal']&(states.scene!=0))

def labels(states,aux,legal,seat,danger):
    rows=jnp.arange(seat.shape[0]);distance=aux['expert']['distance'][rows,seat]
    danger=jnp.maximum(danger,(states.core.blast_linger>0).astype(jnp.float32))
    target,valid=action_set(states.pos[rows,seat],distance,legal,danger,states.wall|states.brick,states.fuse)
    return target,jax.lax.stop_gradient(gate(states,aux,seat)&valid)
