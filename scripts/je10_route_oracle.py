"""Set-valued route oracle on continuous learner states and native topology."""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import INF

DELTA=jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])


def route_allowed(carrying,enemy_alive,enemy_trapped,distance):
    controlled=~enemy_alive|(enemy_trapped>0)
    return jnp.where(carrying,controlled|(distance>4.),controlled)


def eligible(scene,alive,terminal,trapped,carrying,window,custody,age):
    return (scene!=0)&alive&~terminal&(trapped==0)&jnp.where(carrying>=0,custody,window)


def continuous_distance(pos,distance,blocked):
    # Shared topology-continuous field, successor-only recipe.
    from scripts.continuous_route_metric import continuous_distance as shared
    return shared(pos,distance,blocked)


def action_set(pos,distance,legal,danger,blocked,fuse):
    n,h,w=distance.shape;r=jnp.arange(n)[:,None]
    cell=jnp.clip(jnp.floor(pos).astype(jnp.int32),jnp.array([0,0]),jnp.array([h-1,w-1]))
    neighbor=jnp.clip(cell[:,None]+DELTA,jnp.array([0,0]),jnp.array([h-1,w-1]))
    # Endpoint cell safety prevents steering into a blocked next grid square
    # even if the first small movement remains in the current square.
    safe=(~blocked[r,neighbor[...,0],neighbor[...,1]]
        &(danger[r,neighbor[...,0],neighbor[...,1]]==0)
        &(fuse[r,neighbor[...,0],neighbor[...,1]]<=0))
    candidates=pos[:,None]+.18*DELTA.astype(jnp.float32)
    repeated=jnp.repeat(distance,5,axis=0);barriers=jnp.repeat(blocked,5,axis=0)
    after=continuous_distance(candidates.reshape(-1,2),repeated,barriers).reshape(n,5)
    before=continuous_distance(pos,distance,blocked)
    gain=before[:,None]-after
    allowed=legal&safe&(gain>1e-5)
    best=jnp.max(jnp.where(allowed,gain,-1e9),-1,keepdims=True)
    chosen=allowed&(gain>=best-1e-5)
    target=chosen/jnp.maximum(chosen.sum(-1,keepdims=True),1)
    active=chosen.any(-1)&(before<INF)&(danger[jnp.arange(n),cell[:,0],cell[:,1]]==0)
    return jax.lax.stop_gradient(target),jax.lax.stop_gradient(active)


def supervised_loss(move,legal,target,active,coefficient=0.,temperature=1.):
    accepted=legal&(target>0)
    logits=jnp.where(legal,move/temperature,-1e9)
    # -log(sum p(legal oracle actions)): all acceptable directions comprise
    # one event. No artificial uniform distribution over tied routes.
    loss=jax.nn.logsumexp(logits,-1)-jax.nn.logsumexp(jnp.where(accepted,logits,-1e9),-1)
    valid=active&accepted.any(-1)
    mean=jnp.where(valid,loss,0.).sum()/jnp.maximum(valid.sum(),1)
    return coefficient*mean,jnp.array([mean,valid.mean()])


def labels(states,aux,legal,seat,danger):
    from scripts import je09_stage_h5c_dagger as dagger
    r=jnp.arange(seat.size);enemy=1-seat;d=aux['dagger']
    empty=states.bun_carried[r,seat]<0
    enemy_distance=jnp.abs(states.pos[r,seat]-states.pos[r,enemy]).sum(-1)
    allowed=route_allowed(~empty,states.alive[r,enemy],states.trapped[r,enemy],enemy_distance)
    native_window=~states.alive[r,enemy]|(states.trapped[r,enemy]>0)
    allow=eligible(states.scene,states.alive[r,seat],aux['terminal'],states.trapped[r,seat],
        states.bun_carried[r,seat],native_window,d['custody'][r,seat],d['age'][r,seat])&allowed
    loose=d['loose'][r,seat].any((-2,-1))&native_window
    stock=states.bun_stored[r,enemy,enemy]>0
    kind=jnp.where(empty,jnp.where(loose,jnp.where(d['loose_origin'][r,seat]==seat,4,1),jnp.where(stock,2,0)),3)
    channel=jnp.where((kind==1)|(kind==4),seat,jnp.where(kind==2,2+seat,4+seat))
    distance=d['distance'][r,channel]
    danger=jnp.maximum(danger,(states.core.blast_linger>0).astype(jnp.float32))
    target,reachable=action_set(states.pos[r,seat],distance,legal,danger,d['blocked'],states.fuse)
    cell=jnp.clip(states.pos[r,seat].astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    dist=distance[r,cell[:,0],cell[:,1]]
    goal=(dist==0)&legal[:,4]&(danger[r,cell[:,0],cell[:,1]]==0)&(states.fuse[r,cell[:,0],cell[:,1]]<=0)
    target=jnp.where(goal[:,None],jax.nn.one_hot(jnp.full(seat.shape,4),5),target)
    active=allow&(kind>0)
    from jax_bomb import je07_bun_env as env
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    home=bases[seat,cell[:,0],cell[:,1]]
    return jax.lax.stop_gradient(jnp.concatenate([target,(active&(reachable|goal))[:,None],
        kind[:,None],active[:,None],(dist<INF)[:,None],home[:,None],
        d['custody'][r,seat,None],d['origin'][r,seat,None]],-1).astype(jnp.float32))
