"""Prepared PBRS alignment with the existing own-recovery target chain.

The successor installer supplies the hook only after fixed-run termination.
Reuse reverse-origin drop and home caches without a new grid search.
"""
import jax
import jax.numpy as jnp


def align_recovery(states,route,drop,values):
    from scripts.continuous_route_metric import sampled_distance
    phase,identity,base,distance,provenance=values
    n,h,w=states.wall.shape
    rows=jnp.arange(n)[:,None];team=jnp.arange(2)[None]
    target=route['target']
    inside=jnp.all((target>=0)&(target<jnp.array([h,w])),-1)
    cells=jnp.clip(target,jnp.array([0,0]),jnp.array([h-1,w-1]))
    present=states.bun_loose[rows,cells[...,0],cells[...,1],team]>0
    native_window=~states.alive[:,::-1]|(states.trapped[:,::-1]>0)
    credited=(drop['credit']==team)&(drop['nonce']>0)&(route['nonce']==drop['nonce'])
    choose=(phase<=1)&states.alive&(states.trapped==0)&(states.bun_carried<0)
    choose&=native_window&inside&present&credited
    # Player 0's own origin map is routing channel 1, and vice versa.
    channels=jnp.broadcast_to(1-team,(n,2))
    pos=states.pos.reshape(n,2,2)
    continuous=sampled_distance(pos,route['distance'],states.wall,channels)
    current=jnp.clip(jnp.floor(pos).astype(jnp.int32),jnp.array([0,0]),jnp.array([h-1,w-1]))
    grid=route['distance'][rows,channels,current[...,0],current[...,1]]
    recovery_base=jnp.where(grid<32767,-jnp.minimum(continuous,64)/64.,-1.)
    # Negative namespace cannot alias stock/enemy-drop/carry identities.
    recovery_identity=-1-(2*drop['nonce']+team)
    result=(jnp.where(choose,2,phase),jnp.where(choose,recovery_identity,identity),
        jnp.where(choose,recovery_base,base),jnp.where(choose,grid,distance),
        jnp.where(choose,drop['nonce'],provenance))
    # Native deposit returns either origin to the carrier's home. Parent PBRS
    # covers enemy-origin cargo only; own recovery must continue after pickup.
    # This adapter is inside the parent phase function, so the existing outer
    # threat gate still governs both sorts of cargo identically.
    own_carry=(phase<=1)&states.alive&(states.trapped==0)&(states.bun_carried==team)
    home_channels=jnp.broadcast_to(team+2,(n,2))
    home=sampled_distance(pos,route['distance'],states.wall,home_channels)
    home_grid=route['distance'][rows,home_channels,current[...,0],current[...,1]]
    home_base=jnp.where(home_grid<32767,-jnp.minimum(home,64)/64.,-1.)
    home_phase=jnp.where(home_grid==0,4,3).astype(jnp.int8)
    home_identity=home_phase.astype(jnp.int32)*100000+team+1
    result=tuple(jnp.where(own_carry,new,old) for new,old in zip(
        (home_phase,home_identity,home_base,home_grid,jnp.full_like(provenance,3)),result))
    return jax.tree.map(jax.lax.stop_gradient,result)
