"""Uninstalled candidate: describe certified return sets that require a detour."""
import jax.numpy as jnp


def detour_flags(data):
    """Classify native labels; never construct a new direction or safety label.

    A detour set contains only moves increasing Manhattan distance to the
    observed home cage rectangle. This is a sampling stratum, not a path or
    reachability oracle. Neutral/tied-toward sets and delivery are excluded.
    """
    obs=data['obs']
    height,width=obs.shape[-2:]
    rr,cc=jnp.meshgrid(jnp.arange(height),jnp.arange(width),indexing='ij')
    grid=jnp.stack([rr,cc],-1)
    player=obs[:,0]
    pos=(player[...,None]*(grid+.5)).sum((1,2))/jnp.maximum(player.sum((1,2))[:,None],1e-9)
    cage=obs[:,14]>0
    lower=jnp.where(cage[...,None],grid,jnp.array([height,width])).min((1,2))
    upper=jnp.where(cage[...,None],grid+1,jnp.array([0,0])).max((1,2))

    def distance(points):
        return (jnp.maximum(lower[:,None]-points,0)
                +jnp.maximum(points-upper[:,None],0)).sum(-1)

    delta=jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])
    before=distance(pos[:,None])
    after=distance(pos[:,None]+.18*delta)
    accepted=(data['joint_target']&data['legal'][:,:,None]
              &data['ability_legal'][:,None,:]).any(-1)
    away=(after-before)>1e-5
    eligible=data['valid']&(data['kind']==3)&~data['preparation']&(data['gv'][:,13]>0)
    return eligible&cage.any((1,2))&accepted.any(-1)&jnp.all(~accepted|away,-1)
