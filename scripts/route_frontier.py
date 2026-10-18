"""Proposal-only reachable brick frontiers for disconnected native routes.

This planner supplies a movement objective. It grants neither a movement
through brick nor permission/reward to place a bomb; native joint safety and
ability supervision remain separate work before any training integration.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import INF,shortest


class FrontierPlan(NamedTuple):
    needed:object
    staging:object
    distance:object
    first_bricks:object


def _neighbors(values,padding):
    grid=jnp.pad(values,((0,0),(1,1),(1,1)),constant_values=padding)
    return jnp.stack([grid[:,:-2,1:-1],grid[:,2:,1:-1],
        grid[:,1:-1,:-2],grid[:,1:-1,2:]],-1)


def plan(pos,hard,brick,goals):
    """Use existing open routes, otherwise approach optimal first-brick stages.

    All equal minimum-cost staging cells remain valid. Route cost matches
    production's brick-entry weight four; only currently open reachable cells
    can be staging goals. Replanning after a real opening returns to the home
    goal as soon as it is connected, including a longer existing detour.
    """
    n,h,w=hard.shape;rows=jnp.arange(n)
    cell=jnp.floor(pos).astype(jnp.int32)
    inside=jnp.all((cell>=0)&(cell<jnp.array([h,w])),-1)
    cell=jnp.clip(cell,jnp.array([0,0]),jnp.array([h-1,w-1]))
    blocked=hard|brick;clear=jnp.zeros_like(brick)
    valid_start=inside&~blocked[rows,cell[:,0],cell[:,1]]
    start=jnp.zeros_like(brick).at[rows,cell[:,0],cell[:,1]].set(valid_start)
    home=shortest(blocked,clear,goals)
    reach=shortest(blocked,clear,start)
    weighted=shortest(hard,brick,goals)
    adjacent=_neighbors(weighted,INF)
    breakable=_neighbors(brick&~hard,False)
    candidates=(reach<INF)[...,None]&breakable&(adjacent<INF)
    # Reach contains only open cells. The first brick pays its actual entry
    # weight once; weighted[brick] covers the remaining path after that entry.
    score=jnp.where(candidates,reach[...,None]+4+adjacent,1000000000)
    best=jnp.min(score,axis=(1,2,3),keepdims=True)
    disconnected=home[rows,cell[:,0],cell[:,1]]>=INF
    needed=valid_start&disconnected&(best[:,0,0,0]<1000000000)
    chosen=candidates&(score==best)&needed[:,None,None,None]
    staging=chosen.any(-1)
    y,x=jnp.indices((h,w))
    dy=jnp.array([-1,1,0,0]);dx=jnp.array([0,0,-1,1])
    target_y=jnp.clip(y[...,None]+dy,0,h-1)
    target_x=jnp.clip(x[...,None]+dx,0,w-1)
    first_bricks=jnp.zeros_like(brick).at[
        jnp.arange(n)[:,None,None,None],target_y[None],target_x[None]].max(chosen)
    frontier=shortest(blocked,clear,staging)
    distance=jnp.where(needed[:,None,None],frontier,home)
    return jax.lax.stop_gradient(FrontierPlan(needed,staging,distance,first_bricks))
