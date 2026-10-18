"""Proposal-only corridor objectives before acquiring enemy inventory.

Native carriers cannot place bombs. The caller must supply active only for a
free, empty, alive learner in a legal native window with real enemy inventory.
No native action, reward, learner observation or training state is changed.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from scripts.je08_bun_potential import shortest,INF
from scripts.route_frontier import plan as frontier_plan


class StockPreparation(NamedTuple):
    ready:object
    preparing:object
    objective:object
    staging:object
    first_bricks:object
    distance:object


def plan(pos,hard,brick,home_goals,stock_goals,active):
    n,h,w=hard.shape;rows=jnp.arange(n)
    cell=jnp.floor(pos).astype(jnp.int32)
    inside=jnp.all((cell>=0)&(cell<jnp.array([h,w])),-1)
    cell=jnp.clip(cell,jnp.array([0,0]),jnp.array([h-1,w-1]))
    blocked=hard|brick;clear=jnp.zeros_like(brick)
    home=shortest(blocked,clear,home_goals)
    stock=shortest(blocked,clear,stock_goals)
    home_reachable=home[rows,cell[:,0],cell[:,1]]<INF
    stock_reachable=stock[rows,cell[:,0],cell[:,1]]<INF
    legal_start=inside&~blocked[rows,cell[:,0],cell[:,1]]
    ready=legal_start&home_reachable&stock_reachable
    preparing=active&~ready
    # Near disconnected enemy stock, walking through its cage could acquire a
    # bun before opening is possible. Exclude that cage during home preparation.
    toward_home=~home_reachable
    preparation_hard=hard|(stock_goals&toward_home[:,None,None])
    destination=jnp.where(toward_home[:,None,None],home_goals,stock_goals)
    frontier=frontier_plan(pos,preparation_hard,brick,destination)
    valid=preparing&frontier.needed
    objective=jnp.where(active&ready,1,jnp.where(valid,jnp.where(toward_home,2,3),0))
    staging=frontier.staging&valid[:,None,None]
    first_bricks=frontier.first_bricks&valid[:,None,None]
    distance=jnp.where((active&ready)[:,None,None],stock,
        jnp.where(valid[:,None,None],frontier.distance,INF))
    return jax.lax.stop_gradient(StockPreparation(ready,preparing,objective,staging,first_bricks,distance))


def native_inputs(states,aux,seat):
    """Use the existing teacher's target priority and real native inventory."""
    from jax_bomb import je07_bun_env as env
    rows=jnp.arange(seat.size);enemy=1-seat
    window=~states.alive[rows,enemy]|(states.trapped[rows,enemy]>0)
    active=(states.scene!=0)&~aux['terminal']&states.alive[rows,seat]
    active&=(states.trapped[rows,seat]==0)&(states.bun_carried[rows,seat]<0)&window
    active&=states.bun_stored[rows,enemy,enemy]>0
    active&=~aux['dagger']['loose'][rows,seat].any((-2,-1))
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    # Planning cannot assume a crate has already been pushed. Recompute after
    # the actual native transition changes this factual blocked topology.
    return states.pos[rows,seat],states.wall|states.core.pushable,states.brick,bases[seat],bases[enemy],active


def native_plan(states,aux,seat):
    return plan(*native_inputs(states,aux,seat))


def movement_labels(pos,hard,brick,home_goals,stock_goals,active,legal,danger,fuse):
    """Proposal endpoint labels; subsequent native joint verification is required.

    Subcell centering at a frontier is allowed even next to a brick. Check the
    proposed endpoint rather than pretending the adjacent grid cell was entered.
    Never label wait at staging: goal-directed ability supervision owns that step.
    """
    from scripts.continuous_route_metric import continuous_distance
    result=plan(pos,hard,brick,home_goals,stock_goals,active)
    n,h,w=hard.shape;rows=jnp.arange(n)[:,None]
    blocked=hard|brick|(stock_goals&(result.objective==2)[:,None,None])
    delta=jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]],jnp.float32)
    endpoints=pos[:,None]+.18*delta
    cells=jnp.floor(endpoints).astype(jnp.int32)
    inside=jnp.all((cells>=0)&(cells<jnp.array([h,w])),-1)
    cells=jnp.clip(cells,jnp.array([0,0]),jnp.array([h-1,w-1]))
    safe=inside&~blocked[rows,cells[...,0],cells[...,1]]
    safe&=(danger[rows,cells[...,0],cells[...,1]]==0)&(fuse[rows,cells[...,0],cells[...,1]]<=0)
    before=continuous_distance(pos,result.distance,blocked)
    after=continuous_distance(endpoints.reshape(-1,2),jnp.repeat(result.distance,5,0),
        jnp.repeat(blocked,5,0)).reshape(n,5)
    gain=before[:,None]-after
    start=jnp.clip(jnp.floor(pos).astype(jnp.int32),jnp.array([0,0]),jnp.array([h-1,w-1]))
    clear=(danger[jnp.arange(n),start[:,0],start[:,1]]==0)&(fuse[jnp.arange(n),start[:,0],start[:,1]]<=0)
    allowed=legal&safe&(gain>1e-5)&active[:,None]&(result.objective>0)[:,None]&clear[:,None]
    best=jnp.max(jnp.where(allowed,gain,-1e9),-1,keepdims=True)
    chosen=allowed&(gain>=best-1e-5)
    target=chosen/jnp.maximum(chosen.sum(-1,keepdims=True),1)
    return jax.lax.stop_gradient(target),jax.lax.stop_gradient(chosen.any(-1)),result


def selection_state(n,h=13,w=15):
    """Prepared auxiliary schema only; no production hook or Actor input."""
    grid=(n,h,w)
    return dict(objective=jnp.zeros(n,jnp.int8),revision=jnp.zeros(n,jnp.int32),
        staging=jnp.zeros(grid,bool),first_bricks=jnp.zeros(grid,bool),
        distance=jnp.full(grid,INF,jnp.int32),hard=jnp.zeros(grid,bool),brick=jnp.zeros(grid,bool))


def select(previous,proposed,hard,brick,pos,terminal):
    """Retain a reachable stage until topology, phase or reachability changes.

    Revision advances on selection edges, not on sustained preparation. It is
    scoped to one learner episode; terminal/reset restores the empty schema.
    """
    n,h,w=hard.shape;rows=jnp.arange(n)
    current=jnp.floor(pos).astype(jnp.int32)
    inside=jnp.all((current>=0)&(current<jnp.array([h,w])),-1)
    current=jnp.clip(current,jnp.array([0,0]),jnp.array([h-1,w-1]))
    objective=jnp.where(proposed.objective>1,proposed.objective,0).astype(jnp.int8)
    topology_same=~jnp.any((hard!=previous['hard'])|(brick!=previous['brick']),(-2,-1))
    reachable=previous['distance'][rows,current[:,0],current[:,1]]<INF
    reachable&=inside&~(hard|brick)[rows,current[:,0],current[:,1]]
    keep=(objective>0)&(objective==previous['objective'])&topology_same&reachable
    keep&=previous['staging'].any((-2,-1))
    staging=jnp.where(keep[:,None,None],previous['staging'],proposed.staging)
    first_bricks=jnp.where(keep[:,None,None],previous['first_bricks'],proposed.first_bricks)
    distance=jnp.where(keep[:,None,None],previous['distance'],proposed.distance)
    staging&=(objective>0)[:,None,None];first_bricks&=(objective>0)[:,None,None]
    distance=jnp.where((objective>0)[:,None,None],distance,INF)
    changed=(objective!=previous['objective'])|jnp.any(staging!=previous['staging'],(-2,-1))
    changed|=jnp.any(first_bricks!=previous['first_bricks'],(-2,-1))
    result=dict(objective=objective,revision=previous['revision']+changed.astype(jnp.int32),
        staging=staging,first_bricks=first_bricks,distance=distance,hard=hard,brick=brick)
    empty=selection_state(n,h,w)
    result=jax.tree.map(lambda x,z:jnp.where(terminal.reshape((n,)+(1,)*(x.ndim-1)),z,x),result,empty)
    return jax.tree.map(jax.lax.stop_gradient,result)


def potential_values(pos,hard,brick,selected,parent):
    """Prepared stage potential; only replace an existing valid stock target.

    The shared continuous field follows the exact selected staging map. Its
    edge revision supplies the identity consumed by the existing PBRS ledger.
    Production integration must use this selection for its movement/joint sets.
    """
    from scripts.continuous_route_metric import continuous_distance
    phase,identity,base,distance,provenance=parent
    n,h,w=hard.shape;rows=jnp.arange(n)
    cell=jnp.clip(jnp.floor(pos).astype(jnp.int32),jnp.array([0,0]),jnp.array([h-1,w-1]))
    grid=selected['distance'][rows,cell[:,0],cell[:,1]]
    blocked=hard|brick|(selected['distance']>=INF)
    continuous=continuous_distance(pos,selected['distance'],blocked)
    value=jnp.where(grid<INF,-jnp.minimum(continuous,64)/64.,-1.)
    choose=(phase==1)&(selected['objective']>1)
    new_phase=jnp.where(selected['objective']==2,5,6).astype(jnp.int8)
    result=tuple(jnp.where(choose,new,old) for new,old in zip(
        (new_phase,500000+selected['revision'],value,grid,selected['revision']),parent))
    return jax.tree.map(jax.lax.stop_gradient,result)
