"""Prepared shared native objective state; no production recipe hook.

Both seats use the same selected corridor for potential and future joint labels.
Loose priority comes from factual routing/drop IDs, including the transition
before the cached DAgger loose flags have been refreshed.
"""
import jax
import jax.numpy as jnp
from scripts.prestock_route_preparation import selection_state,select,plan,potential_values,StockPreparation


def state(n,h=13,w=15):
    result=jax.tree.map(lambda x:x.reshape((n,2)+x.shape[1:]),selection_state(n*2,h,w))
    result.update(active=jnp.zeros((n,2),bool),ready=jnp.zeros((n,2),bool))
    return result


def flatten(values):
    return jax.tree.map(lambda x:x.reshape((-1,)+x.shape[2:]),values)


def attributed_loose(states,aux):
    n,h,w=states.wall.shape;rows=jnp.arange(n)[:,None];origin=jnp.arange(2)[None]
    route=aux['routing'];drop=aux['drop'];target=route['target']
    inside=jnp.all((target>=0)&(target<jnp.array([h,w])),-1)
    cell=jnp.clip(target,jnp.array([0,0]),jnp.array([h-1,w-1]))
    present=states.bun_loose[rows,cell[...,0],cell[...,1],origin]>0
    valid=inside&present&(drop['nonce']>0)&(route['nonce']==drop['nonce'])
    credited=drop['credit'][:,None,:]==jnp.arange(2)[None,:,None]
    return (valid[:,None,:]&credited).any(-1)


def update(states,aux,previous):
    from jax_bomb import je07_bun_env as env
    n,h,w=states.wall.shape;rows=jnp.arange(n)[:,None];team=jnp.arange(2)[None];enemy=1-team
    window=~states.alive[:,::-1]|(states.trapped[:,::-1]>0)
    active=(states.scene!=0)[:,None]&~aux['terminal'][:,None]&states.alive
    active&=(states.trapped==0)&(states.bun_carried<0)&window
    active&=(states.bun_stored[rows,enemy,enemy]>0)&~attributed_loose(states,aux)
    hard=jnp.broadcast_to((states.wall|states.core.pushable)[:,None],(n,2,h,w)).reshape(-1,h,w)
    brick=jnp.broadcast_to(states.brick[:,None],(n,2,h,w)).reshape(-1,h,w)
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    home=jnp.broadcast_to(bases[None],(n,2,h,w));stock=home[:,::-1]
    pos=states.pos.reshape(-1,2)
    proposed=plan(pos,hard,brick,home.reshape(-1,h,w),stock.reshape(-1,h,w),active.reshape(-1))
    reset=aux['terminal'][:,None]|~states.alive|(states.trapped>0)
    selected=select(flatten(previous),proposed,hard,brick,pos,reset.reshape(-1))
    selected=jax.tree.map(lambda x:x.reshape((n,2)+x.shape[1:]),selected)
    selected.update(active=active&~reset,ready=proposed.ready.reshape(n,2)&~reset)
    return jax.tree.map(jax.lax.stop_gradient,selected)


def prepared(selected):
    return StockPreparation(selected['ready'],selected['active']&~selected['ready'],
        selected['objective'],selected['staging'],selected['first_bricks'],selected['distance'])


def potential(states,selected,parent):
    n,h,w=states.wall.shape
    hard=jnp.broadcast_to((states.wall|states.core.pushable)[:,None],(n,2,h,w)).reshape(-1,h,w)
    brick=jnp.broadcast_to(states.brick[:,None],(n,2,h,w)).reshape(-1,h,w)
    result=potential_values(states.pos.reshape(-1,2),hard,brick,flatten(selected),flatten(parent))
    result=jax.tree.map(lambda x:x.reshape(n,2),result)
    # A physically impossible hard disconnection has no stock-direction
    # fallback. Preserve the existing inactive-potential ledger behavior.
    unavailable=(parent[0]==1)&selected['active']&~selected['ready']&(selected['objective']==0)
    inactive=(jnp.zeros_like(result[0]),jnp.full_like(result[1],-1),jnp.zeros_like(result[2]),
        jnp.full_like(result[3],32767),jnp.zeros_like(result[4]))
    return jax.tree.map(lambda x,z:jnp.where(unavailable,z,x),result,inactive)
