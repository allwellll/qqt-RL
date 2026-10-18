"""Candidate native certificates using the shared pre-stock objective.

The production verifier can reuse its five native one-step forks and complete
temporal joint safety result. Opening witnesses run serially over selected
rows, with scalar conditions, so navigation rows do not execute long searches.
This module is not installed in a running recipe.
"""
import jax
import jax.numpy as jnp


def navigation(states,after,seat,selected,legal,safety):
    from jax_bomb import je07_bun_env as env
    from scripts.native_route_certificate import certify_native_positions
    n=seat.size;rows=jnp.arange(n)
    physical=states.wall|states.brick|states.core.pushable
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    avoid=bases[1-seat]&(selected['objective']==2)[:,None,None]
    blocked=physical|avoid
    stable=jnp.all((after.wall|after.brick|after.core.pushable).reshape(n,5,13,15)==physical[:,None],(-2,-1))
    next_rows=jnp.arange(n*5);next_seat=jnp.repeat(seat,5)
    free=after.alive[next_rows,next_seat]&(after.trapped[next_rows,next_seat]==0)
    free&=after.bun_carried[next_rows,next_seat]<0
    controlled=~after.alive[next_rows,1-next_seat]|(after.trapped[next_rows,1-next_seat]>0)
    active=selected['active']&~selected['ready']&(selected['objective']>1)
    safe=safety[:,:,0]&(free&controlled).reshape(n,5)&active[:,None]
    safe&=jnp.arange(5)[None]!=4
    # A staging cell can still need native subcell centering. Only strict
    # positive progress is navigation; wait and placement are never a fallback.
    return certify_native_positions(states.pos[rows,seat],
        after.pos[next_rows,next_seat].reshape(n,5,2),selected['distance'],
        blocked,legal,safe,jnp.zeros(n,bool),stable)


def targets(states,seat,selected,move,move_valid,safety,home=None,stock=None):
    from jax_bomb import je07_bun_env as env
    from scripts.prestock_route_preparation import StockPreparation
    from scripts.prestock_joint_oracle import preparation_targets
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    if home is None:home=bases[seat]
    if stock is None:stock=bases[1-seat]

    def one(values):
        state,who,chosen,movement,valid,safe,home_goals,stock_goals=values
        prepared=StockPreparation(chosen['ready'],chosen['active']&~chosen['ready'],
            chosen['objective'],chosen['staging'],chosen['first_bricks'],chosen['distance'])
        joint,_=preparation_targets(state,who,prepared,home_goals,stock_goals,movement,valid)
        joint&=safe
        return joint,joint.any()

    # vmap(cond) would evaluate the expensive opening branch for every row.
    # Scalar lax.map preserves conditional execution for the bounded slots.
    return jax.lax.map(one,(states,seat,selected,move,move_valid,safety,home,stock))
