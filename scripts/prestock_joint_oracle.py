"""Proposal-only joint movement/ability sets for empty-hand preparation.

No online query budget, replay storage, optimizer or running recipe is hooked
here. Each set retains native joint pairs rather than their separate marginals.
"""
import jax
import jax.numpy as jnp


def joint_set_loss(move,ability,legal_move,legal_ability,target,active,coefficient=1.):
    move_logp=jax.nn.log_softmax(jnp.where(legal_move,move,-1e9),-1)
    ability_logp=jax.nn.log_softmax(jnp.where(legal_ability,ability,-1e9),-1)
    accepted=target&legal_move[..., :,None]&legal_ability[...,None,:]
    pair_logp=move_logp[..., :,None]+ability_logp[...,None,:]
    loss=-jax.nn.logsumexp(jnp.where(accepted,pair_logp,-1e9),axis=(-2,-1))
    valid=active&accepted.any((-2,-1))
    mean=jnp.where(valid,loss,0.).sum()/jnp.maximum(valid.sum(),1)
    return coefficient*mean,jnp.array([mean,valid.mean()])


def preparation_targets(state,seat,prepared,home_goals,stock_goals,move_target,movement_active):
    """One state, mutually exclusive navigation or restricted opening witness."""
    from jax_bomb import je07_bun_env as env
    from scripts.frontier_native_witness import placement_witness
    mm,am=env.legal_mask(state)
    cell=jnp.clip(jnp.floor(state.pos[seat]).astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    eligible=prepared.preparing&(prepared.objective>1)&state.alive[seat]
    eligible&=(state.trapped[seat]==0)&(state.bun_carried[seat]<0)
    eligible&=~state.alive[1-seat]|(state.trapped[1-seat]>0)
    navigation=(move_target>0)&mm[seat]&jnp.array([True,True,True,True,False])
    nav=eligible&movement_active&navigation.any()&am[seat,0]
    staged=prepared.staging[cell[0],cell[1]]
    goals=jnp.where(prepared.objective==2,home_goals,stock_goals)

    def opening(_):
        return placement_witness(state,seat,goals,prepared.first_bricks).joint_set

    target=jax.lax.cond(nav,
        lambda _:jnp.zeros((5,3),bool).at[:,0].set(navigation),
        lambda _:jax.lax.cond(eligible&staged,opening,
            lambda _:jnp.zeros((5,3),bool),None),None)
    target&=mm[seat,:,None]&am[seat,None,:]
    return jax.lax.stop_gradient(target),jax.lax.stop_gradient(target.any())
