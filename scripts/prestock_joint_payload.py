"""Prepared replay schema adapter; no production collector or optimizer hook.

All legal ability choices remain acceptable for a legacy movement label. Only
stock preparation owns a certified joint replacement, including an empty set
when no safe native opening/navigation has been certified yet.
"""
import jax
import jax.numpy as jnp


def merge(move_target,active,kind,legal_move,legal_ability,preparing,preparation_joint):
    """active is selected candidate eligibility, not old stock reachability.

    The caller supplies already certified preparation_joint sets. An empty
    parent movement target must not suppress a reachable preparation target.
    """
    legacy=(move_target>0)[..., :,None]&legal_ability[...,None,:]
    owns=preparing&(kind==2)&active
    target=jnp.where(owns[...,None,None],preparation_joint,legacy)
    target&=legal_move[..., :,None]&legal_ability[...,None,:]&active[...,None,None]
    return jax.lax.stop_gradient(target),jax.lax.stop_gradient(target.any((-2,-1)))
