"""Proposal-only native-step progress certificate; not installed in r01/r02."""
import jax
import jax.numpy as jnp


def certify_set(proposed,safe,before,after,at_goal,tolerance=1e-5):
    gain=before[:,None]-after
    finite=(before[:,None]<32767)&(after<32767)&jnp.isfinite(gain)
    allowed=(proposed>0)&safe&finite&(gain>tolerance)
    best=jnp.max(jnp.where(allowed,gain,-jnp.inf),-1,keepdims=True)
    selected=allowed&(gain>=best-tolerance)
    goal_wait=at_goal&(proposed[:,4]>0)&safe[:,4]&(before<32767)
    selected=jnp.where(at_goal[:,None],jax.nn.one_hot(jnp.full(before.shape,4),5,dtype=bool)&goal_wait[:,None],selected)
    target=selected.astype(proposed.dtype)/jnp.maximum(selected.sum(-1,keepdims=True),1)
    return jax.lax.stop_gradient(target),jax.lax.stop_gradient(selected.any(-1))


def certify_native_positions(before_pos,after_pos,distance,blocked,legal,safe,at_goal,
                             topology_stable=None):
    """Set of best safe native endpoints, considering every legal movement.

    Inputs are the existing strict verifier's real one-step counterfactuals.
    Opponent waits; ability0 and the existing safety certificate are retained.
    Stale distance grids after a topology change cannot certify an endpoint.
    """
    from scripts.je10_route_oracle import continuous_distance
    n=before_pos.shape[0]
    before=continuous_distance(before_pos,distance,blocked)
    after=continuous_distance(after_pos.reshape(-1,2),jnp.repeat(distance,5,axis=0),
                              jnp.repeat(blocked,5,axis=0)).reshape(n,5)
    if topology_stable is not None:safe=safe&topology_stable
    return certify_set(legal.astype(jnp.float32),safe,before,after,at_goal)
