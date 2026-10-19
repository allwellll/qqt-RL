"""Uninstalled candidate for native return movement retention only.

Consumes existing collector certificates; creates no custody, target or label.
It has no effect unless a successor explicitly installs the PPO call site.
"""
import jax
import jax.numpy as jnp


def movement_scale(context,global_state,legal_move,legal_ability):
    # Older collectors cannot prove this exception. Keep their exact baseline.
    if context.shape[-1]<114:
        return jnp.ones(context.shape[0],jnp.float32)
    joint=context[:,95:110].reshape((-1,5,3))>0
    accepted=(joint&legal_move[:,:,None]&legal_ability[:,None,:]).any((-2,-1))
    scene=context[:,0].astype(jnp.int32)
    native_scene=(scene!=0)&~jnp.isin(scene,jnp.array([2,3,4,6]))
    free_alive=global_state[:,9]==1
    enemy_carry=(global_state[:,13]>.5)&(global_state[:,14]>.5)
    safe_native_return=(native_scene&free_alive&enemy_carry
        &(context[:,1]==0)&(context[:,47]==0)
        &(context[:,54]==3)&(context[:,55]>0)&(context[:,58]>0)
        &(context[:,110]==0)&accepted)
    return jax.lax.stop_gradient((~safe_native_return).astype(jnp.float32))


def retained_kl(move_kl,bomb_kl,weights,scale):
    return jnp.mean(weights*(scale*move_kl+bomb_kl))


def exposure_summary(context,global_state,legal_move,legal_ability):
    """Passive host count over the rollout consumed once by PPO Actor indices.

    The count precedes gradient/Adam aggregation and is not a causal estimate
    of an objective's parameter update. It creates no labels or training gate.
    """
    import numpy as np
    c,g,m,a=map(np.asarray,(context,global_state,legal_move,legal_ability))
    shape=c.shape[:-1]
    if c.shape[-1]<114:
        exempt=np.zeros(shape,bool)
    else:
        joint=c[...,95:110].reshape(*shape,5,3)>0
        accepted=(joint&m[..., :,None]&a[...,None,:]).any((-2,-1))
        scene=c[...,0].astype(np.int32)
        exempt=((scene!=0)&~np.isin(scene,[2,3,4,6])&(g[...,9]==1)
            &(g[...,13]>.5)&(g[...,14]>.5)&(c[...,1]==0)&(c[...,47]==0)
            &(c[...,54]==3)&(c[...,55]>0)&(c[...,58]>0)&(c[...,110]==0)&accepted)
    return dict(actor_rollout_rows=int(exempt.size),original_denominator_rows=int(exempt.size),
        certified_return_movement_exempt_rows=int(exempt.sum()),
        exempt_rows_by_device=exempt.reshape(shape[0],-1).sum(-1).tolist(),
        fraction=float(exempt.mean()),controls_training=False,
        scope='Actual rollout rows visited once by the PPO Actor permutation; pre-optimizer count, not loss/gradient magnitude or independent episodes')
