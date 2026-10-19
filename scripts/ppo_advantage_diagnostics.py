"""Read-only PPO advantage telemetry, not an optimizer or collector hook."""
import jax
import jax.numpy as jnp


def replica_moments(advantages):
    """The same mean/std reduction used inside one PPO pmap replica."""
    return jnp.stack([advantages.mean(),advantages.std()])


def device_moments(advantages):
    """Reduce [device, tick, env] on device; transfer only two scalars each.

    PPO computes ad.mean() and ad.std() separately inside each pmap replica.
    Keep that scope and its population standard deviation exactly.
    """
    advantages=jnp.asarray(advantages)
    assert advantages.ndim==3,'expected [device, tick, environment] advantages'
    return jax.vmap(replica_moments)(advantages)


def normalized_diagnostic(raw_advantage,moments):
    """Use saved device moments and one already selected raw GAE scalar."""
    import numpy as np
    moments=np.asarray(moments)
    assert moments.shape==(2,)
    scalar=moments.dtype.type
    mean,std=moments
    normalized=(scalar(raw_advantage)-mean)/(std+scalar(1e-8))
    return dict(collector_advantage_device_mean=float(mean),
        collector_advantage_device_std=float(std),
        collector_ppo_normalized_advantage=float(normalized),
        collector_normalization_scope='PPO per-device rollout tick/environment population; before ratio/clipping, not action Q or full loss gradient')
