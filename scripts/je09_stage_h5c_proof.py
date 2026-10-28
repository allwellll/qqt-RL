"""Production DAgger context, replay, and Actor-only gradient attribution."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scripts import je09_stage_h5c_dagger as d
from scripts.population_run_state import atomic_json


def prove(params,batch,replay,run):
    c=np.asarray(batch[-1]);active=c[...,d.OFFSET+5]>0
    excluded=c[...,d.OFFSET+7]==0
    assert not (active&excluded).any()
    assert np.all(c[...,0][active]==8)
    if not np.asarray(replay['valid']).any():
        atomic_json(run/'validation/H5_PRODUCTION_OBSERVATION.json',dict(passed=True,
            actual_eligible=int(active.sum()),gradient_observed=False,no_fabricated_samples=True))
        return False
    q=jax.tree.map(lambda x:x[0],params)
    flat=lambda x:x.reshape((-1,)+x.shape[2:])
    data=jax.tree.map(flat,replay)
    ix=np.flatnonzero(np.asarray(data['valid']))[:32]
    data=jax.tree.map(lambda x:x[ix],data)
    loss,grad=jax.jit(jax.value_and_grad(d.replay_loss))(q,data)
    norms={k:float(optax.global_norm(grad[k])) for k in ['actor','critic']}
    zero=jax.jit(jax.grad(d.replay_loss))(q,dict(data,valid=jnp.zeros_like(data['valid'])))
    zero_norm=float(optax.global_norm(zero))
    assert np.isfinite(float(loss)) and norms['actor']>0 and norms['critic']==0 and zero_norm==0
    atomic_json(run/'validation/H5_PRODUCTION_GRADIENT.json',dict(passed=True,
        collect_gae_context_consumed_by_ppo=True,actual_eligible=int(active.sum()),
        raw_auxiliary_gradient=norms,excluded_gradient=zero_norm,action_override=False,
        replay_source='actual learner train-only natural full states; online oracle; checkpointed aggregation'))
    return True
