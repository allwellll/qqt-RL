"""Loss selectors and matched displacements for non-adopted update diagnosis."""
import numpy as np
import jax
import jax.numpy as j
import optax
import pickle
import json
from pathlib import Path


def branch_loss(base, metrics, branch):
    # Frozen R4 objective metrics: PPO,value,entropy,supervised,coverage,item,KL,gate.
    if branch == 'joint_anchor':
        return base
    if branch == 'ppo_anchor':
        return base - .2 * metrics[3]
    if branch == 'dagger_anchor':
        return .2 * metrics[3] + .05 * metrics[6]
    raise ValueError('unknown update mechanism branch: ' + branch)


def random_displacement(before, joint, *, seed):
    """Fixed random direction with each module's actual joint L2 displacement.

    This is a parameter counterfactual, not an optimizer/Adam-matched trained arm.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for module in ['actor', 'critic']:
        old, structure = jax.tree.flatten(before[module])
        new = jax.tree.leaves(joint[module])
        scale = np.sqrt(sum(np.sum((np.asarray(a,float)-np.asarray(b,float))**2)
                            for a,b in zip(old,new)))
        draws = [rng.standard_normal(a.shape) for a in old]
        length = np.sqrt(sum(np.sum(a*a) for a in draws))
        result[module] = jax.tree.unflatten(structure,[
            (np.asarray(a)+d*(scale/max(length,1e-30))).astype(np.asarray(a).dtype)
            for a,d in zip(old,draws)])
    return result


def make_update(objective, optimizer, branch, preserve_critic):
    """Frozen R4 minibatch/shuffle/normalization, with one loss intervention.

    All input batches are the original learner on-policy batch. Historical
    replay enters only the supervised component of the supplied objective.
    Nothing returned by this diagnostic is adopted by a training campaign.
    """
    if branch not in ['joint_anchor', 'ppo_anchor', 'dagger_anchor']:
        raise ValueError('unknown active branch: ' + branch)
    def update(params, adam, batch, replay, key):
        batch=tuple(x.reshape((-1,)+x.shape[2:]) for x in batch)
        advantage=batch[4]
        mean=jax.lax.pmean(advantage.mean(),'device')
        square=jax.lax.pmean((advantage**2).mean(),'device')
        advantage=(advantage-mean)/j.sqrt(j.maximum(square-mean**2,0)+1e-8)
        batch=batch[:4]+(advantage,)+batch[5:]
        order=jax.random.permutation(key,len(advantage)).reshape(-1,32)
        def step(carry,indices):
            p,a=carry
            mb=tuple(x[indices] for x in batch)
            rb=tuple(x[indices] for x in replay)
            def loss(q):
                base,metrics=objective(q,mb,rb,'anchor')
                return branch_loss(base,metrics,branch),metrics
            (value,metrics),grad=jax.value_and_grad(loss,has_aux=True)(p)
            grad=jax.lax.pmean(grad,'device')
            delta,na=optimizer.update(grad,a,p)
            pn=optax.apply_updates(p,delta)
            if branch=='dagger_anchor':
                pn=dict(pn,critic=p['critic'])
                na=preserve_critic(na,a)
            diagnostics=j.concatenate([value[None],metrics,
                optax.global_norm(grad)[None],
                optax.global_norm(grad['actor']['heads']['wb'])[None]])
            return (pn,na),jax.lax.pmean(diagnostics,'device')
        return jax.lax.scan(step,(params,adam),order)
    return jax.pmap(update,axis_name='device')


def load_committed_update(checkpoint, *, expected=None):
    """Resume verified diagnostics without re-executing an adopted update."""
    from scripts.population_run_state import sha
    from scripts.terminal_checkpoint_restore import digest
    checkpoint=Path(checkpoint)
    receipt=json.loads((checkpoint/'COMMITTED.json').read_text())
    assert sha(checkpoint/'state.pkl')==receipt['state_sha256'],'saved branch SHA mismatch'
    value=pickle.load((checkpoint/'state.pkl').open('rb'))
    for key,label in [('params','params_digest'),('adam','Adam_digest')]:
        assert digest(value[key])==receipt[label],'saved branch digest mismatch: '+key
        if expected is not None:
            assert digest(value[key])==digest(expected[key]),'factual branch differs from source: '+key
    return value,receipt
