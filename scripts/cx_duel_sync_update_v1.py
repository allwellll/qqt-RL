"""Real cross-device PPO: global mode moments and averaged gradients before Adam."""
import jax
import jax.numpy as j
from scripts.cx_duel_updater_v1 import objective


def global_normalize(adv,modes,live):
    normalized=j.zeros_like(adv);counts=[]
    for mode in (0,1):
        use=live&(modes==mode)
        count=jax.lax.psum(use.sum(),'replicas');den=j.maximum(count,1)
        mean=jax.lax.psum(j.where(use,adv,0.).sum(),'replicas')/den
        variance=jax.lax.psum(j.where(use,(adv-mean)**2,0.).sum(),'replicas')/den
        normalized=j.where(use,(adv-mean)/j.sqrt(variance+1e-8),normalized)
        counts.append(count)
    return normalized,j.stack(counts)


def build(minibatch,optimizer,coef=.5):
    import optax
    def update(params,adam,batch,key):
        flat=tuple(v.reshape((-1,)+v.shape[2:]) for v in batch)
        order=jax.random.permutation(key,len(flat[4])).reshape(-1,minibatch)
        def body(carry,indices):
            p,a=carry;mb=tuple(v[indices] for v in flat)
            adv,counts=global_normalize(mb[4],mb[8],mb[9])
            (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(p,mb[:4]+(adv,)+mb[5:],coef)
            # Equal local minibatch sizes: pmean is the exact global-batch mean.
            gradient=jax.lax.pmean(gradient,'replicas')
            changes={};next_adam={}
            for branch in ('actor','critic'):
                changes[branch],next_adam[branch]=optimizer.update(gradient[branch],a[branch],p[branch])
            signed=[]
            for mode in (0,1):
                for ability in (0,1):
                    use=mb[9]&(mb[8]==mode)&(mb[2][:,1]==ability)
                    count=jax.lax.psum(use.sum(),'replicas');den=j.maximum(count,1)
                    raw=jax.lax.psum(j.where(use,mb[4],0.).sum(),'replicas')/den
                    normalized=jax.lax.psum(j.where(use,adv,0.).sum(),'replicas')/den
                    signed.extend([count,raw,normalized])
            metrics=jax.lax.pmean(j.concatenate([loss[None],metrics]),'replicas')
            values=j.concatenate([metrics,optax.global_norm(gradient)[None],
                optax.global_norm(gradient['actor']['heads']['wb'])[None],counts,j.stack(signed)])
            return (optax.apply_updates(p,changes),next_adam),values
        return jax.lax.scan(body,(params,adam),order)
    return jax.pmap(update,axis_name='replicas')


def verify_collectives():
    """Heterogeneous data: aggregate gradients == concatenated-batch reference."""
    import numpy as np
    x=j.array([[1.,3.,9.,11.],[-5.,2.,4.,7.]])
    modes=j.array([[0,1,0,1],[0,0,1,1]],j.int32);live=j.ones_like(modes,bool)
    def test(v,m,l):
        normalized,counts=global_normalize(v,m,l)
        gradient=jax.grad(lambda theta:j.square(theta*v-j.arange(4)).mean())(j.float32(.3))
        return normalized,counts,gradient,jax.lax.pmean(gradient,'replicas')
    normalized,counts,local,global_gradient=jax.pmap(test,axis_name='replicas')(x,modes,live)
    expected=np.zeros((2,4),np.float32)
    for m in (0,1):
        use=np.array(modes)==m;v=np.array(x)[use];expected[use]=(v-v.mean())/np.sqrt(v.var()+1e-8)
    reference=jax.grad(lambda theta:j.square(theta*x-j.arange(4)[None]).mean())(j.float32(.3))
    assert not np.isclose(np.array(local)[0],np.array(local)[1])
    assert np.allclose(np.array(global_gradient),float(reference),rtol=1e-6,atol=1e-6)
    assert np.allclose(np.array(normalized),expected,rtol=1e-6,atol=1e-6)
    return dict(status='PASS_REAL_PSUM_PMEAN',different_local_gradients=np.array(local).tolist(),
        aggregated_gradients=np.array(global_gradient).tolist(),concatenated_reference=float(reference),
        mode_counts=np.array(counts).tolist(),global_moment_maxabs=float(abs(np.array(normalized)-expected).max()))
