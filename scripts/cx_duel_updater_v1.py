"""Original full-game SIL, native duel PPO; real minibatch mode normalization."""
import jax
import jax.numpy as j
from scripts import r4_k4_collector as common


def objective(params,batch,coef):
    from scripts.r4_k4_observation import actor_critic_forward
    from jax_bomb.jax_train import hl_gauss_value_loss
    obs,g,actions,old,adv,ret,mm,bm,modes=batch[:9]
    mv,ab,_,vl=actor_critic_forward(params,obs,g)
    lm=jax.nn.log_softmax(j.where(mm,mv,-1e9));la=jax.nn.log_softmax(j.where(bm,ab,-1e9))
    rows=j.arange(len(old));selected=lm[rows,actions[:,0]]+la[rows,actions[:,1]]
    ratio=j.exp(selected-old)
    policy=j.maximum(-adv*ratio,-adv*j.clip(ratio,.9,1.1)).mean()
    value=hl_gauss_value_loss(vl,ret)
    entropy=-(j.sum(j.exp(lm)*lm,-1)+j.sum(j.exp(la)*la,-1)).mean()
    # Preserve the inherited full-game term; the new task has no imitation term.
    use=(modes==0)&(actions[:,1]==1)&bm[:,1].astype(bool)
    sil=-(j.where(use,jax.nn.relu(adv),0.)*la[:,1]).sum()/len(old)
    loss=policy+.5*value-.005*entropy+coef*sil
    return loss,j.array([policy,value,entropy,(old-selected).mean(),(j.abs(ratio-1.)>.1).mean()])


def updater(coef):
    import optax
    def build(minibatch,optimizer):
        def update(params,adam,batch,key):
            flat=tuple(v.reshape((-1,)+v.shape[2:]) for v in batch)
            order=jax.random.permutation(key,len(flat[4])).reshape(-1,minibatch)
            def body(carry,indices):
                p,a=carry;mb=tuple(v[indices] for v in flat)
                adv,norm=common.normalize_by_mode(mb[4],mb[8],mb[9])
                normbatch=mb[:4]+(adv,)+mb[5:]
                (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(p,normbatch,coef)
                changes={};next_adam={}
                for branch in ('actor','critic'):
                    changes[branch],next_adam[branch]=optimizer.update(gradient[branch],a[branch],p[branch])
                signed=[]
                for mode in (0,1):
                    for ability in (0,1):
                        use=mb[9]&(mb[8]==mode)&(mb[2][:,1]==ability)
                        count=use.sum();den=j.maximum(count,1)
                        signed.extend([count,j.where(use,mb[4],0.).sum()/den,j.where(use,adv,0.).sum()/den])
                values=j.concatenate([loss[None],metrics,optax.global_norm(gradient)[None],
                    optax.global_norm(gradient['actor']['heads']['wb'])[None],
                    norm['mode_samples'],norm['mode_mean_abs_adv'],j.stack(signed)])
                return (optax.apply_updates(p,changes),next_adam),values
            return jax.lax.scan(body,(params,adam),order)
        return jax.jit(update)
    return build
