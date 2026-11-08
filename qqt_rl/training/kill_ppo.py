"""Synchronous data-parallel PPO; all gradients aggregate before clipping/Adam."""
from functools import partial
import jax
import jax.numpy as j
import optax


def gae(reward, value, done, bootstrap, gamma=.997, lam=.95):
    next_value=j.concatenate([value[1:],bootstrap[None]],axis=0)
    def body(carry,row):
        r,v,nv,d=row
        adv=r+gamma*(1-d)*nv-v+gamma*lam*(1-d)*carry
        return adv,adv
    _,adv=jax.lax.scan(body,j.zeros_like(bootstrap),(reward,value,next_value,done),reverse=True)
    return adv,adv+value


def masked_logits(move,ability,mm,bm):
    return jax.nn.log_softmax(j.where(mm,move,-1e9)),jax.nn.log_softmax(j.where(bm,ability,-1e9))


def build(forward, devices, minibatch, lr=3e-5, critic_lr=1e-4, clip=.15, kl_coef=.01):
    actor_opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(lr,eps=1e-8))
    critic_opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(critic_lr,eps=1e-8))
    def init(params):
        return dict(actor=actor_opt.init(params['actor']),critic=critic_opt.init(params['critic']))

    @partial(jax.pmap,devices=devices)
    def inference(params,opponent,obs,state,mm,bm,key,temperature):
        # Inputs contain learner rows first, then opponent rows.
        n=obs.shape[0]//2
        mv,ab,value,_=forward(params,obs[:n],state[:n])
        om,oa,_,_=forward(dict(actor=opponent,critic=opponent),obs[n:],state[n:])
        move=j.concatenate([mv,om]);ability=j.concatenate([ab,oa])
        lm,la=masked_logits(move/temperature,ability/temperature,mm,bm)
        key,km,ka=jax.random.split(key,3)
        action=j.stack([jax.random.categorical(km,lm),jax.random.categorical(ka,la)],-1)
        rows=j.arange(2*n)
        logp=lm[rows,action[:,0]]+la[rows,action[:,1]]
        return action,logp[:n],value,key

    @partial(jax.pmap,axis_name='cards',devices=devices)
    def update(params,adam,reference,batch,key,warmup):
        live=batch['live']
        total=jax.lax.psum(live.sum(),'cards')
        mean=jax.lax.psum((batch['adv']*live).sum(),'cards')/j.maximum(total,1)
        var=jax.lax.psum((((batch['adv']-mean)**2)*live).sum(),'cards')/j.maximum(total,1)
        batch=dict(batch,adv=(batch['adv']-mean)*jax.lax.rsqrt(var+1e-8))
        order=jax.random.permutation(key,live.size).reshape(-1,minibatch)
        def step(carry,indices):
            p,a=carry;b=jax.tree.map(lambda x:x[indices],batch)
            count=jax.lax.psum(b['live'].sum(),'cards')
            cards=jax.lax.psum(j.array(1.),'cards')
            def objective(q):
                mv,ab,value,vl=forward(q,b['obs'],b['state'])
                lm,la=masked_logits(mv,ab,b['mm'],b['bm'])
                rows=j.arange(minibatch)
                logp=lm[rows,b['action'][:,0]]+la[rows,b['action'][:,1]]
                ratio=j.exp(logp-b['logp'])
                policy=j.maximum(-b['adv']*ratio,-b['adv']*j.clip(ratio,1-clip,1+clip))
                entropy=-(j.sum(j.exp(lm)*lm,-1)+j.sum(j.exp(la)*la,-1))
                rm,ra,_,_=forward(dict(actor=reference,critic=reference),b['obs'],b['state'])
                rlm,rla=masked_logits(rm,ra,b['mm'],b['bm'])
                kl=j.sum(j.exp(rlm)*(rlm-lm),-1)+j.sum(j.exp(rla)*(rla-la),-1)
                # Categorical critic uses the inherited [-20,20] support.
                centers=j.linspace(-20.,20.,vl.shape[-1])
                target=jax.nn.softmax(-.5*((centers[None]-b['returns'][:,None])/1.5)**2,axis=-1)
                value_loss=-(target*jax.nn.log_softmax(vl)).sum(-1).mean()
                denom=j.maximum(count,1)
                ps=cards*j.sum((policy-.005*entropy+kl_coef*kl)*b['live'])/denom
                loss=j.where(warmup,0.,ps)+.5*value_loss
                metrics=j.stack([ps,value_loss,entropy.mean(),kl.mean(),
                    j.mean(j.abs(ratio-1)>clip),j.mean((value-b['returns'])**2)])
                return loss,metrics
            (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(p)
            gradient=jax.lax.pmean(gradient,'cards')
            ca,na=critic_opt.update(gradient['critic'],a['critic'],p['critic'])
            def learn_actor(_):
                delta,state=actor_opt.update(gradient['actor'],a['actor'],p['actor'])
                return optax.apply_updates(p['actor'],delta),state
            actor,actor_state=jax.lax.cond(warmup|(count==0),lambda _:(p['actor'],a['actor']),learn_actor,None)
            q=dict(actor=actor,critic=optax.apply_updates(p['critic'],ca))
            state=dict(actor=actor_state,critic=na)
            row=jax.lax.pmean(j.concatenate([loss[None],metrics,optax.global_norm(gradient)[None]]),'cards')
            return (q,state),row
        return jax.lax.scan(step,(params,adam),order)
    return init,inference,update
