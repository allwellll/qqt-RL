"""Fused DCU-resident rule bots, frozen selfplay, rollout and GAE."""
from functools import partial
import jax
import jax.numpy as j
import numpy as np
from scripts.r4_k4_observation import actor_critic_forward,forward
from qqt_rl.training.kill_ppo import masked_logits,gae
from qqt_rl.training import jax_kill_env as E


def build(devices,count,ticks,*,bot_fraction=.5):
    E.prepare()
    from jax_bomb import bun_jax_bots as bots
    # Half rules bots; half model opponents. All decisions stay on device.
    if count%16:raise ValueError('JAX collector envs/card must be divisible by16')
    if bot_fraction not in (0.,.5):raise ValueError('Expected 0 or .5 bot fraction')
    bot_count=count//2 if bot_fraction else 0
    ids=j.arange(count);seats=ids%2
    categories=np.zeros(count,np.int32)
    categories[:bot_count]=-1
    remaining=count-bot_count
    a=bot_count+remaining//2;b=a+remaining//4;c=b+remaining//8
    categories[bot_count:a]=0;categories[a:b]=1;categories[b:c]=2;categories[c:]=3
    groups=[np.flatnonzero(categories==i) for i in range(4)]
    tiers=j.where(j.arange(bot_count)<bot_count//2,3,4).astype(j.int32)
    def select(mask,new,old):
        return jax.tree.map(lambda a,b:j.where(mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)),a,b),new,old)
    def one_input(world,seat):return E.inputs(world,seat)

    @partial(jax.pmap,devices=devices)
    def init(keys):return jax.vmap(E.fresh)(jax.random.split(keys,count))

    @partial(jax.pmap,devices=devices)
    def collect(p,world,bank,key):
        def body(carry,_):
            world,bank,key=carry
            key,km,ka,kb,ke,kr=jax.random.split(key,6)
            obs,gv,mm,bm,live=one_input(world,seats)
            mv,ab,value,_=actor_critic_forward(p,obs,gv)
            lm,la=masked_logits(mv,ab,mm,bm)
            own=j.stack([jax.random.categorical(km,lm),jax.random.categorical(ka,la)],-1)
            logp=lm[ids,own[:,0]]+la[ids,own[:,1]]
            eo,eg,emm,ebm,_=one_input(world,1-seats)
            enemy=j.zeros_like(own)
            if bot_count:
                sub=jax.tree.map(lambda x:x[:bot_count],world.state)
                enemy=enemy.at[:bot_count].set(bots.rule_bot_actions(sub,1-seats[:bot_count],
                    emm[:bot_count],ebm[:bot_count],kb,tiers))
            for kind,indices in enumerate(groups):
                if not len(indices):continue
                ix=j.asarray(indices)
                mv,ab,_,_=forward(jax.tree.map(lambda x:x[kind],bank),eo[ix],eg[ix])
                lm,la=masked_logits(mv,ab,emm[ix],ebm[ix]);k0,k1=jax.random.split(jax.random.fold_in(kb,kind+1))
                action=j.stack([jax.random.categorical(k0,lm),jax.random.categorical(k1,la)],-1)
                enemy=enemy.at[ix].set(action)
            actions=j.zeros((count,2,2),j.int32).at[ids,seats].set(own).at[ids,1-seats].set(enemy)
            after,reward,done,metrics=jax.vmap(E.step)(world,actions,jax.random.split(ke,count))
            # Worlds start together and share the fixed horizon. Refresh the
            # frozen current/history banks only between complete games.
            def refresh(bank):
                return jax.tree.map(lambda slots,actor:slots.at[1].set(slots[0]).at[0].set(actor),bank,p['actor'])
            bank=jax.lax.cond(j.all(done),refresh,lambda b:b,bank)
            # All worlds terminate on the same tick; avoid constructing a fresh
            # world on every nonterminal tick, while keeping reset on device.
            def reset(_):return jax.vmap(E.fresh)(jax.random.split(kr,count))
            after=jax.lax.cond(j.all(done),reset,lambda _:after,None)
            metric=j.stack([metrics[ids,seats].sum(0),metrics[ids,1-seats].sum(0)])
            data=dict(obs=obs,state=gv,mm=mm,bm=bm,live=live.astype(j.float32),
                action=own,logp=logp,value=value,reward=reward[ids,seats],done=done,metric=metric,
                overflow=j.max(after.state.native_items.overflow))
            return (after,bank,key),data
        (world,bank,key),data=jax.lax.scan(body,(world,bank,key),None,length=ticks)
        obs,gv,_,_,_=one_input(world,seats)
        _,_,bootstrap,_=actor_critic_forward(p,obs,gv)
        adv,returns=gae(data['reward'],data['value'],data['done'],bootstrap)
        batch={k:data[k] for k in ('obs','state','mm','bm','live','action','logp')}
        batch.update(adv=adv,returns=returns)
        batch={k:v.reshape((-1,)+v.shape[2:]) for k,v in batch.items()}
        return world,bank,key,batch,data['metric'].sum(0),j.max(data['overflow'])
    return init,collect
