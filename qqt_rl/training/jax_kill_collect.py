"""Fused DCU-resident rule bots, frozen selfplay, rollout and GAE."""
from functools import partial
import jax
import jax.numpy as j
import numpy as np
from scripts.r4_k4_observation import actor_critic_forward,forward
from qqt_rl.training.kill_ppo import masked_logits,gae
from qqt_rl.training import jax_kill_env as E


def consistency_check(devices,minibatch):
    @partial(jax.pmap,devices=devices)
    def check(params,batch):
        order=j.arange(batch['value'].size).reshape(-1,minibatch)
        def body(_,indices):
            b=jax.tree.map(lambda x:x[indices],batch)
            mv,ab,value,_=actor_critic_forward(params,b['obs'],b['state'])
            lm,la=masked_logits(mv,ab,b['mm'],b['bm'])
            rows=j.arange(minibatch)
            logp=lm[rows,b['action'][:,0]]+la[rows,b['action'][:,1]]
            return None,j.stack([j.max(j.abs(logp-b['logp'])),
                                 j.max(j.abs(value-b['value']))])
        _,errors=jax.lax.scan(body,None,order)
        finite=j.all(j.stack([j.all(j.isfinite(batch[k]))
                              for k in ('value','logp','adv','returns')]))
        return errors.max(0),finite
    return check


def opponent_allocation(count,bot_fraction=.1):
    if count%2:raise ValueError('JAX collector envs/card must be even')
    if abs(bot_fraction-.1)>1e-9:raise ValueError('The evaluated league uses a 10% rule-bot fraction')
    fractions=np.asarray([.10,.10,.35,.35,.05,.05],np.float64)
    bot_count=int(round(count*bot_fraction/2))*2
    pairs=(count-bot_count)//2
    exact=pairs*fractions
    pair_counts=np.floor(exact).astype(np.int32)
    remainder=pairs-int(pair_counts.sum())
    order=np.argsort(-(exact-pair_counts),kind='stable')
    pair_counts[order[:remainder]]+=1
    counts=pair_counts*2
    if counts.sum()+bot_count!=count:
        raise AssertionError('Opponent allocation lost environments')
    return bot_count,counts


def build(devices,count,ticks,*,minibatch=512,bot_fraction=.1):
    E.prepare()
    from jax_bomb import bun_jax_bots as bots
    # Sample a league of fixed strong anchors plus recent frozen self-play.
    bot_count,counts=opponent_allocation(count,bot_fraction)
    ids=j.arange(count);seats=ids%2
    categories=np.full(count,-1,np.int32)
    start=bot_count;groups=[]
    for category,n in enumerate(counts):
        categories[start:start+n]=category
        groups.append(np.arange(start,start+n,dtype=np.int32))
        start+=n
    tiers=j.where(j.arange(bot_count)<bot_count//2,3,4).astype(j.int32)
    def select(mask,new,old):
        return jax.tree.map(lambda a,b:j.where(mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)),a,b),new,old)
    def one_input(world,seat):return E.inputs(world,seat)

    def learner_forward(params, obs, state):
        if count <= minibatch:
            return actor_critic_forward(params, obs, state)[:3]

        # On ROCm the low-precision forward can change with the batch size.
        # Use PPO's batch shape while keeping all worlds in the same rollout.
        def chunks(x):
            padding = (-count) % minibatch
            if padding:
                x = j.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
            return x.reshape((-1, minibatch) + x.shape[1:])

        def infer(rows):
            return actor_critic_forward(params, *rows)[:3]

        outputs = jax.lax.map(infer, (chunks(obs), chunks(state)))
        return tuple(x.reshape((-1,) + x.shape[2:])[:count] for x in outputs)

    @partial(jax.pmap,devices=devices)
    def init(keys):return jax.vmap(E.fresh)(jax.random.split(keys,count))

    @partial(jax.pmap,devices=devices)
    def collect(p,world,bank,key):
        def body(carry,_):
            world,bank,key=carry
            key,km,ka,kb,ke,kr=jax.random.split(key,6)
            obs,gv,mm,bm,live=one_input(world,seats)
            mv,ab,value=learner_forward(p,obs,gv)
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
        _,_,bootstrap=learner_forward(p,obs,gv)
        adv,returns=gae(data['reward'],data['value'],data['done'],bootstrap)
        batch={k:data[k] for k in ('obs','state','mm','bm','live','action','logp','value')}
        batch.update(adv=adv,returns=returns)
        batch={k:v.reshape((-1,)+v.shape[2:]) for k,v in batch.items()}
        return world,bank,key,batch,data['metric'].sum(0),j.max(data['overflow'])
    return init,collect
