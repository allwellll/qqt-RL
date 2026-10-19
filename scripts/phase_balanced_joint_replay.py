"""Successor candidate: stage allocation inside each current/history half.

No collector, buffer capacity/additions, objective, safety, loss coefficient or
production hook changes. Opening gets one of 32 slots when alternatives exist;
ordinary routes retain half the slots while navigation is present. The old
uniform sampler is exact when no preparation labels are available.
"""
import jax
import jax.numpy as jnp
from scripts.prestock_joint_replay import aggregate as _UNIFORM_AGGREGATE


def selected_ppo_diagnostic(batch,index):
    """Host evidence for selected rows; never substitute zero-value reward GAE."""
    import numpy as np
    logp,advantage,target=(float(np.asarray(batch[i][index])) for i in [3,4,5])
    return dict(collector_gae_advantage=advantage,collector_value_target=target,
        collector_value_estimate=target-advantage,behavior_action_log_probability=logp,
        collector_advantage_scope='Critic-backed rollout GAE before per-device PPO normalization; not action Q or counterfactual')


def choose(data,key,num=32):
    valid=data['valid']
    ordinary=valid&~data['preparation']
    opening=valid&data['opening']
    navigation=valid&data['preparation']&~data['opening']

    def draw(mask,k):
        mass=mask.astype(jnp.float32)
        return jax.random.choice(k,mass.size,(num,),p=jnp.where(mass.sum()>0,
            mass/jnp.maximum(mass.sum(),1),jnp.ones_like(mass)/mass.size))

    def balanced(_):
        has_ordinary=ordinary.any();has_navigation=navigation.any();has_opening=opening.any()
        opening_slots=jnp.where(has_opening,jnp.where(has_ordinary|has_navigation,max(1,num//32),num),0)
        ordinary_slots=jnp.where(has_ordinary,jnp.where(has_navigation,num//2,num-opening_slots),0)
        masks=jnp.stack([ordinary&((data['kind']==1)|(data['kind']==4)),
            ordinary&(data['kind']==2),ordinary&(data['kind']==3)])
        present=masks.any(-1)
        # Reserve three quarters of ordinary slots for return when both return and
        # acquisition labels exist; absent groups redistribute within the half.
        return_slots=jnp.where(present[2],jnp.where(present[:2].any(),ordinary_slots*3//4,ordinary_slots),0)
        other_order=jnp.argsort(~present[:2],stable=True)
        other=other_order[(jnp.arange(num)-return_slots)%jnp.maximum(present[:2].sum(),1)]
        group=jnp.where(jnp.arange(num)<return_slots,2,other)
        ordinary_ids=jnp.stack([draw(mask,jax.random.fold_in(key,i)) for i,mask in enumerate(masks)])
        ordinary_ids=ordinary_ids[group,jnp.arange(num)]
        nav_ids=draw(navigation,jax.random.fold_in(key,3))
        open_ids=draw(opening,jax.random.fold_in(key,4))
        positions=jnp.arange(num)
        return jnp.where(positions<ordinary_slots,ordinary_ids,
            jnp.where(positions<ordinary_slots+opening_slots,open_ids,nav_ids))

    ids=jax.lax.cond((navigation|opening).any(),balanced,lambda _:draw(valid,key),None)
    return {name:value[ids] for name,value in data.items()}


def aggregate(old,batch,key,generation,offset=48,joint_offset=95):
    from scripts import je09_stage_h5c_dagger as legacy
    updated,_,stats=_UNIFORM_AGGREGATE(old,batch,key,generation,offset,joint_offset)
    capacity=old['obs'].shape[0]
    additions=legacy.ADDITIONS if capacity==legacy.CAPACITY else 2
    slots=(old['cursor']+jnp.arange(additions))%capacity
    new={name:value[slots] for name,value in updated.items() if name not in ['cursor','additions']}
    history_data={name:old[name] for name in new}
    k1,k2=jax.random.split(key);size=32 if capacity==legacy.CAPACITY else 2
    recent=choose(new,k1,size);history=choose(history_data,k2,size)
    has_old=old['valid'].any();has_new=new['valid'].any()
    history=jax.tree.map(lambda h,n:jnp.where(has_old,h,n),history,recent)
    recent=jax.tree.map(lambda n,h:jnp.where(has_new,n,h),recent,history)
    replay=jax.tree.map(lambda a,b:jnp.concatenate([a,b],0),recent,history)
    stats=dict(stats,history_fraction=jnp.mean((replay['round']<generation)&replay['valid']),
        new_fraction=jnp.mean((replay['round']==generation)&replay['valid']))
    # Observability only: count the actual selected batch, including fallback
    # duplicates, rather than inferring exposure from FIFO membership.
    for half,data in [('current',recent),('history',history)]:
        valid=data['valid'];ordinary=valid&~data['preparation']
        groups=dict(valid=valid,ordinary=ordinary,returning=ordinary&(data['kind']==3),
            drop=ordinary&((data['kind']==1)|(data['kind']==4)),
            stock_ready=ordinary&(data['kind']==2),
            navigation=valid&data['preparation']&~data['opening'],opening=valid&data['opening'])
        stats.update({'replay_'+half+'_'+name:mask.sum() for name,mask in groups.items()})
    return updated,replay,stats
