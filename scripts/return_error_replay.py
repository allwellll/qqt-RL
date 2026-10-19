"""Uninstalled successor candidate: error-weighted native return replay only."""
import jax
import jax.numpy as jnp
from jax_bomb.jax_net import net_forward
from scripts.phase_balanced_joint_replay import choose as baseline_choose

EPSILON=.05


def replace_return_draws(data,key,target_mass,num=32):
    """Preserve baseline positions/classes, replace only ordinary-return rows.

    Existing native legal sets remain authoritative. Every return row keeps
    positive support; current fit changes sampling, not labels or x2 loss.
    Equal fit is exactly the baseline draw, including its random realization.
    """
    selected=baseline_choose(data,key,num)
    returning=data['valid']&(data['kind']==3)&~data['preparation']
    score=jax.lax.stop_gradient(jnp.clip(target_mass,0.,1.))
    weights=returning*(EPSILON+1.-score)
    low=jnp.min(jnp.where(returning,weights,jnp.inf))
    high=jnp.max(jnp.where(returning,weights,-jnp.inf))
    slots=selected['valid']&(selected['kind']==3)&~selected['preparation']

    def replace(_):
        ids=jax.random.choice(jax.random.fold_in(key,17),weights.size,(num,),
            p=weights/jnp.maximum(weights.sum(),1e-9))
        return {name:jnp.where(slots.reshape((num,)+(1,)*(value.ndim-1)),
            value[ids],selected[name]) for name,value in data.items()}

    return jax.lax.cond(returning.any()&(high>low),replace,lambda _:selected,None)


def target_mass(actor,data):
    move,ability,_,_=net_forward(actor,'transformer',data['obs'],data['gv'])
    pm=jax.nn.softmax(jnp.where(data['legal'],move,-1e9))
    pa=jax.nn.softmax(jnp.where(data['ability_legal'],ability,-1e9))
    accepted=data['joint_target']&data['legal'][:,:,None]&data['ability_legal'][:,None,:]
    return jax.lax.stop_gradient((pm[:,:,None]*pa[:,None,:]*accepted).sum((-2,-1)))


def choose(actor,data,key,num):
    mass=target_mass(actor,data)
    return replace_return_draws(dict(data,_actor_target_mass=mass),key,mass,num)


def aggregate(old,batch,key,generation,actor,offset=48,joint_offset=95):
    from scripts import phase_balanced_joint_replay as phase
    from scripts import je09_stage_h5c_dagger as legacy
    updated,_,stats=phase._UNIFORM_AGGREGATE(old,batch,key,generation,offset,joint_offset)
    capacity=old['obs'].shape[0]
    additions=legacy.ADDITIONS if capacity==legacy.CAPACITY else 2
    slots=(old['cursor']+jnp.arange(additions))%capacity
    new={name:value[slots] for name,value in updated.items() if name not in ['cursor','additions']}
    history_data={name:old[name] for name in new}
    k1,k2=jax.random.split(key);size=64 if capacity==legacy.CAPACITY else 2
    recent=choose(actor,new,k1,size);history=choose(actor,history_data,k2,size)
    has_old=old['valid'].any();has_new=new['valid'].any()
    history=jax.tree.map(lambda h,n:jnp.where(has_old,h,n),history,recent)
    recent=jax.tree.map(lambda n,h:jnp.where(has_new,n,h),recent,history)
    stats=dict(stats)
    for half,data in [('current',recent),('history',history)]:
        valid=data['valid'];ordinary=valid&~data['preparation']
        groups=dict(valid=valid,ordinary=ordinary,returning=ordinary&(data['kind']==3),
            drop=ordinary&((data['kind']==1)|(data['kind']==4)),
            stock_ready=ordinary&(data['kind']==2),
            navigation=valid&data['preparation']&~data['opening'],opening=valid&data['opening'])
        stats.update({'replay_'+half+'_'+name:mask.sum() for name,mask in groups.items()})
        ret=groups['returning'];mass=data.pop('_actor_target_mass')
        stats['replay_'+half+'_return_target_mass_sum']=jnp.where(ret,mass,0.).sum()
        stats['replay_'+half+'_return_error_sum']=jnp.where(ret,1.-mass,0.).sum()
    replay=jax.tree.map(lambda a,b:jnp.concatenate([a,b],0),recent,history)
    stats.update(history_fraction=jnp.mean((replay['round']<generation)&replay['valid']),
        new_fraction=jnp.mean((replay['round']==generation)&replay['valid']))
    return updated,replay,stats
