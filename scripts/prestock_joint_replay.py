"""Prepared joint FIFO and loss; no running worker/collector hooks.

Keeps the existing additions, capacity, random keys and current/history split.
The extended context must contain an already verified 5x3 set at joint_offset;
ordinary movement labels accept every legal ability. Native-state verification
and first-deviation query authority remain responsibilities of the collector.
"""
import jax
import jax.numpy as jnp
from scripts import je09_stage_h5c_dagger as legacy
_LEGACY_BUFFER=legacy.buffer


def buffer(capacity=legacy.CAPACITY,additions=legacy.ADDITIONS,obs_shape=(24,13,15),global_size=24):
    result=_LEGACY_BUFFER(capacity,additions,obs_shape,global_size)
    result.update(ability_legal=jnp.zeros((capacity,3),bool),joint_target=jnp.zeros((capacity,5,3),bool))
    result.update(preparation=jnp.zeros(capacity,bool),revision=jnp.zeros(capacity,jnp.int32),
        objective=jnp.zeros(capacity,jnp.int32),opening=jnp.zeros(capacity,bool))
    return result


def aggregate(old,batch,key,generation,offset=legacy.OFFSET,joint_offset=95):
    flatten=lambda x:x.reshape((-1,)+x.shape[2:])
    obs,gv,legal,ability,ctx=map(flatten,(batch[0],batch[1],batch[6][0],batch[6][1],batch[-1]))
    labels=ctx[:,offset:offset+12]
    joint=ctx[:,joint_offset:joint_offset+15].reshape(-1,5,3)>0
    joint&=legal[:,:,None]&ability[:,None,:]
    valid=(labels[:,5]>0)&joint.any((-2,-1))
    count=old['obs'].shape[0];k=legacy.ADDITIONS if count==legacy.CAPACITY else 2
    _,ids=jax.lax.top_k(jnp.where(valid,jax.random.uniform(key,(valid.size,))+1.,-1.),k)
    new=dict(obs=obs[ids],gv=gv[ids],legal=legal[ids],target=labels[ids,:5],valid=valid[ids],
        kind=labels[ids,6].astype(jnp.int32),round=jnp.full(k,generation,jnp.int32),
        ability_legal=ability[ids],joint_target=joint[ids])
    metadata=ctx[:,joint_offset+15:joint_offset+19] if ctx.shape[-1]>=joint_offset+19 else jnp.zeros((ctx.shape[0],4))
    new.update(preparation=metadata[ids,0]>0,revision=metadata[ids,1].astype(jnp.int32),
        objective=metadata[ids,2].astype(jnp.int32),opening=metadata[ids,3]>0)
    # Keep parity with the existing per-generation aggregation keys and split.
    # Tests compare every legacy FIFO field, replay field and statistic exactly.
    def choose(data,random_key,num):
        mass=data['valid'].astype(jnp.float32)
        ids=jax.random.choice(random_key,mass.size,(num,),p=jnp.where(mass.sum()>0,
            mass/jnp.maximum(mass.sum(),1),jnp.ones_like(mass)/mass.size))
        return {name:data[name][ids] for name in new}
    k1,k2=jax.random.split(key);size=32 if count==legacy.CAPACITY else 2
    recent=choose(new,k1,size);history=choose(old,k2,size)
    has_old=old['valid'].any();has_new=new['valid'].any()
    history=jax.tree.map(lambda h,n:jnp.where(has_old,h,n),history,recent)
    recent=jax.tree.map(lambda n,h:jnp.where(has_new,n,h),recent,history)
    replay=jax.tree.map(lambda a,b:jnp.concatenate([a,b],0),recent,history)
    slots=(old['cursor']+jnp.arange(k))%count
    updated={name:old[name].at[slots].set(new[name]) for name in new}
    updated.update(cursor=(old['cursor']+k)%count,additions=old['additions'])
    return updated,replay,dict(visited=jnp.float32(valid.size),eligible=labels[:,7].sum(),query=valid.sum(),
        covered=new['valid'].sum(),aggregate=updated['valid'].sum(),
        history_fraction=jnp.mean((replay['round']<generation)&replay['valid']),
        new_fraction=jnp.mean((replay['round']==generation)&replay['valid']),
        path_a=(((labels[:,6]==1)|(labels[:,6]==4))&valid).sum(),
        path_a_enemy=((labels[:,6]==1)&valid).sum(),path_a_own_recovery=((labels[:,6]==4)&valid).sum(),
        path_b=((labels[:,6]==2)&valid).sum(),returning=((labels[:,6]==3)&valid).sum(),
        cage_contact=((labels[:,9]>0)&(labels[:,7]>0)).sum(),actual_custody=(labels[:,10]>0).sum(),
        noneligible_label=((~(labels[:,7]>0))&valid).sum())


def replay_loss(params,replay):
    from jax_bomb import jax_train as train
    from scripts.return_dagger_weight import weighted_joint_loss
    move,ability,_,_=train.net_forward(params,'transformer',replay['obs'],replay['gv'])
    returning=(replay['kind']==3)&~replay['preparation']
    return weighted_joint_loss(move/legacy.TEMPERATURE,ability/legacy.TEMPERATURE,
        replay['legal'],replay['ability_legal'],replay['joint_target'],replay['valid'],
        returning,2.,legacy.COEFFICIENT)[0]
