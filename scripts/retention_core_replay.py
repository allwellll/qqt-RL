"""Persistent causal return states; replaces existing history slots only.

Four equal strata, deterministic bottom-k content priorities and duplicate
removal. No teacher logits, new labels, environment resets or extra loss.
"""
import jax
import jax.numpy as jnp
from scripts import return_error_replay as baseline
from scripts.prestock_joint_replay import buffer as fifo_buffer
_BASE_AGGREGATE=baseline.aggregate


def eligible(data):
    g=data['gv'];scene=data['scene']
    accepted=(data['joint_target']&data['legal'][:,:,None]&data['ability_legal'][:,None,:]).any((-2,-1))
    return (data['valid']&(data['kind']==3)&~data['preparation']&data['custody']
        &~jnp.isin(scene,jnp.array([0,2,3,4,6]))&(g[:,9]==1)&(g[:,13]>.5)&(g[:,14]>.5)&accepted)


def stratum(data):
    return (data['scene']==8).astype(jnp.int32)*2+(data['gv'][:,10]>.5).astype(jnp.int32)


def fingerprints(data):
    # Exclude elapsed episode time; include every other observed feature and
    # authoritative label/mask to keep each joint target indivisible.
    parts=[data['obs'].reshape((data['obs'].shape[0],-1)),data['gv'][:,1:],
        data['legal'],data['ability_legal'],data['joint_target'].reshape((-1,15))]
    value=jnp.rint(jnp.concatenate(parts,-1)*64).astype(jnp.int32).astype(jnp.uint32)
    index=jnp.arange(value.shape[1],dtype=jnp.uint32)+1
    def mixed(x):
        x=(x^(x>>16))*jnp.uint32(0x7feb352d)
        x=(x^(x>>15))*jnp.uint32(0x846ca68b)
        return x^(x>>16)
    return (mixed((value*mixed(index)).sum(-1)),
        mixed((value*mixed(index+jnp.uint32(0x9e3779b9))).sum(-1)))


def empty(template,per_stratum=128):
    return {name:jnp.zeros((4*per_stratum,)+value.shape[1:],value.dtype) for name,value in template.items()}


def merge(old,new):
    data=jax.tree.map(lambda a,b:jnp.concatenate([a,b]),old,new)
    h1,h2=fingerprints(data);group=stratum(data);valid=eligible(data)
    # Stable ordering prefers the already resident record of a duplicate.
    order=jnp.lexsort((h2,h1,~valid,group))
    h1s,h2s,gs=h1[order],h2[order],group[order]
    duplicate=jnp.concatenate([jnp.array([False]),(h1s[1:]==h1s[:-1])&(h2s[1:]==h2s[:-1])&(gs[1:]==gs[:-1])])
    distinct=valid[order]&~duplicate;quota=old['valid'].size//4
    ids=[];masks=[]
    for layer in range(4):
        selected=distinct&(gs==layer)
        positions=jnp.argsort(~selected,stable=True)[:quota]
        ids.append(order[positions]);masks.append(selected[positions])
    ids=jnp.concatenate(ids);mask=jnp.concatenate(masks)
    result={name:jnp.where(mask.reshape((-1,)+(1,)*(value.ndim-1)),value[ids],jnp.zeros_like(value[ids])) for name,value in data.items()}
    result['valid']=mask
    return result


def replace_history(replay,bank,key,history_start):
    valid=eligible(bank)
    groups=stratum(bank)
    counts=jnp.bincount(groups,weights=valid.astype(jnp.float32),length=4)
    weights=valid/jnp.maximum(counts[groups],1)
    weights=jnp.where(weights.sum()>0,weights/jnp.maximum(weights.sum(),1),jnp.ones_like(weights)/weights.size)
    ids=jax.random.choice(key,weights.size,(replay['valid'].size,),p=weights)
    returning=replay['valid']&(replay['kind']==3)&~replay['preparation']&(jnp.arange(replay['valid'].size)>=history_start)
    slots=returning&(jnp.cumsum(returning)%2==1)&valid.any()
    output={name:jnp.where(slots.reshape((-1,)+(1,)*(value.ndim-1)),bank[name][ids],value) for name,value in replay.items()}
    return output,slots.sum()


def buffer(*args,**kwargs):
    fifo=fifo_buffer(*args,**kwargs)
    template={k:v for k,v in fifo.items() if k not in ['cursor','additions']}
    template.update(scene=jnp.zeros(template['valid'].size,jnp.int32),custody=jnp.zeros(template['valid'].size,bool))
    return dict(fifo,retention_core=empty(template))


def aggregate(old,batch,key,generation,actor,offset=48,joint_offset=95):
    fifo={k:v for k,v in old.items() if k!='retention_core'}
    updated,replay,stats=_BASE_AGGREGATE(fifo,batch,key,generation,actor,offset,joint_offset)
    from scripts import je09_stage_h5c_dagger as legacy
    additions=legacy.ADDITIONS if fifo['obs'].shape[0]==legacy.CAPACITY else 2
    positions=(fifo['cursor']+jnp.arange(additions))%fifo['obs'].shape[0]
    new={k:v[positions] for k,v in updated.items() if k not in ['cursor','additions']}
    # Exact same top-k indices as the authoritative FIFO writer. Proof is the
    # collector's native custody field, never inferred from scene or carry.
    ctx=batch[-1].reshape((-1,batch[-1].shape[-1]))
    lm=batch[6][0].reshape((-1,5));la=batch[6][1].reshape((-1,3))
    joint=ctx[:,joint_offset:joint_offset+15].reshape((-1,5,3))>0
    authority=(ctx[:,offset+5]>0)&(joint&lm[:,:,None]&la[:,None,:]).any((-2,-1))
    _,ids=jax.lax.top_k(jnp.where(authority,jax.random.uniform(key,(authority.size,))+1.,-1.),additions)
    new.update(scene=ctx[ids,0].astype(jnp.int32),custody=ctx[ids,offset+10]>0)
    bank=merge(old['retention_core'],new)
    # Only pre-existing history contributes; do not recategorize current rows.
    replay,replaced=replace_history(replay,old['retention_core'],jax.random.fold_in(key,0xC0AE),replay['valid'].size//2)
    stats=dict(stats,core_replaced_history_return=replaced,core_size=bank['valid'].sum(),
        core_oldest_round_device_sum=jnp.min(jnp.where(bank['valid'],bank['round'],generation)))
    for layer in range(4):stats['core_layer_'+str(layer)]=(bank['valid']&(stratum(bank)==layer)).sum()
    # Historical exposure values must describe the actual final batch.
    mass=baseline.target_mass(actor,replay)
    hist_return=replay['valid']&(replay['kind']==3)&~replay['preparation']&(jnp.arange(replay['valid'].size)>=replay['valid'].size//2)
    core_slots=hist_return&(jnp.cumsum(hist_return)%2==1)&eligible(old['retention_core']).any()
    stats['core_selected_joint_mass_sum']=jnp.where(core_slots,mass,0.).sum()
    for half,sl in [('current',slice(0,replay['valid'].size//2)),('history',slice(replay['valid'].size//2,None))]:
        ret=replay['valid'][sl]&(replay['kind'][sl]==3)&~replay['preparation'][sl]
        stats['replay_'+half+'_return_target_mass_sum']=jnp.where(ret,mass[sl],0).sum()
        stats['replay_'+half+'_return_error_sum']=jnp.where(ret,1-mass[sl],0).sum()
    stats.update(history_fraction=jnp.mean((replay['round']<generation)&replay['valid']),new_fraction=jnp.mean((replay['round']==generation)&replay['valid']))
    return dict(updated,retention_core=bank),replay,stats


def install():
    from scripts import je09_stage_h5c_dagger as dagger
    dagger.buffer=buffer
    baseline.aggregate=aggregate
