"""Online native-event stage labels and bounded iterative DAgger storage."""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import shortest, INF
from scripts.je09_return_expert import action_set, supervised_loss

OFFSET=48
CAPACITY=4096
ADDITIONS=256
COEFFICIENT=.05
TEMPERATURE=1.


def mode(enemy_alive, enemy_trapped, attributed):
    return attributed & (~enemy_alive | (enemy_trapped>0))


def eligible(scene, alive, terminal, trapped, carrying, window, custody, age):
    return ((scene==8)&alive&~terminal&(trapped==0)
            &(((carrying<0)&window)|((carrying>=0)&custody&(age<120))))


def state(n):
    return dict(window=jnp.zeros((n,2),bool),custody=jnp.zeros((n,2),bool),
        age=jnp.zeros((n,2),jnp.int32),origin=jnp.zeros((n,2),jnp.int32),
        loose_origin=jnp.full((n,2),-1,jnp.int32),ready=jnp.zeros(n,bool),blocked=jnp.zeros((n,13,15),bool),
        loose=jnp.zeros((n,2,13,15),bool),distance=jnp.full((n,6,13,15),INF,jnp.int32))


def update(before,after,info,aux,old):
    n=before.pos.shape[0];r=jnp.arange(n)[:,None];s=jnp.arange(2)[None];e=1-s
    attribution=(after.trapped_causal[r,e,s]|after.trapped_source[r,e,s])&(after.trapped[r,e]>0)
    owned=old['window']|attribution|info['surviving_kill']
    window=mode(after.alive[:,::-1],after.trapped[:,::-1],owned)
    stock=(before.bun_stored[r,e,e]>after.bun_stored[r,e,e])&info['steal']
    picked=jnp.maximum(info['pickup_origin'],0)
    loose=info['pickup_loose']
    causal_loose=loose&(aux['drop']['credit'][r,picked]==s)&(aux['drop']['nonce'][r,picked]>0)
    start=(before.bun_carried<0)&(after.bun_carried>=0)&before.alive&after.alive
    acquired=start&(((stock|causal_loose)&old['window'])|(info['native_transfer']&owned))
    custody=(old['custody']|acquired)&after.alive&(after.bun_carried>=0)
    age=jnp.where(acquired,0,jnp.minimum(old['age']+1,1000000))
    origin=jnp.where(acquired,jnp.where(stock,2,jnp.where(after.bun_carried==s,4,1)),old['origin'])
    blocked=after.wall|after.brick|after.core.pushable
    targets=aux['routing']['target']
    native_present=after.bun_loose[r,jnp.clip(targets[...,0],0,12),jnp.clip(targets[...,1],0,14),s]>0
    credited=(aux['drop']['credit'][r,e]==s)&(aux['drop']['nonce'][r,e]>0)&jnp.all(targets[r,e]>=0,-1)&native_present[r,e]
    chosen=jnp.where(credited,e,s)
    target=targets[r,chosen]
    cell=jnp.clip(target,jnp.array([0,0]),jnp.array([12,14]))
    attributable=(aux['drop']['credit'][r,chosen]==s)&(aux['drop']['nonce'][r,chosen]>0)&jnp.all(target>=0,-1)
    present=after.bun_loose[r,cell[...,0],cell[...,1],chosen]>0
    loose_goals=jnp.zeros((n,2,13,15),bool).at[r,s,cell[...,0],cell[...,1]].set(attributable&present)
    dirty=(~old['ready']|jnp.any(blocked!=old['blocked'],(-2,-1))
           |jnp.any(loose_goals!=old['loose'],(1,2,3)))
    from jax_bomb import je07_bun_env as env
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    goals=jnp.concatenate([loose_goals,jnp.broadcast_to(bases[::-1],(n,2,13,15)),
                           jnp.broadcast_to(bases,(n,2,13,15))],1)
    def build(_):
        result=shortest(blocked[:,None],jnp.zeros_like(blocked[:,None]),goals)
        return jnp.where(dirty[:,None,None,None],result,old['distance'])
    distance=jax.lax.cond(jnp.any(dirty),build,lambda _:old['distance'],None)
    return dict(window=window,custody=custody,age=age,origin=origin,ready=jnp.ones(n,bool),
                blocked=blocked,loose=loose_goals,loose_origin=chosen,distance=distance)


def labels(states,aux,legal,seat,danger):
    r=jnp.arange(seat.shape[0]);e=1-seat;d=aux['dagger'];empty=states.bun_carried[r,seat]<0
    allow=eligible(states.scene,states.alive[r,seat],aux['terminal'],states.trapped[r,seat],
        states.bun_carried[r,seat],d['window'][r,seat],d['custody'][r,seat],d['age'][r,seat])
    a=jnp.any(d['loose'][r,seat],(-2,-1))&(states.bun_respawn[r,e]>0)
    stock=states.bun_stored[r,e,e]>0
    kind=jnp.where(empty,jnp.where(a,jnp.where(d['loose_origin'][r,seat]==seat,4,1),jnp.where(stock,2,0)),3)
    channel=jnp.where((kind==1)|(kind==4),seat,jnp.where(kind==2,2+seat,4+seat))
    distance=d['distance'][r,channel]
    danger=jnp.maximum(danger,(states.core.blast_linger>0).astype(jnp.float32))
    target,reachable=action_set(states.pos[r,seat],distance,legal,danger,d['blocked'],states.fuse)
    cell=jnp.clip(states.pos[r,seat].astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    dist=distance[r,cell[:,0],cell[:,1]]
    at_goal=(dist==0)&legal[:,4]&(danger[r,cell[:,0],cell[:,1]]==0)&(states.fuse[r,cell[:,0],cell[:,1]]<=0)
    target=jnp.where(at_goal[:,None],jax.nn.one_hot(jnp.full(seat.shape,4),5),target)
    gate=allow&(kind>0)&(reachable|at_goal)
    active=allow&(kind>0)
    in_cage=env_base_contact(states,seat)
    # No future state, curriculum ID, or oracle action enters Actor inputs.
    return jax.lax.stop_gradient(jnp.concatenate([target,gate[:,None],kind[:,None],
        active[:,None],(dist<INF)[:,None],in_cage[:,None],d['custody'][r,seat,None],
        d['origin'][r,seat,None]],-1).astype(jnp.float32))


def env_base_contact(states,seat):
    from jax_bomb import je07_bun_env as env
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    cell=jnp.clip(states.pos[jnp.arange(seat.size),seat].astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    return bases[1-seat,cell[:,0],cell[:,1]]


def buffer(capacity=CAPACITY,additions=ADDITIONS,obs_shape=(24,13,15),global_size=24):
    return dict(obs=jnp.zeros((capacity,)+obs_shape),gv=jnp.zeros((capacity,global_size)),
        legal=jnp.zeros((capacity,5),bool),target=jnp.zeros((capacity,5)),
        valid=jnp.zeros(capacity,bool),kind=jnp.zeros(capacity,jnp.int32),
        round=jnp.zeros(capacity,jnp.int32),cursor=jnp.int32(0),additions=jnp.int32(additions))


def aggregate(old,batch,key,generation,offset=OFFSET):
    flatten=lambda x:x.reshape((-1,)+x.shape[2:])
    obs,gv,legal,ctx=map(flatten,(batch[0],batch[1],batch[6][0],batch[-1]))
    c=ctx[:,offset:offset+12];valid=c[:,5]>0
    count=old['obs'].shape[0];k=ADDITIONS if count==CAPACITY else 2
    _,ix=jax.lax.top_k(jnp.where(valid,jax.random.uniform(key,(valid.size,))+1.,-1.),k)
    new=dict(obs=obs[ix],gv=gv[ix],legal=legal[ix],target=c[ix,:5],valid=valid[ix],
             kind=c[ix,6].astype(jnp.int32),round=jnp.full(k,generation,jnp.int32))
    # Half current learner states, half previous learner rounds; fallback only
    # when a half has no eligible samples. Replay never enters PPO or Critic.
    def choose(data,random_key,num):
        mass=data['valid'].astype(jnp.float32)
        ids=jax.random.choice(random_key,mass.size,(num,),p=jnp.where(mass.sum()>0,mass/jnp.maximum(mass.sum(),1),jnp.ones_like(mass)/mass.size))
        return {name:data[name][ids] for name in new}
    k1,k2=jax.random.split(key);size=32 if count==CAPACITY else 2
    recent=choose(new,k1,size);history=choose(old,k2,size)
    has_old=old['valid'].any();has_new=new['valid'].any()
    history=jax.tree.map(lambda h,n:jnp.where(has_old,h,n),history,recent)
    recent=jax.tree.map(lambda n,h:jnp.where(has_new,n,h),recent,history)
    replay=jax.tree.map(lambda a,b:jnp.concatenate([a,b],0),recent,history)
    slots=(old['cursor']+jnp.arange(k))%count
    updated={name:old[name].at[slots].set(new[name]) for name in new}
    updated.update(cursor=(old['cursor']+k)%count,additions=old['additions'])
    return updated,replay,dict(visited=jnp.float32(valid.size),eligible=c[:,7].sum(),query=valid.sum(),
        covered=new['valid'].sum(),aggregate=updated['valid'].sum(),
        history_fraction=jnp.mean((replay['round']<generation)&replay['valid']),
        new_fraction=jnp.mean((replay['round']==generation)&replay['valid']),
        path_a=(((c[:,6]==1)|(c[:,6]==4))&valid).sum(),
        path_a_enemy=((c[:,6]==1)&valid).sum(),path_a_own_recovery=((c[:,6]==4)&valid).sum(),path_b=((c[:,6]==2)&valid).sum(),
        returning=((c[:,6]==3)&valid).sum(),cage_contact=((c[:,9]>0)&(c[:,7]>0)).sum(),
        actual_custody=(c[:,10]>0).sum(),noneligible_label=((~(c[:,7]>0))&valid).sum())


def replay_loss(params,replay):
    from jax_bomb import jax_train as train
    move=train.net_forward(params,'transformer',replay['obs'],replay['gv'])[0]
    return supervised_loss(move,replay['legal'],replay['target'],replay['valid'],COEFFICIENT,TEMPERATURE)[0]
