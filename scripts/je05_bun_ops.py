"""Jittable retention-v3 reward ledger, funnel rollout, scene KL and PFSP.

All reward history and adaptive statistics are explicit checkpointed arrays.
Historical web evaluation code is deliberately not modified.
"""
import jax
import jax.numpy as jnp
import optax
from jax_bomb import je05_bun_env as env,jax_train as train,bun_jax_bots
from scripts import je05_bun_transport as transport, je05_bun_scenes as scenes
from scripts import je04_bun_ops as legacy_ops
from scripts.je03_bun_selfplay_ops_20261012 import balanced_indices

TASK_EVENTS=('route_success','enter_enemy_half','enter_enemy_base','steal','carry_cross_home','return_bun','capture','recover')
BASE_COMPONENTS=('lesson_task','global_safety','settled_kill','settled_shaping','clawback','blocked_route','total')
FUNNEL=('valid','approach','contact','legal_attack','request_bomb','placed_bomb','threat','pressure','kill','settled_survival','break_route','enter_base','steal','carry','return','capture','use_item','death','self','trade')
TRANSPORT_COMPONENTS=('transport_shape','transport_clawback','transport_capture')
COMPONENTS=BASE_COMPONENTS+TRANSPORT_COMPONENTS+('combined_total',)
BASE_FUNNEL=FUNNEL
FUNNEL=BASE_FUNNEL+('sugar_window_steal','dead_window_steal','unconditional_steal','drop','repeat_pickup','progress25','progress50','progress75','home_region','reverse_region','region_advance')
BASE=jnp.array([.08,.05,.05,.08,.08,.02,.08,.08,.07,.04,.07,.04,.03,.05,.05,.13],jnp.float32)
COMBAT=jnp.array([False,False,False,False,True,True,True,True,False,False,False,False,True,False,False,True])

def curriculum_weights(counts):
    target=jnp.array([9,12,15,15,15,12,15,15,15])
    return scenes.curriculum_weights(jnp.stack([counts[:,0],counts[jnp.arange(9),target]],-1))

def pfsp_weights(counts):
    # columns valid, safe settled kill, death, self, approach, threat
    valid=jnp.maximum(counts[:,0],1)
    success=(counts[:,1]+1)/(counts[:,1]+counts[:,2]+2)
    deficit=1-jnp.minimum(1,20*counts[:,5]/valid)
    score=.05+(1-success)**2+.5*counts[:,3]/jnp.maximum(1,counts[:,2])+ .25*deficit
    return score/score.sum()

def reward_state(n):
    aux=legacy_ops.reward_state(n)
    aux['transport']=jax.tree.map(lambda x:x.reshape((n,2)+x.shape[1:]),transport.transport_state(n*2))
    return aux


def combined_reward(states,next_states,info,aux,settled,coefficient=1.):
    # Preserve JE04 combat/safety settlement but replace legacy transport event
    # bonuses: no unconditional steal or column-based crossing rewards.
    taskinfo=dict(info)
    for name in ['steal','carry_cross_home','return_bun','capture','enter_enemy_half','enter_enemy_base']:
        taskinfo[name]=jnp.zeros_like(info[name])
    oldaux={k:v for k,v in aux.items() if k!='transport'}
    base,new,parts=legacy_ops.retention_reward(taskinfo,oldaux,settled)
    if coefficient==0:
        reward,old,original=legacy_ops.retention_reward(info,oldaux,settled)
        return reward,dict(old,transport=aux['transport']),jnp.concatenate([original,jnp.zeros(parts.shape[:-1]+(3,)),reward[...,None]],-1),{}
    before,unused=scenes.lookup(states);after,home=scenes.lookup(next_states)
    team=jnp.arange(2)[None,:];enemy=1-team
    origin=jnp.where(next_states.bun_carried>=0,next_states.bun_carried,states.bun_carried)
    enemycarry=next_states.bun_carried==enemy
    # Theft begins AFTER Bun interaction, so movement into the cage cannot also
    # receive pre-custody transport progress on the steal tick.
    before=jnp.where(info['steal'],after,before)
    events=dict(steal=info['steal'],eligible=info['steal_window_trap']|info['steal_window_dead'],
        capture=info['capture'],death=info['death'],drop=info['drop']|info['carrier_death'],
        trade=jnp.broadcast_to(info['mutual_death'][:,None],info['death'].shape),pickup=info['pickup_loose'],
        settled=jnp.broadcast_to(settled[:,None],info['death'].shape),carried=enemycarry,before_region=before,after_region=after,home_half=home,token=origin)
    flatten=lambda x:x.reshape((-1,)+x.shape[2:])
    r,ledger,metrics=transport.transport_reward(jax.tree.map(flatten,events),jax.tree.map(flatten,aux['transport']))
    reshape=lambda x:x.reshape((states.pos.shape[0],2)+x.shape[1:])
    new['transport']=jax.tree.map(reshape,ledger);metrics=jax.tree.map(reshape,metrics)
    r=r.reshape(base.shape);total=base+r
    extra=jnp.stack([metrics['shaping'],metrics['clawback'],metrics['capture']],-1)
    return total,new,jnp.concatenate([parts,extra,total[...,None]],-1),metrics

def factor_kl(move,ability,ref_move,ref_ability,mm,bm):
    def kl(a,b,mask):
        a=jax.nn.log_softmax(jnp.where(mask,a,-1e9));b=jax.nn.log_softmax(jnp.where(mask,b,-1e9));p=jnp.exp(a)
        return jnp.sum(p*(a-b),-1)
    # Slice before softmax => use_item's logit has EXACTLY zero KL gradient.
    move_kl=kl(move,ref_move,mm)
    bomb_kl=kl(ability[:,:2],ref_ability[:,:2],bm[:,:2])
    full_kl=kl(ability,ref_ability,bm)
    return jnp.stack([move_kl,bomb_kl,full_kl],-1)

def collect(params,states,aux,key,pool,frozen,course,pfsp,num_steps,coefficient=1.):
    if coefficient==0:
        # Caller binds legacy env/reset and provides the unmodified JE04 state
        # and aux. Whole algorithm bypass avoids extra RNG or arithmetic.
        return legacy_ops.collect(params,states,aux,key,pool,frozen,course,pfsp,num_steps)
    n=states.pos.shape[0];rows=jnp.arange(n);seat=rows%2;idx=rows+seat*n;oppidx=rows+(1-seat)*n
    key,kindkey,poolkey=jax.random.split(key,3)
    kinds=jax.random.choice(kindkey,4,(n,),p=jnp.array([.30,.20,.20,.30]))
    # One league member per device rollout, independently sampled; all devices
    # contribute to checkpointed PFSP stats. Avoid 11 network forwards/tick.
    member=jax.random.choice(poolkey,len(pool['tok'][0]),p=pfsp)
    selected=jax.tree.map(lambda x:x[member],pool)
    counts=jnp.zeros((9,4,len(FUNNEL)))
    poolcounts=jnp.zeros((11,6));component=jnp.zeros((len(COMPONENTS),))
    def body(carry,_):
        states,aux,key,kinds,counts,poolcounts,component=carry
        key,ka,kg,ks,kl,kr,kstep,kreset=jax.random.split(key,8)
        obs=train.both_perspectives(states);gv=train.both_states(states);mm,bm=train.both_masks(states)
        # During residual settlement actor can still escape; forbid new bombs
        # for BOTH seats so terminal chains cannot be prolonged indefinitely.
        terminal=jnp.concatenate([aux['terminal'],aux['terminal']])
        bm=bm.at[:,1:].set(bm[:,1:] & ~terminal[:,None])
        acts,lps,vals=train.sample_actions(params,'transformer',obs,(mm,bm),ka,state=gv)
        def sample(q,k):return train.sample_actions(q,'transformer',obs[oppidx],(mm[oppidx],bm[oppidx]),k,state=gv[oppidx])[0]
        g0=sample(frozen['g0'],kg);s1=sample(frozen['stage1_teacher'],ks);league=sample(selected,kl)
        # Frozen responsive JAX training bots, explicitly NOT web Hunter.
        rule=bun_jax_bots.rule_bot_actions(states,1-seat,mm[oppidx],bm[oppidx],kr,2+rows%3)
        # Guard own cage, chase enemy carriers and intercept respawn routes;
        # only override unconstrained safe movement with no live hazard.
        opponent=1-seat;cells=jnp.clip(states.pos[rows,opponent].astype(jnp.int32),jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
        neighbors=jnp.clip(cells[:,None]+jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]]),jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
        variant=scenes.OPEN_SCENE[states.scene].astype(jnp.int32)
        home_distance=scenes.DISTANCE[variant[:,None],opponent[:,None],neighbors[:,:,0],neighbors[:,:,1]]
        chase=jnp.sum(jnp.abs(neighbors-states.pos[rows,seat,None]),-1)
        intercept=jnp.sum(jnp.abs(neighbors-states.bun_spawn_pos[rows,seat,None]),-1)
        objective=jnp.where((rows%3==0)[:,None],home_distance,jnp.where((rows%3==1)[:,None],chase,intercept))
        directed=jnp.argmin(jnp.where(mm[oppidx],objective,1e8),-1)
        safe=~jnp.any(states.fuse>0,axis=(1,2))&~jnp.any(states.core.blast_linger>0,axis=(1,2))&(rule[:,1]==0)&(states.trapped[rows,opponent]==0)
        rule=rule.at[:,0].set(jnp.where(safe,directed,rule[:,0]))
        other=jnp.stack([rule,g0,s1,league])[kinds,rows]
        a0=jnp.where((seat==0)[:,None],acts[:n],other);a1=jnp.where((seat==1)[:,None],acts[n:],other)
        joint=jnp.stack([a0,a1],1)
        next_states,natural_done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(states,joint,jax.random.split(kstep,n))
        settled=~(jnp.any(next_states.fuse>0,axis=(1,2))|jnp.any(next_states.core.blast_linger>0,axis=(1,2))|jnp.any(next_states.trapped>0,axis=1))
        deferred=aux['terminal']|natural_done|(next_states.t>=scenes.HORIZONS[states.scene])
        done=deferred & settled
        # Slow glue on an actor's current carry-home route is a factual native
        # item event; do not reward arbitrary trap placement or route blocking.
        info['blocked_own_route']=(joint[:,:,1]==2)&(states.held_item==env.ITEM_SLOW_GLUE)&(states.bun_carried>=0)
        rewards,newaux,parts,transport_metrics=combined_reward(states,next_states,info,aux,settled)
        newaux['terminal']=deferred
        dist=jnp.sum(jnp.abs(states.pos[:,0]-states.pos[:,1]),-1)
        after=jnp.sum(jnp.abs(next_states.pos[:,0]-next_states.pos[:,1]),-1)
        take=lambda name:info[name][rows,seat]
        valid=states.alive[rows,seat]
        feat=jnp.stack([valid,(after<dist)&valid,(dist<=3)&valid,bm[idx,1]&valid,
            acts[idx,1]==1,take('bomb_placed'),take('threat_bomb_placed'),take('threat_bomb_placed')&aux['previous_threat'][rows,seat],
            take('surviving_kill'),parts[rows,seat,2]>0,take('route_success')|take('bridge_route_open'),take('enter_enemy_base'),
            take('steal'),states.bun_carried[rows,seat]>=0,take('return_bun')|take('carry_cross_home'),take('capture'),acts[idx,1]==2,
            take('death'),take('own_bomb_defeat'),info['mutual_death']],-1).astype(jnp.float32)
        tm=lambda name:transport_metrics[name][rows,seat]
        bonus=jnp.stack([take('steal')&take('steal_window_trap'),take('steal')&take('steal_window_dead'),
            tm('unconditional_steal'),take('drop')|take('carrier_death'),tm('repeat_pickup'),
            (tm('progress_marks')&1)>0,(tm('progress_marks')&2)>0,(tm('progress_marks')&4)>0,
            tm('home'),tm('reverse'),tm('region_advance')],-1).astype(jnp.float32)
        feat=jnp.concatenate([feat,bonus],-1)
        counts=counts.at[states.scene,kinds].add(feat)
        selected_counts=feat[:,jnp.array([0,9,17,18,1,6])]*(kinds==3)[:,None]
        poolcounts=poolcounts.at[member].add(selected_counts.sum(0))
        component+=parts[rows,seat].sum(0)
        fresh=scenes.init_batch(kreset,n,course)
        next_states=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),next_states,fresh)
        empty=reward_state(n)
        newaux=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),newaux,empty)
        key,kk=jax.random.split(key);newkinds=jax.random.choice(kk,4,(n,),p=jnp.array([.30,.20,.20,.30]))
        kinds=jnp.where(done,newkinds,kinds)
        context=jnp.stack([states.scene.astype(jnp.float32),take('danger_exposure').astype(jnp.float32)],-1)
        # Only the actual learner actions/log-prob enter PPO. Opponent rows
        # never leak into Actor gradients; whole transition/seat remains paired.
        data=(obs[idx],gv[idx],acts[idx],lps[idx],vals[idx],rewards[rows,seat],done,(mm[idx],bm[idx]),context)
        return (next_states,newaux,key,kinds,counts,poolcounts,component),data
    (states,aux,key,_,counts,poolcounts,component),data=jax.lax.scan(body,(states,aux,key,kinds,counts,poolcounts,component),None,length=num_steps)
    obs,gv,acts,lps,vals,reward,done,masks,context=data
    tailobs=train.both_perspectives(states)[idx];tailgv=train.both_states(states)[idx]
    _,_,tail,_=train.net_forward(params,'transformer',tailobs,tailgv)
    adv=train.compute_gae(reward,vals,jnp.concatenate([vals[1:],tail[None]]),done,.995,.95)
    return states,aux,key,(obs,gv,acts,lps,adv,adv+vals,masks,context),(counts,poolcounts,component,jnp.array([reward.mean(),done.mean()]))

def ppo(params,adam,batch,key,ema,opt,frozen,minibatch,axis_name='dev'):
    obs,gv,acts,old,ad,ret,masks,context=batch
    total=ad.size;half=minibatch//2
    norm=(ad-ad.mean())/(ad.std()+1e-8)
    flatten=lambda x:x.reshape((-1,)+x.shape[2:])
    obs,gv,acts,context=map(flatten,(obs,gv,acts,context));mm,bm=map(flatten,masks)
    old=old.reshape(-1);ad=norm.reshape(-1);ret=ret.reshape(-1)
    ka,kc=jax.random.split(key)
    indices=jnp.concatenate([jax.random.permutation(ka,total).reshape(-1,half),jax.random.permutation(kc,total).reshape(-1,half)],1)
    def loss(p,ii):
        o=obs[ii];s=gv[ii];m=mm[ii];b=bm[ii]
        move,ability,_,vl=train.net_forward(p,'transformer',o,s)
        lm=jax.nn.log_softmax(jnp.where(m,move,-1e9));lb=jax.nn.log_softmax(jnp.where(b,ability,-1e9))
        a=acts[ii[:half]];logp=lm[jnp.arange(half),a[:,0]]+lb[jnp.arange(half),a[:,1]]
        ratio=jnp.exp(logp-old[ii[:half]])
        pol=jnp.maximum(-ad[ii[:half]]*ratio,-ad[ii[:half]]*jnp.clip(ratio,.9,1.1)).mean()
        value=train.hl_gauss_value_loss(vl[half:],ret[ii[half:]])
        entropy=-(jnp.sum(jnp.exp(lm[:half])*lm[:half],-1)+jnp.sum(jnp.exp(lb[:half])*lb[:half],-1)).mean()
        lessons=context[ii[:half],0].astype(jnp.int32);danger=context[ii[:half],1]>0
        combat=(lessons==0)|danger
        klstats=[];penalty=0.
        for name,strong,weak in [('g0',.10,.025),('stage1_teacher',.04,.01)]:
            rm,rb,_,_=train.net_forward(frozen[name],'transformer',o[:half],s[:half])
            kl=factor_kl(move[:half],ability[:half],rm,rb,m[:half],b[:half])
            penalty+=jnp.mean(jnp.where(combat,strong,weak)*(kl[:,0]+kl[:,1]))
            klstats.append(kl.mean(0))
        approxkl=jnp.mean((ratio-1)-(logp-old[ii[:half]]))
        return pol+.5*value-.005*entropy+penalty,jnp.concatenate([jnp.array([pol,value,entropy,approxkl,penalty]),*klstats])
    def update(carry,ii):
        p,a=carry;(l,diag),grad=jax.value_and_grad(loss,has_aux=True)(p,ii)
        if axis_name:grad=jax.lax.pmean(grad,axis_name)
        gn=optax.global_norm(grad);change,a=opt.update(grad,a,p);p=optax.apply_updates(p,change)
        return (p,a),jnp.concatenate([jnp.array([l,gn]),diag])
    (params,adam),diagnostic=jax.lax.scan(update,(params,adam),indices)
    ema=jax.tree.map(lambda e,p:.995*e+.005*p,ema,params)
    return params,adam,ema,diagnostic.mean(0)
