"""Stage E production adapter. Original Stage C/D operators remain immutable.

Collect is source-identical apart from audit-only pickup-distance context.
PPO is source-identical apart from fixed-coefficient return coverage mixture.
"""
import jax
import jax.numpy as jnp
import optax
from jax_bomb import je07_bun_env as env,jax_train as train,bun_jax_bots
from scripts.je09_bun_ops import *
from scripts import je09_bun_scenes as scenes,je07_bun_curriculum as curriculum,je09_drop_expert as expert,je09_chase_expert as chase_expert
from scripts.je09_stage_e_coverage import teacher as return_expert
def collect(params,states,aux,key,pool,frozen,course,pfsp,num_steps,coefficient=1.,assist_probability=.7,expert_hazard_mode="global",bridge_config=None,return_config=None,chase_stage=None,chase_supervision=True):
    if coefficient==0:
        result=predecessor_ops.collect(params,states,{k:v for k,v in aux.items() if k not in ('potential','routing')},key,pool,frozen,course,pfsp,num_steps,assist_probability=assist_probability)
        st,a,k,b,stats=result
        counts,poolcounts,parts,means=stats
        counts=jnp.pad(counts,[(0,0),(0,0),(0,len(REWARD_METRICS))])
        parts=jnp.concatenate([parts[:-1],jnp.zeros(len(REWARD_METRICS)),parts[-1:]])
        return st,dict(a,potential=aux['potential'],routing=aux['routing']),k,b,(counts,poolcounts,parts,means)
    n=states.pos.shape[0];rows=jnp.arange(n);seat=rows%2;idx=rows+seat*n;oppidx=rows+(1-seat)*n
    key,kindkey,poolkey=jax.random.split(key,3)
    kinds=jax.random.choice(kindkey,4,(n,),p=jnp.array([.30,.20,.20,.30]))
    # One league member per device rollout, independently sampled; all devices
    # contribute to checkpointed PFSP stats. Avoid 11 network forwards/tick.
    member=jax.random.choice(poolkey,len(pool['tok'][0]),p=pfsp)
    selected=jax.tree.map(lambda x:x[member],pool)
    counts=jnp.zeros((10,4,len(FUNNEL)))
    poolcounts=jnp.zeros((11,6));component=jnp.zeros((len(COMPONENTS),))
    def body(carry,_):
        states,aux,key,kinds,counts,poolcounts,component=carry
        key,ka,kg,ks,kl,kr,kstep,kreset,kassist,kwalk=jax.random.split(key,10)
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
        carrying=states.bun_carried[rows,seat]==1-seat
        assist,assist_state=curriculum.assist_gate(aux['assist'],carrying,states.alive[rows,seat],
            aux['terminal']|(states.t>=scenes.HORIZONS[states.scene]),kassist,assist_probability)
        if coefficient==0:assist=jnp.zeros_like(assist)
        other=curriculum.assisted_actions(other,mm[oppidx],assist,kwalk)
        a0=jnp.where((seat==0)[:,None],acts[:n],other);a1=jnp.where((seat==1)[:,None],acts[n:],other)
        joint=jnp.stack([a0,a1],1)
        next_states,natural_done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(states,joint,jax.random.split(kstep,n))
        settled=~(jnp.any(next_states.fuse>0,axis=(1,2))|jnp.any(next_states.core.blast_linger>0,axis=(1,2))|jnp.any(next_states.trapped>0,axis=1))
        deferred=aux['terminal']|natural_done|(next_states.t>=scenes.HORIZONS[states.scene])
        done=deferred & settled
        # Slow glue on an actor's current carry-home route is a factual native
        # item event; do not reward arbitrary trap placement or route blocking.
        info['blocked_own_route']=(joint[:,:,1]==2)&(states.held_item==env.ITEM_SLOW_GLUE)&(states.bun_carried>=0)
        rewards,newaux,parts,transport_metrics=combined_reward(states,next_states,info,aux,settled,coefficient,bridge_config)
        newaux['terminal']=deferred
        if chase_stage is not None:
            newaux['chase']=chase_expert.update(next_states,newaux,aux['chase'],jnp.asarray(scenes.DISTANCE_BUCKETS)[chase_stage])
        ended=~(next_states.bun_carried[rows,seat]==1-seat)|~next_states.alive[rows,seat]|deferred
        newaux['assist']={name:jnp.asarray(jnp.where(ended,jnp.zeros_like(value),value),dtype=aux['assist'][name].dtype)
                          for name,value in assist_state.items()}
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
            tm('home'),tm('reverse'),tm('region_advance'),tm('drop_recovery'),tm('unattributed_pickup'),tm('own_recover'),assist],-1).astype(jnp.float32)
        bonus=jnp.concatenate([bonus,jnp.stack([tm(k) for k in REWARD_METRICS],-1)],-1)
        feat=jnp.concatenate([feat,bonus],-1)
        counts=counts.at[states.scene,kinds].add(feat)
        selected_counts=feat[:,jnp.array([0,9,17,18,1,6])]*(kinds==3)[:,None]
        poolcounts=poolcounts.at[member].add(selected_counts.sum(0))
        component+=parts[rows,seat].sum(0)
        fresh=scenes.init_batch(kreset,n,course,chase_stage)
        next_states=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),next_states,fresh)
        empty=reward_state(n)
        newaux=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),newaux,empty)
        newaux['assist']={name:jnp.asarray(value,dtype=aux['assist'][name].dtype)
                          for name,value in newaux['assist'].items()}
        key,kk=jax.random.split(key);newkinds=jax.random.choice(kk,4,(n,),p=jnp.array([.30,.20,.20,.30]))
        kinds=jnp.where(done,newkinds,kinds)
        labels,supervised=expert.labels(states,aux,mm[idx],seat,obs[idx,5],expert_hazard_mode)
        context=jnp.stack([states.scene.astype(jnp.float32),take('danger_exposure').astype(jnp.float32),labels.astype(jnp.float32),supervised.astype(jnp.float32)],-1)
        if return_config is not None:
            target,eligible=return_expert.labels(states,aux,mm[idx],seat,obs[idx,5],return_config['window'])
            cell=states.pos[rows,seat].astype(jnp.int32)
            dist=aux['return_expert']['distance'][rows,seat,cell[:,0],cell[:,1]]
            context=jnp.concatenate([context,target,eligible[:,None].astype(jnp.float32),dist[:,None].astype(jnp.float32),
                (cell[:,0]*env.W+cell[:,1])[:,None].astype(jnp.float32),
                return_expert.gate(states,aux,seat,return_config['window'])[:,None].astype(jnp.float32)],-1)
        if chase_stage is not None:
            target,eligible=chase_expert.labels(states,aux,mm[idx],seat,obs[idx,5])
            eligible=eligible&chase_supervision
            cell=states.pos[rows,seat].astype(jnp.int32)
            d=aux['expert']['distance'][rows,seat,cell[:,0],cell[:,1]]
            context=jnp.concatenate([context,target,eligible[:,None].astype(jnp.float32),d[:,None].astype(jnp.float32),
                aux['chase']['age'][rows,seat,None].astype(jnp.float32),aux['chase']['initial'][rows,seat,None].astype(jnp.float32)],-1)
        context=jnp.concatenate([context,aux['bridge']['start_distance'][rows,seat,None]],-1)
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

def ppo(params,adam,batch,key,ema,opt,frozen,minibatch,axis_name='dev',gradient_accumulation=1,expert_coefficient=.2,expert_temperature=1.,return_coefficient=0.,return_temperature=1.,return_replay=None,long_fraction=.5):
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
        parent_kl=jnp.float32(0.)
        if 'parent' in frozen:
            rm,rb,_,_=train.net_forward(frozen['parent'],'transformer',o[:half],s[:half])
            parent_kl=parent_retention_kl(move[:half],ability[:half],rm,rb,m[:half],b[:half],s[:half])
            penalty+=.05*parent_kl
        auxloss,expert_stats=expert.supervised_loss(move[:half],m[:half],
            context[ii[:half],2].astype(jnp.int32),context[ii[:half],3]>0,expert_coefficient,expert_temperature) if context.shape[-1]>=4 else (jnp.float32(0.),jnp.zeros(2))
        if context.shape[-1]>=22:
            auxloss,expert_stats=chase_expert.supervised_loss(move[:half],m[:half],context[ii[:half],13:18],
                context[ii[:half],18]>0,expert_coefficient,expert_temperature)
        returnloss,returnstats=return_expert.supervised_loss(move[:half],m[:half],context[ii[:half],4:9],
            context[ii[:half],9]>0,return_coefficient,return_temperature) if context.shape[-1]>=10 else (jnp.float32(0.),jnp.zeros(2))
        if return_replay is not None:
            # Same .05 coefficient; only the state distribution is broadened.
            # Replay rows never enter PPO ratio, value, reward, or observations.
            losses=[]
            for group in ['short','long']:
                ro,rg,rmask,rtarget=return_replay[group]
                ridx=ii[:half]%ro.shape[0]
                replay_move=train.net_forward(p,'transformer',ro[ridx],rg[ridx])[0]
                replay_loss,_=return_expert.supervised_loss(replay_move,rmask[ridx],rtarget[ridx],jnp.ones(half,bool),return_coefficient,return_temperature)
                origin=context[ii[:half],22]
                chosen=(origin>=24)&(origin<=36) if group=='long' else (origin>0)&(origin<24)
                online_loss,_=return_expert.supervised_loss(move[:half],m[:half],context[ii[:half],4:9],(context[ii[:half],9]>0)&chosen,return_coefficient,return_temperature)
                losses.append(.5*(online_loss+replay_loss))
            returnloss=(1-long_fraction)*losses[0]+long_fraction*losses[1]
        approxkl=jnp.mean((ratio-1)-(logp-old[ii[:half]]))
        return pol+.5*value-.005*entropy+penalty+auxloss+returnloss,jnp.concatenate([jnp.array([pol,value,entropy,approxkl,penalty]),*klstats,parent_kl[None],auxloss[None],expert_stats,returnloss[None],returnstats])
    assert total%(half*gradient_accumulation)==0
    indices=indices.reshape((-1,gradient_accumulation,minibatch))
    def update(carry,micros):
        p,a=carry
        def accumulate(acc,ii):
            gradients,metrics=acc
            (l,diag),grad=jax.value_and_grad(loss,has_aux=True)(p,ii)
            return (jax.tree.map(lambda x,y:x+y,gradients,grad),metrics+jnp.concatenate([l[None],diag])),None
        zero=jax.tree.map(jnp.zeros_like,p)
        (grad,metrics),_=jax.lax.scan(accumulate,(zero,jnp.zeros(19)),micros)
        grad=jax.tree.map(lambda x:x/gradient_accumulation,grad)
        metrics=metrics/gradient_accumulation;l,diag=metrics[0],metrics[1:]
        if axis_name:grad=jax.lax.pmean(grad,axis_name)
        gn=optax.global_norm(grad);change,a=opt.update(grad,a,p);p=optax.apply_updates(p,change)
        return (p,a),jnp.concatenate([jnp.array([l,gn]),diag])
    (params,adam),diagnostic=jax.lax.scan(update,(params,adam),indices)
    ema=jax.tree.map(lambda e,p:.995*e+.005*p,ema,params)
    return params,adam,ema,diagnostic.mean(0)
