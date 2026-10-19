"""Jittable retention-v3 reward ledger, funnel rollout, scene KL and PFSP.

All reward history and adaptive statistics are explicit checkpointed arrays.
Historical web evaluation code is deliberately not modified.
"""
import jax
import jax.numpy as jnp
import optax
from jax_bomb import je07_bun_env as env,jax_train as train,bun_jax_bots
from scripts import je05_bun_transport as transport, je07_bun_scenes as scenes
from scripts import je04_bun_ops as legacy_ops
from scripts import je05_bun_ops as previous_ops, je07_bun_curriculum as curriculum
from scripts import je07_bun_ops as predecessor_ops,je08_bun_potential as potential
from scripts.je03_bun_selfplay_ops_20261012 import balanced_indices

TASK_EVENTS=('route_success','enter_enemy_half','enter_enemy_base','steal','carry_cross_home','return_bun','capture','recover')
BASE_COMPONENTS=('lesson_task','global_safety','settled_kill','settled_shaping','clawback','blocked_route','total')
FUNNEL=('valid','approach','contact','legal_attack','request_bomb','placed_bomb','threat','pressure','kill','settled_survival','break_route','enter_base','steal','carry','return','capture','use_item','death','self','trade')
TRANSPORT_COMPONENTS=('transport_shape','transport_clawback','transport_capture')
COMPONENTS=BASE_COMPONENTS+TRANSPORT_COMPONENTS+('drop_bonus','drop_clawback','combined_total')
POTENTIAL_METRICS=('drop_approach_positive','drop_approach_negative','drop_approach_net','carry_home_positive','carry_home_negative','carry_home_net',
    'potential_stationary','potential_phase_start','potential_phase_end','potential_cap_hit','potential_topology_rebase','potential_clawback')
POTENTIAL_METRICS+=tuple(phase+'_'+name for phase in ['drop_approach','carry_home'] for name in
    ['stationary','phase_start','phase_end','cap_hit','topology_rebase','clawback'])
COMPONENTS=COMPONENTS[:-1]+POTENTIAL_METRICS+('combined_total',)
BASE_FUNNEL=FUNNEL
FUNNEL=BASE_FUNNEL+('sugar_window_steal','dead_window_steal','unconditional_steal','drop','repeat_pickup','progress25','progress50','progress75','home_region','reverse_region','region_advance','drop_recovery','unattributed_loose','own_recover','assist_tick')
FUNNEL=FUNNEL+POTENTIAL_METRICS
BASE=jnp.array([.08,.05,.05,.08,.08,.02,.08,.08,.07,.04,.07,.04,.03,.05,.05,.13],jnp.float32)
COMBAT=jnp.array([False,False,False,False,True,True,True,True,False,False,False,False,True,False,False,True])

def curriculum_weights(counts):
    target=jnp.array([9,12,15,15,15,12,15,15,15,15])
    return scenes.curriculum_weights(jnp.stack([counts[:,0],counts[jnp.arange(10),target]],-1))

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
    aux['drop']=jax.tree.map(lambda x:jnp.broadcast_to(x,(n,)+x.shape),curriculum.drop_state())
    aux['assist']=curriculum.assist_state(n)
    aux['potential']=potential.state(n)
    aux['routing']=potential.routing_state(n)
    return aux


def combined_reward(states,next_states,info,aux,settled,coefficient=1.):
    # Preserve JE04 combat/safety settlement but replace legacy transport event
    # bonuses: no unconditional steal or column-based crossing rewards.
    taskinfo=dict(info)
    for name in ['steal','carry_cross_home','return_bun','capture','enter_enemy_half','enter_enemy_base']:
        taskinfo[name]=jnp.zeros_like(info[name])
    oldaux={k:v for k,v in aux.items() if k not in ('transport','drop','assist','potential','routing')}
    base,new,parts=legacy_ops.retention_reward(taskinfo,oldaux,settled)
    if coefficient==0:
        reward,old,parts,metrics=predecessor_ops.combined_reward(states,next_states,info,
            {k:v for k,v in aux.items() if k not in ('potential','routing')},settled)
        metrics.update({name:jnp.zeros_like(reward) for name in POTENTIAL_METRICS})
        return reward,dict(old,potential=aux['potential'],routing=aux['routing']),jnp.concatenate(
            [parts[...,:-1],jnp.zeros(parts.shape[:-1]+(len(POTENTIAL_METRICS),)),reward[...,None]],-1),metrics
    before,unused=scenes.lookup(states);after,home=scenes.lookup(next_states)
    team=jnp.arange(2)[None,:];enemy=1-team
    origin=jnp.where(next_states.bun_carried>=0,next_states.bun_carried,states.bun_carried)
    enemycarry=next_states.bun_carried==enemy
    # Theft begins AFTER Bun interaction, so movement into the cage cannot also
    # receive pre-custody transport progress on the steal tick.
    before=jnp.where(info['steal']|info['pickup_loose']|info['native_transfer'],after,before)
    events=dict(steal=info['steal'],eligible=info['steal_window_trap']|info['steal_window_dead'],
        capture=info['capture'],death=info['death'],drop=info['drop']|info['carrier_death'],
        trade=jnp.broadcast_to(info['mutual_death'][:,None],info['death'].shape),pickup=info['pickup_loose'],
        settled=jnp.broadcast_to(settled[:,None],info['death'].shape),carried=enemycarry,before_region=before,after_region=after,home_half=home,token=origin)
    # Region/highwater/quarter milestones are replaced, not stacked with the
    # dense potential. Conditional steal .75 shares the same carry cap4.
    events.update(before_region=jnp.full_like(before,-1),after_region=jnp.full_like(after,-1),home_half=jnp.zeros_like(home))
    flatten=lambda x:x.reshape((-1,)+x.shape[2:])
    r,ledger,metrics=transport.transport_reward(jax.tree.map(flatten,events),jax.tree.map(flatten,aux['transport']))
    reshape=lambda x:x.reshape((states.pos.shape[0],2)+x.shape[1:])
    new['transport']=jax.tree.map(reshape,ledger);metrics=jax.tree.map(reshape,metrics)
    drop_events=dict(drop=info['drop'],drop_origin=info['drop_origin'],drop_credit=info['drop_credit'],
        pickup=info['pickup_loose'],pickup_origin=info['pickup_origin'],carried=next_states.bun_carried,
        death=info['death'],trade=info['mutual_death'],capture=info['capture'],settled=settled)
    drop_reward,drop_state,drop_metrics=jax.vmap(curriculum.drop_reward)(drop_events,aux['drop'])
    new['drop']=drop_state;new['assist']=aux['assist'];metrics.update(drop_metrics)
    route,changed=potential.refresh(states,next_states,info,aux['routing'],drop_state,jnp.asarray(scenes.goals))
    rows=jnp.arange(states.pos.shape[0])[:,None];teams=jnp.arange(2)[None]
    oldcell=jnp.clip(states.pos.astype(jnp.int32),jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
    cell=jnp.clip(next_states.pos.astype(jnp.int32),jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
    common=dict(bad=info['death']|info['mutual_death'][:,None],settled=jnp.broadcast_to(settled[:,None],info['death'].shape),
        pickup=info['pickup_loose'],capture=info['capture'],topology_changed=jnp.broadcast_to(changed[:,None],info['death'].shape))
    target=route['target'][:,::-1];target_safe=jnp.clip(target,jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
    enemy=jnp.broadcast_to(1-teams,states.bun_carried.shape)
    loose=next_states.bun_loose[rows,target_safe[...,0],target_safe[...,1],enemy]>0
    priorloose=states.bun_loose[rows,target_safe[...,0],target_safe[...,1],enemy]>0
    nonce=drop_state['nonce'][rows,enemy]
    eligible=loose&(nonce>0)&(drop_state['credit'][rows,enemy]==teams)&jnp.all(target>=0,-1)&next_states.alive&(next_states.bun_carried<0)&~drop_state['cycle_paid'][rows,teams,enemy]
    # New drop gets registered only after native death/loose/pickup phases.
    # Locked per-origin target/nonce never switches to a nearest loose object.
    d=route['distance'][rows,teams,oldcell[...,0],oldcell[...,1]]
    afterd=route['distance'][rows,teams,cell[...,0],cell[...,1]]
    drop_e=dict(common,eligible=eligible,identity=nonce,before=d.astype(jnp.float32),after=afterd.astype(jnp.float32),
        boundary=info['pickup_loose']|info['native_transfer']|info['capture'])
    dr,drop_phase,dm=jax.vmap(jax.vmap(lambda e,s:potential.phase_reward(e,s,.05,2.)))(drop_e,aux['potential']['drop'])
    d=route['distance'][rows,teams+2,oldcell[...,0],oldcell[...,1]]
    afterd=route['distance'][rows,teams+2,cell[...,0],cell[...,1]]
    acquired=(states.bun_carried!=enemy)&enemycarry
    carry_e=dict(common,bad=common['bad']|info['drop']|info['carrier_death'],eligible=enemycarry&next_states.alive,identity=enemy+1,before=d.astype(jnp.float32),after=afterd.astype(jnp.float32),
        boundary=acquired|info['pickup_loose']|info['steal']|info['native_transfer']|info['capture']|info['drop'])
    cap=jnp.maximum(0,4.-new['transport']['episode_gross'])
    cr,carry_phase,cm=jax.vmap(jax.vmap(lambda e,s,cap:potential.phase_reward(e,s,.0625,cap,allow_restart=True)))(carry_e,aux['potential']['carry'],cap)
    new['routing']=route;new['potential']=dict(drop=drop_phase,carry=carry_phase)
    metrics.update(drop_approach_positive=dm['positive'],drop_approach_negative=dm['negative'],drop_approach_net=dm['net'],
        carry_home_positive=cm['positive'],carry_home_negative=cm['negative'],carry_home_net=cm['net'])
    for name in ['stationary','phase_start','phase_end','cap_hit','topology_rebase','clawback']:
        metrics['potential_'+name]=dm[name].astype(jnp.float32)+cm[name].astype(jnp.float32)
        metrics['drop_approach_'+name]=dm[name].astype(jnp.float32)
        metrics['carry_home_'+name]=cm[name].astype(jnp.float32)
    r=r.reshape(base.shape);total=base+r+drop_reward+dr+cr
    extra=jnp.stack([metrics['shaping'],metrics['clawback'],metrics['capture'],
        drop_metrics['drop_bonus'],drop_metrics['drop_clawback']],-1)
    extras=jnp.stack([metrics[k] for k in POTENTIAL_METRICS],-1)
    return total,new,jnp.concatenate([parts,extra,extras,total[...,None]],-1),metrics

def factor_kl(move,ability,ref_move,ref_ability,mm,bm):
    def kl(a,b,mask):
        a=jax.nn.log_softmax(jnp.where(mask,a,-1e9));b=jax.nn.log_softmax(jnp.where(mask,b,-1e9));p=jnp.exp(a)
        return jnp.sum(p*(a-b),-1)
    # Slice before softmax => use_item's logit has EXACTLY zero KL gradient.
    move_kl=kl(move,ref_move,mm)
    bomb_kl=kl(ability[:,:2],ref_ability[:,:2],bm[:,:2])
    full_kl=kl(ability,ref_ability,bm)
    return jnp.stack([move_kl,bomb_kl,full_kl],-1)

def collect(params,states,aux,key,pool,frozen,course,pfsp,num_steps,coefficient=1.,assist_probability=.7):
    if coefficient==0:
        result=predecessor_ops.collect(params,states,{k:v for k,v in aux.items() if k not in ('potential','routing')},key,pool,frozen,course,pfsp,num_steps,assist_probability=assist_probability)
        st,a,k,b,stats=result
        counts,poolcounts,parts,means=stats
        counts=jnp.pad(counts,[(0,0),(0,0),(0,len(POTENTIAL_METRICS))])
        parts=jnp.concatenate([parts[:-1],jnp.zeros(len(POTENTIAL_METRICS)),parts[-1:]])
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
        rewards,newaux,parts,transport_metrics=combined_reward(states,next_states,info,aux,settled,coefficient)
        newaux['terminal']=deferred
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
        bonus=jnp.concatenate([bonus,jnp.stack([tm(k) for k in POTENTIAL_METRICS],-1)],-1)
        feat=jnp.concatenate([feat,bonus],-1)
        counts=counts.at[states.scene,kinds].add(feat)
        selected_counts=feat[:,jnp.array([0,9,17,18,1,6])]*(kinds==3)[:,None]
        poolcounts=poolcounts.at[member].add(selected_counts.sum(0))
        component+=parts[rows,seat].sum(0)
        fresh=scenes.init_batch(kreset,n,course)
        next_states=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),next_states,fresh)
        empty=reward_state(n)
        newaux=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),newaux,empty)
        newaux['assist']={name:jnp.asarray(value,dtype=aux['assist'][name].dtype)
                          for name,value in newaux['assist'].items()}
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

def ppo(params,adam,batch,key,ema,opt,frozen,minibatch,axis_name='dev',gradient_accumulation=1):
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
    assert total%(half*gradient_accumulation)==0
    indices=indices.reshape((-1,gradient_accumulation,minibatch))
    def update(carry,micros):
        p,a=carry
        def accumulate(acc,ii):
            gradients,metrics=acc
            (l,diag),grad=jax.value_and_grad(loss,has_aux=True)(p,ii)
            return (jax.tree.map(lambda x,y:x+y,gradients,grad),metrics+jnp.concatenate([l[None],diag])),None
        zero=jax.tree.map(jnp.zeros_like,p)
        (grad,metrics),_=jax.lax.scan(accumulate,(zero,jnp.zeros(12)),micros)
        grad=jax.tree.map(lambda x:x/gradient_accumulation,grad)
        metrics=metrics/gradient_accumulation;l,diag=metrics[0],metrics[1:]
        if axis_name:grad=jax.lax.pmean(grad,axis_name)
        gn=optax.global_norm(grad);change,a=opt.update(grad,a,p);p=optax.apply_updates(p,change)
        return (p,a),jnp.concatenate([jnp.array([l,gn]),diag])
    (params,adam),diagnostic=jax.lax.scan(update,(params,adam),indices)
    ema=jax.tree.map(lambda e,p:.995*e+.005*p,ema,params)
    return params,adam,ema,diagnostic.mean(0)
