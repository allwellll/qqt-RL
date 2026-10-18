"""Jittable retention-v3 reward ledger, funnel rollout, scene KL and PFSP.

All reward history and adaptive statistics are explicit checkpointed arrays.
Historical web evaluation code is deliberately not modified.
"""
import jax
import jax.numpy as jnp
import optax
from jax_bomb import bun_env as env,jax_train as train,bun_jax_bots
from scripts.je03_bun_selfplay_ops_20261012 import balanced_indices

TASK_EVENTS=('route_success','enter_enemy_half','enter_enemy_base','steal','carry_cross_home','return_bun','capture','recover')
COMPONENTS=('lesson_task','global_safety','settled_kill','settled_shaping','clawback','blocked_route','total')
FUNNEL=('valid','approach','contact','legal_attack','request_bomb','placed_bomb','threat','pressure','kill','settled_survival','break_route','enter_base','steal','carry','return','capture','use_item','death','self','trade')
BASE=jnp.array([.08,.05,.05,.08,.08,.02,.08,.08,.07,.04,.07,.04,.03,.05,.05,.13],jnp.float32)
COMBAT=jnp.array([False,False,False,False,True,True,True,True,False,False,False,False,True,False,False,True])

def curriculum_weights(counts):
    # Every lesson retains support. Reweight exclusively from online funnel,
    # not asynchronous CPU authority results; preserve 45% combat reset share.
    target=jnp.array([15,14,12,10,8,6,6,8,11,11,15,14,8,8,8,9])
    conversion=counts[jnp.arange(16),target]/jnp.maximum(1,counts[:,0])
    weights=BASE*jnp.clip(1.5-20*conversion,.5,1.5)
    return jnp.where(COMBAT,.45*weights/jnp.sum(jnp.where(COMBAT,weights,0)),.55*weights/jnp.sum(jnp.where(~COMBAT,weights,0)))

def pfsp_weights(counts):
    # columns valid, safe settled kill, death, self, approach, threat
    valid=jnp.maximum(counts[:,0],1)
    success=(counts[:,1]+1)/(counts[:,1]+counts[:,2]+2)
    deficit=1-jnp.minimum(1,20*counts[:,5]/valid)
    score=.05+(1-success)**2+.5*counts[:,3]/jnp.maximum(1,counts[:,2])+ .25*deficit
    return score/score.sum()

def reward_state(n):
    return dict(seen=jnp.zeros((n,2,len(TASK_EVENTS)),jnp.bool_),paid=jnp.zeros((n,2)),task_paid=jnp.zeros((n,2)),shape_paid=jnp.zeros((n,2)),
        pending_shape=jnp.zeros((n,2)),pending_kill=jnp.zeros((n,2)),dead=jnp.zeros((n,2),jnp.bool_),
        terminal=jnp.zeros(n,jnp.bool_),previous_threat=jnp.zeros((n,2),jnp.bool_))

def retention_reward(info,aux,settled,legacy=None,coefficient=1.):
    """Lesson task + global safety, added LAST, never overwritten by lessons.

    Positive chain rewards are paid only after physical bombs/flames settle.
    Episode death cancels pending credit and reclaims all paid positives.
    """
    bad=info['death'] | info['mutual_death'][:,None]
    dead=aux['dead'] | bad
    events=jnp.stack([info[name] for name in TASK_EVENTS],-1)
    task=(events & ~aux['seen']).astype(jnp.float32)@jnp.array([.5,.25,.5,1.,.5,.5,2.,.25])
    task=jnp.minimum(task,jnp.maximum(0,4-aux['task_paid']))
    task=jnp.where(dead,0.,task)
    threat=info['threat_bomb_placed'] & info['safe_bomb_placed']
    # New staggered bomb must again reduce real enemy safe moves. No tickwise
    # threat reward; one event per actual native placement/chain mark.
    pressure=threat & aux['previous_threat']
    reduction=jnp.maximum(0,info['tactical_bomb_enemy_safe_moves_before']-info['tactical_bomb_enemy_safe_moves_after'])
    shaping=.2*threat+.2*pressure+.1*(threat & (reduction>0))+.5*info['danger_safe_resolution']
    pending_shape=jnp.where(dead,0.,aux['pending_shape']+shaping)
    pending_kill=jnp.where(dead,0.,aux['pending_kill']+16.*info['surviving_kill'])
    shape=jnp.where(settled[:,None]&~dead,jnp.minimum(pending_shape,jnp.maximum(0,8-aux['shape_paid'])),0.)
    kill=jnp.where(settled[:,None]&~dead,pending_kill,0.)
    reclaim=jnp.where(bad,-aux['paid'],0.)
    safety=-12.*info['death']-16.*info['own_bomb_defeat']-12.*info['mutual_death'][:,None]-4.*info['avoidable_danger_death']-.5*info['drop']
    blocked=-.25*info.get('blocked_own_route',jnp.zeros_like(bad))
    terms=jnp.stack([task,safety,kill,shape,reclaim,blocked],-1)
    total=jnp.sum(terms,-1)
    # Exact legacy fallback used by tests and ablations; no extra floating ops.
    reward=legacy if coefficient==0 else coefficient*total
    new=dict(aux,seen=aux['seen']|events,paid=jnp.where(bad,0,aux['paid']+task+shape+kill),
        task_paid=aux['task_paid']+task,shape_paid=aux['shape_paid']+shape,dead=dead,
        pending_shape=jnp.where(settled[:,None],0,pending_shape),pending_kill=jnp.where(settled[:,None],0,pending_kill),
        previous_threat=jnp.where(settled[:,None],False,aux['previous_threat']|threat))
    return reward,new,jnp.concatenate([terms,total[...,None]],-1)

def factor_kl(move,ability,ref_move,ref_ability,mm,bm):
    def kl(a,b,mask):
        a=jax.nn.log_softmax(jnp.where(mask,a,-1e9));b=jax.nn.log_softmax(jnp.where(mask,b,-1e9));p=jnp.exp(a)
        return jnp.sum(p*(a-b),-1)
    # Slice before softmax => use_item's logit has EXACTLY zero KL gradient.
    move_kl=kl(move,ref_move,mm)
    bomb_kl=kl(ability[:,:2],ref_ability[:,:2],bm[:,:2])
    full_kl=kl(ability,ref_ability,bm)
    return jnp.stack([move_kl,bomb_kl,full_kl],-1)

def collect(params,states,aux,key,pool,frozen,course,pfsp,num_steps):
    n=states.pos.shape[0];rows=jnp.arange(n);seat=rows%2;idx=rows+seat*n;oppidx=rows+(1-seat)*n
    key,kindkey,poolkey=jax.random.split(key,3)
    kinds=jax.random.choice(kindkey,4,(n,),p=jnp.array([.30,.20,.20,.30]))
    # One league member per device rollout, independently sampled; all devices
    # contribute to checkpointed PFSP stats. Avoid 11 network forwards/tick.
    member=jax.random.choice(poolkey,len(pool['tok'][0]),p=pfsp)
    selected=jax.tree.map(lambda x:x[member],pool)
    counts=jnp.zeros((16,4,len(FUNNEL)))
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
        other=jnp.stack([rule,g0,s1,league])[kinds,rows]
        a0=jnp.where((seat==0)[:,None],acts[:n],other);a1=jnp.where((seat==1)[:,None],acts[n:],other)
        joint=jnp.stack([a0,a1],1)
        next_states,natural_done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(states,joint,jax.random.split(kstep,n))
        settled=~(jnp.any(next_states.fuse>0,axis=(1,2))|jnp.any(next_states.core.blast_linger>0,axis=(1,2)))
        deferred=aux['terminal']|natural_done
        done=deferred & settled
        # Slow glue on an actor's current carry-home route is a factual native
        # item event; do not reward arbitrary trap placement or route blocking.
        info['blocked_own_route']=(joint[:,:,1]==2)&(states.held_item==env.ITEM_SLOW_GLUE)&(states.bun_carried>=0)
        rewards,newaux,parts=retention_reward(info,aux,settled)
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
        counts=counts.at[states.lesson,kinds].add(feat)
        selected_counts=feat[:,jnp.array([0,9,17,18,1,6])]*(kinds==3)[:,None]
        poolcounts=poolcounts.at[member].add(selected_counts.sum(0))
        component+=parts[rows,seat].sum(0)
        fresh=env.init_batch(kreset,n,course)
        next_states=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),next_states,fresh)
        empty=reward_state(n)
        newaux=jax.tree.map(lambda a,b:jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)),b,a),newaux,empty)
        key,kk=jax.random.split(key);newkinds=jax.random.choice(kk,4,(n,),p=jnp.array([.30,.20,.20,.30]))
        kinds=jnp.where(done,newkinds,kinds)
        context=jnp.stack([states.lesson.astype(jnp.float32),take('danger_exposure').astype(jnp.float32)],-1)
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
        combat=COMBAT[lessons]|danger
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
