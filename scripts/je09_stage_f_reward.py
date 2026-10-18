"""Stage F only gates pre-pickup payments; unchanged combat/carry ledgers."""
from scripts.je09_bun_ops import *
from scripts.je09_stage_f_mode import transition_allowed

def combined_reward(states,next_states,info,aux,settled,coefficient=1.,bridge_config=None):
    mode=transition_allowed(states,next_states)
    # Preserve JE04 combat/safety settlement but replace legacy transport event
    # bonuses: no unconditional steal or column-based crossing rewards.
    taskinfo,suppressed=fixture_retention_info(states,info)
    for name in ['steal','carry_cross_home','return_bun','capture','enter_enemy_half','enter_enemy_base']:
        taskinfo[name]=jnp.zeros_like(info[name])
    oldaux={k:v for k,v in aux.items() if k not in ('transport','drop','assist','potential','routing')}
    base,new,parts=legacy_ops.retention_reward(taskinfo,oldaux,settled)
    if coefficient==0:
        reward,old,parts,metrics=predecessor_ops.combined_reward(states,next_states,info,
            {k:v for k,v in aux.items() if k not in ('potential','routing')},settled)
        metrics.update({name:jnp.zeros_like(reward) for name in REWARD_METRICS})
        return reward,dict(old,potential=aux['potential'],routing=aux['routing'],bridge=aux['bridge']),jnp.concatenate(
            [parts[...,:-1],jnp.zeros(parts.shape[:-1]+(len(REWARD_METRICS),)),reward[...,None]],-1),metrics
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
        pickup=info['pickup_loose']&(mode|(info['pickup_origin']==jnp.arange(2)[None,:])),pickup_origin=info['pickup_origin'],carried=next_states.bun_carried,
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
    eligible=mode&loose&(nonce>0)&(drop_state['credit'][rows,enemy]==teams)&jnp.all(target>=0,-1)&next_states.alive&(next_states.bun_carried<0)&~drop_state['cycle_paid'][rows,teams,enemy]
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
    new['expert']=expert.refresh(next_states,route,aux['expert'],changed)
    new['return_expert']=return_expert.refresh(next_states,aux['return_expert'],changed,jnp.asarray(scenes.goals))
    new['routing']=route;new['potential']=dict(drop=drop_phase,carry=carry_phase)
    metrics.update(drop_approach_positive=dm['positive'],drop_approach_negative=dm['negative'],drop_approach_net=dm['net'],
        carry_home_positive=cm['positive'],carry_home_negative=cm['negative'],carry_home_net=cm['net'])
    for name in ['stationary','phase_start','phase_end','cap_hit','topology_rebase','clawback']:
        metrics['potential_'+name]=dm[name].astype(jnp.float32)+cm[name].astype(jnp.float32)
        metrics['drop_approach_'+name]=dm[name].astype(jnp.float32)
        metrics['carry_home_'+name]=cm[name].astype(jnp.float32)
    r=r.reshape(base.shape);total=base+r+drop_reward+dr+cr
    # ``drop_recovery`` is the later first-pickup event.  Bridge eligibility
    # must be the preceding factual enemy drop attributed to this learner;
    # otherwise an unattributed loose pickup can open the bridge ledger.
    attributed_drop=jnp.any(info['drop'][:,None,:]
        & (info['drop_origin'][:,None,:]==enemy[:,:,None])
        & (info['drop_credit'][:,None,:]==teams[:,:,None]),axis=-1)
    bridge_info=dict(
        drop=attributed_drop,
        pickup_loose=drop_metrics['drop_recovery'],
        before_distance=d,after_distance=afterd,carried=enemycarry,
        topology_changed=common['topology_changed'],home_half=home,
        lost=info['drop']|info['carrier_death'],settled=common['settled'],
        capture=info['capture'],
        death=info['death'], self=info['own_bomb_defeat'],
        trade=jnp.broadcast_to(info['mutual_death'][:, None], info['death'].shape),
        timeout=jnp.broadcast_to((settled & (aux['terminal'] | (next_states.t>=scenes.HORIZONS[states.scene])))[:,None],info['death'].shape),
    )
    bridge_reward,bridge_state,bridge_metrics=bridge.jax_bridge_reward(bridge_info,aux['bridge'],**(bridge_config or {}))
    total=total+bridge_reward
    new['bridge']=bridge_state
    metrics.update(bridge_metrics)
    metrics['fixture_kill_suppressed']=suppressed.astype(jnp.float32)
    metrics.update(bridge_drop=bridge_metrics['bridge_events'][...,0],
        bridge_pickup=bridge_metrics['bridge_events'][...,1],bridge_switch=bridge_metrics['bridge_events'][...,2])
    metrics['progress_marks']=(bridge_metrics['bridge_events'][...,3].astype(jnp.uint8)
        +2*bridge_metrics['bridge_events'][...,4].astype(jnp.uint8))
    metrics['home']=bridge_metrics['bridge_events'][...,5]
    extra=jnp.stack([metrics['shaping'],metrics['clawback'],metrics['capture'],
        drop_metrics['drop_bonus'],drop_metrics['drop_clawback']],-1)
    extras=jnp.stack([metrics[k] for k in REWARD_METRICS],-1)
    return total,new,jnp.concatenate([parts,extra,extras,total[...,None]],-1),metrics
