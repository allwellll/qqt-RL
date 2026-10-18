"""Integrated control credit, continuous PBRS and learner-state route DAgger."""
import jax
import jax.numpy as jnp
from scripts import je09_stage_h8_ops as h8
from scripts import je09_stage_h5c_ops as base
from scripts import je09_stage_h6_ops as h6
from scripts import je09_stage_h5c_dagger as dagger
from scripts import je09_route_pbrs as pbrs
from scripts import je10_control_credit as credit
from scripts import je10_route_oracle as oracle

for _name in dir(h8):
    if not _name.startswith('_'): globals()[_name]=getattr(h8,_name)

PARENT_REWARD=h8.combined_reward
PARENT_STATE=h8.reward_state
PARENT_DAGGER_UPDATE=dagger.update
PARENT_PHASE_VALUES=pbrs.phase_values
COMPONENTS=h8.COMPONENTS[:-1]+credit.METRICS+pbrs.METRICS+('combined_total',)
ROUTE_COEFFICIENT=.5
TRACE_NAMES=('reward_total','reward_control','reward_route','phi_before','phi_after',
    'route_phase','route_identity','route_distance_before','route_distance_after',
    'control_victim_event','control_owner','control_episode','state_tick')


def reward_state(n):
    aux=PARENT_STATE(n)
    aux['control_credit']=credit.state(n)
    aux['route_pbrs']=pbrs.state(n)
    # Shared pre-stock auxiliary state.
    from scripts.prestock_training_state import state as prestock_state
    aux['prestock']=prestock_state(n)
    return aux


def combined_reward(states,after,info,aux,settled,coefficient=1.,bridge_config=None):
    reward,new,parts,metrics=PARENT_REWARD(states,after,credit.parent_reward_events(states,info),aux,settled,coefficient,bridge_config)
    events=credit.native_events(states,after,info,aux['terminal']|info['h_route_terminal'])
    ledger,extra,cm=credit.transition(events,aux['control_credit'])
    new['control_credit']=ledger
    reward=reward+extra
    parts=jnp.concatenate([parts[...,:-1],jnp.stack([cm[k] for k in credit.METRICS],-1),reward[...,None]],-1)
    reward,new,parts,rm=pbrs.apply_reward(states,after,info,aux,reward,new,parts,ROUTE_COEFFICIENT)
    parts=parts.at[...,-1].set(reward)
    metrics.update(cm);metrics.update(rm)
    metrics['control_reward_total']=extra
    return reward,new,parts,metrics


def earliest_fragment(ctx,offset=48):
    d=ctx[...,offset:offset+12];kind=d[...,6].astype(jnp.int32)
    objective=jnp.where((kind==3)&(d[...,9]>0),4,jnp.where(kind==3,3,
        jnp.where(kind==2,2,jnp.where((kind==1)|(kind==4),1,0))))
    return jnp.where(d[...,7]>0,objective,0).astype(jnp.int32)


def collect(*args,**kwargs):
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    old_step=env.step
    def step(*a,**k):
        after,done,info=old_step(*a,**k)
        terminal=done|info['capture'].any(-1)|(after.t>=scenes.HORIZONS[after.scene.astype(jnp.int32)])
        return after,done,dict(info,h_route_terminal=terminal)
    env.step=step
    base.combined_reward=combined_reward
    base.reward_state=reward_state
    base.COMPONENTS=COMPONENTS
    dagger.labels=oracle.labels
    dagger.eligible=oracle.eligible
    h6.earliest_fragment=earliest_fragment
    try:
        states,aux,key,batch,stats=h8.collect(*args,**kwargs)
    finally:
        env.step=old_step
    obs,gv,acts,logp,adv,returns,masks,ctx=batch
    # Trace is emitted at every real learner transition before reset. Zero-
    # value reward GAE is a labelled approximation, never a counterfactual Q.
    values=ctx[...,79:79+len(TRACE_NAMES)]
    extra=values[...,1]+values[...,2]
    done=ctx[...,47]>0
    zeros=jnp.zeros_like(extra)
    extra_adv=train.compute_gae(extra,zeros,zeros,done,.995,.95)
    ctx=jnp.concatenate([ctx,extra_adv[...,None],(adv-extra_adv)[...,None]],-1)
    return states,aux,key,(obs,gv,acts,logp,adv,returns,masks,ctx),stats


def update_custody(before,after,info,aux,old):
    updated=PARENT_DAGGER_UPDATE(before,after,info,aux,old)
    r=jnp.arange(before.alive.shape[0])[:,None];team=jnp.arange(2)[None];enemy=1-team
    native_window=~before.alive[:,::-1]|(before.trapped[:,::-1]>0)
    stock=(before.bun_stored[r,enemy,enemy]>after.bun_stored[r,enemy,enemy])&info['steal']
    origin=jnp.maximum(info['pickup_origin'],0)
    causal_loose=info['pickup_loose']&(aux['drop']['credit'][r,origin]==team)&(aux['drop']['nonce'][r,origin]>0)
    start=(before.bun_carried<0)&(after.bun_carried>=0)&before.alive&after.alive
    acquired=start&native_window&(stock|causal_loose|info['native_transfer'])
    updated['window']=~after.alive[:,::-1]|(after.trapped[:,::-1]>0)
    updated['custody']=(old['custody']|acquired)&after.alive&(after.bun_carried>=0)
    updated['age']=jnp.where(acquired,0,updated['age'])
    updated['origin']=jnp.where(acquired,jnp.where(stock,2,jnp.where(after.bun_carried==team,4,1)),updated['origin'])
    return updated


def safe_phase_values(states,route,drop):
    phase,identity,base,distance,provenance=PARENT_PHASE_VALUES(states,route,drop)
    separation=jnp.abs(states.pos[:,0]-states.pos[:,1]).sum(-1)[:,None]
    carrying=states.bun_carried>=0
    allowed=oracle.route_allowed(carrying,states.alive[:,::-1],states.trapped[:,::-1],separation)
    return (jnp.where(allowed,phase,0),identity,jnp.where(allowed,base,0.),distance,provenance)


def install():
    pbrs.phase_values=safe_phase_values
    dagger.labels=oracle.labels
    dagger.update=update_custody
    dagger.eligible=oracle.eligible
    dagger.action_set=oracle.action_set
    dagger.supervised_loss=oracle.supervised_loss
    dagger.COEFFICIENT=.2
    from scripts import prestock_joint_replay as joint_replay
    dagger.buffer=joint_replay.buffer
    from scripts import phase_balanced_joint_replay as phase_balanced
    dagger.aggregate=phase_balanced.aggregate
    dagger.replay_loss=joint_replay.replay_loss
    base.reward_state=reward_state
    base.COMPONENTS=COMPONENTS
    # Route and return teachers share the same set-mass loss and geometry.
    from scripts import je09_return_expert as teacher
    teacher.action_set=oracle.action_set
    teacher.supervised_loss=oracle.supervised_loss
