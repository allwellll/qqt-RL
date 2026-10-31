"""Custody revision 2. Process-local, never loaded by R2-a or historical runs.

The user explicitly selected conservative *undiscounted* differences. This is
not discounted PBRS and carries no policy-invariance claim. Geometry is a
deterministic function of state, including topology; it has no resettable gain.
"""
import inspect
import jax
import jax.numpy as j

VERSION='custody-graded-conservative-v2'
KAPPA=1.
_INSTALLED=False


def graded_phi(home,enemy_carry,loose,progress_loose,progress_home):
    return KAPPA*j.where(home,0.,j.where(enemy_carry,-.30-.50*j.clip(progress_home,0.,1.),
        j.where(loose,-.10-.10*j.clip(progress_loose,0.,1.),-.80)))


def delta(before,after,terminal=False):
    # Delivered/timeout terminal does not refund a failed defense. No tail pay.
    return j.where(terminal,0.,after-before)


def cache_state(n):
    return dict(ready=j.zeros(n,bool),blocked=j.zeros((n,13,15),bool),
        target=j.full((n,2,2),-1,j.int32),fields=j.full((n,4,13,15),32767,j.int32))


def potential(states,cache):
    """Wall/brick/pushable shortest paths, normalized by cage separation.

    An own carrier is treated as a moving loose package until return. No path
    means zero progress. All geometry choices are frozen in the amendment.
    """
    from scripts.carrier_mechanism_engine import env
    from scripts.je08_bun_potential import shortest,INF
    n=len(states.t);rows=j.arange(n)[:,None];team=j.arange(2)[None];enemy=1-team
    # Reward-only milligrid canonicalization avoids float32/float64 selecting
    # opposite tiles at an exact grid line. Native movement is untouched.
    cells=j.clip(j.floor(j.round(states.pos*1000.)/1000.).astype(j.int32),j.array([0,0]),j.array([12,14]))
    own=states.bun_carried==team
    locations=j.argmax(states.bun_loose.reshape(n,195,2),axis=1)
    loose=(states.bun_loose.sum((1,2))>0)|own
    target=j.stack([locations//15,locations%15],-1)
    target=j.where(own[...,None],cells,target)
    target=j.where(loose[...,None],target,-1)
    safe=j.maximum(target,0)
    obj=j.zeros((n,2,13,15),bool).at[rows,team,safe[...,0],safe[...,1]].set(loose)
    goals=j.stack([env._base_mask(0),env._base_mask(1)])
    allgoals=j.concatenate([j.broadcast_to(goals,(n,2,13,15)),obj],1)
    blocked=states.wall|states.brick|states.core.pushable
    dirty=~cache['ready']|j.any(blocked!=cache['blocked'],(-2,-1))|j.any(target!=cache['target'],(-2,-1))
    fields=jax.lax.cond(dirty.any(),lambda _:j.where(dirty[:,None,None,None],
        shortest(blocked[:,None],j.zeros_like(blocked[:,None]),allgoals),cache['fields']),lambda _:cache['fields'],None)
    # Minimum actual cage-to-cage route, not Euclidean or a single step score.
    span=j.min(j.where(goals[None],fields[:,[1,0]],INF),(-2,-1))
    span=j.maximum(span,1)
    enemycell=cells[:,::-1]
    dh=fields[rows,enemy,enemycell[...,0],enemycell[...,1]]
    dl=fields[rows,team+2,enemycell[...,0],enemycell[...,1]]
    q=j.where((dh<INF)&(span<INF),j.clip(1.-dh/span,0.,1.),0.)
    p=j.where((dl<INF)&(span<INF),j.clip(1.-dl/span,0.,1.),0.)
    home=states.bun_stored[rows,team,team]>0
    carry=states.bun_carried[:,::-1]==team
    phi=graded_phi(home,carry,loose,p,q)
    return phi,dict(ready=j.ones(n,bool),blocked=blocked,target=target,fields=fields),dict(p=p,q=q,home=home,enemy_carry=carry,loose=loose)


def victim_factors(states):
    from scripts.carrier_mechanism_engine import env
    pos=j.floor(states.pos[:,::-1]).astype(j.int32);base=j.asarray(env._BUN_BASES)[None]
    distance=(j.maximum(base-pos,0)+j.maximum(pos-(base+2),0)).sum(-1)
    return j.where((states.bun_carried[:,::-1]>=0)|(distance<=3),0.,1.)


def install():
    global _INSTALLED
    if _INSTALLED:return
    from scripts import bun_native_v2_reward as old,je10_ops as ops,je04_bun_ops as legacy
    from scripts import je10_control_credit as credit,je05_bun_transport as transport
    old.install(False)
    # Exclude defensive kills at factual accrual, not after delayed payment.
    fn=legacy.retention_reward;scope=dict(fn.__globals__);code=inspect.getsource(fn)
    assert code.count("16.*info['surviving_kill']")==1
    exec(code.replace("16.*info['surviving_kill']","16.*info['surviving_kill']*info['custody_kill_factor']"),scope)
    legacy.retention_reward=scope['retention_reward']
    native=credit.native_events
    def events(before,after,info,terminal):
        return dict(native(before,after,info,terminal),custody_factor=victim_factors(before)[:,::-1])
    credit.native_events=events
    scope=dict(credit.__dict__);code=inspect.getsource(credit.transition)
    code=code.replace("(owner_hot*attributed[...,None]).sum(1)*ENTRY_REWARD","(owner_hot*attributed[...,None]*e['custody_factor'][...,None]).sum(1)*ENTRY_REWARD")
    code=code.replace("extra=residual_hot.sum(1)*RESIDUAL_REWARD","extra=(residual_hot*e['custody_factor'][...,None]).sum(1)*RESIDUAL_REWARD")
    exec(code,scope);credit.transition=scope['transition']
    # Both pending and immediate transport capture use conserved origin.
    tr=transport.transport_reward
    def transport_reward(e,s,*args,**kwargs):
        # Stage F flattens [batch,physical-seat] in row-major order.
        team=j.arange(len(e['token']))%2
        hostile=e['token']!=team
        safe=dict(e,capture=e['capture']&hostile,steal=e['steal']&hostile,carried=e['carried']&hostile,
            before_region=j.where(hostile,e['before_region'],-1),after_region=j.where(hostile,e['after_region'],-1))
        return tr(safe,s,*args,**kwargs)
    transport.transport_reward=transport_reward
    rs=ops.reward_state
    def reward_state(n):return dict(rs(n),custody_v2=cache_state(n))
    ops.reward_state=reward_state
    original=ops.combined_reward
    def combined(before,after,info,aux,settled,coefficient=1.,bridge_config=None):
        # Recover is own-loose pickup in the real simulator, not a combat save.
        clean=dict(info,custody_kill_factor=victim_factors(before),recover=j.zeros_like(info['recover']))
        reward,new,parts,metrics=original(before,after,clean,aux,settled,coefficient,bridge_config)
        ids=j.arange(2)[None]
        recovering=(before.bun_carried==ids)|(after.bun_carried==ids)|info['return_bun']|info['recover']
        # The empty-hand leg to an own-origin loose package is also recovery,
        # not an offensive enemy-stock trip. Do not stack old route credit.
        own_loose=(before.bun_loose.sum((1,2))>0)|(after.bun_loose.sum((1,2))>0)
        recovering|=own_loose&(before.bun_carried!=1-ids)&(after.bun_carried!=1-ids)
        # Own-origin travel is entirely represented by the custody potential.
        # Historical route offsets are retained, but their cash delta is zero.
        cancel=j.where(recovering,metrics['route_shaped_reward'],0.)
        reward=reward-cancel
        k=ops.COMPONENTS.index('route_shaped_reward');parts=parts.at[...,k].add(-cancel)
        metrics['route_shaped_reward']=metrics['route_shaped_reward']-cancel
        phi0,c0,_=potential(before,aux['custody_v2'])
        phi1,c1,observed=potential(after,c0)
        first=~aux['terminal'];terminal=info['h_route_terminal']
        custody=delta(phi0,phi1,terminal[:,None])*first[:,None]
        terminal_pay=j.where(terminal[:,None]&first[:,None],
            20.*(info['winner'][:,None]==ids)-20.*(info['winner'][:,None]==1-ids),0.)
        reward=reward+custody+terminal_pay
        new['custody_v2']=c1
        metrics.update(custody_phi_before=phi0,custody_phi_after=phi1,custody_delta=custody,
            custody_terminal_pay=terminal_pay,custody_own_route_cancel=cancel,custody_home=observed['home'],custody_q=observed['q'],custody_p=observed['p'])
        info.update({k:v for k,v in metrics.items() if k.startswith('custody_')})
        # Named diagnostics are separate; preserve the existing parts ABI.
        return reward,new,parts.at[...,-1].set(reward),metrics
    ops.combined_reward=combined
    _INSTALLED=True
