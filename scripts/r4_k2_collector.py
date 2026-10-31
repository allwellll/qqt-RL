# 实验中 / EXPERIMENTAL: opt-in only; activation requires frozen gates.
"""K2-only factual event/GAE wiring. Never imported by K1/K3 workers."""
import jax
import jax.numpy as j
from scripts import r4_k2_carrier_reward as credit


def state(n):
    return dict(advance=credit.state(n),pending_batch=j.full(n,-1,j.int32),
        pending_age=j.full(n,-1,j.int32),pending_used=j.zeros(n,bool),
        trap_batch=j.full(n,-1,j.int32),life_age=j.zeros(n,j.int32))


def source_batch(before,after,info,aux,seats):
    """Unique actual exploding source hitting the victim's physical body.

    Mixed/ambiguous batches and later contact popping stay unknown (-1).
    No placement request, logit, or merely nearby source is promoted to proof.
    """
    from scripts.carrier_mechanism_engine import env
    n=len(seats);rows=j.arange(n);enemy=1-seats
    birth=aux['combat_mix']['birth']
    pending=aux['combat_mix']['k2']['pending_batch']
    # The selected birth may consist of multiple genuine same-tick sources.
    exploded=(before.fuse>0)&(after.fuse==0)
    selected=(birth==pending[:,None,None])&exploded&(before.core.owner==seats[:,None,None])
    def one(st,at,mask):
        _,h,v=env.base._rays(mask,st.fuse>0,st.core.bomb_blast,st.wall,st.brick,return_dirs=True)
        return env.base._is_hit_by_explosion(at[None],h,v,env.base.RADIUS)[0]
    hit=jax.vmap(one)(before,after.pos[rows,enemy],selected)
    other=jax.vmap(one)(before,after.pos[rows,enemy],exploded&~selected)
    sole=info['causal_damage_source'][rows,enemy,seats]& (info['causal_damage_source'][rows,enemy].sum(-1)==1)
    # Player-level causal uniqueness is insufficient: two different own
    # batches can hit together. Reject these instead of inventing provenance.
    return j.where(hit&~other&sole&(pending>=0),pending,-1)


def evidence(before,after,info,aux,seats):
    from scripts.bomb_threat_r1_progress_reward import native_safe_exits,native_safety
    n=len(seats);rows=j.arange(n);enemy=1-seats
    p=j.clip(j.floor(before.pos[rows,seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    installed=info['bomb_placed'][rows,seats]&(after.fuse[rows,p[:,0],p[:,1]]>0)
    def adjudicate(_):
        without=after._replace(core=after.core._replace(fuse=after.fuse.at[rows,p[:,0],p[:,1]].set(
            j.where(installed,0,after.fuse[rows,p[:,0],p[:,1]]))))
        old,_=native_safe_exits(without,seats);new,_=native_safe_exits(after,seats)
        safety=native_safety()
        safe=jax.vmap(lambda s,a:safety.analyze_selected_actions(s,a).selected_survivable)(before,info['h_joint_actions'])[rows,seats]
        return installed&old.any(-1)&~new.any(-1)&safe
    forced=jax.lax.cond(installed.any(),adjudicate,lambda _:j.zeros(n,bool),None)
    return forced,j.where(forced,before.t,-1)


def settle(s,*,forced,batch,kill_batch,causal_kill,enemy_carry,enemy_died,eligible,terminal,terminal_close=None,scale=1.):
    """Pure event settlement ABI, used identically by transition and tests."""
    flush=terminal if terminal_close is None else terminal_close
    advance,carrier_pay,cm=credit.advance(s['advance'],forced=forced,source_batch=batch,
        enemy_carrying_own=enemy_carry,eligible=eligible,causal_kill=causal_kill,
        kill_batch=kill_batch,enemy_died=enemy_died,terminal=flush)
    start=forced&eligible&~s['pending_used']&~flush&(batch>=0)
    pending=j.where(start,batch,s['pending_batch']);age=j.where(start,0,j.where(s['pending_age']>=0,s['pending_age']+1,-1))
    matched=(pending>=0)&causal_kill&(kill_batch==pending)&(age<=20)&eligible&~enemy_carry
    bonus=j.where(causal_kill&eligible&~enemy_carry&~terminal,1/3,0.)+j.where(matched&~terminal,1/15,0.)
    closed=enemy_died|flush|(age>=20)
    new=dict(s,advance=advance,pending_batch=j.where(closed,-1,pending),pending_age=j.where(closed,-1,age),
        pending_used=j.where(enemy_died,False,s['pending_used']|start))
    payment=(carrier_pay+bonus)*scale
    return new,payment,dict(r4_k2_forced=forced.astype(j.int32),r4_k2_actual_extra=payment,
        r4_k2_generic_kill_pay=j.where(causal_kill&eligible&~enemy_carry&~terminal,1/3,0.),
        r4_k2_forced_realized=matched.astype(j.int32),**{'r4_k2_'+k:v for k,v in cm.items()})


def install(module):
    if getattr(module,'_k2_installed',False):return
    credit.install()
    original_metadata,original_transition=module.metadata,module.transition
    def metadata(st,*args,**kwargs):
        return dict(original_metadata(st,*args,**kwargs),k2=state(len(st.t)))
    def transition(before,aux,actions,keys,coefficient):
        after,na,reward,done,info,parts,metrics=original_transition(before,aux,actions,keys,coefficient)
        seats=aux['combat_mix']['learner_seat'];rows=j.arange(len(seats));enemy=1-seats;s=aux['combat_mix']['k2']
        forced,batch=evidence(before,after,info,aux,seats)
        source=source_batch(before,after,info,aux,seats)
        trapped=(before.trapped[rows,enemy]==0)&(after.trapped[rows,enemy]>0)
        trap_batch=j.where(trapped,source,s['trap_batch'])
        popped=info['native_trap_popper'][rows,enemy]>=0
        actual_batch=j.where(popped,-1,j.where(before.trapped[rows,enemy]>0,trap_batch,source))
        epos=j.floor(before.pos[rows,enemy]).astype(j.int32)
        distance=j.abs(epos-j.floor(before.bun_spawn_pos[rows,enemy]).astype(j.int32)).sum(-1)
        from scripts.carrier_mechanism_engine import env
        in_cage=jax.vmap(lambda p:env._base_mask(0)[p[0],p[1]]|env._base_mask(1)[p[0],p[1]])(j.clip(epos,j.array([0,0]),j.array([12,14])))
        age=s['life_age']+before.alive[rows,enemy].astype(j.int32)
        eligible=(before.core.invuln[rows,enemy]==0)&(age>=15)&(distance>2)&~in_cage&before.alive[rows,seats]&(before.trapped[rows,seats]==0)
        terminal=aux['terminal'];died=info['death'][rows,enemy]
        ns,pay,m=settle(s,forced=forced,batch=batch,kill_batch=actual_batch,
            causal_kill=info['surviving_kill'][rows,seats],enemy_carry=before.bun_carried[rows,enemy]==seats,
            enemy_died=died,eligible=eligible,terminal=terminal,terminal_close=na['terminal'])
        ns=dict(ns,trap_batch=j.where(died,-1,trap_batch),life_age=j.where(died,0,age))
        reward=reward.at[rows,seats].add(pay);parts=parts.at[rows,seats,-1].set(reward[rows,seats])
        na=dict(na,combat_mix=dict(na['combat_mix'],k2=ns));metrics.update(m)
        return after,na,reward,done,info,parts,metrics
    module.metadata=module._impl.metadata=metadata
    module.transition=module._impl.transition=transition
    from types import ModuleType
    source=module.src.replace('            diagnostics.update(policy_diag)',
        "            diagnostics.update(policy_diag)\n            diagnostics.update({k:v.sum(0) for k,v in sm.items() if k.startswith('r4_k2_')})")
    impl=ModuleType('k2_private_collector');exec(compile(source,__file__,'exec'),impl.__dict__)
    for name in ('fresh','metadata','transition','inputs','prediction'):setattr(impl,name,getattr(module._impl,name))
    module._impl=module._base._impl=impl
    for name in ('choose','choose_with_diagnostics'):setattr(module,name,getattr(impl,name))
    module._k2_installed=True
