"""Real parent-state on-policy wiring probe, CPU only and no training credit.

This diagnostic cannot authorize K2 from a zero intervention. Parent states
are genuine trajectories; no carried package or death is fabricated.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAXBOMB_RULE='bun')
import json
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import optax
from scripts.population_run_state import atomic_json,sha,lock

OUT=Path('runs/r4_combat_20261031/k2_wiring_acceptance')


def run():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'OWNER.lock');began=time.monotonic()
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    from scripts.bun_spawn_side import install as ownside
    ownside()
    from scripts import combat_item_engine as T,r3_items_collector as M,r4_k2_collector as K
    T.configure();M.configure(True,True);K.install(M)
    cp=Path('runs/r3_items_route_20261030/formal/r3_a_ownside_c000001/u004002')
    cm=json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
    payload=pickle.load((cp/'state.pkl').open('rb'));count=16;steps=64
    single=jax.tree.map(lambda x:j.asarray(x[0]),payload['states'])
    # Fixed deterministic spread, selected before observing any K2 reward.
    ids=np.linspace(0,len(single.t)-1,count,dtype=np.int32)
    st=jax.tree.map(lambda x:x[ids],single);ax=jax.tree.map(lambda x:j.asarray(x[0])[ids],payload['aux'])
    ax=dict(ax,combat_mix=dict(ax['combat_mix'],k2=K.state(count)))
    keys=jax.random.split(jax.random.PRNGKey(2026331102),count)
    params=jax.tree.map(j.asarray,payload['params']);ring=jax.tree.map(j.asarray,payload['combat_ring'][0]);fixed=jax.tree.map(j.asarray,payload['combat_fixed_actor'])
    groups={'bot':np.arange(count),'current':np.array([],int),'ring':np.array([],int),'fixed':np.array([],int)}
    # Keep the exact real native state and its complete historical reward aux;
    # the diagnostic opponent is the existing native rule bot, declared here.
    choose=jax.jit(lambda s,a,k:M.choose(params,ring,fixed,s,a,k,groups))
    transition=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,0.))
    chunks=[];rows=[]
    atomic_json(OUT/'PROTOCOL.json',dict(status='FROZEN',parent=cm,ids=ids.tolist(),steps=steps,
        environment_steps=count*steps,seed=2026331102,policy='frozen parent categorical on genuine saved states',
        opponent='existing rule bot; probe only, not new training configuration',
        no_fabricated_carriers=True,not_added_to_formal_budget=True,
        sources={n:sha(n) for n in ['scripts/r4_k2_collector.py','scripts/r4_k2_carrier_reward.py']},unix=time.time()))
    for tick in range(steps):
        nk,ke,act,data=choose(st,ax,keys)
        after,na,r,done,info,parts,metrics=transition(st,ax,act,ke)
        jax.block_until_ready(r)
        o,g,mm,bm,own,lp,v=data;seat=ax['combat_mix']['learner_seat'];sel=j.arange(count)
        nv=M.prediction(params,after,seat,na['combat_mix']['bomb_threat']['r3'])[2]
        extra=metrics['r4_k2_actual_extra']
        row=dict(tick=tick+1,actual_extra=np.asarray(extra).tolist(),
            forced=np.asarray(metrics['r4_k2_forced']).tolist(),
            generic=np.asarray(metrics['r4_k2_generic_kill_pay']).tolist(),
            realized=np.asarray(metrics['r4_k2_forced_realized']).tolist(),
            custody=np.asarray(info['custody_delta'][sel,seat]).tolist(),
            reward=np.asarray(r[sel,seat]).tolist(),terminal=np.asarray(na['terminal']).tolist())
        rows.append(row);atomic_json(OUT/'PROGRESS.json',dict(ticks=tick+1,nonzero_ticks=sum(any(x!=0 for x in q['actual_extra']) for q in rows),unix=time.time()))
        # Episodes already ended remain a ledger tail, not invented resets.
        chunks.append((o,g,own,lp,v,nv,r[sel,seat],done,mm,bm,extra))
        st,ax,keys=after,na,nk
    data=tuple(j.stack([b[i] for b in chunks]) for i in range(11))
    o,g,act,lp,v,nv,reward,done,mm,bm,extra=data
    adv=M.E.train.compute_gae(reward,v,nv,done,.995,.95)
    without=M.E.train.compute_gae(reward-extra,v,nv,done,.995,.95)
    assert float(j.abs(adv-without).sum())>0,'K2 intervention never reached GAE; coverage BLOCKED'
    batch=tuple(x.reshape((-1,)+x.shape[2:]) for x in (o,g,act,lp,adv,adv+v,mm,bm))
    from scripts.ppo_scale_longrun_train import pure_ppo_objective,norms
    loss,grads=jax.jit(jax.value_and_grad(lambda p:pure_ppo_objective(p,batch)[0]))(params)
    jax.block_until_ready(grads)
    opt=optax.adam(7.5e-5,eps=1e-8);adam=opt.init(params)
    updates,adam=opt.update(grads,adam,params);after=optax.apply_updates(params,updates)
    delta=norms(params,after)
    ability=[float(j.linalg.norm(a)) for path,a in jax.tree_util.tree_flatten_with_path(grads['actor'])[0]
        if any(getattr(k,'key',None) in ('bomb','ab','ability') for k in path)]
    atomic_json(OUT/'RAW.json',rows)
    result=dict(status='PASS',genuine_parent_states=True,on_policy_categorical=True,teacher=False,
        ticks=steps,environment_steps=count*steps,loss=float(loss),update_l2=delta,
        actual_extra_absolute_sum=float(j.abs(extra).sum()),gae_delta_l1=float(j.abs(adv-without).sum()),
        actor_gradient_l2=float(j.sqrt(sum(j.square(a).sum() for a in jax.tree.leaves(grads['actor'])))),
        ability_gradient_candidates=ability,raw_sha256=sha(OUT/'RAW.json'),wall_seconds=time.monotonic()-began,
        limitation='CPU wiring diagnostic, not an R4 formal update or capability evaluation',unix=time.time())
    assert min(delta.values())>0 and all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(after))
    atomic_json(OUT/'ACCEPTANCE.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',error=type(error).__name__,reason=str(error),unix=time.time()));raise
