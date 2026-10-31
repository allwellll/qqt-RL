"""CPU witnesses for A3S default parity and genuine Bank custody restoration."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time,copy
from pathlib import Path
import jax,jax.numpy as j,numpy as np,optax
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest


def main():
    from scripts import ppo_scale_longrun_train as P,bomb_threat_collector_diagnostics_v1 as old
    from scripts import bomb_threat_seat5050_collector as S,bomb_threat_seat5050_reward as SR,bomb_threat_reward as R
    from scripts import bomb_threat_iter2_recipe as recipe
    from scripts.carrier_mechanism_analysis import bank,merged
    from scripts.carrier_mechanism_stats import validate_bank_record
    from scripts.bomb_threat_runtime import install
    from scripts.checkpoint_policy_artifact import load_actor
    base=Path('runs/bomb_threat_v1');out=base/'a3s';out.mkdir(exist_ok=True)
    P.T.configure();P.C.install_provenance();install()
    cfg=json.loads((base/'iter2/TRAINING_PROTOCOL.json').read_text());items=[x for x in bank() if x['record']['split']=='train']
    for x in items:validate_bank_record(x['record'])
    bs,ba=merged(items,'states'),merged(items,'aux');n=4;seed=2026112907
    keys=jax.random.split(jax.random.PRNGKey(seed),n);spawns=json.loads(Path(cfg['spawn_manifest']).read_text())['spawn_cells']
    designated=j.zeros(n,bool);actor=load_actor(Path(cfg['policy_source']['checkpoint']));params=dict(actor=actor,critic=copy.deepcopy(actor))
    rep=lambda x:jax.tree.map(lambda a:j.asarray(a)[None],x)
    def exact(a,b):
        assert jax.tree.structure(a)==jax.tree.structure(b)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)
    before=time.monotonic();st,ax=jax.jit(old.arena_fresh)(keys,bs,ba,designated,j.asarray(spawns))
    kinds=j.full(n,3);ax=dict(ax,combat_mix=old.metadata(st,kinds,designated));weights=[0,0,0,1,0,0]
    args=(rep(params),rep(st),rep(ax),rep(keys),bs,ba,rep(actor),rep(actor))
    baseline=old.collector(2,n,seed,0.,0.,spawns,weights)(*args)
    default=S.collector(2,n,seed,0.,0.,spawns,weights,seat_fraction=0)(*args);jax.block_until_ready(default);exact(baseline,default)
    seats=j.asarray(S.seat_allocation(n,seed,.5));st2,ax2=jax.jit(S.fresh)(keys,bs,ba,designated,j.asarray(spawns),seats)
    ax2=dict(ax2,combat_mix=S.metadata(st2,kinds,designated,seats))
    mixed=S.collector(2,n,seed,0.,0.,spawns,weights)(rep(params),rep(st2),rep(ax2),rep(keys),bs,ba,rep(actor),rep(actor));jax.block_until_ready(mixed)
    assert list(np.asarray(mixed[2]['seat_ticks'])[0])==[4,4]
    assert float(np.asarray(mixed[2]['bank_seat_ticks'])[0,1])==0
    # Same physical P0 event: generalized threat projection equals frozen A3 math.
    nk,ke,acts,_=old.choose(params,actor,actor,st,ax,keys,dict(current=np.arange(n),ring=[],fixed=[],bot=[]))
    original_settle=R.settle;R.settle=recipe.settlement(5.,.6)
    try:
        standard=old.transition(st,ax,acts,ke,0.)
        withseat=dict(ax,combat_mix=dict(ax['combat_mix'],learner_seat=j.zeros(n,j.int32)))
        projected=S.transition(st,withseat,acts,ke,0.)
        exact(standard[0],projected[0]);exact(standard[2:],projected[2:])
        stripped=dict(projected[1],combat_mix={k:v for k,v in projected[1]['combat_mix'].items() if k!='learner_seat'})
        exact(standard[1],stripped)
    finally:R.settle=original_settle
    opt=optax.adam(7.5e-5,eps=1e-8);adam={k:opt.init(v) for k,v in params.items()}
    update=recipe.updater(4,opt,lambda p,b,e:P.pure_ppo_objective(p,b));rk=jax.random.split(jax.random.PRNGKey(99),1)
    a=update(rep(params),rep(adam),baseline[1],rk,.005,1.,False);b=update(rep(params),rep(adam),default[1],rk,.005,1.,False)
    jax.block_until_ready(b);exact(a,b)
    u=update(rep(params),rep(adam),mixed[1],rk,.005,1.,False);jax.block_until_ready(u)
    delta=P.norms(params,P.one(u[0][0]));assert min(delta.values())>0 and all(np.isfinite(x) for x in delta.values())
    # Replay ALL actual train-Bank acquisition events, then ensure selected Bank
    # states and complete auxiliary credit/custody bytes survive fresh unchanged.
    parity=[]
    step=jax.jit(S.T.OLD_TRANSITION)
    for x in items:
        af,aa,r,d,info,parts=step(x['before'],x['preaux'],x['acquisition_actions'],x['acquisition_envkeys'])
        exact(af,x['states']);exact(aa,x['aux'])
        assert bool(S.E.causal_pickup(x['before'],af,info)[0])
        parity.append(dict(source_episode=x['record']['source_episode'],state_digest=digest(af),aux_digest=digest(aa),real_steal=True))
    testkeys=jax.random.split(jax.random.PRNGKey(543),128);ss=j.asarray(S.seat_allocation(128,seed,.5))
    fst,fax=jax.jit(S.fresh)(testkeys,bs,ba,j.zeros(128,bool),j.asarray(spawns),ss)
    ids=np.asarray(fax['provenance_source']);selected=np.flatnonzero(ids>=0)
    assert len(selected)>0 and not np.any(np.asarray(ss)[selected])
    for i in selected:
        exact(jax.tree.map(lambda x:x[i:i+1],fst),jax.tree.map(lambda x:x[ids[i]:ids[i]+1],bs))
        for key in ba:exact(jax.tree.map(lambda x:x[i:i+1],fax[key]),jax.tree.map(lambda x:x[ids[i]:ids[i]+1],ba[key]))
    evidence=dict(status='PASS',default_zero_seat_exact_actions_env_aux_RNG_batch=True,default_zero_exact_update_Adam_loss=True,
        physical_P0_A3_reward_parts_aux_exact=True,mixed_seat_update_l2=delta,
        mixed_seat_ticks=np.asarray(mixed[2]['seat_ticks']).tolist(),bank_selected=len(selected),bank_P1=0,
        bank_full_state_and_aux_unchanged=True,real_acquisition_events=len(parity),bank=parity,
        elapsed_seconds=time.monotonic()-before,source_sha256={p:sha(p) for p in ['scripts/bomb_threat_seat5050_collector.py','scripts/bomb_threat_seat5050_reward.py']},unix=time.time())
    atomic_json(out/'CPU_PARITY.json',evidence);print(json.dumps(evidence))


if __name__=='__main__':main()
