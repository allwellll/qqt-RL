"""Bounded frozen Bank-ratio PPO; one GPU slot or explicit serial fallback."""
from __future__ import annotations
import argparse,json,os,pickle,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/authorized_campaign_20261026/bank_ratio_ab_v1')
CP=Path('runs/authorized_campaign_20261026')
PROTO=CP/'protocols/bank_ratio_ab_v1.json'


def read(p):return json.loads(Path(p).read_text())
def notify(kind,value):atomic_json(B/'notifications'/f'{time.time_ns()}_{kind}.json',dict(kind=kind,**value,unix=time.time()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['control','reduced']);ap.add_argument('seed',type=int)
    ap.add_argument('--devices',type=int,default=1);ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    out=B/'formal'/f'{a.arm}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
    os.environ.update(JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR=str(B/'cache'/f'd{a.devices}_{a.arm}'))
    import numpy as np,jax,jax.numpy as j,optax
    from scripts import ppo_scale_longrun_train as P
    from scripts.carrier_mechanism_analysis import bank,merged,shard
    from scripts.carrier_mechanism_stats import validate_bank_record
    from scripts.campaign_paired_train_canonical import canonical_collector
    from scripts.terminal_checkpoint_restore import digest
    guard=lock(out/'worker.lock') if not a.verify else lock(out/'verify.lock')
    assert jax.local_device_count()==a.devices
    proto=read(PROTO);seal=read(B/'SOURCE_SEAL.json')
    for p,h in seal['files'].items():assert sha(p)==h,p
    ratio=proto['arms'][a.arm]['bank_probability'];P.T.configure();P.C.install_provenance(ratio)
    items=[x for x in bank() if x['record']['split']=='train']
    for item in items:validate_bank_record(item['record'])
    bs,ba=merged(items,'states'),merged(items,'aux')
    source=proto['sources'][str(2026102951+proto['paired_seeds'].index(a.seed))]
    src=Path(source['checkpoint']);assert sha(src/'state.pkl')==source['state_sha256']
    checkpoints=sorted(out.glob('u*/COMMITTED.json'))
    if checkpoints:
        cp=checkpoints[-1].parent;marker=read(cp/'COMMITTED.json')
        assert sha(cp/'state.pkl')==marker['state_sha256'] and sha(cp/'policy.pt')==marker['policy_sha256']
        with (cp/'state.pkl').open('rb') as f:x=pickle.load(f)
        start=x['update']
    else:
        with (src/'state.pkl').open('rb') as f:x=pickle.load(f)
        start=0
    def reshard(tree):
        return jax.tree.map(lambda v:np.asarray(v).reshape((a.devices,128//a.devices)+np.asarray(v).shape[2:]),tree)
    st,ax=reshard(x['states']),reshard(x['aux'])
    keys=reshard(x['keys']) if checkpoints else shard(np.asarray(jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(a.seed),i))(np.arange(128,dtype=np.uint32))),a.devices)
    if a.verify:
        token=out/f'NEXT_STEP_u{start:06d}.json'
        # COMMITTED seals state before the read-only next-step calculation.
        # A verifier may start in that interval; wait for the matching token.
        deadline=time.monotonic()+600
        while not token.exists():
            if time.monotonic()>deadline:raise TimeoutError('Next-step token not committed')
            time.sleep(2)
        exp=read(token)
        state,aux,key=(P.one(v) for v in (st,ax,keys))
        nk,ke,act,_=jax.jit(P.C.E.choose)(x['params'],state,aux,key)
        nxt=jax.jit(P.T.transition)(state,aux,act,ke)
        assert digest((nk,ke,act,nxt))==exp['digest'];assert digest(x['adam'])==exp['Adam_digest']
        atomic_json(out/'RESTORE.json',dict(passed=True,fresh_process=True,next_step_env_aux_RNG_exact=True,Adam_exact=True,checkpoint=str(cp),update=start));return
    rep=lambda t:jax.tree.map(lambda v:j.stack([j.asarray(v)]*a.devices),t)
    p,adam=rep(x['params']),rep(x['adam'])
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    collect=canonical_collector(P.C.collector(128,reset_telemetry=True),out/'INITIAL_PLACEMENT.json')
    updater=P.scale_updater(512//a.devices,opt)
    def commit(u,row):
        cp=out/f'u{u:06d}';cp.mkdir(exist_ok=True)
        value=dict(params=P.one(p),adam=P.one(adam),states=jax.device_get(st),aux=jax.device_get(ax),keys=jax.device_get(keys),
            history=x.get('history'),update=u,seed=a.seed,arm=a.arm,environment_steps=u*16384,optimizer_steps=u*32,devices=a.devices)
        for name,v in [('state.pkl',value),('policy.pt',value['params']['actor'])]:
            tmp=cp/(name+'.tmp')
            with tmp.open('wb') as f:pickle.dump(v,f,protocol=5);f.flush();os.fsync(f.fileno())
            tmp.replace(cp/name)
        marker=dict(passed=True,checkpoint=str(cp),update=u,environment_steps=u*16384,optimizer_steps=u*32,
            state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),metrics=row)
        atomic_json(cp/'COMMITTED.json',marker)
        return cp,marker
    if not checkpoints:
        cp,marker=commit(0,dict(source_params_Adam_exact=True,source_env_aux_exact_after_reshape=True,paired_training_RNG_reset=True))
        atomic_json(out/'SOURCE.json',dict(**source,paired_seed=a.seed,bank_probability=ratio,devices=a.devices,
            source_RNG_digest=digest(x['keys']),starting_env_aux_digest=digest((x['states'],x['aux'])),protocol_sha256=sha(PROTO)))
    # Fixed-key GPU fresh sample measures actual reset wrapper, not a CPU approximation.
    if not (out/'RESET_FREQUENCY.json').exists():
        total=0;selected=0
        fresh=jax.jit(P.T.fresh)
        for i in range(64):
            kk=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(a.seed),0xBA00+i),128)
            ss,aa=fresh(kk,bs,ba);jax.block_until_ready(ss.t)
            selected+=int(np.count_nonzero(np.asarray(aa['provenance_source'])>=0));total+=128
        from scipy.stats import binomtest
        test=binomtest(selected,total,ratio)
        atomic_json(out/'RESET_FREQUENCY.json',dict(passed=True,actual_GPU_fresh_calls=64,n=total,bank_resets=selected,
            frequency=selected/total,expected=ratio,binomial_two_sided_p=float(test.pvalue),compatible_at_001=bool(test.pvalue>=.01),
            seed=a.seed,training_RNG_unchanged=True,not_episode_occupancy=True))
        if test.pvalue<.01:raise RuntimeError('Reset frequency incompatible with frozen probability')
    cumulative=np.zeros(2,np.int64)
    for u in range(start+1,321):
        while True:
            if any((CP/f).exists() for f in ('USER_STOP.json','USER_PAUSE.json')) or (B/'USER_STOP.json').exists():
                cp,marker=commit(u-1,dict(user_paused=True));atomic_json(out/'PAUSED.json',marker);return
            command=read(out/'COMMAND.json') if (out/'COMMAND.json').exists() else dict(target_update=0)
            if command.get('exit_at_boundary'):
                cp,marker=commit(u-1,dict(scheduling_fallback=True));atomic_json(out/'SAFE_EXIT.json',marker);return
            if u<=command.get('target_update',0):break
            atomic_json(out/'HELD.json',dict(update=u-1,safe_atomic_boundary=True,unix=time.time()));time.sleep(2)
        before=P.one(p);began=time.monotonic()
        (st,ax,keys),batch,diag=collect(p,st,ax,keys,bs,ba);jax.block_until_ready(batch)
        rollout=time.monotonic()-began
        random=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(a.seed),64+u),a.devices))
        (p,adam),loss=updater(p,adam,batch,batch,random);jax.block_until_ready(p)
        after=P.one(p);delta=P.norms(before,after)
        assert all(v>0 for v in delta.values()) and np.isfinite(np.asarray(loss)).all()
        assert all(np.isfinite(v).all() for v in jax.tree.leaves((after,P.one(adam))))
        cumulative+=np.asarray(diag[7]).sum((0,1,2)).astype(np.int64)
        row=dict(passed=True,finite=True,new_update=u,environment_steps=u*16384,optimizer_steps=u*32,
            update_l2=delta,actual_environment_steps_this_update=16384,actual_optimizer_steps_this_update=32,
            cycle_seconds=time.monotonic()-began,rollout_seconds=rollout,loss=np.asarray(loss).mean((0,1)).tolist(),
            bank_probability=ratio,devices=a.devices,global_batch=16384,global_minibatch=512,dagger=0,combat_kl=0,
            actual_resets_this_process=int(cumulative[0]),actual_bank_resets_this_process=int(cumulative[1]),unix=time.time())
        row['env_per_second']=16384/row['cycle_seconds']
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        command=read(out/'COMMAND.json')
        if u==1 or u%64==0 or u==320 or u>=command['target_update']:
            cp,marker=commit(u,row)
            if u==1 or u>=command['target_update']:
                state,aux,key=(P.one(v) for v in (st,ax,keys))
                nk,ke,act,_=jax.jit(P.C.E.choose)(after,state,aux,key);nxt=jax.jit(P.T.transition)(state,aux,act,ke)
                atomic_json(out/f'NEXT_STEP_u{u:06d}.json',dict(digest=digest((nk,ke,act,nxt)),Adam_digest=digest(P.one(adam))))
            if u in (64,128,192,256,320):atomic_json(B/'eval_queue'/f'{out.name}_u{u:06d}.json',dict(**marker,stage='formal',final=u==320,arm=a.arm,seed=a.seed))
        atomic_json(out/'PROGRESS.json',dict(iterations=u,last=row))
        if u==1:notify('FIRST_UPDATE',dict(run=out.name,row=row,atomic_state_sha256=marker['state_sha256']))
        # Keep host/device placement common with validated campaign optimization.
        p,adam=rep(after),rep(P.one(adam))
    atomic_json(out/'COMPLETE.json',dict(**marker,actual_environment_steps=5242880,actual_optimizer_steps=10240))
    notify('RUN_COMPLETE',dict(run=out.name,update=320,checkpoint=str(cp),state_sha256=marker['state_sha256']))

if __name__=='__main__':main()
