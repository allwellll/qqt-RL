"""Independent bounded pure-PPO campaign worker with complete atomic resume."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('protocol');parser.add_argument('stage',choices=['smoke','restore','formal'])
    parser.add_argument('arm');parser.add_argument('seed',type=int);args=parser.parse_args()
    protocol=json.loads(Path(args.protocol).read_text());base=ROOT/protocol['namespace']
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR=str(base/'cache'/args.arm))
    import jax
    import jax.numpy as j
    import numpy as np
    import optax
    from scripts import ppo_scale_longrun_train as P
    from scripts import campaign_reward_v2 as reward
    from scripts.carrier_mechanism_analysis import bank,merged
    from scripts.population_run_state import atomic_json,sha,lock
    from scripts.terminal_checkpoint_restore import digest
    run=base/(args.stage if args.stage!='formal' else 'formal')/f'{args.arm}_s{args.seed}'
    if args.stage=='restore':run=base/'smoke'/f'{args.arm}_s{args.seed}'
    run.mkdir(parents=True,exist_ok=True);worker_lock=lock(base/'worker.lock')
    assert jax.default_backend()=='gpu' and jax.local_device_count()==8
    for f,h in protocol['source_sha256'].items():assert sha(ROOT/f)==h,f
    assert sha(ROOT/'runs/retention_20261021/USER_GPU_PAUSE.json')==protocol['pause_receipt_sha256']
    P.T.configure();P.C.install_provenance()
    if args.arm=='task_semantics':reward.install(P.C)
    source=protocol['sources'][str(args.seed)]
    assert sha(ROOT/source['checkpoint']/'state.pkl')==source['state_sha256']
    state=pickle.load((ROOT/source['checkpoint']/'state.pkl').open('rb'))
    if args.arm=='task_semantics':
        state['aux']=dict(state['aux'],campaign_task_paid=np.maximum(0,state['aux']['paid']-state['aux']['shape_paid']))
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(protocol['common']['lr'],eps=1e-8))
    items=[x for x in bank() if x['record']['split']=='train'];bs,ba=merged(items,'states'),merged(items,'aux')
    def safe():
        return not any((ROOT/'runs/authorized_campaign_20261026'/f).exists() for f in ('USER_STOP.json','USER_PAUSE.json')) and \
            sha(ROOT/'runs/retention_20261021/USER_GPU_PAUSE.json')==protocol['pause_receipt_sha256']
    def save(index,p,a,st,ax,keys,history,row):
        cp=run/f'u{index:06d}';cp.mkdir(exist_ok=True)
        payload=dict(params=P.one(p),adam=P.one(a),states=jax.device_get(st),aux=jax.device_get(ax),
            keys=jax.device_get(keys),history=history,update=index,seed=args.seed,arm=args.arm)
        for filename,value in [('state.pkl',payload),('policy.pt',payload['params']['actor'])]:
            temp=cp/(filename+'.tmp')
            with temp.open('wb') as out:pickle.dump(value,out,protocol=pickle.HIGHEST_PROTOCOL);out.flush();os.fsync(out.fileno())
            temp.replace(cp/filename)
        per_steps=1024 if args.stage!='formal' else 16384
        per_optim=16 if args.stage!='formal' else 32
        marker=dict(passed=True,new_update=index,environment_steps=index*per_steps,optimizer_steps=index*per_optim,
            state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),metrics=row)
        atomic_json(cp/'COMMITTED.json',marker)
        return cp,marker
    if args.stage=='restore':
        cp=run/'u000001';marker=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'state.pkl')==marker['state_sha256']
        x=pickle.load((cp/'state.pkl').open('rb'));st=P.one(x['states']);ax=P.one(x['aux']);keys=P.one(x['keys'])
        nk,ke,act,_=jax.jit(P.C.E.choose)(x['params'],st,ax,keys)
        result=jax.jit(P.T.transition)(st,ax,act,ke)
        expected=json.loads((run/'EXPECTED.json').read_text())
        assert digest((nk,ke,act,result))==expected['next_step_digest']
        assert digest(x['adam'])==expected['adam_digest']
        atomic_json(run/'RESTORE.json',dict(passed=True,fresh_process=True,full_state_next_step_exact=True,Adam_exact=True));return
    if args.stage=='formal':
        for arm in protocol['arms']:
            smoke=base/'smoke'/f'{arm}_s{protocol["paired_seeds"][0]}'
            assert json.loads((smoke/'COMPLETE.json').read_text())['passed']
            assert json.loads((smoke/'RESTORE.json').read_text())['passed']
    start=0
    checkpoints=sorted(run.glob('u*/COMMITTED.json'))
    if checkpoints:
        cp=checkpoints[-1].parent;marker=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'state.pkl')==marker['state_sha256']
        state=pickle.load((cp/'state.pkl').open('rb'));start=state['update']
    p,a=P.rep(state['params']),P.rep(state['adam']);st,ax,keys=state['states'],state['aux'],state['keys'];history=state['history']
    steps=8 if args.stage=='smoke' else 128;mb=8 if args.stage=='smoke' else 64
    collect=P.C.collector(steps);update=P.scale_updater(mb,opt)
    strategy='baseline'
    decision=ROOT/'runs/authorized_campaign_20261026/GPU_analysis/COMPLETE.json'
    if decision.exists():strategy=json.loads(decision.read_text())['selected_common_implementation']
    total=1 if args.stage=='smoke' else protocol['updates']
    if start==0:save(0,p,a,st,ax,keys,history,dict(fresh_parent=True))
    for index in range(start+1,total+1):
        if not safe():
            cp,marker=save(index-1,p,a,st,ax,keys,history,dict(user_paused=True))
            atomic_json(run/'PAUSED.json',dict(last_update=index-1,checkpoint=str(cp.relative_to(ROOT)),**marker));return
        began=time.monotonic();before=P.one(p)
        (st,ax,keys),batch,diag=collect(p,st,ax,keys,bs,ba);jax.block_until_ready(batch)
        rollout=time.monotonic()-began
        history,_,_=P.C.aggregate(batch,history,args.seed,64+index)
        current=update if strategy=='hoisted_updater' else P.scale_updater(mb,opt)
        (p,a),loss=current(p,a,batch,batch,np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed),64+index),8)))
        jax.block_until_ready(p);after=P.one(p);delta=P.norms(before,after)
        assert all(v>0 for v in delta.values()) and np.isfinite(np.asarray(loss)).all()
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(after))
        row=dict(passed=True,new_update=index,finite=True,update_l2=delta,
            actual_environment_steps_this_update=steps*128,actual_optimizer_steps_this_update=steps*16//mb,
            environment_steps=index*steps*128,cycle_seconds=time.monotonic()-began,rollout_seconds=rollout,
            env_per_second=steps*128/(time.monotonic()-began),loss=np.asarray(loss).mean((0,1)).tolist(),
            reward_sum=float(np.asarray(diag[3]).sum()),nonzero_rewards=int(np.count_nonzero(diag[3])),
            implementation=strategy,dagger=0,combat_kl=0,actual_opponent_exposure=dict(current=0,ring=0,fixed=steps*128,identity='JAX rule3'))
        with (run/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        snapshot_needed=index==1 or index%64==0 or index==total
        if snapshot_needed:
            cp,marker=save(index,p,a,st,ax,keys,history,row)
            atomic_json(base/'eval_queue'/f'{args.arm}_s{args.seed}_u{index:06d}.json',dict(checkpoint=str(cp.relative_to(ROOT)),state_sha256=marker['state_sha256'],policy_sha256=marker['policy_sha256'],stage=args.stage,final=index==total))
        if strategy!='resident_params_adam':p,a=P.rep(after),P.rep(P.one(a))
        atomic_json(run/'PROGRESS.json',dict(iterations=index,last=row))
    if args.stage=='smoke':
        state=pickle.load((cp/'state.pkl').open('rb'));ss=P.one(state['states']);aa=P.one(state['aux']);kk=P.one(state['keys'])
        nk,ke,act,_=jax.jit(P.C.E.choose)(state['params'],ss,aa,kk);result=jax.jit(P.T.transition)(ss,aa,act,ke)
        atomic_json(run/'EXPECTED.json',dict(next_step_digest=digest((nk,ke,act,result)),adam_digest=digest(state['adam'])))
    atomic_json(run/'COMPLETE.json',dict(passed=True,updates=total,actual_environment_steps=total*steps*128,
        checkpoint=str(cp.relative_to(ROOT)),state_sha256=marker['state_sha256'],policy_sha256=marker['policy_sha256'],implementation=strategy))

if __name__=='__main__':main()
