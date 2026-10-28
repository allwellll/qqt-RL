"""Independent single-factor PPO scaling runs after frozen reward V2 ends."""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/authorized_campaign_20261026'


def dimensions(spec):
    devices=int(spec['devices']); envs=int(spec['envs_per_device'])
    ticks=int(spec['rollout_ticks']); mb=int(spec['global_minibatch'])
    batch=devices*envs*ticks
    if devices<=0 or envs<=0 or ticks<=0 or mb%devices or batch%mb:
        raise ValueError('Invalid fixed batch/minibatch/device partition')
    return batch,mb//devices,batch//mb


def selected_devices(spec, override=None):
    if override is not None and int(spec['devices']) != 1:
        raise ValueError('physical device override is only supported for a single-device worker')
    return str(override) if override is not None else ','.join(str(i) for i in range(int(spec['devices'])))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('protocol');p.add_argument('arm');p.add_argument('seed',type=int)
    p.add_argument('--verify-restore',action='store_true')
    args=p.parse_args(); protocol=json.loads(Path(args.protocol).read_text())
    spec={**protocol['common'],**protocol['arms'][args.arm]}
    batch_size,local_mb,optim_steps=dimensions(spec)
    output=ROOT/protocol['namespace']/f'{args.arm}_s{args.seed}'
    output.mkdir(parents=True,exist_ok=True);guard=lock(output/'worker.lock')
    if (output/'COMPLETE.json').exists() and not args.verify_restore:
        done=json.loads((output/'COMPLETE.json').read_text())
        assert done['passed']
        assert sha(ROOT/done['checkpoint']/'state.pkl')==done['state_sha256']
        return
    device_override = os.environ.get('JAXBOMB_DEVICE_OVERRIDE')
    visible_devices = selected_devices(spec, device_override)
    os.environ.update(CUDA_VISIBLE_DEVICES=visible_devices,
        JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR=str(output.parent/'cache'/args.arm))
    import jax
    import jax.numpy as j
    import numpy as np
    import optax
    from scripts import ppo_scale_longrun_train as P
    from scripts.carrier_mechanism_analysis import bank,merged,shard
    from scripts.campaign_paired_train_canonical import canonical_collector
    from scripts.terminal_checkpoint_restore import digest
    for filename,expected in protocol['source_sha256'].items(): assert sha(ROOT/filename)==expected,filename
    assert jax.local_device_count()==spec['devices']
    P.T.configure();P.C.install_provenance()
    if 'bank_probability' in spec and spec['bank_probability']!=.25:
        raise ValueError('Bank ratio intervention needs separately validated fresh/provenance wrapper')
    source=protocol['sources'][str(args.seed)]
    assert sha(ROOT/source['checkpoint']/'state.pkl')==source['state_sha256']
    x=pickle.load((ROOT/source['checkpoint']/'state.pkl').open('rb'))
    items=[a for a in bank() if a['record']['split']=='train'];bs,ba=merged(items,'states'),merged(items,'aux')
    devices=spec['devices'];n=devices*spec['envs_per_device']
    start=0;environment_steps=0;optimizer_steps=0
    existing=sorted(output.glob('u*/COMMITTED.json'))
    if existing:
        marker=json.loads(existing[-1].read_text());cp=existing[-1].parent
        assert sha(cp/'state.pkl')==marker['state_sha256']
        x=pickle.load((cp/'state.pkl').open('rb'));start=x['update']
        environment_steps=x['environment_steps'];optimizer_steps=x['optimizer_steps']
        st,ax,keys=x['states'],x['aux'],x['keys']
    else:
        keys=np.asarray(jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed),i))(np.arange(n,dtype=np.uint32)))
        st,ax=jax.jit(P.T.fresh)(keys,bs,ba)
        st,ax,keys=shard(st,devices),shard(ax,devices),shard(keys,devices)
    if args.verify_restore:
        if not existing:raise ValueError('No atomic checkpoint to restore')
        one_state,one_aux,one_key=(P.one(a) for a in (st,ax,keys))
        nk,ke,act,_=jax.jit(P.C.E.choose)(x['params'],one_state,one_aux,one_key)
        next_step=jax.jit(P.T.transition)(one_state,one_aux,act,ke)
        expected=json.loads((output/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert digest((nk,ke,act,next_step))==expected['digest']
        assert digest(x['adam'])==expected['Adam_digest']
        atomic_json(output/'RESTORE.json',dict(passed=True,fresh_process=True,
            full_env_aux_RNG_next_step_exact=True,Adam_exact=True,checkpoint=str(cp.relative_to(ROOT))))
        return
    rep=lambda tree:jax.tree.map(lambda a:j.stack([j.asarray(a)]*devices),tree)
    par,adam=rep(x['params']),rep(x['adam'])
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(spec['lr'],eps=1e-8))
    collect=canonical_collector(P.C.collector(spec['rollout_ticks']),output/'INITIAL_PLACEMENT.json')
    update=P.scale_updater(local_mb,opt)
    target=protocol['environment_steps_per_run']; assert target%batch_size==0
    total=target//batch_size
    def save(index,row):
        cp=output/f'u{index:06d}';cp.mkdir(exist_ok=True)
        payload=dict(params=P.one(par),adam=P.one(adam),states=jax.device_get(st),aux=jax.device_get(ax),
            keys=jax.device_get(keys),history=x.get('history'),update=index,environment_steps=environment_steps,
            optimizer_steps=optimizer_steps,seed=args.seed,arm=args.arm)
        for filename,value in [('state.pkl',payload),('policy.pt',payload['params']['actor'])]:
            tmp=cp/(filename+'.tmp')
            with tmp.open('wb') as f:pickle.dump(value,f,protocol=5);f.flush();os.fsync(f.fileno())
            tmp.replace(cp/filename)
        result=dict(passed=True,checkpoint=str(cp.relative_to(ROOT)),state_sha256=sha(cp/'state.pkl'),
            policy_sha256=sha(cp/'policy.pt'),update=index,environment_steps=environment_steps,
            optimizer_steps=optimizer_steps,metrics=row)
        atomic_json(cp/'COMMITTED.json',result)
        atomic_json(output/'eval_queue'/f'u{index:06d}.json',result)
        return result
    if start==0:save(0,dict(fresh_parent=True,matching_parent_params_Adam=True))
    for index in range(start+1,total+1):
        if any((BASE/f).exists() for f in ('USER_STOP.json','USER_PAUSE.json')):
            receipt=save(index-1,dict(user_paused=True));atomic_json(output/'PAUSED.json',receipt);return
        before=P.one(par);began=time.monotonic()
        (st,ax,keys),batch,diag=collect(par,st,ax,keys,bs,ba);jax.block_until_ready(batch)
        rollout=time.monotonic()-began
        # No DAgger/replay gradient. Unused history is retained, not fed into PPO.
        random=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed),64+index),devices))
        (par,adam),loss=update(par,adam,batch,batch,random);jax.block_until_ready(par)
        after=P.one(par);delta=P.norms(before,after)
        finite=all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(after)) and bool(np.isfinite(np.asarray(loss)).all())
        if not finite or not all(v>0 for v in delta.values()):raise FloatingPointError('Nonfinite or zero real actor/critic update')
        environment_steps+=batch_size;optimizer_steps+=optim_steps
        row=dict(passed=True,finite=finite,new_update=index,environment_steps=environment_steps,
            optimizer_steps=optimizer_steps,update_l2=delta,cycle_seconds=time.monotonic()-began,
            rollout_seconds=rollout,actual_environment_steps_this_update=batch_size,
            actual_optimizer_steps_this_update=optim_steps,loss=np.asarray(loss).mean((0,1)).tolist(),
            categorical=True,dagger=0,combat_kl=0,configuration=spec,unix=time.time())
        with (output/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        if index==1 or index==total//2 or environment_steps%1048576==0 or index==total:
            receipt=save(index,row)
            # Fresh next-step parity token stored without advancing training RNG.
            one_state,one_aux,one_key=(P.one(a) for a in (st,ax,keys))
            nk,ke,act,_=jax.jit(P.C.E.choose)(after,one_state,one_aux,one_key)
            next_step=jax.jit(P.T.transition)(one_state,one_aux,act,ke)
            atomic_json(output/f'NEXT_STEP_u{index:06d}.json',dict(digest=digest((nk,ke,act,next_step)),Adam_digest=digest(P.one(adam))))
        atomic_json(output/'PROGRESS.json',dict(iterations=index,last=row))
        par,adam=rep(after),rep(P.one(adam))
    atomic_json(output/'COMPLETE.json',dict(**receipt,actual_environment_steps=environment_steps,actual_optimizer_steps=optimizer_steps))


if __name__=='__main__':main()
