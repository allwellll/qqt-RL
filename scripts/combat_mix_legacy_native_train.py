"""Isolated COMBAT-MIX-V1 full-state continuation; frozen predecessors untouched."""
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
    p.add_argument('--verify-restore',action='store_true');p.add_argument('--smoke',action='store_true')
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
        JAX_COMPILATION_CACHE_DIR=str(output.parent/'cache_native_v2'/args.arm))
    import jax
    import jax.numpy as j
    import numpy as np
    import optax
    from scripts import ppo_scale_longrun_train as P
    from scripts.carrier_mechanism_analysis import bank,merged,shard
    from scripts.campaign_paired_train_canonical import canonical_collector
    from scripts.terminal_checkpoint_restore import digest
    from scripts import combat_mix_legacy_batched as L
    from scripts.combat_mix_opponents import fixed_allocation
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
    ax=L.base_aux(ax)
    for item in protocol['legacy_inputs'].values():
        assert sha(ROOT/item['path'])==item['sha256'],'legacy frozen input SHA mismatch'
    historical=pickle.load((ROOT/protocol['legacy_inputs']['initial_pool']['path']).open('rb'))
    frozen=pickle.load((ROOT/protocol['legacy_inputs']['frozen']['path']).open('rb'))
    frozen={k:frozen[k] for k in ('g0','stage1_teacher')}
    frozen['parent']=pickle.load((ROOT/source['checkpoint']/'state.pkl').open('rb'))['params']['actor']
    frozen=x.get('legacy_frozen',frozen)
    pool=x.get('legacy_pool',L.initial_pool(historical,x['params']['actor']))
    pool_counts=j.asarray(x.get('legacy_pool_counts',np.zeros((11,6),np.float32)))
    pfsp=j.asarray(x.get('legacy_pfsp',P.C.E.ops.pfsp_weights(pool_counts)))
    def probe(params,states,aux,keys,pool_value,generation):
        ss,aa,kk=jax.tree.map(lambda a:j.asarray(a[0]),(states,aux,keys))
        return digest(probe_raw(params,ss,aa,kk,pool_value,pfsp,j.int32(generation)))
    probe_raw=jax.jit(lambda q,s,a,k,pool_value,pf,generation:L.raw(1,q,s,a,k,pool_value,frozen,pf,generation))
    if args.verify_restore:
        if not existing:raise ValueError('No atomic checkpoint to restore')
        expected=json.loads((output/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert probe(x['params'],st,ax,keys,pool,start+1)==expected['digest']
        assert digest(x['adam'])==expected['Adam_digest']
        assert digest((st,ax,keys))==expected['full_environment_digest']
        assert digest(pool)==expected['pool_digest']
        atomic_json(output/'RESTORE.json',dict(passed=True,fresh_process=True,
            full_env_aux_RNG_exact=True,next_step_exact=True,Adam_exact=True,league_PFSP_exact=True,full_batch_next_step_exact=True,
            checkpoint=str(cp.relative_to(ROOT)),state_sha256=sha(cp/'state.pkl')))
        return
    rep=lambda tree:jax.tree.map(lambda a:j.stack([j.asarray(a)]*devices),tree)
    par,adam=rep(x['params']),rep(x['adam'])
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(spec['lr'],eps=1e-8))
    collect=canonical_collector(L.collector(spec['rollout_ticks']),output/'INITIAL_PLACEMENT.json')
    update=P.scale_updater(local_mb,opt)
    target=protocol['environment_steps_per_run']; assert target%batch_size==0
    total=target//batch_size
    def save(index,row):
        cp=output/f'u{index:06d}';cp.mkdir(exist_ok=True)
        payload=dict(params=P.one(par),adam=P.one(adam),states=jax.device_get(st),aux=jax.device_get(ax),
            keys=jax.device_get(keys),history=x.get('history'),update=index,environment_steps=environment_steps,
            optimizer_steps=optimizer_steps,seed=args.seed,arm=args.arm,
            legacy_frozen=jax.device_get(frozen),legacy_pool=jax.device_get(pool),legacy_pool_counts=jax.device_get(pool_counts),legacy_pfsp=jax.device_get(pfsp),protocol_sha256=sha(Path(args.protocol)))
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
    if start==160 and not existing:
        save(160,dict(fresh_parent=True,matching_parent_params_Adam=True,passive_metadata_added=True))
    smoke_end=min(total,int(protocol.get('technical_smoke_end_update',162)))
    if not args.smoke:
        assert (output/'SMOKE.json').exists() and (output/'RESTORE.json').exists(), 'Smoke and fresh restore required'
    stop=smoke_end if args.smoke else total
    for index in range(start+1,stop+1):
        if any((b/f).exists() for b in (BASE,ROOT/'runs/combat_mix_v1') for f in ('USER_STOP.json','USER_PAUSE.json')):
            receipt=save(index-1,dict(user_paused=True));atomic_json(output/'PAUSED.json',receipt);return
        before=P.one(par);began=time.monotonic()
        extras=dict(frozen=frozen,pfsp=pfsp,generation=j.int32(index))
        (st,ax,keys),batch,diag=collect(par,st,ax,keys,bs,ba,rep(pool),rep(extras));jax.block_until_ready(batch)
        rollout=time.monotonic()-began
        # No DAgger/replay gradient. Unused history is retained, not fed into PPO.
        random=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed),64+index),devices))
        (par,adam),loss=update(par,adam,batch,batch,random);jax.block_until_ready(par)
        after=P.one(par);delta=P.norms(before,after)
        hostdiag=jax.tree.map(lambda a:np.asarray(a).sum(0),jax.device_get(diag))
        pool_counts=.95*pool_counts+.05*j.asarray(hostdiag['league_counts'])
        pfsp=P.C.E.ops.pfsp_weights(pool_counts)
        pool=L.advance_pool(pool,after['actor'],index-160)
        finite=all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(after)) and bool(np.isfinite(np.asarray(loss)).all())
        if not finite or not all(v>0 for v in delta.values()):raise FloatingPointError('Nonfinite or zero real actor/critic update')
        environment_steps+=batch_size;optimizer_steps+=optim_steps
        row=dict(passed=True,finite=finite,new_update=index,environment_steps=environment_steps,
            optimizer_steps=optimizer_steps,update_l2=delta,cycle_seconds=time.monotonic()-began,
            rollout_seconds=rollout,actual_environment_steps_this_update=batch_size,
            actual_optimizer_steps_this_update=optim_steps,loss=np.asarray(loss).mean((0,1)).tolist(),
            categorical=True,dagger=0,combat_kl=0,configuration=spec,unix=time.time(),
            exposure=jax.tree.map(lambda a:np.asarray(a).sum(0).tolist(),jax.device_get(diag)))
        with (output/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        if index==161 or index==162 or (args.smoke and index==smoke_end) or index==total//2 or environment_steps%1048576==0 or index==total:
            receipt=save(index,row)
            # Fresh next-step parity token stored without advancing training RNG.
            atomic_json(output/f'NEXT_STEP_u{index:06d}.json',dict(
                digest=probe(after,st,ax,keys,pool,index+1),Adam_digest=digest(P.one(adam)),
                full_environment_digest=digest(jax.device_get((st,ax,keys))),pool_digest=digest(pool)))
        atomic_json(output/'PROGRESS.json',dict(iterations=index,last=row))
        par,adam=rep(after),rep(P.one(adam))
    if args.smoke:
        rows=[json.loads(l) for l in (output/'updates.jsonl').read_text().splitlines()]
        row=next(r for r in rows if r['new_update']==smoke_end)
        baseline=protocol['warm_baseline_env_per_second']
        throughput=batch_size/row['cycle_seconds']
        passed=row['finite'] and all(v>0 for v in row['update_l2'].values()) and throughput>=.85*baseline
        atomic_json(output/'SMOKE.json',dict(passed=passed,status='PASS' if passed else 'BLOCKED_THROUGHPUT',
            warm_env_per_second=throughput,baseline_env_per_second=baseline,
            fraction=throughput/baseline,checkpoint=receipt,exposure=row['exposure'],
            note='technical check only; first compiled cycle reported separately'))
        assert passed,'Warm throughput below frozen .85 baseline; preserve smoke and fix performance'
        return
    atomic_json(output/'COMPLETE.json',dict(**receipt,actual_environment_steps=environment_steps,actual_optimizer_steps=optimizer_steps))


if __name__=='__main__':main()
