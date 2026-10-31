"""Bounded, D-sized timing probe, executable only after all D runs release GPUs.

This program never participates in the D owner or any training/eval queue.
It produces diagnostic updates, not adopted policy checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'runs/expert_reward_20261025/D_ablation_v1'


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_handoff():
    """Read-only handoff; never interrupts D or treats its absence as release."""
    if not (D/'TRAINING_COMPLETE.json').exists():
        raise RuntimeError('D fixed six-run queue has not completed; GPU probe stays deferred')
    done=json.loads((D/'TRAINING_COMPLETE.json').read_text())
    if not done.get('passed') or done.get('runs')!=6:
        raise RuntimeError('D completion is not a verified six-run terminal')
    for seed in (2026102851,2026102852,2026102853):
        for arm in ('legacy','minimal'):
            receipt=json.loads((D/'formal'/f'{arm}_s{seed}'/'COMPLETE.json').read_text())
            if receipt['new_update']!=320 or receipt['environment_steps']!=5242880:
                raise RuntimeError('D terminal budget mismatch')
            cp=ROOT/receipt['saved_checkpoint']
            for filename,field in (('state.pkl','state_sha256'),('policy.pt','policy_sha256')):
                if digest_file(cp/filename)!=receipt[field]:
                    raise RuntimeError('D terminal hash mismatch')
    for line in subprocess.check_output(['ps','-eo','args='],text=True).splitlines():
        words=line.split()
        if words and Path(words[0]).name.startswith('python') and \
                '-m scripts.expert_reward_ablation_resume' in line:
            raise RuntimeError('Existing D owner/worker still active; GPU probe stays deferred')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if apps or any(float(v)>0 for v in memory.splitlines()):
        raise RuntimeError('GPUs not fully released; refusing to share or preempt')


def main():
    process_started=time.monotonic()
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('variant',choices=('baseline','hoisted_updater','resident_params_adam'))
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    contract=json.loads(args.manifest.read_text())
    verify_handoff()
    # Recheck old global pause receipt: a new user pause always wins.
    pause=ROOT/'runs/retention_20261021/USER_GPU_PAUSE.json'
    if digest_file(pause)!=contract['global_pause_receipt_sha256']:
        raise RuntimeError('Pause receipt changed; explicit authorization must be reconsidered')
    for filename,expected in contract['source_sha256'].items():
        if digest_file(ROOT/filename)!=expected:
            raise RuntimeError('Performance probe source differs from preregistration')
    parity=json.loads((ROOT/contract['cpu_parity_result']).read_text())
    if not parity.get('passed'):
        raise RuntimeError('CPU semantic parity must pass before the GPU timing probe')
    if contract['gpu_execution_status']!='DEFERRED_UNTIL_D_RELEASE':
        raise RuntimeError('Unexpected performance manifest state')
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR=str((ROOT/contract['cache']/args.variant).resolve()))
    import fcntl
    lock_path=ROOT/contract['namespace']/'probe.lock'
    owner_lock=lock_path.open('a');fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import jax
    import numpy as np
    import optax
    from scripts import ppo_scale_longrun_train as P
    from scripts import expert_reward_ablation as reward_adapter
    from scripts import je10_control_credit as control_credit
    from scripts.carrier_mechanism_analysis import bank,merged
    from scripts.population_run_state import atomic_json
    from scripts.gpu_efficiency_parity import tree_digest
    assert jax.default_backend()=='gpu' and jax.local_device_count()==8
    args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/contract['source_checkpoint']/'state.pkl'
    if digest_file(source)!=contract['source_state_sha256']:
        raise RuntimeError('Performance source checkpoint mismatch')
    with source.open('rb') as stream:state=pickle.load(stream)
    P.T.configure();P.C.install_provenance()
    reward_adapter.bind_reward_arm(P.T.E.ops,control_credit,'legacy')
    items=[x for x in bank() if x['record']['split']=='train']
    bs,ba=merged(items,'states'),merged(items,'aux')
    optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    collect=P.C.collector(128)
    hoisted=P.scale_updater(64,optimizer)
    params,adam=P.rep(state['params']),P.rep(state['adam'])
    states,aux,keys,history=state['states'],state['aux'],state['keys'],state['history']
    records=[]
    startup_seconds=time.monotonic()-process_started
    for index in range(contract['timing_iterations_per_process']):
        if digest_file(pause)!=contract['global_pause_receipt_sha256']:
            atomic_json(args.output/'PAUSED.json',dict(safe_update_boundary=True,completed_iterations=index))
            return
        times={};cycle_start_unix=time.time();started=time.monotonic();t=started
        before=P.one(params);times['pre_rollout_params_device_get']=time.monotonic()-t;t=time.monotonic()
        (states,aux,keys),batch,diag=collect(params,states,aux,keys,bs,ba)
        jax.block_until_ready(batch);times['collector_dispatch_and_wait']=time.monotonic()-t;t=time.monotonic()
        history,_,_=P.C.aggregate(batch,history,2026102851,65+index)
        times['aggregate_batch_get_and_CPU_history']=time.monotonic()-t;t=time.monotonic()
        update=hoisted if args.variant=='hoisted_updater' else P.scale_updater(64,optimizer)
        updatekeys=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(2026102851),65+index),8))
        (params,adam),loss=update(params,adam,batch,batch,updatekeys)
        jax.block_until_ready(params);times['updater_creation_dispatch_compile_and_wait']=time.monotonic()-t;t=time.monotonic()
        after=P.one(params)
        deltas=P.norms(before,after)
        assert all(v>0 for v in deltas.values()) and np.isfinite(np.asarray(loss)).all()
        snapshot=dict(params=after,adam=P.one(adam),states=jax.device_get(states),
                      aux=jax.device_get(aux),keys=jax.device_get(keys),history=history)
        times['metrics_and_full_state_device_get']=time.monotonic()-t;t=time.monotonic()
        if args.variant!='resident_params_adam':
            params,adam=P.rep(after),P.rep(snapshot['adam'])
            jax.block_until_ready((params,adam))
        times['params_Adam_replicate_and_wait']=time.monotonic()-t
        cycle=time.monotonic()-started
        row=dict(iteration=index+1,variant=args.variant,environment_steps=16384,optimizer_steps=32,
            cycle_start_unix=cycle_start_unix,
            first_call_includes_compile=index==0,stages=times,cycle_seconds=cycle,
            env_per_second=16384/cycle,update_per_second=1/cycle,
            float_tolerances=contract['float_tolerances'],update_l2=deltas,finite=True,
            actual_opponent_exposure=dict(current=0,ring=0,fixed=16384,identity='JAX rule3'),
            snapshot_digest=tree_digest(snapshot),
            batch_digest=tree_digest(jax.device_get(batch)),diag_digest=tree_digest(jax.device_get(diag)))
        row['total_cycle_with_passive_witness_seconds']=time.monotonic()-started
        row['passive_witness_overhead_seconds']=row['total_cycle_with_passive_witness_seconds']-cycle
        records.append(row)
        atomic_json(args.output/'PROGRESS.json',dict(iterations=len(records),last=row,controls_D=False))
    # Retain the final real arrays for field-level inspection if a floating
    # digest differs. No optimizer output is promoted to a training parent.
    with (args.output/'FINAL_WITNESS.pkl').open('wb') as stream:
        pickle.dump(dict(snapshot=snapshot,batch=jax.device_get(batch),
            diag=jax.device_get(diag),loss=jax.device_get(loss)),stream,
            protocol=pickle.HIGHEST_PROTOCOL)
    atomic_json(args.output/'COMPLETE.json',dict(passed=True,variant=args.variant,rows=records,
        manifest_sha256=digest_file(args.manifest),new_environment_steps=len(records)*16384,
        new_optimizer_steps=len(records)*32,adopted_training_updates=0,controls_D=False,
        startup_before_first_iteration_seconds=startup_seconds,
        process_wall_seconds=time.monotonic()-process_started,
        final_witness_sha256=digest_file(args.output/'FINAL_WITNESS.pkl')))


if __name__=='__main__':
    main()
