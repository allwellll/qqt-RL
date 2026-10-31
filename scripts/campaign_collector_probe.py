"""Bounded input-placement probe; never adopts diagnostic policy updates."""
import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import pickle
import time


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('protocol');args=parser.parse_args()
    m=json.loads(Path(args.protocol).read_text());base=Path(m['namespace']);base.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR=m['cache'])
    import jax
    import numpy as np
    import optax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec
    from scripts import ppo_scale_longrun_train as P
    from scripts.carrier_mechanism_analysis import bank,merged
    from scripts.gpu_efficiency_parity import tree_digest
    from scripts.population_run_state import atomic_json,sha,lock
    worker_lock=lock(base/'worker.lock')
    assert jax.default_backend()=='gpu' and jax.local_device_count()==8
    for name,h in m['source_sha256'].items():assert sha(name)==h,name
    assert sha(Path(m['source_checkpoint'])/'state.pkl')==m['source_state_sha256']
    assert sha('runs/retention_20261021/USER_GPU_PAUSE.json')==m['pause_receipt_sha256']
    compiler=[]
    class PassiveCompiler(logging.Handler):
        def emit(self,record):
            text=record.getMessage()
            if 'jit(_fun)' in text and ('Compiling' in text or 'Finished' in text):
                row=dict(unix=time.time(),kind='compile_signature' if text.startswith('Compiling') else 'compile_timing')
                if row['kind']=='compile_timing':row['message']=text
                else:
                    import hashlib,re
                    row.update(signature_sha256=hashlib.sha256(text.encode()).hexdigest(),leaf_count=len(re.findall(r'ShapedArray\(',text)))
                compiler.append(row);atomic_json(base/'COMPILER.json',dict(events=compiler))
    logger=logging.getLogger('jax');logger.addHandler(PassiveCompiler());logger.propagate=False
    jax.config.update('jax_log_compiles',True)
    P.T.configure();P.C.install_provenance()
    state=pickle.load((Path(m['source_checkpoint'])/'state.pkl').open('rb'))
    items=[x for x in bank() if x['record']['split']=='train'];bs,ba=merged(items,'states'),merged(items,'aux')
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    collect=P.C.collector(128);update=P.scale_updater(64,opt)
    p,a=P.rep(state['params']),P.rep(state['adam']);history=state['history']
    placement=NamedSharding(Mesh(np.asarray(jax.devices()),('_internal_pmap_axis_name',)),PartitionSpec('_internal_pmap_axis_name'))
    def stage(index,name):atomic_json(base/'ACTIVE_STAGE.json',dict(iteration=index,stage=name,unix=time.time()))
    stage(0,'canonicalize_existing_state_only')
    original=(state['states'],state['aux'],state['keys'])
    st,ax,keys=jax.tree.map(lambda x:jax.device_put(x,placement),original)
    placed=jax.device_get((st,ax,keys))
    exact=dict(passed=jax.tree.structure(original)==jax.tree.structure(placed) and all(
        np.asarray(x).dtype==np.asarray(y).dtype and np.array_equal(x,y)
        for x,y in zip(jax.tree.leaves(original),jax.tree.leaves(placed))))
    assert exact['passed']
    reference=json.loads(Path(m['reference_complete']).read_text())['rows'];records=[]
    for i in range(m['iterations']):
        if any((Path('runs/authorized_campaign_20261026')/f).exists() for f in ('USER_STOP.json','USER_PAUSE.json')):
            atomic_json(base/'PAUSED.json',dict(completed_iterations=i,no_formal_policy_adoption=True));return
        started=time.monotonic();times={};t=started
        before=P.one(p)
        stage(i+1,'collector_dispatch_trace_and_wait')
        (st,ax,keys),batch,diag=collect(p,st,ax,keys,bs,ba);jax.block_until_ready(batch)
        times['collector_including_dispatch_trace_compile']=time.monotonic()-t;t=time.monotonic()
        stage(i+1,'CPU_history')
        history,_,_=P.C.aggregate(batch,history,2026102851,65+i)
        times['CPU_history']=time.monotonic()-t;t=time.monotonic()
        stage(i+1,'hoisted_updater')
        updatekeys=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(2026102851),65+i),8))
        (p,a),loss=update(p,a,batch,batch,updatekeys);jax.block_until_ready(p)
        times['hoisted_updater']=time.monotonic()-t;t=time.monotonic()
        after=P.one(p);delta=P.norms(before,after)
        assert all(v>0 for v in delta.values()) and np.isfinite(np.asarray(loss)).all()
        snapshot=dict(params=after,adam=P.one(a),states=jax.device_get(st),aux=jax.device_get(ax),keys=jax.device_get(keys),history=history)
        times['full_state_get']=time.monotonic()-t;t=time.monotonic()
        p,a=P.rep(after),P.rep(snapshot['adam']);jax.block_until_ready((p,a))
        times['replicate']=time.monotonic()-t
        cycle=time.monotonic()-started
        digests=dict(snapshot_digest=tree_digest(snapshot),batch_digest=tree_digest(jax.device_get(batch)),diag_digest=tree_digest(jax.device_get(diag)))
        match={k:v==reference[i][k] for k,v in digests.items()}
        row=dict(iteration=i+1,finite=True,update_l2=delta,environment_steps=16384,optimizer_steps=32,cycle_seconds=cycle,stages=times,
            original_reference_parity=match,**digests)
        records.append(row);atomic_json(base/'PROGRESS.json',dict(iterations=i+1,last=row))
    atomic_json(base/'RESULT.json',dict(passed=True,rows=records,initial_state_exact=exact,
        all_reference_digests_equal=all(all(x['original_reference_parity'].values()) for x in records),compiler_events=compiler,
        reference=m['reference_complete'],reference_sha256=sha(m['reference_complete']),
        diagnostic_environment_steps=16384*len(records),diagnostic_optimizer_steps=32*len(records),adopted_policy_updates=0,
        conclusion='Input placement intervention only; compare collector first/second/third timings with registered original. No automatic production adoption.'))
    atomic_json(base/'COMPLETE.json',dict(passed=True,result_sha256=sha(base/'RESULT.json'),adopted_policy_updates=0))
    stage(len(records),'COMPLETE')


if __name__=='__main__':main()
