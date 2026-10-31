"""CPU-only fixed-seed semantic checks for a future isolated performance probe.

No D worker, authorization, queue, reward source, or training entry is modified.
The reduced shape is a technical witness, never a throughput or skill result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-6


def compare_arrays(a, b, *, exact):
    a,b = np.asarray(a),np.asarray(b)
    same = a.shape == b.shape and a.dtype == b.dtype
    finite = not (a.dtype.kind in 'fc' and (not np.isfinite(a).all() or not np.isfinite(b).all()))
    equal = bool(same and finite and (np.array_equal(a,b) if exact else
                 np.allclose(a,b,atol=FLOAT_ATOL,rtol=FLOAT_RTOL)))
    return dict(passed=equal,bitwise_equal=bool(same and a.tobytes()==b.tobytes()),
        exact_required=exact,shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype,
        max_abs_error=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))
            if same and finite else None)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def compare_trees(a,b,*,exact=False):
    import jax
    left,ls = jax.tree.flatten_with_path(a)
    right,rs = jax.tree.flatten_with_path(b)
    if ls != rs:
        return dict(passed=False,reason='tree structure differs')
    rows = []
    for (path,x),(_,y) in zip(left,right):
        r = compare_arrays(x,y,exact=exact or np.asarray(x).dtype.kind not in 'fc')
        rows.append(dict(field=jax.tree_util.keystr(path),**r))
    return dict(passed=all(r['passed'] for r in rows),leaf_count=len(rows),
                bitwise_equal=all(r['bitwise_equal'] for r in rows),fields=rows)


def tree_digest(tree):
    import jax
    digest=hashlib.sha256()
    leaves,structure=jax.tree.flatten(tree)
    digest.update(str(structure).encode())
    for leaf in leaves:
        array=np.asarray(leaf)
        digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--source-selection',choices=('parent_first_two','real_train_bank'),
                        default='parent_first_two')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        JAX_COMPILATION_CACHE_DIR=str(args.cache.resolve()))
    # Keep this CPU technical witness away from the live GPU training threads.
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,allowed[-4:])
    import jax
    import jax.numpy as jnp
    import optax
    from scripts import ppo_scale_longrun_train as P
    from scripts import expert_reward_ablation as reward_adapter
    from scripts import je10_control_credit as control_credit
    from scripts.carrier_mechanism_analysis import bank,merged
    from scripts.population_run_state import atomic_json

    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():
        raise RuntimeError('Refusing to rerun completed parity namespace')
    jax.config.update('jax_compilation_cache_dir',str(args.cache.resolve()))
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes',0)
    assert jax.default_backend()=='cpu' and jax.local_device_count()==1
    source=ROOT/'runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102751/u064/state.pkl'
    manifest=dict(schema='qqt.semantic-performance-cpu-parity/v1',
        purpose='Reduced CPU technical witness; not a formal training or GPU performance result',
        source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),
        seed=2026102697,source_selection=args.source_selection,
        source_stream=('first two actual parent checkpoint environments, aux and PRNG retained'
            if args.source_selection=='parent_first_two' else
            'first two hash-verified genuine acquisition train Bank snapshots and their stored RNG; no fabricated carry'),
        variants=['per-iteration updater creation plus host roundtrip','hoisted updater only','resident params/Adam only'],
        device='CPU',devices=1,envs_per_device=2,rollout_ticks=8,iterations_per_variant=2,
        local_minibatch=8,optimizer_steps_per_iteration=2,
        reward='frozen D legacy',lr=7.5e-5,clip_grad_norm=.5,adam_eps=1e-8,
        dagger=0.,combat_kl=0.,float_atol=FLOAT_ATOL,float_rtol=FLOAT_RTOL,
        exact_fields=['integers','masks','actions','teacher labels','PRNG keys','actual rewards','sample order'],
        source_sha256_files={f:sha(ROOT/f) for f in (
            'scripts/gpu_efficiency_parity.py','scripts/ppo_scale_longrun_train.py',
            'scripts/dagger_rl_r4_train.py','scripts/expert_reward_ablation.py',
            'scripts/carrier_mechanism_engine.py','scripts/combat_item_engine.py')},
        gpu_claim=False,does_not_modify_D=True,created_unix=time.time())
    atomic_json(args.output/'MANIFEST.json',manifest)
    atomic_json(args.output/'STATUS.json',dict(stage='LOADING',controls_gpu=False))
    with source.open('rb') as stream:original=pickle.load(stream)
    P.T.configure();P.C.install_provenance()
    reward_adapter.bind_reward_arm(P.T.E.ops,control_credit,'legacy')
    items=[x for x in bank() if x['record']['split']=='train']
    bs,ba=merged(items,'states'),merged(items,'aux')
    rep=lambda tree:jax.tree.map(lambda a:jnp.asarray(a)[None],tree)
    host=lambda tree:jax.tree.map(lambda a:np.asarray(a).copy(),jax.device_get(tree))
    st=jax.tree.map(lambda a:jnp.asarray(a[:1,:2]),original['states'])
    ax=jax.tree.map(lambda a:jnp.asarray(a[:1,:2]),original['aux'])
    keys=jnp.asarray(original['keys'][:1,:2])
    if args.source_selection=='real_train_bank':
        selected=items[:2]
        from scripts.carrier_mechanism_stats import validate_bank_record
        for item in selected:validate_bank_record(item['record'])
        st=rep(merged(selected,'states'))
        bank_aux=dict(merged(selected,'aux'),tactics=P.T.events.state(2),
                      provenance_source=jnp.arange(2,dtype=jnp.int32))
        ax=rep(bank_aux)
        keys=rep(merged(selected,'keys'))
        atomic_json(args.output/'BANK_SELECTION.json',dict(
            source='hash-verified existing causal acquisition Bank',split='train',
            snapshots=[item['record'] for item in selected],
            native_states_and_existing_reward_aux_retained=True,
            added_tactics_ledger='same zero tactics ledger added by existing T.fresh to certified Bank states',
            limitations='Existing Bank is single map/P0/allfar; no other coverage claimed'))
    initial=(rep(original['params']),rep(original['adam']),st,ax,keys)
    optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    collect=P.C.collector(8)
    hoisted=P.scale_updater(8,optimizer)
    runs={}
    variants=('baseline','hoisted_updater','resident_params_adam')
    for variant in variants:
        p,a,st,ax,keys=initial
        records=[]
        for iteration in range(2):
            atomic_json(args.output/'STATUS.json',dict(stage='PARITY',variant=variant,
                iteration=iteration,controls_gpu=False))
            started=time.monotonic()
            (st,ax,keys),batch,diag=collect(p,st,ax,keys,bs,ba)
            jax.block_until_ready(batch)
            rollout_seconds=time.monotonic()-started
            updater=hoisted if variant=='hoisted_updater' else P.scale_updater(8,optimizer)
            update_keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(2026102697),65+iteration),1)
            (p,a),loss=updater(p,a,batch,batch,update_keys)
            jax.block_until_ready(p)
            before_transfer=time.monotonic()
            # Copying outputs for evidence does not become an optimizer input.
            sample_indices=jax.vmap(lambda k:jax.random.permutation(k,16).reshape(-1,8))(update_keys)
            evidence=host(dict(params=p,adam=a,states=st,aux=ax,keys=keys,batch=batch,
                diag=diag,loss=loss,sample_indices=sample_indices))
            transfer_seconds=time.monotonic()-before_transfer
            if variant != 'resident_params_adam':
                p,a=(jax.tree.map(jnp.asarray,evidence['params']),
                     jax.tree.map(jnp.asarray,evidence['adam']))
            records.append(dict(evidence=evidence,rollout_seconds=rollout_seconds,
                timed_update_and_dispatch_seconds=before_transfer-started-rollout_seconds,
                evidence_host_copy_seconds=transfer_seconds,cycle_seconds=time.monotonic()-started))
        runs[variant]=records
    comparisons=[]
    for variant in variants[1:]:
        for iteration,(left,right) in enumerate(zip(runs['baseline'],runs[variant])):
            l,r=left['evidence'],right['evidence']
            checked={key:compare_trees(l[key],r[key]) for key in l}
            # Float-encoded categorical actions/teacher flags and rewards have
            # exact semantics even though their ABI is numeric float arrays.
            exact=dict(actions=compare_trees(l['batch'][2],r['batch'][2],exact=True),
                masks_and_labels=compare_trees(l['batch'][6:],r['batch'][6:],exact=True),
                rewards=compare_trees(l['diag'][3],r['diag'][3],exact=True),
                PRNG=compare_trees(l['keys'],r['keys'],exact=True),
                sample_order=compare_trees(l['sample_indices'],r['sample_indices'],exact=True))
            comparisons.append(dict(variant=variant,iteration=iteration,fields=checked,
                exact_semantics=exact,passed=all(v['passed'] for v in [*checked.values(),*exact.values()])))
    payload=args.output/'RAW.pkl'
    with payload.open('wb') as stream:pickle.dump(runs,stream,protocol=pickle.HIGHEST_PROTOCOL)
    result=dict(schema='qqt.semantic-performance-parity-result/v1',passed=all(c['passed'] for c in comparisons),
        manifest_sha256=sha(args.output/'MANIFEST.json'),raw_sha256=sha(payload),comparisons=comparisons,
        output_digest=tree_digest({v:[r['evidence'] for r in values] for v,values in runs.items()}),
        timing={v:[{k:x for k,x in row.items() if k!='evidence'} for row in values] for v,values in runs.items()},
        updates_are_technical_only=True,formal_new_steps=0,controls_gpu=False,
        source_selection=args.source_selection,
        limitation='CPU one-device reduced-shape witness; full 8-GPU D-sized timing and parity remain pending D release',
        unix=time.time())
    atomic_json(args.output/'COMPLETE.json',result)
    atomic_json(args.output/'STATUS.json',dict(stage='COMPLETE' if result['passed'] else 'PARITY_FAILED',controls_gpu=False))
    print(json.dumps(dict(passed=result['passed'],output_digest=result['output_digest'],controls_gpu=False)))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
