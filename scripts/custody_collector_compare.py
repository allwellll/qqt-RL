"""Matched 32-tick actual-policy collector comparison, zero PPO updates."""
import os
import sys
from pathlib import Path
from types import ModuleType


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('JAXBOMB_DEVICE_OVERRIDE', '2'),
        JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast import install as fast
    fast()
    from scripts.bun_native_v2_custody_fast_train import worker_source
    source = worker_source()
    source = source.replace("out=ROOT/protocol['namespace']/args.arm;",
        "out=ROOT/'runs/bun_native_v2_20261030/custody_performance_v2/collector_compare'/args.arm;")
    anchor = '    def objective(p,b,ent):'
    assert source.count(anchor) == 1
    probe = '''    from scripts import bun_native_v2_custody_fast2 as second
    measurements={};reference_digest=None
    # Explicit compilation is measured separately. Reuse identical canonical
    # device inputs, avoiding the first-call placement change of earlier probe.
    from jax.sharding import Mesh,NamedSharding,PartitionSpec
    placement=NamedSharding(Mesh(np.asarray(jax.devices()),('_internal_pmap_axis_name',)),PartitionSpec('_internal_pmap_axis_name'))
    st,ax,keys=jax.tree.map(lambda x:jax.device_put(x,placement),(st,ax,keys))
    inputs=(par,st,ax,keys,bs,ba,rep(ring[start%4]),rep(fixed))
    assert all(Path('runs/bun_native_v2_20261030/custody_performance_v2/PARITY_PROFILE.json').exists() for _ in [0])
    for label in ('sparse64','sparse64_eight_sweep'):
        if label=='sparse64_eight_sweep':second.install()
        atomic_json(out/'PROGRESS.json',dict(stage=label,phase='EXPLICIT_COMPILE',training_updates=0,unix=time.time()))
        raw=M.collector(32,n,seed,.10,0.,spawns,protocol['A1_weights'])
        began=time.monotonic();executable=raw.lower(*inputs).compile();compile_seconds=time.monotonic()-began
        times=[]
        for iteration in range(4):
            atomic_json(out/'PROGRESS.json',dict(stage=label,phase='EXECUTION',iteration=iteration+1,timing=times,training_updates=0,unix=time.time()))
            began=time.monotonic();result=executable(*inputs);jax.block_until_ready(result);times.append(time.monotonic()-began)
            if iteration==0:
                observed=digest(jax.device_get(result))
                if reference_digest is None:reference_digest=observed
                else:assert observed==reference_digest,'collector state/batch/GAE/diagnostics differ'
            del result
        measurements[label]=dict(compile_seconds=compile_seconds,first_execution_seconds=times[0],warm_seconds=times[1:],
            warm_median_seconds=float(np.median(times[1:])),output_sha256=observed)
        del executable,raw
    atomic_json(out/'COMPLETE.json',dict(status='PASS',source=cm,environments=n,ticks=32,training_updates=0,
        fixed_student_policy=True,no_training_checkpoint_writes=True,all_outputs_digest_exact=True,results=measurements,
        speedup=measurements['sparse64']['warm_median_seconds']/measurements['sparse64_eight_sweep']['warm_median_seconds'],
        limitations=['32tick matched diagnostic, not128tick training throughput','GPU2 co-tenants may affect timings; explicit compile excluded'],unix=time.time()))
    return
'''
    source = source.replace(anchor, probe + anchor)
    module = ModuleType('custody_collector_compare')
    module.__file__ = str(Path(__file__))
    exec(compile(source, __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import time
        from scripts.population_run_state import atomic_json
        atomic_json(Path('runs/bun_native_v2_20261030/custody_performance_v2/COLLECTOR_COMPARE_BLOCKED.json'),
            dict(status='BLOCKED',reason='inspect COLLECTOR_COMPARE.log; no training worker signalled',unix=time.time()))
        raise
