"""Bounded matched real-policy collector profile, no optimizer/state writes."""
from pathlib import Path
from types import ModuleType
import os
import sys


def main():
    device=os.environ.get('JAXBOMB_DEVICE_OVERRIDE','2')
    os.environ.update(CUDA_VISIBLE_DEVICES=device,JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast import install as fast
    fast()
    from scripts.bun_native_v2_custody_fast_train import worker_source
    s=worker_source()
    # All writes/locks belong to the probe namespace, never the real run.
    s=s.replace("out=ROOT/protocol['namespace']/args.arm;", "out=ROOT/'runs/bun_native_v2_20261030/custody_performance_v1/collector_probe'/args.arm;")
    anchor='    def objective(p,b,ent):'
    assert s.count(anchor)==1
    probe='''    timing=[]
    for iteration in range(4):
        began=time.monotonic()
        # Identical checkpoint/RNG start each repetition measures one frozen
        # student-policy rollout; it is NOT new on-policy training progress.
        result=collect(par,st,ax,keys,bs,ba,rep(ring[start%4]),rep(fixed))
        jax.block_until_ready(result)
        timing.append(time.monotonic()-began)
        atomic_json(out/'PROGRESS.json',dict(iteration=iteration+1,timing=timing,training_updates=0,unix=time.time()))
    atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',gpu=int(os.environ['JAXBOMB_DEVICE_OVERRIDE']),
        source=cm,environments=n,ticks=spec['rollout_ticks'],timing=timing,cold_seconds=timing[0],
        warm_seconds=timing[1:],warm_env_per_second=batch_size/float(np.median(timing[1:])),
        seed=seed,actual_frozen_student_policy=True,no_optimizer_updates=True,no_training_checkpoint_writes=True,unix=time.time()))
    return
'''
    s=s.replace(anchor,probe+anchor)
    m=ModuleType('custody_collector_profile');m.__file__=str(Path(__file__))
    exec(compile(s,__file__,'exec'),m.__dict__);m.main()


if __name__=='__main__':main()
