"""Real saved-state/native tick exact parity and GPU custody microprofile."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('JAXBOMB_DEVICE_OVERRIDE','2'),JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
    XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
import json
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json,sha,lock

OUT=Path('runs/bun_native_v2_20261030/custody_performance_v1')


def main():
    guard=lock(OUT/'PROFILE.lock')
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts import carrier_mechanism_engine as E,combat_item_engine as T
    from scripts import bun_native_v2_custody_fast as F
    T.configure()
    slots=json.loads(Path('runs/bun_native_v2_20261030/R2_SLOTS.json').read_text());folder=Path('runs/bun_native_v2_20261030/formal')/slots['b']['run']
    cps=[p for p in sorted(folder.glob('u*/COMMITTED.json')) if (p.parent/'state.pkl').exists()]
    cp=cps[-1].parent;receipt=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'state.pkl')==receipt['state_sha256']
    payload=pickle.load((cp/'state.pkl').open('rb'))
    states=jax.tree.map(lambda x:j.asarray(x[0]),payload['states']);cache=jax.tree.map(lambda x:j.asarray(x[0]),payload['aux']['custody_v2'])
    keys=jax.tree.map(lambda x:j.asarray(x[0]),payload['keys']);n=len(states.t)
    old=jax.jit(F.REFERENCE);new=jax.jit(F.potential)
    step=jax.jit(jax.vmap(lambda s,a,k:E.env.step(s,a,k,auto_reset=False,return_info=True)))
    samples=[];cases=[];began=time.time()
    for tick in range(32):
        a=old(states,cache);b=new(states,cache);jax.block_until_ready((a,b))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
        blocked=states.wall|states.brick|states.core.pushable
        topology=~cache['ready']|j.any(blocked!=cache['blocked'],(-2,-1))
        samples.append((states,cache));cases.append(dict(tick=tick,environments=n,topology_dirty=int(topology.sum()),
            object_target_dirty=int(j.any(a[1]['target']!=cache['target'],(-2,-1)).sum())))
        cache=a[1]
        move=jax.vmap(lambda k:jax.random.randint(k,(2,),0,5))(keys)
        ability=jax.vmap(lambda k:jax.random.bernoulli(jax.random.fold_in(k,17),.05,(2,)))(keys).astype(j.int32)
        actions=j.stack([move,ability],-1)
        keys,envkeys=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,2))(keys),0,1)
        states,done,info=step(states,actions,envkeys);jax.block_until_ready(states)
        atomic_json(OUT/'PROGRESS.json',dict(status='PARITY_TICKS',completed=tick+1,budget=32,cases=cases[-1],unix=time.time()))
    timing={}
    for label,fn in [('old',old),('sparse',new)]:
        fn(*samples[0]);jax.block_until_ready(fn(*samples[0]));times=[]
        for st,ca in samples:
            start=time.perf_counter();value=fn(st,ca);jax.block_until_ready(value);times.append(time.perf_counter()-start)
        timing[label]=dict(warm_seconds=sum(times),mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),samples=times)
    speedup=timing['old']['warm_seconds']/timing['sparse']['warm_seconds']
    result=dict(status='PASS',input=receipt,input_state_sha256=sha(cp/'state.pkl'),gpu=2,environments=n,ticks=32,
        native_transitions=True,actions='fixed independently keyed diagnostic random actions, no policy learning',
        phi_and_cache_and_observations_bitwise_exact=True,cases=cases,timing=timing,microkernel_speedup=speedup,
        rollout_throughput_not_yet_proven=True,wall_seconds=time.time()-began,
        source_sha256={f:sha(Path(f)) for f in ['scripts/bun_native_v2_custody_reward.py','scripts/bun_native_v2_custody_fast.py']},unix=time.time())
    atomic_json(OUT/'PARITY_PROFILE.json',result)
    print(json.dumps(dict(status=result['status'],n=n,ticks=32,speedup=speedup,timing=timing)))


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',reason='inspect PROFILE.log; old workers untouched',unix=time.time()));raise
