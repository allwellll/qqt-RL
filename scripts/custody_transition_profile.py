"""GPU2 matched saved-state component profile; no training updates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('JAXBOMB_DEVICE_OVERRIDE','2'),JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
import json,pickle,time,statistics
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json,lock,sha

OUT=Path('runs/bun_native_v2_20261030/custody_performance_v1')


def main():
 guard=lock(OUT/'TRANSITION_PROFILE.lock')
 from scripts.bun_native_v2_runtime import install
 install()
 from scripts.bun_native_v2_custody_fast import install as fast
 fast()
 from scripts import carrier_mechanism_engine as E,combat_item_engine as T,bun_native_v2_custody_collector as M
 from scripts import bun_native_v2_custody_fast as F
 T.configure();M.configure(True,True)
 source=json.loads((OUT/'ATOMIC_SWITCH_READY.json').read_text())['states']['c']['checkpoint'];cp=Path(source['checkpoint'])
 assert sha(cp/'state.pkl')==source['state_sha256'];payload=pickle.load((cp/'state.pkl').open('rb'))
 st=jax.tree.map(lambda x:j.asarray(x[0]),payload['states']);ax=jax.tree.map(lambda x:j.asarray(x[0]),payload['aux'])
 keys=j.asarray(payload['keys'][0]);n=len(st.t);actions=j.zeros((n,2,2),j.int32).at[:,:,0].set(4)
 cache=ax['custody_v2'];core={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr','r1_attribute_highwater')}
 funcs={
 'phi_reference':lambda:F.REFERENCE(st,cache),
 'phi_sparse':lambda:F.potential(st,cache),
 'native_physics':lambda:jax.vmap(lambda s,a,k:E.env.step(s,a,k,auto_reset=False,return_info=True))(st,actions,keys),
 'native_reward_custody':lambda:T.transition(st,core,actions,keys,'control'),
 'full_transition_r1_custody':lambda:M.transition(st,ax,actions,keys,0.),
 }
 # Explicit arguments prevent constant-folding the saved trajectory.
 def build(label):
  if label=='phi_reference':return jax.jit(lambda s,a,k:F.REFERENCE(s,a['custody_v2']))
  if label=='phi_sparse':return jax.jit(lambda s,a,k:F.potential(s,a['custody_v2']))
  if label=='native_physics':return jax.jit(lambda s,a,k:jax.vmap(lambda ss,ac,kk:E.env.step(ss,ac,kk,auto_reset=False,return_info=True))(s,actions,k))
  if label=='native_reward_custody':return jax.jit(lambda s,a,k:T.transition(s,{kk:v for kk,v in a.items() if kk not in ('combat_mix','a3sr','r1_attribute_highwater')},actions,k,'control'))
  return jax.jit(lambda s,a,k:M.transition(s,a,actions,k,0.))
 results={}
 for label in funcs:
  atomic_json(OUT/'TRANSITION_PROFILE_PROGRESS.json',dict(stage=label,status='COMPILE',unix=time.time()))
  fn=build(label);began=time.perf_counter();value=fn(st,ax,keys);jax.block_until_ready(value);cold=time.perf_counter()-began
  for _ in range(3):jax.block_until_ready(fn(st,ax,keys))
  times=[]
  for _ in range(10):
   t=time.perf_counter();jax.block_until_ready(fn(st,ax,keys));times.append(time.perf_counter()-t)
  results[label]=dict(cold_seconds=cold,warm_median=statistics.median(times),warm_seconds=times)
  atomic_json(OUT/'TRANSITION_PROFILE_PROGRESS.json',dict(stage=label,status='MEASURED',results=results,unix=time.time()))
 atomic_json(OUT/'TRANSITION_PROFILE.json',dict(status='COMPLETE',source=source,environments=n,actions='joint wait; diagnostic only',
  results=results,training_updates=0,gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),limitations=['one matched wait-action state, not full student rollout','concurrent evaluator and R3 qualification may add contention'],unix=time.time()))
 print(json.dumps(results),flush=True)


if __name__=='__main__':
 try:main()
 except Exception:
  atomic_json(OUT/'TRANSITION_PROFILE_BLOCKED.json',dict(status='BLOCKED',reason='inspect TRANSITION_PROFILE.log; training untouched',unix=time.time()));raise
