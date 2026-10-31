"""Bounded measurement of direct historical batched collector, no optimizer."""
import os,json,pickle,time,argparse
from pathlib import Path
os.environ.update(JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
import jax
import jax.numpy as j
import numpy as np
from scripts import combat_mix_legacy as L,ppo_scale_longrun_train as P
from scripts.population_run_state import atomic_json,sha,lock

def main():
 p=argparse.ArgumentParser();p.add_argument('--envs',type=int,default=2048);p.add_argument('--steps',type=int,default=128);a=p.parse_args()
 b=Path('runs/combat_mix_v1/legacy_native_probe_v1');b.mkdir(exist_ok=True);guard=lock(b/'OWNER.lock')
 doc=json.loads(Path('runs/combat_mix_v1/protocols/legacy_mix_s2026103051.json').read_text())
 source=Path(doc['sources']['2026103051']['checkpoint']);assert sha(source/'state.pkl')==doc['sources']['2026103051']['state_sha256']
 x=pickle.load((source/'state.pkl').open('rb'));P.T.configure();P.C.install_provenance()
 st,ax=jax.tree.map(lambda z:j.asarray(z[0,:a.envs]),(x['states'],x['aux']));ax=L.base_aux(ax)
 key=j.asarray(x['keys'][0,0]);pool=L.initial_pool(pickle.load(Path(doc['legacy_inputs']['initial_pool']['path']).open('rb')),x['params']['actor'])
 frozen=pickle.load(Path(doc['legacy_inputs']['frozen']['path']).open('rb'));frozen={k:frozen[k] for k in ('g0','stage1_teacher')};frozen['parent']=x['params']['actor']
 pf=L.ops.pfsp_weights(j.zeros((11,6)));course=L.course()
 atomic_json(b/'PROTOCOL.json',dict(status='BOUNDED_TECHNICAL_PROBE',iterations=3,envs=a.envs,ticks=a.steps,no_optimizer=True,no_training_claim=True,source=str(source),source_sha256=sha(source/'state.pkl'),direct_original_ops_collect=True,seat='historical alternating P0/P1; not equivalent to failed per-env P0 adapter',purpose='verify batching and historical parameter-sharing cost before any training amendment'))
 # Materialize only PPO batch and diagnostics, as production does; unused labels stay dead.
 def collect(q,s,aux,k,pool,pf):
  ss,aa,kk,batch,stats=L.ops.collect(q,s,aux,k,pool,frozen,course,pf,a.steps,assist_probability=.1,expert_hazard_mode='local_safe',bridge_config=dict(window=120,scale=2.),return_config=dict(window=120),chase_stage=j.int32(0),chase_supervision=False,dagger_round=j.int32(163))
  return (ss,aa,kk),batch[:6]+(batch[6][0],batch[6][1]),stats
 call=jax.jit(collect);rows=[]
 for i in range(3):
  t=time.monotonic();(st,ax,key),batch,stats=call(x['params'],st,ax,key,pool,pf);jax.block_until_ready((batch,st));elapsed=time.monotonic()-t
  row=dict(iteration=i+1,cycle_seconds=elapsed,env_per_second=a.envs*a.steps/elapsed,finite=all(np.isfinite(np.asarray(z)).all() for z in jax.tree.leaves(batch)),environment_steps=a.envs*a.steps,unix=time.time());rows.append(row)
  atomic_json(b/'PROGRESS.json',dict(rows=rows))
 atomic_json(b/'COMPLETE.json',dict(passed=all(r['finite'] for r in rows),rows=rows,protocol_sha256=sha(b/'PROTOCOL.json'),no_optimizer=True))
if __name__=='__main__':main()
