"""Isolated full-game GPU/CPU evaluation; original native transition is unmodified."""
import argparse,json,os,sys,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/iter_pbt_v1'

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('request');p.add_argument('--gpu',type=int,default=0);p.add_argument('--backend',choices=('gpu','cpu'),default='gpu');p.add_argument('--probe-games',type=int);p.add_argument('--probe-output');args=p.parse_args()
 request=json.loads(Path(args.request).read_text());name=request['name'];split=request.get('split','dev');seed=20261101 if split=='dev' else 20261001
 output=ROOT/args.probe_output if args.probe_output else BASE/'eval'/name/'native'
 output.mkdir(parents=True,exist_ok=True);guard=lock(output/'native_worker.lock')
 gpu_guard=lock(BASE/f'NATIVE_GPU{args.gpu}.lock') if args.backend=='gpu' else None
 if (output/'NATIVE_COMPLETE.json').exists():
  marker=json.loads((output/'NATIVE_COMPLETE.json').read_text());assert sha(output/'NATIVE_SUMMARY.json')==marker['summary_sha256'];return
 cp=ROOT/request['checkpoint'];marker=json.loads((cp/'COMMITTED.json').read_text());assert sha(cp/'state.pkl')==marker['state_sha256'];assert sha(cp/'policy.pt')==marker['policy_sha256']
 settings=dict(CUDA_VISIBLE_DEVICES=str(args.gpu) if args.backend=='gpu' else '',JAX_PLATFORMS='cuda' if args.backend=='gpu' else 'cpu',JAX_PLATFORM_NAME='cuda' if args.backend=='gpu' else 'cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.03',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAXBOMB_RULE='bun',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR=str(BASE/'native_gpu/cache'),FULLMATCH_OUTPUT=str(output),FULLMATCH_GAMES_PER_MODE=str(args.probe_games or (64 if split=='dev' else 304)),FULLMATCH_SEED_START=str(seed))
 os.environ.update(settings)
 import jax
 jax.config.update('jax_default_matmul_precision','highest')
 # Initialize the requested device before the historical CPU-only module sets
 # its import defaults. Restore environment afterwards for any child imports.
 assert ('gpu' if jax.default_backend()=='gpu' else 'cpu')==args.backend
 from scripts import expert_reward_alignment_fullmatch as F
 os.environ.update(settings)
 assert ('gpu' if jax.default_backend()=='gpu' else 'cpu')==args.backend
 F.MODELS={'candidate':dict(state=str((cp/'state.pkl').relative_to(ROOT)),checkpoint=str(cp.relative_to(ROOT)),run_seed=0)}
 F.T.configure();F.provenance.install_provenance();scan,names=F.make_scan();params=F.load_params(F.MODELS['candidate'])
 atomic_json(output/'GPU_MANIFEST.json',dict(checkpoint=request['checkpoint'],policy_sha256=marker['policy_sha256'],state_sha256=marker['state_sha256'],backend=args.backend,gpu=args.gpu if args.backend=='gpu' else None,seed=seed,natural=True,teacher=False,opponent='fixed JAX rule3; not JS Hunter',source_sha256={f:sha(ROOT/f) for f in ['scripts/iter_pbt_native_eval.py','scripts/expert_reward_alignment_fullmatch.py']},memory_fraction=.03,preallocate=False,matmul_precision='highest',probe=bool(args.probe_games),unix=time.time()))
 # A single fixed-state diagnostic isolates backend policy arithmetic from
 # the later closed-loop rollout. It never supplies actions to a game.
 import numpy as np
 ds,da,dk=F.initial_batch(F.jnp.asarray(list(range(args.probe_games or 8)),F.jnp.int32))
 dp=F.E.prediction(params,ds)
 atomic_json(output/'INITIAL_POLICY_PROBE.json',dict(actor_keys=np.asarray(dk).tolist(),move_logits=np.asarray(dp[0]).tolist(),ability_logits=np.asarray(dp[1]).tolist(),position=np.asarray(ds.pos).tolist(),wall_sha256=__import__('hashlib').sha256(np.asarray(ds.wall).tobytes()).hexdigest(),brick_sha256=__import__('hashlib').sha256(np.asarray(ds.brick).tobytes()).hexdigest()))
 started=time.time();natural={};files={};backend_coverage={};modes=('categorical',) if split=='dev' else ('categorical','greedy');games=args.probe_games or (64 if split=='dev' else 304)
 # Cooperative yielding is checked between fixed scan chunks, never signals
 # any training process. Partial evaluation is rerun, completed episodes retained.
 original_scan=scan;scan_times=[]
 def guarded_scan(*a,**k):
  if (output/'YIELD_GPU.json').exists() or any((BASE/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json')):raise InterruptedError('GPU evaluation yielded to throughput/memory guard')
  tick_start=time.time();result=original_scan(*a,**k);jax.block_until_ready(result[0][0].t);scan_times.append(time.time()-tick_start)
  atomic_json(output/'NATIVE_PROGRESS.json',dict(phase='native_full_batch',backend=args.backend,gpu=args.gpu,scan_chunks_completed=len(scan_times),last_scan_seconds=scan_times[-1],first_scan_compile_execute_seconds=scan_times[0],elapsed_seconds=time.time()-started,unix=time.time()))
  return result
 from scripts.iter_pbt_native_gpu_policy import batches
 for mode in modes:
  by_index={};backends={}
  for path in sorted(output.glob(f'{mode}_*.json')):
   v=json.loads(path.read_text());assert v['policy_sha256']==marker['policy_sha256']
   for e in v['episodes']:
    if e['episode_index']>=games:continue
    assert sha(ROOT/e['provenance']['trace'])==e['provenance']['trace_sha256']
    if e['episode_index'] in by_index:raise ValueError('duplicate native episode')
    by_index[e['episode_index']]=e;backends[e['episode_index']]=v.get('backend','cpu')
   files[path.name]=sha(path)
  for indices in batches(games,set(by_index)):
   atomic_json(output/'NATIVE_PROGRESS.json',dict(phase='native_full_batch',backend=args.backend,gpu=args.gpu,batch_games=len(indices),mode=mode,unix=time.time()))
   try:rows=F.run_batch(params,'candidate',mode,indices,guarded_scan,names)
   except InterruptedError:
    atomic_json(output/'YIELDED.json',dict(reason='evaluation-only cooperative GPU guard',unix=time.time()));return
   path=output/f'{mode}_gpu_batch_{indices[0]:04d}.json';atomic_json(path,dict(policy_sha256=marker['policy_sha256'],backend=args.backend,episodes=rows));files[path.name]=sha(path)
   by_index.update({e['episode_index']:e for e in rows});backends.update({e['episode_index']:args.backend for e in rows})
  assert set(by_index)==set(range(games))
  rows=[by_index[i] for i in range(games)]
  backend_coverage[mode]={b:sum(v==b for v in backends.values()) for b in sorted(set(backends.values()))}
  natural[mode]=dict(games=games,capture_episodes=sum(e['events']['capture']>0 for e in rows),death_episodes=sum(e['events']['death']>0 for e in rows))
  atomic_json(output/'NATIVE_PROGRESS.json',dict(phase='mode_complete',backend=args.backend,gpu=args.gpu,mode=mode,games=games,elapsed_seconds=time.time()-started,unix=time.time()))
 value=dict(natural_jax=natural,episode_backend_counts=backend_coverage,policy_sha256=marker['policy_sha256'],backend=args.backend,seed=seed,elapsed_seconds=time.time()-started,probe=bool(args.probe_games),scan_seconds=scan_times,unix=time.time())
 atomic_json(output/'NATIVE_SUMMARY.json',value);atomic_json(output/'NATIVE_COMPLETE.json',dict(passed=True,summary_sha256=sha(output/'NATIVE_SUMMARY.json'),files=files,unix=time.time()))
if __name__=='__main__':main()
