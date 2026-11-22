"""Fused native JAX kill-mode training on ROCm; Actor/Critic-only checkpoints."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time

ROOT=Path(__file__).resolve().parents[1]
U930_PATH=ROOT/'data/carry-training-20261115/u930_actor.pt'
U930_SHA='1b1079d213a971ed7013739868d3b9848484ecf48ade25418142e16fb3a06d27'
U840_PATH=ROOT/'runs/kill-jax-8card-e512-20261115/u000840/actor.pt'
U840_SHA='dcc148a17acea599555c468c4295c8578bd515ff584035397b9920d684bda7f5'
# These reviewed versions use the same game rules and Actor/Critic parameter ABI.
# Only the launcher and learner inference batching differ from this version.
RUNNER_SOURCE='scripts/train_dcu_jax_kill.py'
COLLECTOR_SOURCE='qqt_rl/training/jax_kill_collect.py'
E512_RUNNER_SHA='93ff11c298ec0a98145a055a4739eab34ddc2af2b1593a23f8d31190abcc50ba'
UNBATCHED_RUNNER_SHA='429abececbe5ebd8fae551fb1b5bd2975e315f60935fd419ae82d29c7bbf5251'
UNBATCHED_COLLECTOR_SHA='f95976c6ce6b03b921ef9ac8a62e7f430bbc8714efb3f44926ae38634a46a427'
SOURCES=('qqt_rl/training/jax_kill_env.py','qqt_rl/training/jax_kill_collect.py',
 'qqt_rl/training/kill_ppo.py','qqt_rl/training/checkpoints.py','scripts/train_dcu_jax_kill.py',
 'scripts/r4_kill_arena.py','scripts/r4_k4_observation.py','scripts/r3_items_observation.py',
 'scripts/retention_phase_conditioning.py','jax_bomb/jax_net.py','jax_bomb/bun_jax_bots.py',
 'jax_bomb/je07_bun_env_r1_items.py','jax_bomb/je05_core_env_r1_items.py',
 'jax_bomb/je07_native_items.py','web/assets/maps/levels.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_checkpoint_sources(sources):
 reviewed_unbatched=(sources.get(RUNNER_SOURCE) in (E512_RUNNER_SHA,UNBATCHED_RUNNER_SHA)
                     and sources.get(COLLECTOR_SOURCE)==UNBATCHED_COLLECTOR_SHA)
 for file,expected in sources.items():
  if reviewed_unbatched and file in (RUNNER_SOURCE,COLLECTOR_SOURCE):
   continue
  if sha(ROOT/file)!=expected:
   raise RuntimeError('Checkpoint source mismatch: '+file)


def load_pinned_actor(path,expected,template,migrate):
 path=Path(path)
 if path.is_symlink() or not path.is_file() or sha(path)!=expected:
  raise ValueError('Historical opponent SHA mismatch: '+str(path))
 actor=migrate(pickle.loads(path.read_bytes()))
 import jax
 import numpy as np
 if jax.tree.structure(actor)!=jax.tree.structure(template):
  raise ValueError('Historical opponent tree mismatch: '+str(path))
 for actual,reference in zip(jax.tree.leaves(actor),jax.tree.leaves(template)):
  if actual.shape!=reference.shape or actual.dtype!=reference.dtype or not np.isfinite(actual).all():
   raise ValueError('Historical opponent ABI/finite mismatch: '+str(path))
 return actor

def arguments(argv=None):
 p=argparse.ArgumentParser(__doc__)
 p.add_argument('mode',choices=('plan','train'))
 p.add_argument('--devices',default='0,1,2,3,4,5,6,7')
 p.add_argument('--envs',type=int,default=2048,help='parallel environments per DCU')
 p.add_argument('--ticks',type=int,default=64)
 p.add_argument('--minibatch',type=int,default=2048,help='PPO samples per DCU per optimizer step')
 p.add_argument('--critic-warmup',type=int,default=2)
 p.add_argument('--max-updates',type=int,default=100000)
 p.add_argument('--hours',type=float,default=5)
 p.add_argument('--deadline-utc',help='absolute wall deadline including compilation, ISO UTC')
 p.add_argument('--out',type=Path)
 p.add_argument('--checkpoint',type=Path)
 p.add_argument('--initial-model',choices=('u930','DUEL_latest'),default='u930')
 p.add_argument('--u930',type=Path,default=U930_PATH)
 p.add_argument('--u840',type=Path,default=U840_PATH)
 p.add_argument('--seed',type=int,default=2026111502)
 p.add_argument('--acknowledge-training',action='store_true')
 a=p.parse_args(argv)
 try:a.device_ids=[int(x) for x in a.devices.split(',')]
 except ValueError:p.error('invalid devices')
 if not a.device_ids or min(a.device_ids)<0 or len(set(a.device_ids))!=len(a.device_ids):p.error('invalid devices')
 if a.envs<16 or a.envs%16 or a.ticks<1 or a.minibatch<1 or a.envs*a.ticks%a.minibatch:p.error('invalid batch geometry')
 if not 0<a.hours<=5 or a.max_updates<1 or a.critic_warmup<0:p.error('invalid budget')
 if not 0<=a.seed<=0xffffffff:p.error('invalid seed')
 if a.mode=='train' and (not a.acknowledge_training or a.out is None):p.error('explicit acknowledgement and new output required')
 if a.out and a.out.exists():p.error('output exists')
 return a

def plan(a):
 return dict(schema='qqt.dcu-jax-kill/v1',devices=a.device_ids,envs_per_device=a.envs,
  global_environments=len(a.device_ids)*a.envs,ticks=a.ticks,
  global_samples=len(a.device_ids)*a.envs*a.ticks,minibatch_per_device=a.minibatch,
  global_minibatch=len(a.device_ids)*a.minibatch,
  learner_inference_batch=min(a.envs,a.minibatch),
  simulator='existing native JAX core + map806 kill adapter',host_callbacks=False,
  rollout_gae_on_device=True,rule_bot_fraction=.1,
  opponent_model_fractions=dict(current=.09,recent=.09,u930=.315,u840=.315,SP=.045,G=.045),
  opponent_pool=dict(frozen_anchors=['u930','u840','SP','G'],candidate_auto_admission=False),
  initial_actor=a.initial_model,
  source_sha256={f:sha(ROOT/f) for f in SOURCES},deadline_utc=a.deadline_utc,
  optimizer_saved=False,precise_resume=False,checkpoint_policy=dict(every_updates=5,seconds=120,keep=64,budget_gib=4),
  checkpoint=str(a.checkpoint) if a.checkpoint else None)

def run(a,started):
 import numpy as np
 import jax
 import jax.numpy as j
 from scripts import portable_training as portable
 from scripts.r4_k4_observation import migrate
 from scripts.train_dcu_kill import require_local_launch
 from scripts.r4_k4_observation import actor_critic_forward
 from qqt_rl.training.kill_ppo import build
 from qqt_rl.training.jax_kill_collect import build as collector,consistency_check
 from qqt_rl.training.checkpoints import CheckpointStore,durable_json
 require_local_launch(a)
 devices=jax.local_devices();cards=len(devices)
 if cards!=len(a.device_ids) or any(d.platform!='gpu' for d in devices):raise RuntimeError('Expected explicit ROCm DCUs')
 # Build static map/schema constants on CPU once; rollouts stay on DCU.
 with jax.default_device(jax.devices('cpu')[0]):
  portable.verify_bundle();protocol=portable.read(portable.BUNDLE/'protocols/DUEL.json')
  template=portable.load_actor(protocol,'DUEL_latest')
  initial_actor=(load_pinned_actor(a.u930,U930_SHA,template,migrate) if a.initial_model=='u930'
                 else template)
  params=portable.fresh_parameters(initial_actor,a.seed)
  actor_source=str(a.checkpoint) if a.checkpoint else a.initial_model
  initial_actor_sha=U930_SHA if a.initial_model=='u930' else protocol['models']['DUEL_latest']['sha256']
  if a.checkpoint:
   meta=json.loads((a.checkpoint/'CHECKPOINT.json').read_text())
   # Reviewed cross-backend warm start: verify all recorded source files and
   # payloads before loading. Environment/RNG and Adam are intentionally fresh.
   verify_checkpoint_sources(meta['source_sha256'])
   initial_actor_sha=meta['actor_sha256']
   for b in ('actor','critic'):
    file=a.checkpoint/(b+'.pt')
    if sha(file)!=meta[b+'_sha256']:raise RuntimeError('Checkpoint hash mismatch')
    value=pickle.loads(file.read_bytes())
    if jax.tree.structure(value)!=jax.tree.structure(params[b]):raise RuntimeError('Checkpoint tree mismatch')
    for x,y in zip(jax.tree.leaves(value),jax.tree.leaves(params[b])):
     if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all():raise RuntimeError('Checkpoint schema/finite mismatch')
    params[b]=jax.tree.map(j.asarray,value)
  fixed={n:portable.load_actor(protocol,n) for n in ('SP_latest','G_frozen')}
  u930=load_pinned_actor(a.u930,U930_SHA,template,migrate)
  u840=load_pinned_actor(a.u840,U840_SHA,template,migrate)
  init_opt,_,update=build(actor_critic_forward,devices,a.minibatch)
  print(json.dumps(dict(event='prepare_native_jax',cards=cards)),flush=True)
  init_world,collect=collector(devices,a.envs,a.ticks,minibatch=a.minibatch)
  check_rollout=consistency_check(devices,a.minibatch)
  adam_host=init_opt(params)
  bank_host=jax.tree.map(lambda current,recent,best,prior,sp,g:
      j.stack([current,recent,best,prior,sp,g]),params['actor'],params['actor'],
      u930,u840,fixed['SP_latest'],fixed['G_frozen'])
 rep=lambda x:jax.device_put_replicated(x,devices)
 p=rep(params);adam=rep(adam_host);reference=rep(params['actor']);bank=rep(bank_host)
 keys=jax.random.split(jax.random.PRNGKey(a.seed),cards)
 print(json.dumps(dict(event='initialize_device_worlds',envs=a.envs,cards=cards)),flush=True)
 world=init_world(keys);jax.block_until_ready(world)
 finite=jax.pmap(lambda q,opt:j.all(j.stack([j.all(j.isfinite(x)) for x in jax.tree.leaves((q,opt))])),devices=devices)
 summary=jax.pmap(lambda q:j.stack([j.stack([j.sum(x),j.sum(x*x)]) for x in jax.tree.leaves(q)]),devices=devices)
 out=a.out;out.mkdir(parents=True,exist_ok=False)
 config=plan(a);config.update(device_kinds=[d.device_kind for d in devices],
  jax=jax.__version__,initial_actor=actor_source,initial_actor_sha256=initial_actor_sha,
  opponent_pool_sha256=dict(u930=U930_SHA,u840=U840_SHA),
  warm_start_from=str(a.checkpoint) if a.checkpoint else None,
  actor_critic_warm_start=bool(a.checkpoint),fresh_adam=True)
 durable_json(out/'PLAN.json',config)
 deadline=started+a.hours*3600
 if a.deadline_utc:
  absolute=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))
  if absolute.tzinfo is None:raise ValueError('deadline must specify UTC offset')
  deadline=min(deadline,time.monotonic()+absolute.timestamp()-time.time())
 stop=[False]
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
 def stopped():
  require_local_launch(a)
  return stop[0] or time.monotonic()>=deadline or (out/'USER_STOP').exists()
 store=CheckpointStore(out,64,4,8);last_saved=None;last_save=0.;completed=0;steps=0
 initial_hash=None
 def checkpoint(index):
  nonlocal last_saved,last_save,initial_hash
  if last_saved==index:return
  sums=np.asarray(summary(p))
  if not np.isfinite(sums).all() or not np.array_equal(sums,np.broadcast_to(sums[0],sums.shape)):raise RuntimeError('Model replicas differ')
  host=jax.tree.map(lambda x:np.asarray(x[0]),p)
  meta=dict(schema='qqt.kill-actor-critic/v1',update=index,source_sha256=config['source_sha256'],
   global_environment_steps=steps,optimizer_saved=False,precise_resume=False,
   rule='jax-native-map806-kill',mode_row=0,cards=cards,replica_summaries_equal=True,
   actor_source_checkpoint=actor_source)
  receipt=store.publish(index,host,meta);last_saved=index;last_save=time.monotonic()
  if index==0:initial_hash=json.loads((out/'LATEST.json').read_text())
  print(json.dumps(dict(event='checkpoint',update=index,**receipt)),flush=True)
 try:
  print(json.dumps(dict(event='ready',**config)),flush=True);checkpoint(0)
  for index in range(1,a.max_updates+1):
   if stopped():break
   begin=time.monotonic()
   if index==1:print(json.dumps(dict(event='compiling_fused_rollout',ticks=a.ticks,envs_per_device=a.envs)),flush=True)
   world,bank,keys,batch,metrics,overflow=collect(p,world,bank,keys)
   jax.block_until_ready((batch,metrics,overflow));collected=time.monotonic()
   if np.asarray(overflow).max():raise RuntimeError('Native item queue overflow')
   if index==1:
    errors,finite_batch=check_rollout(p,batch)
    jax.block_until_ready((errors,finite_batch))
    errors=np.asarray(errors)
    if (not np.asarray(finite_batch).all() or not np.isfinite(errors).all()
        or (errors[:,0]>.03).any() or (errors[:,1]>.1).any()):
     raise RuntimeError('Rollout inference check failed; refusing PPO: '+str(errors.tolist()))
    durable_json(out/'INFERENCE_CHECK.json',dict(
        max_abs_logp_error=float(errors[:,0].max()),
        max_abs_value_error=float(errors[:,1].max()),passed=True))
   if stopped():break
   keys=jax.vmap(lambda k:jax.random.split(k)[0])(keys)
   warm=index<=a.critic_warmup
   if index==1:print(json.dumps(dict(event='compiling_ppo',collect_seconds=collected-begin)),flush=True)
   (p,adam),loss=update(p,adam,reference,batch,keys,np.full(cards,warm))
   jax.block_until_ready(loss);updated=time.monotonic()
   if not np.asarray(finite(p,adam)).all() or not np.isfinite(np.asarray(loss)).all():raise RuntimeError('Nonfinite training state')
   completed=index;steps+=cards*a.envs*a.ticks
   row=dict(update=index,critic_only=warm,cards=cards,environment_steps=steps,
    seconds=updated-begin,collect_seconds=collected-begin,ppo_seconds=updated-collected,
    elapsed_seconds=time.monotonic()-started,remaining_seconds=max(0,deadline-time.monotonic()),
    metrics=np.asarray(metrics).sum(0).tolist(),loss=np.asarray(loss).mean((0,1)).tolist())
   with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(dict(event='update',**row)),flush=True)
   if index==1 or index==a.critic_warmup or index%5==0 or time.monotonic()-last_save>=120:checkpoint(index)
  checkpoint(completed)
  receipt=dict(status='PASS_FUSED_JAX_DCU_UPDATE' if completed else 'NO_UPDATE',updates=completed,
   cards=cards,environment_steps=steps,elapsed_seconds=time.monotonic()-started,
   optimizer_saved=False,automatic_successor=False)
  durable_json(out/'RESULT.json',receipt);print(json.dumps(receipt),flush=True)
 finally:
  durable_json(out/'STOPPED.json',dict(updates=completed,environment_steps=steps,automatic_successor=False))

def main(argv=None):
 started=time.monotonic();a=arguments(argv);os.chdir(ROOT)
 if a.mode=='plan':print(json.dumps(plan(a),indent=2));return
 from qqt_rl.training.dcu import configure
 configure(a.device_ids[0]);visible=','.join(map(str,a.device_ids))
 os.environ.update(HIP_VISIBLE_DEVICES=visible,CUDA_VISIBLE_DEVICES=visible,
  JAX_ENABLE_COMPILATION_CACHE='false',JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='false',
  JAX_PLATFORMS='rocm,cpu')
 run(a,started)

if __name__=='__main__':main()
