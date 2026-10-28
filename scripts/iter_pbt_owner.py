"""Durable ITER-PBT consumer: one slot per free physical GPU, asynchronous CPU selection."""
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.combat_mix_owner import resources,worker_reserved_devices,read
from scripts.iter_pbt_recipe import explore,replacements
from scripts.iter_pbt_eval_scheduling import fresh_cohort
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/iter_pbt_v1'
CPU={0:'32-39',1:'40-47',2:'48-55',3:'72-79',4:'80-87',5:'88-95',6:'96-103',7:'104-111'}

class AdoptedWorker:
 def __init__(self,pid,out):self.proc=Path('/proc')/str(pid);self.out=out;self.returncode=None
 def poll(self):
  try:
   if self.proc.exists() and (self.proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return None
  except OSError:pass
  stage=read(self.out/'LAUNCH.json',{}).get('stage')
  marker={'smoke':'SMOKE.json','fresh_restore':'RESTORE.json','formal':'COMPLETE.json'}.get(stage,'COMPLETE.json')
  self.returncode=0 if read(self.out/marker,{}).get('passed') else 1
  return self.returncode

def notify(key,message,**details):
 p=BASE/'notifications'/f'{key}.json'
 if p.exists():return
 atomic_json(p,dict(message=message,unix=time.time(),**details))
 try:
  r=subprocess.run(['jdme-notify','send',message],capture_output=True,timeout=20)
  atomic_json(p.with_suffix('.delivery.json'),dict(exit_code=r.returncode,unix=time.time()))
 except (OSError,subprocess.TimeoutExpired) as e:atomic_json(p.with_suffix('.delivery.json'),dict(error=type(e).__name__))

def live_workers(module):
 result={};parents={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   args=(d/'cmdline').read_bytes().split(b'\0')
   if module.encode() not in args or (d/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
   paths=[a.decode() for a in args if a.startswith((b'runs/iter_pbt_v1/protocols/',b'runs/iter_pbt_v1/eval_requests/'))]
   if paths:
    name=Path(paths[0]).stem
    pid=int(d.name);stat=(d/'stat').read_text().rsplit(')',1)[1].split();ppid=int(stat[1])
    if name in result:
     # fork-before-exec helper inherits argv briefly; only independent roots
     # are duplicate workers. The worker file lock remains authoritative.
     if ppid==result[name]:continue
     if parents.get(name)==pid:result[name]=pid;parents[name]=ppid;continue
     raise RuntimeError('duplicate independent PBT worker '+name)
    result[name]=pid;parents[name]=ppid
  except (OSError,ValueError):continue
 return result

def source_candidates():
 rows=[]
 for p in (ROOT/'runs/combat_mix_v1').glob('formal*/*/COMPLETE.json'):
  if 'legacy_shared' in str(p):continue
  r=read(p)
  if not r or not r.get('passed'):continue
  cp=ROOT/r['checkpoint']
  if not (cp/'COMMITTED.json').exists():continue
  rank=0 if 'shaping' in str(p) else 1 if 'curriculum' in str(p) else 2 if 'opp_mix_' in str(p) else 3
  rows.append((rank,p.parent.name,r))
 scored=[]
 for rank,name,r in rows:
  selection=read(BASE/'eval'/('initial_'+name)/'SELECTION.json',{})
  scored.append((not selection.get('eligible',False),-selection.get('score',-100000),rank,name,r))
 return [r for *_,r in sorted(scored,key=lambda v:v[:4])]

def is_selection_checkpoint(update,start):
 return update-start==80

def publish_evaluations():
 # Both training and selection are append-only; never await a CPU score to train.
 for p in BASE.glob('formal/*/eval_queue/u*.json'):
  m=read(p);cp=m['checkpoint'];name=p.parent.parent.name+'_'+p.stem
  if m.get('metrics',{}).get('fresh_parent'):continue
  pr=read(BASE/'protocols'/f'{p.parent.parent.name}.json',{})
  if not is_selection_checkpoint(m['update'],pr['chunk_start_update']):continue
  if not (BASE/'eval_requests'/f'{name}.json').exists():
   atomic_json(BASE/'eval_requests'/f'{name}.json',dict(name=name,checkpoint=cp,split='dev',slot=int(p.parent.parent.name[4:6]),chunk=int(p.parent.parent.name.split('_c')[1]),generation=pr.get('generation',0),update=m['update']))
 for cp in (ROOT/'runs/combat_mix_v1').glob('formal*/*/u000320/COMMITTED.json'):
  name='initial_'+cp.parent.parent.name
  target=BASE/'eval_requests'/f'{name}.json'
  if not target.exists():atomic_json(target,dict(name=name,checkpoint=str(cp.parent.relative_to(ROOT)),split='dev',role='available_initial_candidate'))

def ensure_protocol(slot,chunk,source,recipe,decision):
 name=f'slot{slot:02d}_c{chunk:06d}';p=BASE/'protocols'/f'{name}.json'
 if p.exists():return p
 seed=2026110100+slot;start=source['update']
 hashes={f:sha(ROOT/f) for f in read(BASE/'PROTOCOL.json')['source_files']}
 previous=read(BASE/'protocols'/f'slot{slot:02d}_c{chunk-1:06d}.json',{})
 generation=previous.get('generation',0)+(0 if decision['kind']=='technical_dtype_recovery' else 1) if previous else 0
 value=dict(namespace='runs/iter_pbt_v1/formal',generation=generation,common=dict(devices=1,envs_per_device=2048,rollout_ticks=128,global_minibatch=8192,epochs=1,bank_probability=.25,dagger=0,combat_kl=0),
  arms={name:recipe},sources={str(seed):source},source_sha256=hashes,
  environment_steps_per_run=(start+80)*262144,chunk_start_update=start,
  spawn_manifest='runs/combat_mix_v1/js_web_compare_v1/shared/spawns.json',warm_baseline_env_per_second=20255.36135177422,
  fixed_actor_source=read(ROOT/'runs/combat_mix_v1/PROTOCOL.json')['sources']['2026103051'],
  slot=slot,chunk=chunk,decision=decision,additional_environment_steps=20971520,additional_optimizer_steps=2560)
 atomic_json(p,value)
 with (BASE/'LINEAGE.jsonl').open('a') as f:f.write(json.dumps(dict(run=name,source=source,recipe=recipe,decision=decision,unix=time.time()))+'\n');f.flush();os.fsync(f.fileno())
 return p

def cohort_decisions(generation,current_generations=None):
 if current_generations is not None and not fresh_cohort(generation,current_generations):return []
 rows=[]
 for g in range(8):
  matches=[(p,read(p)) for p in sorted((BASE/'protocols').glob(f'slot{g:02d}_c??????.json'))]
  matches=[(p,pr) for p,pr in matches if pr.get('generation')==generation]
  if not matches:return []
  p,pr=matches[-1]
  if not pr:return []
  end=pr['chunk_start_update']+80
  r=read(BASE/'eval'/f'{p.stem}_u{end:06d}'/'SELECTION.json')
  if not r:return []
  rows.append(dict(slot=g,**r))
 return replacements(rows)

def prepare_slot_cache(slot,name,base=BASE):
 """Reuse content-addressed compiler files; never move a live worker cache."""
 mapping=read(base/'SLOT_CACHE_TARGETS.json',{})
 cache=base/'formal'/'cache'/name;cache.parent.mkdir(parents=True,exist_ok=True)
 if cache.exists():
  mapping.setdefault(str(slot),str(cache.relative_to(base)))
 elif str(slot) in mapping:
  target=base/mapping[str(slot)]
  target.mkdir(parents=True,exist_ok=True)
  cache.symlink_to(os.path.relpath(target,cache.parent),target_is_directory=True)
 else:
  cache.mkdir();mapping[str(slot)]=str(cache.relative_to(base))
 atomic_json(base/'SLOT_CACHE_TARGETS.json',mapping)
 return cache

def main():
 guard=lock(BASE/'OWNER.lock');state=read(BASE/'SLOTS.json',{});children={};idle={};last_sample=0
 atomic_json(BASE/'OWNER_ACCEPTANCE.json',dict(unique_lock=True,legacy_queues_resumed=False,protocol_sha256=sha(BASE/'PROTOCOL.json'),unix=time.time()))
 while True:
  publish_evaluations();live=live_workers('scripts.iter_pbt_train');paused=any((BASE/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json'))
  for name,pid in live.items():
   if name not in children:
    stream=(BASE/'logs'/f'{name}.adopt.log').open('ab',buffering=0)
    children[name]=(AdoptedWorker(pid,BASE/'formal'/name),stream)
  for name,(proc,stream) in list(children.items()):
   if proc.poll() is None:continue
   rc=proc.returncode;stream.close();del children[name]
   if rc:
    atomic_json(BASE/'formal'/name/'TECHNICAL_FAILURE.json',dict(exit_code=rc,unix=time.time()))
    notify(name+'_FAILED',f'ITER-PBT技术阻塞：{name} exit={rc}；已保存日志及最近原子状态，其他slot继续。')
  samples=resources();reserved=worker_reserved_devices();available=[]
  if time.time()-last_sample>=10:
   with (BASE/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),gpus=samples))+'\n')
   last_sample=time.time()
  for s in samples:
   g=s['gpu'];busy=s['occupied'] or g in reserved
   if busy:idle.pop(g,None)
   else:
    idle.setdefault(g,time.time());available.append(g)
    if time.time()-idle[g]>300:notify(f'GPU{g}_IDLE_{int(idle[g])}',f'ITER-PBT GPU{g}空闲超过5分钟，持久状态记录准备/技术阻塞原因。',slot=state.get(str(g)))
  # A completed smoke process is a phase gap, not a released legacy candidate.
  # Preserve the authorized fixed-run boundary through smoke/restore handoff.
  boundary=read(ROOT/'runs/combat_mix_v1/USER_PBT_QUEUE_OVERRIDE.json',{}).get('received_active',{})
  from scripts.combat_mix_owner import workers as combat_workers
  actual=combat_workers();remembered=read(BASE/'COMBAT_RESERVATIONS.json',{})
  remembered.update(actual);atomic_json(BASE/'COMBAT_RESERVATIONS.json',remembered)
  for old_name,g in boundary.items():
   g=actual.get(old_name,remembered.get(old_name,g))
   from scripts.combat_mix_owner import run_output,technical_failure
   old_out=run_output(old_name)
   if not (old_out/'COMPLETE.json').exists() and not technical_failure(old_name):
    if g in available:available.remove(g)
  for g in range(8):
   previous=state.get(str(g));name=previous['run'] if previous else None
   if previous and name not in live and (BASE/'formal'/name/'TECHNICAL_FAILURE.json').exists():
    log=(BASE/'logs'/f'{name}.log').read_text(errors='replace')
    if 'carry input and carry output must have equal types' in log and 'int8[2048]' in log:
     prior=read(BASE/'protocols'/f'{name}.json');source=next(iter(prior['sources'].values()))
     path=ensure_protocol(g,previous['chunk']+1,source,prior['arms'][name],dict(kind='technical_dtype_recovery',failed_run=name,zero_effective_updates_in_failed_run=True))
     atomic_json(BASE/'formal'/name/'RECOVERY.json',dict(successor=path.stem,amendment='protocols/amendment02_dtype_recovery.json',unix=time.time()))
     name=path.stem;state[str(g)]=dict(run=name,chunk=previous['chunk']+1,accepted=False);atomic_json(BASE/'SLOTS.json',state);previous=state[str(g)]
   if previous:
    out=BASE/'formal'/name;pr=read(BASE/'protocols'/f'{name}.json');done=read(out/'COMPLETE.json')
    prog=read(out/'PROGRESS.json',{}).get('last',{})
    if prog.get('finite') and all(v>0 for v in prog['update_l2'].values()):
     notify(name+'_FIRST_UPDATE',f"ITER-PBT真实更新：{name} u{prog['new_update']}，Actor/Critic有限非零，{262144/prog['cycle_seconds']:.0f}环境步/秒；CPU评估独立。",evidence=prog)
    if done and not previous.get('accepted'):
     cp=ROOT/done['checkpoint'];assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
     previous['accepted']=True;atomic_json(BASE/'SLOTS.json',state)
     notify(name+'_COMPLETE',f'ITER-PBT chunk完成：{name}，新增20971520环境步/2560 Adam步，双SHA验收；立即接续下一chunk。',checkpoint=done)
    if name in live:continue
    if (out/'TECHNICAL_FAILURE.json').exists():continue
    if done:
     source=done;chunk=previous['chunk']+1;recipe=pr['arms'][name];decision=dict(kind='continue_without_waiting_for_async_scores')
     scored_chunk=None;decisions=[]
     current_generations={k:read(BASE/'protocols'/f"{v['run']}.json").get('generation',0) for k,v in state.items()}
     for c in range(pr['generation'],max(-1,pr['generation']-3),-1):
      decisions=cohort_decisions(c,current_generations)
      if decisions:scored_chunk=c;break
     for move in decisions:
      if move['loser']==g:
       consumed=BASE/'decisions'/f'cohort{scored_chunk:06d}_slot{g:02d}.json'
       if consumed.exists():continue
       donor_name=Path(move['checkpoint']).parent.name
       donor=read(BASE/'protocols'/f'{donor_name}.json')
       source=read(ROOT/move['checkpoint']/'COMMITTED.json')
       recipe=explore(next(iter(donor['arms'].values())),20261101+g+1000*chunk)
       decision=dict(kind='full_state_exploit_explore',scored_cohort=scored_chunk,**move)
       atomic_json(consumed,dict(next_chunk=chunk,decision=decision,unix=time.time()))
    else:
     if g not in available:continue
     source=None;chunk=previous['chunk'];recipe=None;decision=None
   else:
    if g not in available:continue
    candidates=source_candidates()
    if not candidates:continue
    source=candidates[g%len(candidates)];chunk=0
    recipe=dict(lr=7.5e-5,entropy_coefficient=.005,combat_shaping=.1 if g%2 else 0,curriculum_fraction=.25,opponent_weights=[.2,.25,.2,.15,.1,.1])
    if g:recipe=explore(recipe,20261101+g)
    decision=dict(kind='provisional_available_candidate',ranking='newdev scores pending; not a best claim')
   if paused or g not in available:continue
   if source is not None:
    path=ensure_protocol(g,chunk,source,recipe,decision);name=path.stem
    state[str(g)]=dict(run=name,chunk=chunk,accepted=False);atomic_json(BASE/'SLOTS.json',state)
   else:path=BASE/'protocols'/f'{name}.json'
   out=BASE/'formal'/name;spec=read(path);seed=2026110100+g
   args=[str(path.relative_to(ROOT)),name,str(seed)];stage='formal'
   if not (out/'SMOKE.json').exists():args.append('--smoke');stage='smoke'
   elif not read(out/'SMOKE.json')['passed']:continue
   elif not (out/'RESTORE.json').exists():args.append('--verify-restore');stage='fresh_restore'
   prepare_slot_cache(g,name)
   stream=(BASE/'logs'/f'{name}.log').open('ab',buffering=0)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),JAXBOMB_DEVICE_OVERRIDE=str(g),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   proc=subprocess.Popen(['taskset','-c',CPU[g],sys.executable,'-m','scripts.iter_pbt_train']+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   children[name]=(proc,stream);idle.pop(g,None)
   atomic_json(out/'LAUNCH.json',dict(stage=stage,gpu=g,protocol=str(path.relative_to(ROOT)),unix=time.time()))
  progress={};work={}
  for g,entry in state.items():
   progress[g]=dict(run=entry['run'],launch=read(BASE/'formal'/entry['run']/'LAUNCH.json'),progress=read(BASE/'formal'/entry['run']/'PROGRESS.json'))
   executed=[]
   for trace in (BASE/'formal').glob(f'slot{int(g):02d}_c*/updates.jsonl'):
    executed.extend(json.loads(line) for line in trace.read_text().splitlines() if line.strip())
   work[g]=dict(actual_recorded_updates=len(executed),actual_environment_steps=sum(r['actual_environment_steps_this_update'] for r in executed),actual_optimizer_steps=sum(r['actual_optimizer_steps_this_update'] for r in executed),counter_scope='new slot work only; donor history excluded; all recorded executions including a potential replay are retained')
  atomic_json(BASE/'WORK_TOTALS.json',work)
  atomic_json(BASE/'STATUS.json',dict(status='PAUSED' if paused else 'RUNNING',slots=state,progress=progress,work_totals=work,active=list(live),gpu_samples=samples,updated_unix=time.time(),CPU_evaluation_independent=True))
  time.sleep(5)
if __name__=='__main__':main()
