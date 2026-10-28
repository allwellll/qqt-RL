"""Durable, finite A/B/C driver. Never resumes a continuous queue."""
import os
os.environ.setdefault('JAXBOMB_RULE','bun');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import pathlib,sys,json,pickle,time,hashlib,subprocess,argparse,signal
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'runs/carrier_mechanism_20261023/v1'
from scripts.population_run_state import atomic_json,sha,lock

def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('wb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
 t.replace(p)
def notify(text):
 print(text,flush=True);atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=text,unix=time.time()))
 try:subprocess.run(['jdme-notify','send',text],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
 except (OSError,subprocess.TimeoutExpired):pass

def authorize():
 d=read(B/'SCOPED_GPU_AUTHORIZATION.json');assert d['authorized_by']=='explicit_user_instruction' and not d['old_continuous_resume']
 assert sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')==d['old_global_pause_sha256'],'old continuous pause changed'
 assert not (B/'USER_STOP.json').exists(),'diagnostic user stop'

def extract():
 import jax,numpy as np
 from scripts.terminal_checkpoint_restore import digest
 for name,c in read(B/'PROTOCOL.json')['models'].items():
  target=B/'inputs'/f'{name}.pkl'
  if target.exists():continue
  cp=R/c['checkpoint'];assert sha(cp/'state.pkl')==c['state_sha256'];s=pickle.load((cp/'state.pkl').open('rb'))
  assert digest(s['live'])==read(cp/'COMMITTED.json')['full_state_digest']
  params=jax.tree.map(lambda x:np.asarray(x)[0],s['live']['params']);adam=jax.tree.map(lambda x:np.asarray(x)[0],s['live']['adam'])
  save(target,dict(params=params,adam=adam,checkpoint=c,source_state_digest=digest(s['live'])));del s
  print('extracted Actor/Critic/Adam',name,sha(target),flush=True)
 atomic_json(B/'INPUTS_COMPLETE.json',dict(passed=True,models={p.stem:sha(p) for p in (B/'inputs').glob('*.pkl')}))

def collect():
 authorize()
 import jax,jax.numpy as jnp,numpy as np
 from scripts import carrier_mechanism_engine as E
 from scripts.terminal_checkpoint_restore import digest
 from scripts.carrier_mechanism_stats import validate_bank_record
 E.configure();devices=jax.local_device_count();assert devices==8 and jax.default_backend()=='gpu'
 n=16;total=devices*n;chunk=128;protocol=read(B/'PROTOCOL.json');records=[];excluded=[];attempted=[]
 init=jax.pmap(E.initial,in_axes=(0,None))
 rollout=jax.pmap(lambda p,c:E.collection_chunk(p,c,chunk))
 rep=lambda x:jax.tree.map(lambda a:jnp.stack([jnp.asarray(a)]*devices),x)
 for model_index,name in enumerate(protocol['models']):
  data=pickle.load((B/'inputs'/f'{name}.pkl').open('rb'));params=rep(data['params'])
  for layout,split in enumerate(['train','dev','holdout']):
   target=B/'bank'/f'{name}_{split}.pkl';manifest=B/'bank'/f'{name}_{split}.json'
   if target.exists() and manifest.exists():
    records.extend(read(manifest)['records']);continue
   keys=jax.random.split(jax.random.PRNGKey(2026102300+model_index*100+layout),total).reshape(devices,n,2)
   states,aux=init(keys,layout)
   carry=(states,aux,keys,jnp.zeros((devices,n),bool),jnp.zeros((devices,n),bool),states,states,aux,aux,keys,keys,jnp.zeros((devices,n,2,2),jnp.int32),keys)
   prefix=[];prefixkeys=[];start=time.time()
   for ci in range(20):
    carry,trajectory=rollout(params,carry);jax.block_until_ready(carry[0].t)
    acts,ek,hit,reward,done=jax.device_get(trajectory)
    prefix.append(np.transpose(acts,(1,0,2,3,4)).reshape(chunk,total,2,2));prefixkeys.append(np.transpose(ek,(1,0,2,3)).reshape(chunk,total,2))
    got=np.asarray(carry[4]).reshape(-1);ended=np.asarray(carry[3]).reshape(-1)
    atomic_json(B/'STATUS.json',dict(stage='A_COLLECT_REAL_PICKUP',model=name,split=split,source_episodes=total,ticks=(ci+1)*chunk,eligible_acquisitions=int(got.sum()),env_steps=(ci+1)*chunk*total,devices=8,unix=time.time()))
    print('collection',name,split,ci+1,'acquisitions',got.sum(),'ended',ended.sum(),'seconds',round(time.time()-start,1),flush=True)
    if ended.all():break
   host=jax.device_get(carry);flatten=lambda tr:jax.tree.map(lambda a:np.asarray(a).reshape((total,)+a.shape[2:]),tr)
   st,ax,key,end,got,pre,post,preaux,postaux,prekey,postkey,action,ekey=map(flatten,host)
   eligible=np.flatnonzero(got);chosen=eligible[:8];items=[];metadata=[];allacts=np.concatenate(prefix);allkeys=np.concatenate(prefixkeys)
   for i in range(total):
    attempted.append(dict(model=name,split=split,source_episode=f'{name}-{split}-{i:03d}',real_eligible_pickup=bool(got[i]),ended=bool(end[i]),ticks=int(st.t[i])))
    if not got[i]:excluded.append(dict(model=name,split=split,source_episode=f'{name}-{split}-{i:03d}',reason='no genuine eligible stock acquisition in declared source episode budget'))
   for i in chosen:
    take=lambda tr:jax.tree.map(lambda a:a[i:i+1],tr)
    before,after=take(pre),take(post);event_tick=int(before.t[0]);source=f'{name}-{split}-{i:03d}'
    rec=dict(before_carried=int(before.bun_carried[0,0]),after_carried=int(after.bun_carried[0,0]),learner_alive_before=bool(before.alive[0,0]),learner_alive_after=bool(after.alive[0,0]),stock_steal=bool(before.bun_stored[0,1,1]>after.bun_stored[0,1,1]),enemy_alive=bool(before.alive[0,1]),enemy_trapped=int(before.trapped[0,1]),source_episode=source,layout=['native','open','checker'][layout],split=split,provenance='actual_env_step',source_policy=name,source_episode_seed=2026102300+model_index*100+layout,episode_index=int(i),map_id=806,event_tick=event_tick,pre_hash=digest(before),post_hash=digest(after),reward_aux_hash=digest(take(postaux)),prefix_actions_hash=digest(allacts[:event_tick+1,i]),prefix_envkeys_hash=digest(allkeys[:event_tick+1,i]))
    validate_bank_record(rec)
    item=dict(record=rec,before=before,states=after,preaux=take(preaux),aux=take(postaux),prekey=take(prekey),keys=take(postkey),acquisition_actions=take(action),acquisition_envkeys=take(ekey),prefix_actions=allacts[:event_tick+1,i],prefix_envkeys=allkeys[:event_tick+1,i])
    items.append(item);metadata.append(rec)
   save(target,items);atomic_json(manifest,dict(records=metadata,sha256=sha(target),source_episode_count=total,acquisitions=int(got.sum()),selected=len(items),finite_budget=True,wall_seconds=time.time()-start));records.extend(metadata)
   atomic_json(B/'COLLECTION_EXCLUSIONS.json',dict(excluded=excluded,attempted=attempted))
   if len(items)<8:notify(f'A收集覆盖不足：{name}/{split} 实际合法取包{len(items)}/8；保留全部失败与固定预算，不伪造状态。')
 atomic_json(B/'BANK_FROZEN.json',dict(passed=True,states=len(records),independent_source_episodes=len({r['source_episode'] for r in records}),records=records,expected=96,coverage_shortfall=96-len(records),files={p.name:sha(p) for p in (B/'bank').glob('*.pkl')},environment_steps=sum(x['ticks'] for x in attempted),padded_maximum_steps=len(attempted)*2560,source_episodes=len(attempted),map_ids=[806],cross_map_unproven=True))
 notify(f'A真实取包Bank冻结：{len(records)}/96独立来源状态，完整状态/aux/RNG及入站动作证据已保存；开始fresh-process parity与V/Q实测。')


def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','collect','parity','AB','C','owner']);a=p.parse_args()
 if a.stage=='extract':extract()
 elif a.stage=='collect':collect()
 elif a.stage in ['parity','AB','C']:
  from scripts import carrier_mechanism_analysis as analysis
  getattr(analysis,{'parity':'parity','AB':'analyze','C':'train'}[a.stage])()
 else:owner()

def owner():
 authorize();unique=lock(B/'owner.lock');atomic_json(B/'OWNER.json',dict(unique=True,bounded=True,old_continuous_paused=True,unix=time.time()))
 for stage,marker,cpu in [('extract','INPUTS_COMPLETE.json',True),('collect','BANK_FROZEN.json',False),('parity','PARITY_COMPLETE.json',False),('AB','AB_COMPLETE.json',False),('C','C_COMPLETE.json',False)]:
  if (B/marker).exists():continue
  authorize()
  if stage=='collect':
   def external_collection():
    for procdir in pathlib.Path('/proc').glob('[0-9]*'):
     try:
      argv=(procdir/'cmdline').read_bytes().split(b'\0')
      if len(argv)>2 and argv[-2:]==[b'collect',b''] and b'scripts/carrier_mechanism_run.py' in argv:return True
     except (OSError,PermissionError):pass
    return False
   if external_collection():
    atomic_json(B/'OWNER_STAGE.json',dict(stage=stage,state='FOLLOWING_EXISTING_UNIQUE_COLLECTOR',unix=time.time()))
    while external_collection():authorize();time.sleep(5)
    if not (B/marker).exists():
     atomic_json(B/'TECHNICAL_FAILURE.json',dict(stage=stage,reason='existing collector exited without marker',unix=time.time()));notify('真实取包收集技术故障：现有collector未写终态，禁止重复领取或进入C。');return
    continue
  atomic_json(B/'OWNER_STAGE.json',dict(stage=stage,state='RUNNING',unix=time.time()))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else '0,1,2,3,4,5,6,7',JAX_PLATFORMS='cpu' if cpu else 'cuda',JAX_PLATFORM_NAME='cpu' if cpu else 'cuda',JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
  proc=subprocess.Popen([sys.executable,'scripts/carrier_mechanism_run.py',stage],cwd=R,env=env,stdout=(B/f'{stage}.log').open('a'),stderr=subprocess.STDOUT)
  code=proc.wait()
  if code or not (B/marker).exists():
   atomic_json(B/'TECHNICAL_FAILURE.json',dict(stage=stage,exit_code=code,unix=time.time()));notify(f'真实持包诊断技术故障：{stage} exit={code}；旧continuous仍暂停，无自动后继。');return
  notify(f'真实持包诊断阶段终态：{stage}完成；证据{marker}。')
 atomic_json(B/'GPU_PIPELINE_COMPLETE.json',dict(gpu_stages_complete=True,old_continuous_resume=False,automatic_successor=False,unix=time.time()));notify('真实持包A/B/C GPU阶段结束；CPU固定评估尚需验收，不宣称科学诊断完成。旧continuous与储备队列保持暂停，不发布。')

if __name__=='__main__':main()
