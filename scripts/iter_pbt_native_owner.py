"""Durable shared-GPU native evaluator; never sends a signal to training."""
import json,os,signal,subprocess,sys,time,statistics
from pathlib import Path
from scripts.population_run_state import atomic_json,lock
from scripts.iter_pbt_owner import live_workers,notify,read
from scripts.iter_pbt_eval_scheduling import priority
from scripts.iter_pbt_native_gpu_policy import reduce_concurrency
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/iter_pbt_v1';B=BASE/'native_gpu'

class Adopted:
 def __init__(self,identity,out):self.proc=Path('/proc')/str(identity);self.out=out;self.returncode=None
 def poll(self):
  try:
   if self.proc.exists() and (self.proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return None
  except OSError:pass
  self.returncode=0 if (self.out/'NATIVE_COMPLETE.json').exists() or (self.out/'YIELDED.json').exists() else 1
  return self.returncode

def samples_after(gpu,started):
 rows=[]
 for p in (BASE/'formal').glob(f'slot{gpu:02d}_c*/updates.jsonl'):
  if p.stat().st_mtime<started:continue
  for line in p.read_text().splitlines():
   if not line.strip():continue
   r=json.loads(line)
   if r.get('unix',0)>started and 0<r.get('cycle_seconds',0)<30:rows.append(r)
 return sorted(rows,key=lambda r:r['unix'])[-3:]

def main():
 guard=lock(B/'OWNER.lock');children={};last_sample=0
 while not any((BASE/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json')):
  config=read(B/'CONFIG.json',{});enabled=config.get('enabled',False) and read(B/'PARITY_GATE.json',{}).get('status')=='PASS';cap=config.get('max_concurrent',8);found=live_workers('scripts.iter_pbt_native_eval')
  for n,identity in found.items():
   # Probe uses a separate output and is never adopted as production work.
   if n in children or not (BASE/'native_requests'/f'{n}.json').exists():continue
   out=BASE/'eval'/n/'native';launch=read(out/'NATIVE_LAUNCH.json')
   if not launch:continue
   children[n]=(Adopted(identity,out),None,launch['gpu'],launch['unix'],identity)
  for n,(p,stream,gpu,started,identity) in list(children.items()):
   out=BASE/'eval'/n/'native'
   if p.poll() is not None:
    if stream:stream.close()
    del children[n]
    if p.returncode and not (out/'YIELD_GPU.json').exists():
     atomic_json(out/'NATIVE_FAILED.json',dict(exit_code=p.returncode,unix=time.time()))
     notify(n+'_NATIVE_FAILED',f'ITER-PBT GPU原生评估失败：{n}，训练不停止，证据保留。')
    continue
   progress=read(out/'NATIVE_PROGRESS.json',{})
   if time.time()-progress.get('unix',started)>900:notify(n+'_NATIVE_STALE',f'ITER-PBT共卡评估15分钟无scan进展：{n}，检查编译/执行阻塞；训练继续。')
   rows=samples_after(gpu,started)
   rates=[262144/r['cycle_seconds'] for r in rows]
   baseline=config.get('baseline_env_per_second',{}).get(str(gpu),20000)
   newcap=reduce_concurrency(cap,baseline,rates)
   if newcap<cap and not (out/'YIELD_GPU.json').exists():
    config.update(max_concurrent=newcap,last_reduction_unix=time.time());atomic_json(B/'CONFIG.json',config);cap=newcap
    atomic_json(out/'YIELD_GPU.json',dict(reason='training throughput drop >5%; evaluation-only yield',baseline_env_per_second=baseline,observed=rates,unix=time.time()))
    notify(n+'_THROUGHPUT_GUARD',f'ITER-PBT共卡评估自动降并发至{newcap}：GPU{gpu}训练连续3更新中位吞吐低于基线95%，仅评估让出GPU，训练继续。',baseline=baseline,observed=rates)
  if time.time()-last_sample>5:
   r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
   memory={}
   for line in r.stdout.splitlines():
    try:i,m=line.split(',');memory[int(i)]=float(m.strip())
    except ValueError:continue
   samples=[]
   for n,(p,stream,gpu,started,identity) in children.items():
    used=memory.get(identity);samples.append(dict(run=n,gpu=gpu,memory_MiB=used,training_samples=samples_after(gpu,started)))
    if used is not None and used>6144:
     atomic_json(BASE/'eval'/n/'native/YIELD_GPU.json',dict(reason='evaluation memory exceeds 6GiB',memory_MiB=used,unix=time.time()))
     # Evaluation-only process, never the training group; graceful TERM frees
     # an over-budget allocator without waiting for another GPU allocation.
     os.kill(identity,signal.SIGTERM)
     notify(n+'_MEMORY_GUARD',f'ITER-PBT评估显存超6GiB，已停止该评估并保留数据；GPU{gpu}训练保持运行。')
   with (B/'SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),evaluators=samples))+'\n')
   last_sample=time.time()
  used={v[2] for v in children.values()}
  pending=[(q,read(q)) for q in (BASE/'native_requests').glob('*.json')]
  confirmation_scores=[(read(BASE/'eval'/r.get('dev_name','missing')/'SELECTION.json',{}).get('score',-1e9),r['name']) for q,r in pending if r.get('dev_name')]
  best_confirmation=max(confirmation_scores,default=(None,None))[1]
  def native_priority(pair):
   r=pair[1]
   if r.get('dev_name') and r['name']!=best_confirmation and r.get('priority')!='best_candidate':return (4,0,r['name'])
   return priority(r)
  for q,r in sorted(pending,key=native_priority):
   n=r['name'];out=BASE/'eval'/n/'native'
   if n in children or (out/'NATIVE_COMPLETE.json').exists() or (out/'NATIVE_FAILED.json').exists():continue
   if (BASE/'eval'/n/'SKIPPED_SUPERSEDED.json').exists():continue
   if not enabled or len(children)>=cap:break
   gpu=next((g for g in range(8) if g not in used),None)
   if gpu is None:break
   # Prefer the candidate's training GPU, rotate when its evaluator is busy.
   preferred=r.get('slot')
   if preferred in range(8) and preferred not in used:gpu=preferred
   out.mkdir(parents=True,exist_ok=True)
   if (out/'YIELD_GPU.json').exists():
    # A yielded run stays pending under a reduced cap; keep the original receipt.
    (out/'YIELD_GPU.json').rename(out/f'YIELD_GPU_{int(time.time())}.json')
   stream=(B/f'{n}.log').open('ab',buffering=0)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   command=['taskset','-c',str(184+gpu),sys.executable,'-m','scripts.iter_pbt_native_eval',str((BASE/'eval_requests'/f'{n}.json').relative_to(ROOT)),'--gpu',str(gpu)]
   p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   now=time.time();children[n]=(p,stream,gpu,now,p.pid);used.add(gpu)
   atomic_json(out/'NATIVE_LAUNCH.json',dict(gpu=gpu,cpu_affinity=str(184+gpu),unix=now))
  completed={}
  for result in (BASE/'eval').glob('*/native/NATIVE_SUMMARY.json'):
   v=read(result,{})
   if v.get('backend')=='gpu':completed[result.parent.parent.name]=dict(elapsed_seconds=v.get('elapsed_seconds'),games=v.get('natural_jax'),scan_seconds=v.get('scan_seconds'),completed_unix=v.get('unix'))
  atomic_json(B/'REPORT.json',dict(completed_GPU_points=completed,lag=read(BASE/'EVAL_LAG.json',{}),baseline=read(B/'BASELINE.json',{}).get('env_per_second'),max_concurrent=cap,target_lag_chunks=2,updated_unix=time.time()))
  atomic_json(B/'STATUS.json',dict(enabled=enabled,max_concurrent=cap,active={n:dict(gpu=v[2],age_seconds=time.time()-v[3]) for n,v in children.items()},updated_unix=time.time(),training_signal_authority=False))
  time.sleep(5)
if __name__=='__main__':main()
