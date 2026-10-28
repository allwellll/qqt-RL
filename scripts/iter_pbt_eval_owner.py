"""Bounded asynchronous CPU consumer. Never controls or waits GPU workers."""
import os,sys,json,subprocess,time
from pathlib import Path
from scripts.population_run_state import lock,atomic_json,sha
from scripts.combat_mix_owner import read
from scripts.iter_pbt_owner import notify,live_workers
from scripts.iter_pbt_recipe import eligible
from scripts.iter_pbt_eval_scheduling import pending_latest,priority,lag_report
from scripts.combat_mix_js_owner import Adopted
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/iter_pbt_v1'

def main():
 guard=lock(BASE/'EVAL_OWNER.lock');children={};slots=['120-123','124-127','168-171','172-175']+[f'{i}-{i+1}' for i in range(128,144,2)]+[f'{i}-{i+1}' for i in list(range(160,168,2))+list(range(176,184,2))]
 while not any((BASE/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json')):
  for n,(p,stream,cores) in list(children.items()):
   if p.poll() is None:continue
   rc=p.returncode;stream.close();del children[n]
   if rc:
    atomic_json(BASE/'eval'/n/'FAILED.json',dict(exit_code=rc,unix=time.time()));notify(n+'_EVAL_FAILED',f'ITER-PBT异步评估失败：{n}，训练继续，日志已保留。')
  used=[v[2] for v in children.values()]
  requests={p.stem:read(p) for p in (BASE/'eval_requests').glob('*.json')}
  keep=pending_latest(list(requests.values()))
  # Confirm only the strongest scored dev plus the explicitly requested point;
  # old in-flight confirmations finish, but unstarted historical ones cannot
  # monopolize the latest-slot queue.
  confirmations=[]
  for n,r in requests.items():
   if r.get('dev_name'):
    selection=read(BASE/'eval'/r['dev_name']/'SELECTION.json',{})
    if selection.get('eligible'):confirmations.append((selection['score'],n))
  best_pending=max(confirmations,default=(None,None))[1]
  keep={n for n in keep if not requests[n].get('dev_name') or n==best_pending or requests[n].get('priority')=='best_candidate'}

  found=live_workers('scripts.iter_pbt_eval')
  # Adopt all live jobs before allocating any cores to new requests.
  for n,pid in found.items():
   if n in children:continue
   affinity=os.sched_getaffinity(pid)
   cores=next((s for s in slots if set(range(int(s.split('-')[0]),int(s.split('-')[1])+1))==affinity),None)
   if cores is None:raise RuntimeError('unexpected recovered evaluator affinity')
   stream=(BASE/'logs'/f'EVAL_{n}.adopt.log').open('ab',buffering=0)
   children[n]=(Adopted(pid,BASE/'eval'/n),stream,cores);used.append(cores)
  for q in sorted((BASE/'eval_requests'/f'{n}.json' for n in requests),key=lambda p:priority(requests[p.stem])):
   r=read(q);n=r['name'];out=BASE/'eval'/n
   if r.get('selection_eligible') is False:continue
   if (out/'COMPLETE.json').exists():
    v=read(out/'SUMMARY.json');parent=read(BASE/'eval'/('parent_'+r['split'])/'SUMMARY.json')
    if parent and not (out/'SELECTION.json').exists():atomic_json(out/'SELECTION.json',dict(score=v['selection_score'],eligible=eligible(v['natural_jax']['categorical']['capture_episodes'],parent['natural_jax']['categorical']['capture_episodes']),checkpoint=v['checkpoint'],summary_sha256=sha(out/'SUMMARY.json')))
    if r['split']=='dev' and parent and not n.startswith('parent_'):
     prior=read(BASE/'BEST.json',{}).get('dev_score',parent['selection_score'])
     sel=read(out/'SELECTION.json',{})
     if sel.get('eligible') and v['selection_score']>prior:
      hn=n+'_holdout';hp=BASE/'eval_requests'/f'{hn}.json'
      if not hp.exists():atomic_json(hp,dict(name=hn,checkpoint=r['checkpoint'],split='holdout',dev_name=n))
    if r['split']=='holdout' and 'dev_name' in r and parent:
     dev=read(BASE/'eval'/r['dev_name']/'SUMMARY.json');best=read(BASE/'BEST.json',{})
     reference=read(BASE/'eval'/best['holdout_name']/'SUMMARY.json') if best else parent
     natural_ok=all(eligible(v['natural_jax'][m]['capture_episodes'],parent['natural_jax'][m]['capture_episodes']) for m in ('categorical','greedy'))
     if natural_ok and v['selection_score']>reference['selection_score'] and dev['selection_score']>best.get('dev_score',-1e9):
      b=dict(checkpoint=v['checkpoint'],policy_sha256=v['policy_sha256'],holdout_name=n,dev_name=r['dev_name'],dev_score=dev['selection_score'],holdout_score=v['selection_score'],confirmation_summary_sha256=sha(out/'SUMMARY.json'),limitation='adaptive repeated holdout confirmations; descriptive confirmation, not independent final inference',unix=time.time())
      atomic_json(BASE/'BEST.json',b);notify(n+'_BEST',f"ITER-PBT保留集确认新最佳：{r['dev_name']}，dev={b['dev_score']}、holdout={b['holdout_score']}，自然交付保留通过。",evidence=b)
    continue
   if n not in keep and n not in found and n not in children:
    if not (out/'SKIPPED_SUPERSEDED.json').exists():atomic_json(out/'SKIPPED_SUPERSEDED.json',dict(reason='superseded dev checkpoint or lower-priority historical confirmation; historical data retained',unix=time.time()))
    continue
   if n in children or (out/'FAILED.json').exists():continue
   # Recovery adopts no duplicate: active evaluator request path is authoritative.
   found=live_workers('scripts.iter_pbt_eval')
   if q.stem in found:
    pid=found[q.stem]
    affinity=os.sched_getaffinity(pid)
    cores=next((s for s in slots if set(range(int(s.split('-')[0]),int(s.split('-')[1])+1))==affinity),None)
    if cores is None:raise RuntimeError('unexpected recovered evaluator affinity')
    stream=(BASE/'logs'/f'EVAL_{n}.adopt.log').open('ab',buffering=0)
    children[n]=(Adopted(pid,out),stream,cores);used.append(cores);continue
   if (out/'SKIPPED_SUPERSEDED.json').exists():continue
   if len(children)>=len(slots):continue
   cores=next(s for s in slots if s not in used);used.append(cores)
   stream=(BASE/'logs'/f'EVAL_{n}.log').open('ab',buffering=0)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   p=subprocess.Popen(['taskset','-c',cores,sys.executable,'-m','scripts.iter_pbt_eval',str(q.relative_to(ROOT))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   children[n]=(p,stream,cores)
  atomic_json(BASE/'EVAL_LAG.json',lag_report(list(requests.values()),{p.parent.name for p in (BASE/'eval').glob('*/SELECTION.json')}))
  atomic_json(BASE/'EVAL_STATUS.json',dict(js_concurrency_limit=28,js_per_job=4,native_backend='isolated_shared_GPU',native_concurrency_limit=8,active={n:dict(cpu_affinity=v[2]) for n,v in children.items()},cpu_only_JS=True,updated_unix=time.time()))
  time.sleep(5)
if __name__=='__main__':main()
