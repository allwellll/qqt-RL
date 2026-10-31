"""Scheduling-only amendment: bounded, work-conserving fixed-shard evaluation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from scripts import v2_fresh_selection_20261031 as core
from scripts.population_run_state import atomic_json,lock,sha

def main():
 guard=lock(core.OUT/'OWNER.lock');p=core.read(core.OUT/'PROTOCOL.json');ms=core.models(p)
 assert all(sha(Path(f))==h for f,h in core.read(core.OUT/'SOURCE_SEAL.json').items())
 amendment=core.read(core.OUT/'SCHEDULE_AMENDMENT02.json')
 assert amendment['scheduler_sha256']==sha(Path(__file__))
 pending=[j for j in core.jobs(p) if not Path(str(core.path(j))+'.ACCEPTED.json').exists()]
 history=core.read(core.OUT/'SCHEDULER_STATUS.json').get('history',[])
 refs=core.training_snapshot();cohort=core.context();limit=amendment['max_workers'];done0=len(list((core.OUT/'shards').glob('*/*/*.ACCEPTED.json')))*8
 atomic_json(core.OUT/'SCHEDULER2_STARTED.json',dict(status='RUNNING',resumed_accepted_games=done0,remaining_jobs=len(pending),max_workers=limit,amendment_sha256=sha(core.OUT/'SCHEDULE_AMENDMENT02.json'),unix=time.time()))
 last=0;started=time.time();cpu_file=Path('/sys/fs/cgroup/cpu/cpuacct.usage');cpu_last=int(cpu_file.read_text());cpu_time=time.monotonic()
 with ThreadPoolExecutor(max_workers=limit) as pool:
  running={}
  while pending or running:
   while pending and len(running)<limit:
    j=pending.pop(0);running[pool.submit(core.collect,j,ms[j['label']])]=j
   completed,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
   for future in completed:future.result();running.pop(future)
   if time.time()-last>=45 or not (pending or running):
    done=len(list((core.OUT/'shards').glob('*/*/*.ACCEPTED.json')))*8
    now,current=core.training_snapshot(),core.context();ratios=core.comparison(now,refs) if current==cohort else {}
    if current!=cohort:refs,cohort=now,current
    cpu_now=int(cpu_file.read_text());cpu_t=time.monotonic();cores=(cpu_now-cpu_last)/1e9/(cpu_t-cpu_time);cpu_last,cpu_time=cpu_now,cpu_t
    if ratios and min(ratios.values())<.85:limit=max(2,limit//2);refs=now
    elif cores>80:limit=max(2,limit-2)
    record=dict(games=done,budget=p['budget'],workers=limit,running_shards=len(running),ratios=ratios,training_current=now,gpu_cohort=current,cpu_cores=cores,unix=time.time(),scheduler_version=2,new_games_per_minute=(done-done0)*60/max(1,time.time()-started))
    history.append(record);atomic_json(core.OUT/'SCHEDULER_STATUS.json',dict(status='RUNNING',history=history));atomic_json(core.OUT/'EXECUTION_STATUS.json',record);last=time.time()
    print(json.dumps({k:record[k] for k in ('games','workers','running_shards','cpu_cores','new_games_per_minute','ratios')}),flush=True)
 r=core.aggregate(p,ms,True);core.report(r)
 atomic_json(core.OUT/'COMPLETE.json',dict(status='COMPLETE',games=r['games'],results_sha256=sha(core.OUT/'RESULTS.json'),report_sha256=sha(core.REPORT),selection=r['selection'],unix=time.time(),scheduler_version=2))
 atomic_json(core.OUT/'EXECUTION_STATUS.json',dict(status='COMPLETE',games=r['games'],selection=r['selection'],unix=time.time()))
 subprocess.run(['jdme-notify','send',f"v2 fresh256固定矩阵6144局完成，候选={r['selection']['winner']}；报告{core.REPORT}，进入录像和授权发布验收，训练不停。"],check=False)
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic_json(core.OUT/'SCHEDULER2_BLOCKED.json',dict(status='BLOCKED',error_type=type(e).__name__,reason='inspect SCHEDULER2.log; existing accepted shards and training preserved',unix=time.time()));raise
