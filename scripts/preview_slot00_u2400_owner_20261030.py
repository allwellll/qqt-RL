"""Isolated preview owner; bounded two-CPU recording with hash-validated reuse."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
B=Path('runs/bomb_threat_v1/replay_r1_slot00_u2400_preview')
def main():
 guard=lock(B/'OWNER.lock');jobs=[]
 for root in (B,B/'baseline'):
  cfg=json.loads((root/'MANIFEST.json').read_text())
  for g in cfg['games']:jobs.append((root,g['index']))
 atomic_json(B/'EXECUTION.json',dict(status='RECORDING',cpu_only=True,workers=2,affinity='144-145',jobs=len(jobs),unix=time.time()))
 def record(job):
  root,i=job;out=root/f'game_{i:03d}';out.mkdir(exist_ok=True);mp=out/'MATCH.json'
  if mp.exists():
   m=json.loads(mp.read_text());assert m['status']=='MATCH' and sha(out/'review.json')==m['review_sha256'];return
  with (out/'worker.log').open('ab') as f:subprocess.run(['node','scripts/preview_slot00_u2400_record_20261030.js',str(i)],env=dict(os.environ,BT_REPLAY_BASE=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
  assert json.loads(mp.read_text())['status']=='MATCH'
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for n,f in enumerate(concurrent.futures.as_completed([pool.submit(record,j) for j in jobs]),1):f.result();atomic_json(B/'PROGRESS.json',dict(status='RECORDING',completed=n,total=len(jobs),unix=time.time()))
  atomic_json(B/'RECORDING_COMPLETE.json',dict(status='MATCH',jobs=len(jobs),unix=time.time()))
 except Exception as e:
  atomic_json(B/'FAILURE.json',dict(status='BLOCKED',error=str(e),unix=time.time()));raise
if __name__=='__main__':main()
