"""Resumable bounded CPU JS H0/H2/H4 campaign, no training/production owner edits."""
import json,os,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/jitter_followup_u800')
def main():
    guard=lock(B/'OWNER.lock');cfg=json.loads((B/'PROTOCOL.json').read_text());jobs=[(arm,start) for start in range(0,64,8) for arm in cfg['inference_arms']]
    atomic_json(B/'EXECUTION.json',dict(status='RUNNING',protocol_sha256=sha(B/'PROTOCOL.json'),jobs=len(jobs),cpu_only=True,unix=time.time()))
    def job(task,lane):
        arm,start=task;p=B/arm/f'shard_{start:03d}.json';done=p.with_suffix('.json.diagnostic.json')
        p.parent.mkdir(exist_ok=True)
        job_guard=lock(p.with_suffix('.lock'))
        if done.exists():
            x=json.loads(done.read_text());assert len(x['episodes'])==8 and x['model_sha256']==cfg['model']['model_sha256'];return
        with p.with_suffix('.log').open('ab',buffering=0) as f:
            subprocess.run(['taskset','-c',str(cfg['cpu']['affinity'][lane]),'node','scripts/bomb_threat_jitter_hold_eval.js',arm,str(start),str(start+8)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
        assert done.exists()
    def lane(k):
        for task in jobs[k::4]:job(task,k);atomic_json(B/f'LANE{k}.json',dict(last_job=task,unix=time.time()))
    errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(lane,k) for k in range(4)]):
            try:f.result()
            except Exception as e:errors.append(type(e).__name__)
    paths=[B/arm/f'shard_{start:03d}.json.diagnostic.json' for arm,start in jobs]
    if errors or any(not p.exists() for p in paths):
        atomic_json(B/'EXECUTION.json',dict(status='PARTIAL_RETRY_REQUIRED',errors=errors,completed=sum(p.exists() for p in paths),jobs=len(paths),unix=time.time()));return
    atomic_json(B/'HOLD_COMPLETE.json',dict(status='COMPLETE',arms=3,games=192,files={str(p):sha(p) for p in paths},unix=time.time()))
if __name__=='__main__':main()
