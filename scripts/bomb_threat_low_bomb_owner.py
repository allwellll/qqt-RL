"""Bounded CPU-only collection, exact outcomes, resumable per frozen game."""
import json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/low_bomb_diag_u800')
def main():
    guard=lock(B/'OWNER.lock');cfg=json.loads((B/'MANIFEST.json').read_text());ids=[r['index'] for r in cfg['games']]
    atomic_json(B/'EXECUTION.json',dict(status='RUNNING',cpu_only=True,workers=4,games=len(ids),unix=time.time()))
    def run(i):
        out=B/f'game_{i:03d}';out.mkdir(exist_ok=True)
        if (out/'COMPLETE.json').exists():return
        if not (out/'MATCH.json').exists() or not (out/'METRICS.json').exists():
            with (out/'worker.log').open('ab',buffering=0) as f:
                subprocess.run(['node','scripts/bomb_threat_low_bomb_record.js',str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
        match=json.loads((out/'MATCH.json').read_text());m=json.loads((out/'METRICS.json').read_text())
        assert match['status']=='MATCH' and all(match['comparison'].values()),f'replay mismatch {i}'
        assert m['bombs_placed']==match['expected']['placed'][m['seat']] and m['unexplained_rejections']==0
        atomic_json(out/'COMPLETE.json',dict(status='MATCH',index=i,metrics_sha256=sha(out/'METRICS.json'),trace_sha256=sha(out/'TRACE.json'),review_sha256=sha(out/'review.json'),match_sha256=sha(out/'MATCH.json'),unix=time.time()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k,f in enumerate(as_completed([pool.submit(run,i) for i in ids]),1):f.result();atomic_json(B/'PROGRESS.json',dict(completed=k,total=len(ids),unix=time.time()))
    atomic_json(B/'COLLECTION_COMPLETE.json',dict(status='MATCH',games=len(ids),files={str((B/f'game_{i:03d}/COMPLETE.json')):sha(B/f'game_{i:03d}/COMPLETE.json') for i in ids},unix=time.time()))
if __name__=='__main__':main()
