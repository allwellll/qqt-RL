"""Bounded CPU-only replay recording; never starts or stops training."""
import json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.population_run_state import atomic_json,lock
B=Path('runs/bomb_threat_v1/replay_best_u800')

def main():
    guard=lock(B/'OWNER.lock');cfg=json.loads((B/'MANIFEST.json').read_text());ids=sorted({r['index'] for r in cfg['games']})
    atomic_json(B/'EXECUTION.json',dict(status='RUNNING',cpu_only=True,workers=4,unique_games=len(ids),entries=len(cfg['games']),unix=time.time()))
    def run(i):
        out=B/f'game_{i:03d}';out.mkdir(exist_ok=True)
        if (out/'MATCH.json').exists():return
        with (out/'worker.log').open('ab',buffering=0) as f:subprocess.run(['node','scripts/bomb_threat_best_replay_record.js',str(i)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done,f in enumerate(as_completed([pool.submit(run,i) for i in ids]),1):f.result();atomic_json(B/'PROGRESS.json',dict(completed_unique_games=done,total_unique_games=len(ids),unix=time.time()))
    matches=[json.loads((B/f'game_{i:03d}/MATCH.json').read_text()) for i in ids]
    atomic_json(B/'RECORDING_COMPLETE.json',dict(status='MATCH' if all(r['status']=='MATCH' for r in matches) else 'REPLAY_MISMATCH',games=matches,unix=time.time()))

if __name__=='__main__':main()
