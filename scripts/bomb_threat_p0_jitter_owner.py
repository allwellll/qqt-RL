"""Bounded passive CPU audit; one lock, resume completed original-state analyses."""
import json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/p0_jitter_u800')
def main():
    guard=lock(B/'OWNER.lock');cfg=json.loads((B/'PROTOCOL.json').read_text());games=cfg['games']
    atomic_json(B/'EXECUTION.json',dict(status='RUNNING',cpu_only=True,workers=4,affinity=cfg['cpu_affinity'],started=time.time(),protocol_sha256=sha(B/'PROTOCOL.json')))
    def lane(k):
        ids=[str(g['index']) for g in games[k::4]]
        with (B/f'lane{k}.log').open('ab',buffering=0) as f:
            subprocess.run(['taskset','-c',str(cfg['cpu_affinity'][k]),'node','scripts/bomb_threat_p0_jitter_audit.js',*ids],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(lane,k) for k in range(4)]):f.result()
    files={str(B/f"game_{g['index']:03d}.json"):sha(B/f"game_{g['index']:03d}.json") for g in games}
    atomic_json(B/'COLLECTION_COMPLETE.json',dict(status='MATCH',games=len(games),ticks=sum(json.loads(Path(p).read_text())['ticks'] for p in files),files=files,source_sha256={p:sha(Path(p)) for p in ['scripts/bomb_threat_p0_jitter_audit.js','scripts/bomb_threat_p0_jitter_metrics.js']},unix=time.time()))
    atomic_json(B/'EXECUTION.json',dict(status='COLLECTED',cpu_only=True,workers_stopped=True,completed=time.time()))
if __name__=='__main__':main()
