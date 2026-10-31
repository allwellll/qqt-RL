"""Resumable two-lane CPU observer for the fixed nine A3SR terminal games."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/iter2/a3sr');O=B/'brick_observation'
def main():
    guard=lock(O/'worker.lock')
    if (O/'COMPLETE.json').exists():return
    terminal=json.loads((B/'TERMINAL.json').read_text());manifest=json.loads((B/'BRICK_OBSERVATION_MANIFEST.json').read_text())
    from scripts.je08_bun_authority import export
    cp=Path(terminal['last_atomic']['checkpoint']);assert sha(cp/'policy.pt')==terminal['last_atomic']['policy_sha256']
    model=export(cp/'policy.pt',O/'model')
    atomic_json(O/'LAUNCH.json',dict(status='RUNNING',cpu_only=True,workers=2,checkpoint=str(cp),policy_sha256=sha(cp/'policy.pt'),model_sha256=sha(model),unix=time.time()))
    def run(e):
        out=O/f'game_{e["index"]:03d}.json';result=Path(str(out)+'.brick_observation.json')
        if not result.exists():
            with out.with_suffix('.log').open('ab') as f:subprocess.run(['node','scripts/bomb_threat_a3sr_brick_observation.js',str(model),str(out),str(e['index'])],stdout=f,stderr=subprocess.STDOUT,check=True)
        row=json.loads(result.read_text());assert row['index']==e['index'] and row['seed']==e['seed'] and row['done']
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,manifest['games']))
    assert len(rows)==len({r['seed'] for r in rows})==9
    atomic_json(O/'COMPLETE.json',dict(status='COMPLETE',games=9,checkpoint=str(cp),policy_sha256=sha(cp/'policy.pt'),
        rows=rows,files={str(p):sha(p) for p in O.glob('*.brick_observation.json')},
        caveat='descriptive selected same9 originally brick-separated P1 games, not gate or global capability estimate',unix=time.time()))
if __name__=='__main__':main()
