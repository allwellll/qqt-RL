"""Use three freed CPU slots for bounded, locked lookahead; selection stays ordered."""
import json
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,lock,sha

B=Path('runs/new_model_replays_20261031')
LABEL='r3_sp_seed2'
def read(p):return json.loads(Path(p).read_text())

def main():
    guard=lock(B/'PREFETCH_OWNER.lock')
    amendment=read(B/'SCHEDULING_AMENDMENT01.json')
    def worker(core):
        done=[]
        for start in range(amendment['start'],256,8):
            if (B/'scan'/LABEL/'COMPLETE.json').exists():break
            folder=B/'scan'/LABEL/f'shard_{start:03d}'
            # Atomic claims distribute work without touching the scientific data.
            claim=B/'prefetch_claims'/f'{start:03d}'
            try:claim.mkdir(parents=False)
            except FileExistsError:continue
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/f'prefetch_core{core}.log').open('w') as log:
                subprocess.run(['taskset','-c',str(core),'node','scripts/new_model_replay_record_20261031.js',LABEL,str(start),str(start+8)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert (folder/'ACCEPTED.json').exists();done.append(start)
            atomic_json(B/f'PREFETCH_CORE{core}.json',dict(status='ACTIVE',accepted=done,unix=time.time()))
        return dict(core=core,accepted=done)
    (B/'prefetch_claims').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(worker,amendment['freed_cpu_cores']))
    atomic_json(B/'PREFETCH_COMPLETE.json',dict(status='COMPLETE',workers=results,protocol_unchanged=True,unix=time.time()))

if __name__=='__main__':main()
