"""Persist complete CPU reset audit and 1000 initial JS diagnostic fixtures."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time
from pathlib import Path
import numpy as np
from scripts import bomb_threat_a3sr_spawn as S
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/iter2/a3sr')
def main():
    guard=lock(B/'AUDIT.lock');began=time.time();level=S.load_level();hist=np.zeros((2,13,15),int);rates=[];fallback=0;resample=0;connected=0;randomized=0;files={};total=0
    for start in range(0,100000,1000):
        p=B/f'audit_{start:06d}.npz'
        if p.exists():x=dict(np.load(p))
        else:
            x=S.sample_numpy(level,2026103000+start,min(1000,100000-start));np.savez_compressed(p,**x)
        files[str(p)]=sha(p);n=len(x['cells']);total+=n
        for seat in (0,1):np.add.at(hist[seat],(x['cells'][:,seat,0],x['cells'][:,seat,1]),1)
        rates.extend(x['brick_drop_rate'][x['spawn_mode'].astype(bool)].tolist());fallback+=int(x['spawn_fallback'].sum());resample+=int(x['spawn_resample'].sum());connected+=int(x['connected'].sum());randomized+=int(x['spawn_mode'].sum())
        atomic_json(B/'AUDIT_PROGRESS.json',dict(status='RUNNING',resets=total,target=100000,elapsed_seconds=time.time()-began))
        if start==0:
            atomic_json(B/'JS_FIXTURES.json',dict(episodes=[dict(seed=2026103000+i,cells=x['cells'][i].tolist(),brick=x['brick'][i].astype(int).reshape(-1).tolist(),spawn_mode=int(x['spawn_mode'][i])) for i in range(n)]))
    result=dict(status='RESET_AUDIT_COMPLETE_JS_PENDING',resets=total,randomized_resets=randomized,spawn_heatmap=hist.tolist(),drop_rate_histogram=np.histogram(rates,bins=np.linspace(0,.3,11))[0].tolist(),drop_rate_mean=float(np.mean(rates)),spawn_resample=resample,spawn_fallback=fallback,initial_connected_fraction=connected/total,elapsed_seconds=time.time()-began,files=files,JS_self_harm='PENDING',jax_sampler_cross_check='PENDING')
    atomic_json(B/'RESET_AUDIT.json',result)
    atomic_json(B.parent/'A3SR_SPAWN_AUDIT.json',result)
if __name__=='__main__':main()
