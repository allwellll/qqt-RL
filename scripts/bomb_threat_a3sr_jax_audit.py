"""Actual compiled JAX reset sampler audit, independent from NumPy sampler."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from jax_bomb import je07_bun_env as env
from scripts import bomb_threat_a3sr_spawn as S
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/iter2/a3sr')
def main():
    guard=lock(B/'JAX_AUDIT.lock');began=time.time()
    env.prepare();env.configure_training('full=1',1)
    initial=jax.tree.map(lambda a:a[0],env.init_batch(jax.random.PRNGKey(101),1)).core
    sample=jax.jit(jax.vmap(S.sample_jax,in_axes=(0,None)))
    hist=np.zeros((2,13,15),int);total=randomized=fallback=retries=0;rates=[];files={};connected=0
    for start in range(0,100000,1000):
        p=B/f'jax_audit_{start:06d}.npz'
        if p.exists():x=dict(np.load(p))
        else:
            keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(2026103031),i))(j.arange(start,start+1000))
            core,meta=jax.device_get(sample(keys,initial))
            x=dict(cells=np.asarray(core.pos).astype(int),brick=np.asarray(core.brick),**{k:np.asarray(v) for k,v in meta.items()})
            np.savez_compressed(p,**x)
        files[str(p)]=sha(p);n=len(x['cells']);total+=n
        for seat in (0,1):np.add.at(hist[seat],(x['cells'][:,seat,0],x['cells'][:,seat,1]),1)
        randomized+=int(x['spawn_mode'].sum());fallback+=int(x['spawn_fallback'].sum());retries+=int(x['spawn_resample'].sum())
        rates.extend(x['brick_drop_rate'][x['spawn_mode'].astype(bool)].tolist())
        for c,br in zip(x['cells'],x['brick']):connected+=int(tuple(c[1]) in S.flood(~np.asarray(initial.wall)&~br&~np.asarray(initial.pushable),tuple(c[0])))
        atomic_json(B/'JAX_AUDIT_PROGRESS.json',dict(status='RUNNING',resets=total,target=100000,elapsed_seconds=time.time()-began))
    result=dict(status='PASS',scope='actual JIT-vmapped JAX sample_jax, isolated native empty-hand resets; not collector exposure',
        resets=total,randomized_resets=randomized,spawn_heatmap=hist.tolist(),spawn_fallback=fallback,spawn_resample=retries,
        brick_drop_rate_mean=float(np.mean(rates)),drop_rate_histogram=np.histogram(rates,bins=np.linspace(0,.3,11))[0].tolist(),
        initial_connected_fraction=connected/total,source_sha256=sha(Path('scripts/bomb_threat_a3sr_spawn.py')),files=files,elapsed_seconds=time.time()-began)
    atomic_json(B/'JAX_AUDIT.json',result)
if __name__=='__main__':main()
