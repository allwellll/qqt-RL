"""Recover full real episode attribute high-water through frozen prefix replay."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JAXBOMB_RULE='bun')
from scripts.bomb_threat_r1_items_runtime import install
install()
import json,pickle,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from scripts import combat_item_engine as T,carrier_mechanism_engine as E
from scripts.bomb_threat_runtime import install as stock
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_r1_item_progress import fraction
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def main():
    guard=lock(OUT/'BANK_HISTORY.lock');began=time.time();T.configure();stock()
    manifest=json.loads((OUT/'BANK_FROZEN.json').read_text());records=[]
    for seat in (0,1):
        path=Path(manifest[f'p{seat}_file']);assert sha(path)==manifest[f'p{seat}_sha256'];records.extend(pickle.load(path.open('rb')))
    k=len(records);lengths=j.array([len(x['prefix_actions']) for x in records]);h=int(lengths.max())
    initial=jax.tree.map(lambda *v:np.concatenate(v),*[x['initial'] for x in records]);ac=np.zeros((h,k,2,2),np.int32);ek=np.zeros((h,k,2),np.uint32)
    for i,x in enumerate(records):
        l=len(x['prefix_actions']);ac[:l,i]=x['prefix_actions'][:,0];ek[:l,i]=x['prefix_envkeys'][:,0]
    def stats(s):return j.stack([s.core.bombs_cap,s.core.blast_cap,s.core.spd_g],-1)
    def replay(st,ax):
        def body(c,data):
            a,key,t=data;ns,na,*_=T.transition(c[0],c[1],a,key,'control');active=t<lengths
            water=j.maximum(c[2],j.where(active[:,None,None],fraction(stats(ns)),c[2]))
            return (E.select_tree(active,ns,c[0]),E.select_tree(active,na,c[1]),water),None
        return jax.lax.scan(body,(st,ax,fraction(stats(st))),(j.asarray(ac),j.asarray(ek),j.arange(h)))[0]
    ss,aa,water=jax.jit(replay)(initial[0],initial[1]);jax.block_until_ready(water)
    for i,x in enumerate(records):x['attribute_highwater']=np.asarray(water[i:i+1])
    amended=dict(manifest,status='PASS',base_manifest_sha256=sha(OUT/'BANK_FROZEN.json'),history='full natural episode prefix maximum; death never rewinds; existing growth never paid again',wall_seconds=time.time()-began)
    for seat in (0,1):
        path=OUT/f'p{seat}_bank_with_history.pkl';selected=[x for x in records if x['record']['seat']==seat]
        with path.with_suffix('.tmp').open('wb') as f:pickle.dump(selected,f,protocol=5);f.flush();os.fsync(f.fileno())
        path.with_suffix('.tmp').replace(path);amended[f'p{seat}_file']=str(path);amended[f'p{seat}_sha256']=sha(path)
    amended['highwater_above_current_states']=sum(bool((x['attribute_highwater']>np.asarray(fraction(j.stack([x['states'].core.bombs_cap,x['states'].core.blast_cap,x['states'].core.spd_g],-1)))+1e-6).any()) for x in records)
    atomic_json(OUT/'BANK_WITH_HISTORY.json',amended)

if __name__=='__main__':main()
