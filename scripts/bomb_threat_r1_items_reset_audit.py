"""100k actual R1 fresh resets, frozen RNG and bank source verification."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from scripts.bomb_threat_r1_items_runtime import install as native_install
native_install()
import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from scripts import combat_item_engine as T,bomb_threat_r1_progress_collector as M
from scripts.carrier_mechanism_analysis import bank,merged
from scripts.population_run_state import atomic_json,sha,lock
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def main():
    guard=lock(OUT/'RESET_AUDIT.lock');began=time.time();T.configure();M.configure(False)
    bm=json.loads((OUT/'BANK_WITH_HISTORY.json').read_text());items=[]
    bs,ba=M.banks(None,None,bm)
    from scripts.bomb_threat_round15 import read
    source=read(OUT/'SOURCE_AUTHORIZATION.json')['source'];p=read(Path('runs/bomb_threat_v1/protocols')/(Path(source['checkpoint']).parent.name+'.json'))
    spawns=j.asarray(read(Path(p['spawn_manifest']))['spawn_cells']);n=1000;seats=j.arange(n)%2
    # Exact10% clear-scene exposure, balanced by physical seat.
    designated=j.arange(n)%20<2
    sample=jax.jit(lambda keys:M.fresh(keys,bs,ba,designated,spawns,seats))
    counts=np.zeros(2,int);files={};carried=np.zeros(2,int)
    for start in range(0,100000,n):
        f=OUT/'reset_audit'/f'{start:06d}.npz';f.parent.mkdir(exist_ok=True)
        if f.exists():a=np.load(f)
        else:
            keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(2026183033),i))(j.arange(start,start+n))
            st,ax=sample(keys);st,ax=jax.device_get((st,ax));use=np.asarray(ax['provenance_source'])>=0
            a=dict(seats=np.asarray(seats),use=use,carried=np.asarray(st.bun_carried),stored=np.asarray(st.bun_stored),ticks=np.asarray(st.t),
                spawn_mode=np.asarray(ax['a3sr']['spawn_mode']))
            np.savez_compressed(f,**a)
        files[str(f)]=sha(f)
        for seat in (0,1):
            used=np.asarray(a['use'])&(np.asarray(a['seats'])==seat);counts[seat]+=int(used.sum())
            assert np.all(np.asarray(a['carried'])[used,seat]==1-seat),'bank must physically carry enemy origin'
            carried[seat]+=int(np.sum(np.asarray(a['carried'])[used,seat]==1-seat))
        atomic_json(OUT/'RESET_AUDIT_PROGRESS.json',dict(status='RUNNING',resets=start+n,bank_seat_resets=counts.tolist(),wall_seconds=time.time()-began))
    assert all(12000<c<13000 for c in counts) and abs(int(counts[0])-int(counts[1]))<700
    atomic_json(OUT/'RESET_AUDIT.json',dict(status='PASS',actual_resets=100000,bank_seat_resets=counts.tolist(),
        bank_frame_observations_by_seat=carried.tolist(),bank_probability=float(counts.sum()/100000),
        collector_tick_exposure='must independently verify smoke bank_seat_ticks; reset observations are not live rollout ticks',
        physical_carry_origin_verified=True,native_inventory_axes_preserved=True,files=files,
        bank_manifest_sha256=sha(OUT/'BANK_WITH_HISTORY.json'),wall_seconds=time.time()-began,unix=time.time()))

if __name__=='__main__':main()
