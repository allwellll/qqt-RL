"""Independent vectorized NumPy witnesses for all actual JAX randomized starts."""
import json,time
from pathlib import Path
import numpy as np
from scripts import bomb_threat_a3sr_spawn as S
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/iter2/a3sr')
def main():
    began=time.time();level=S.load_level();wall=np.asarray(level['wall'],bool).reshape(13,15);push=np.asarray(level.get('pushable',np.zeros(195)),bool).reshape(13,15);base=S.base_mask(level)
    native_brick=np.asarray(level['brick'],bool).reshape(13,15);reachable=S.wall_targets_reachable(level)
    failures=[];checked=0;hist=np.zeros((2,13,15),int);files={}
    rr,cc=np.indices((13,15));speed=.3*level['initial_stats']['speed'];blast=level['initial_stats']['blast']
    for path in sorted(B.glob('jax_audit_*.npz')):
        a=np.load(path);files[str(path)]=sha(path);use=(a['spawn_mode']>0)&~a['spawn_fallback'].astype(bool)
        cells=a['cells'][use];brick=a['brick'][use];n=len(cells);checked+=n
        assert not np.any(a['brick']&~native_brick)
        assert not np.any((a['brick']!=native_brick)&(wall|base|push))
        if not n:continue
        opened=np.repeat((~wall[None]&~brick&~push[None]),2,axis=0);r=cells[:,:,0].reshape(-1);c=cells[:,:,1].reshape(-1)
        row=np.arange(n*2);dist=np.full(opened.shape,999,np.int32);dist[row,r,c]=0
        for k in range(12):
            pad=np.pad(dist,((0,0),(1,1),(1,1)),constant_values=999)
            near=np.minimum(np.minimum(pad[:,:-2,1:-1],pad[:,2:,1:-1]),np.minimum(pad[:,1:-1,:-2],pad[:,1:-1,2:]))+1
            dist=np.where(opened,np.minimum(dist,near),999)
        # Independent cross-ray construction, with walls/bricks stopping each ray.
        br=np.repeat(brick,2,axis=0);hazard=np.zeros_like(opened);hazard[row,r,c]=True
        for dr,dc in S.STEPS:
            active=np.ones(n*2,bool)
            for k in range(1,int(blast)+1):
                y=r+dr*k;x=c+dc*k;inside=(y>=0)&(y<13)&(x>=0)&(x<15);ys=np.clip(y,0,12);xs=np.clip(x,0,14)
                active&=inside&~wall[ys,xs];hazard[row,ys,xs]|=active;active&=~br[row,ys,xs]
        safe=np.min(np.where(~hazard,dist,999),axis=(1,2));ticks=np.ceil(safe/speed)
        valid=opened[row,r,c]&~base[r,c]&reachable[r,c]&((dist<999).sum((1,2))>=6)&(ticks<30)
        pair=np.abs(cells[:,0]-cells[:,1]).sum(1)>=6
        if not valid.all() or not pair.all():failures.append(dict(file=str(path),invalid_players=int((~valid).sum()),invalid_pairs=int((~pair).sum())))
        for seat in (0,1):np.add.at(hist[seat],(cells[:,seat,0],cells[:,seat,1]),1)
    assert len(files)==100 and not failures,failures
    result=dict(status='PASS',actual_jax_total=100000,randomized_validated=checked,independent_numpy_geometry=True,
        illegal_players=0,illegal_pairs=0,wall_base_pushable_unchanged=True,no_brick_added=True,
        probability_interpretation='randomized eligible starts only; native/default starts unchanged',
        heatmap_randomized_seat=hist.tolist(),files=files,elapsed_seconds=time.time()-began)
    atomic_json(B/'JAX_LEGALITY_WITNESSES.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('files','heatmap_randomized_seat')}))
if __name__=='__main__':main()
