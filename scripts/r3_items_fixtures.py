"""Native diagnostic fixtures only; never inserted into training/Bank."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import time
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json

OUT=Path('runs/r3_items_route_20261030')


def main():
    from scripts.r3_items_runtime import install
    install()
    from scripts import carrier_mechanism_engine as E,combat_item_engine as T,r3_items_collector as M,r3_items_reward as R
    T.configure();M.configure();n=6;keys=jax.random.split(jax.random.PRNGKey(2026208198),n)
    st,ax=E.initial(keys,1);seats=j.zeros(n,j.int32);rows=j.arange(n)
    pos=j.tile(j.array([[[10.5,12.5],[8.5,9.5]]]),(n,1,1))
    fuse=j.zeros_like(st.fuse).at[:,8,4].set(1).at[:,8,6].set(20).at[:,8,8].set(30)
    fuse=fuse.at[4,8,4].set(20).at[4,8,6].set(20).at[4,8,8].set(20)
    owner=j.full_like(st.core.owner,-1).at[:,8,4].set(0).at[:,8,6].set(0).at[:,8,8].set(0)
    owner=owner.at[5,8,4].set(1)
    # Row2 self-hit; row3 empty at home, enemy far away.
    pos=pos.at[2,0].set(j.array([8.5,3.5])).at[3,1].set(j.array([11.5,12.5]))
    st=st._replace(core=st.core._replace(pos=pos,fuse=fuse,owner=owner,bomb_blast=j.where(fuse>0,3,0),wall=j.zeros_like(st.wall),brick=j.zeros_like(st.brick),
        crate=j.zeros_like(st.core.crate),invuln=j.zeros_like(st.core.invuln),blast_linger=j.zeros_like(st.core.blast_linger)),
        tactical_crate=j.zeros_like(st.tactical_crate),field_item=j.zeros_like(st.field_item))
    from scripts.bomb_threat_r1_item_progress import fraction
    meta=M.metadata(st,j.ones(n,j.int32),j.zeros(n,bool),seats)
    s=meta['bomb_threat']['r3'];ids=s['bomb_batch'].at[:,8,4].set(0).at[:,8,6].set(1).at[:,8,8].set(2)
    ids=ids.at[1,8,6].set(0).at[1,8,8].set(0)
    meta=dict(meta,bomb_threat=dict(meta['bomb_threat'],r3=dict(s,bomb_batch=ids)))
    ax=dict(ax,r1_attribute_highwater=fraction(j.stack([st.core.bombs_cap,st.core.blast_cap,st.core.spd_g],-1)),tactics=T.events.state(n),
        provenance_source=j.full(n,-1,j.int32),combat_mix=meta,a3sr={k:j.zeros(n,j.int32) for k in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
    action=j.zeros((n,2,2),j.int32).at[:,:,0].set(4)
    fn=jax.jit(lambda s,a,k:M.transition(s,a,action,k,0.))
    began=time.time();after,new,r,d,info,parts,m=fn(st,ax,keys);jax.block_until_ready(r)
    k=np.asarray(m['r3_actual_cross_batch']);candidate=np.asarray(m['r3_candidate'])
    atomic_json(OUT/'NATIVE_EVENT_FIXTURES_RAW.json',dict(cross_batch=k.tolist(),candidates=candidate.tolist(),early=np.asarray(info['triggered_early']).tolist(),
        trigger_actor=np.asarray(info['trigger_actor']).tolist(),fuse_before=np.asarray(st.fuse).tolist(),fuse_after=np.asarray(after.fuse).tolist()))
    np.testing.assert_array_equal(k[:2],[2,0]);np.testing.assert_allclose(candidate[0,R.CHAIN],.03,atol=1e-7)
    assert candidate[1,R.CHAIN]==0 and candidate[2,R.CHAIN]==0 and candidate[3,R.CHAIN]==0 and candidate[4,R.CHAIN]==0 and candidate[5,R.CHAIN]==0
    # Fixed topology-only counterfactuals: no train sample construction.
    barrier=j.zeros_like(st.brick).at[:,:,7].set(True)
    barrierst=st._replace(core=st.core._replace(fuse=j.zeros_like(st.fuse),brick=barrier))
    a=np.asarray(jax.jit(R.road_distance)(barrierst));gap=barrier.at[:,6,7].set(False)
    b=np.asarray(jax.jit(R.road_distance)(barrierst._replace(core=barrierst.core._replace(brick=gap))))
    assert np.all(b<a)
    atomic_json(OUT/'NATIVE_EVENT_FIXTURES.json',dict(status='PASS',fixture_only=True,training_data=False,real_engine_explosion=True,
        cases=['three batches real chain','same batch','self blast','empty explosion','live stock no explosion','enemy cause'],
        cross_batch_count=k.tolist(),chain_candidates=candidate[:,R.CHAIN].tolist(),road_before=a.tolist(),road_after=b.tolist(),
        limitations=['diagnostic fixture setup, not autonomous ability','synthetic carrier states are never training inputs'],wall_seconds=time.time()-began))
    print('native chain and topology fixtures PASS',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'NATIVE_EVENT_FIXTURES_BLOCKED.json',dict(status='BLOCKED',reason='inspect NATIVE_EVENT_FIXTURES.log',unix=time.time()))
        raise
