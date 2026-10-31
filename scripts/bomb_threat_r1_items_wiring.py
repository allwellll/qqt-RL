"""Native _grow -> separate progress ledger -> learner reward fixture.

Diagnostic fixtures are not Bank or training samples. The real on-policy
optimizer check is separately required by the GPU worker smoke.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from scripts.bomb_threat_r1_items_runtime import install
install()
from pathlib import Path
import time
import numpy as np
import jax,jax.numpy as j
from scripts import combat_item_engine as T,carrier_mechanism_engine as E,bomb_threat_r1_progress_collector as M
from scripts.bomb_threat_runtime import install as stock
from scripts.population_run_state import atomic_json,lock
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def main():
    guard=lock(OUT/'WIRING.lock');began=time.time();T.configure();stock();M.configure(True)
    keys=jax.random.split(jax.random.PRNGKey(2026183099),2);st,ax=E.initial(keys,0);seats=j.array([0,1]);rows=j.arange(2)
    cells=j.floor(st.pos[rows,seats]).astype(j.int32)
    st=st._replace(core=st.core._replace(crate=j.zeros_like(st.core.crate).at[rows,cells[:,0],cells[:,1]].set(3)))
    from scripts.bomb_threat_r1_item_progress import fraction
    ax=dict(ax,r1_attribute_highwater=fraction(j.stack([st.core.bombs_cap,st.core.blast_cap,st.core.spd_g],-1)),tactics=T.events.state(2),provenance_source=j.full(2,-1,j.int32),combat_mix=M.metadata(st,j.ones(2,j.int32),j.zeros(2,bool),seats),
        a3sr={k:j.zeros(2,j.int32) for k in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
    action=j.zeros((2,2,2),j.int32).at[:,:,0].set(4)
    fn=jax.jit(lambda s,a,k:M.transition(s,a,action,k,0.))
    rawfn=jax.jit(lambda s,a,k:T.transition(s,a,action,k,'control'))
    total=np.zeros(2);candidates=np.zeros(2)
    for tick in range(31):
        prior=st;aux={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr','r1_attribute_highwater')}
        st,ax,r,d,info,parts,m=fn(st,ax,keys);jax.block_until_ready(r)
        raw=rawfn(prior,aux,keys)
        np.testing.assert_allclose(np.asarray(r)[rows,seats]-np.asarray(raw[2])[rows,seats],np.asarray(m['pay']),atol=3e-5)
        np.testing.assert_array_equal(np.asarray(r)[rows,1-np.asarray(seats)],np.asarray(raw[2])[rows,1-np.asarray(seats)])
        total+=np.asarray(m['r1_item_pay']);candidates+=np.asarray(m['r1_item_candidate'])
    np.testing.assert_allclose(total,.08/7,rtol=2e-5,atol=2e-7)
    np.testing.assert_allclose(candidates,total,rtol=2e-5,atol=2e-7)
    atomic_json(OUT/'WIRING.json',dict(status='PASS',physical_seats=[0,1],actual_native_speed_growth=True,delayed_item_pay=total.tolist(),
        learner_reward_projection_exact=True,opponent_reward_unchanged=True,ticks=31,fixture_only=True,training_data=False,
        optimizer_wiring='real GPU smoke separately requires nonzero matured item pay in on-policy collector plus finite Actor/Critic updates',wall_seconds=time.time()-began))

if __name__=='__main__':main()
