"""CPU native R1 transition wiring and P1 delivery checks, no GPU allocation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from scripts.bomb_threat_r1_items_runtime import install as native_install
native_install()
import json,pickle,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from scripts.population_run_state import atomic_json,sha
from scripts import combat_item_engine as T,bomb_threat_r1_progress_collector as M
from scripts.bomb_threat_r1_progress_reward import state
from scripts.bomb_threat_runtime import install
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def main():
    began=time.time();T.configure();install();M.configure(True)
    frozen=json.loads((OUT/'BANK_WITH_HISTORY.json').read_text());items=pickle.load(Path(frozen['p1_file']).open('rb'))
    from scripts.carrier_mechanism_analysis import merged
    st=merged(items,'states');ax=merged(items,'aux');n=len(st.t);seats=j.ones(n,j.int32)
    keys=jax.random.split(jax.random.PRNGKey(2026183034),n)
    ax=dict(ax,r1_attribute_highwater=j.asarray(np.concatenate([x['attribute_highwater'] for x in items])),provenance_source=j.arange(n),combat_mix=M.metadata(st,j.ones(n,j.int32),j.zeros(n,bool),seats,j.asarray(np.concatenate([x['attribute_highwater'] for x in items]))),
        a3sr={k:j.zeros(n,j.int32) if k!='brick_drop_rate' else j.zeros(n) for k in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
    action=j.full((n,2,2),0,j.int32).at[:,:,0].set(4)
    step=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,0.))
    after,na,r,d,info,parts,metrics=step(st,ax,action,keys);jax.block_until_ready(r)
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves((after,na,r,parts,metrics)))
    native={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr','r1_attribute_highwater')}
    raw=jax.jit(lambda s,a,ac,k:T.transition(s,a,ac,k,'control'))(st,native,action,keys)
    np.testing.assert_allclose(r[:,1]-raw[2][:,1],metrics['pay'],atol=2e-5)
    np.testing.assert_array_equal(r[:,0],raw[2][:,0])
    # Genuine carried P1 objects must be deposited into the actual P1 cage.
    # Fixed route intervention is diagnostic only; opponent waits, not student
    # capability evidence. No initial position/carry changes.
    initial_stock=np.asarray(st.bun_stored[:,1,0]);captures=np.zeros(n,bool);returned=np.zeros(n,bool)
    from scripts.je10_route_oracle import labels
    def walk(st,ax,keys):
        nk,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,2))(keys),0,1)
        o,g,mm,bm=M.inputs(st,seats);target=labels(st,ax,mm,seats,o[:,5])[:,:5]
        act=j.zeros((n,2,2),j.int32).at[:,:,0].set(4).at[:,1,0].set(target.argmax(-1))
        ns,na,r,d,i,p,m=M.transition(st,ax,act,ke,0.)
        return ns,na,nk,i['capture'][:,1]
    walk=jax.jit(walk)
    for tick in range(240):
        st,ax,keys,capture=walk(st,ax,keys);captures|=np.asarray(capture)
        returned|=np.asarray(st.bun_stored[:,1,0])>initial_stock
        if np.any(returned):break
    assert np.any(returned)&np.any(captures),'no actual P1 home deposit'
    atomic_json(OUT/'MICRO.json',dict(status='PASS',reward_projection_exact=True,opponent_reward_unchanged=True,
        native_transition_finite=True,p1_bank_states=n,p1_actual_deposits=int(returned.sum()),p1_capture_events=int(captures.sum()),
        route_ticks=tick+1,physical_P1_inventory_increase=True,initial_carried_state='restored actual native acquisition only',
        intervention='P1 legal route to own cage, enemy wait; not autonomous capability evidence',wall_seconds=time.time()-began,unix=time.time()))

if __name__=='__main__':main()
