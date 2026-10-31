"""CPU actual transition acceptance for legacy-preserving version and migration."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from scripts.bomb_threat_r1_items_runtime import install
install()
import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from scripts import combat_item_engine as T,carrier_mechanism_engine as E,bomb_threat_r1_preserved_collector as M,bomb_threat_r1_progress_collector as OLD
from scripts import bomb_threat_r1_item_progress as I,bomb_threat_r1_preserved_reward as R
from scripts.bomb_threat_runtime import install as stock
from scripts.population_run_state import atomic_json
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2/conservation_v3')

def main():
    began=time.time();T.configure();stock();M.configure(True);OLD.configure(True)
    keys=jax.random.split(jax.random.PRNGKey(2026183098),2);st,ax=E.initial(keys,0);seats=j.array([0,1]);rows=j.arange(2)
    cells=j.floor(st.pos[rows,seats]).astype(j.int32)
    st=st._replace(core=st.core._replace(crate=j.zeros_like(st.core.crate).at[rows,cells[:,0],cells[:,1]].set(3)))
    ax=dict(ax,r1_attribute_highwater=I.fraction(j.stack([st.core.bombs_cap,st.core.blast_cap,st.core.spd_g],-1)),tactics=T.events.state(2),provenance_source=j.full(2,-1,j.int32),combat_mix=M.metadata(st,j.ones(2,j.int32),j.zeros(2,bool),seats),a3sr={k:j.zeros(2,j.int32) for k in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
    action=j.zeros((2,2,2),j.int32).at[:,:,0].set(4)
    fn=jax.jit(lambda s,a:M.transition(s,a,action,keys,0.));rawfn=jax.jit(lambda s,a:T.transition(s,a,action,keys,'control'))
    # First new episode: actual item growth and delayed collector payment.
    total=np.zeros(2)
    for tick in range(31):
        prior=st;base={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr','r1_attribute_highwater')}
        st,ax,r,d,info,parts,m=fn(st,ax);jax.block_until_ready(r)
        raw=rawfn(prior,base)
        np.testing.assert_allclose(np.asarray(r)[rows,seats]-np.asarray(raw[2])[rows,seats],np.asarray(m['pay']),atol=3e-5)
        np.testing.assert_array_equal(np.asarray(r)[rows,1-np.asarray(seats)],np.asarray(raw[2])[rows,1-np.asarray(seats)])
        total+=np.asarray(m['r1_item_pay'])
    np.testing.assert_allclose(total,.08/7,atol=2e-7)
    # An in-flight old episode is bitwise identical through the reward change.
    oldcredit=ax['combat_mix']['bomb_threat']['pre_revision']
    oldcredit=dict(oldcredit,items=dict(oldcredit['items'],highwater=I.fraction(j.array([[2.,2.,1.3],[2.,2.,1.3]]))))
    migrated=dict(ax['combat_mix']['bomb_threat'],pre_revision=oldcredit,pre_revision_episode=j.ones(2,bool))
    ma=dict(ax,combat_mix=dict(ax['combat_mix'],bomb_threat=migrated))
    oa=dict(ax,combat_mix=dict(ax['combat_mix'],bomb_threat=oldcredit))
    new=fn(st,ma);old=jax.jit(lambda s,a:OLD.transition(s,a,action,keys,0.))(st,oa)
    for x,y in zip(jax.tree.leaves((new[0],new[2:6])),jax.tree.leaves((old[0],old[2:6]))):np.testing.assert_array_equal(x,y)
    for x,y in zip(jax.tree.leaves(new[1]['combat_mix']['bomb_threat']['pre_revision']),jax.tree.leaves(old[1]['combat_mix']['bomb_threat'])):np.testing.assert_array_equal(x,y)
    atomic_json(OUT/'WIRING.json',dict(status='PASS',seats=[0,1],ticks=31,item_pay=total.tolist(),learner_projection_exact=True,opponent_unchanged=True,migrated_old_episode_next_step_reward_env_exact=True,fixture_only=True,training_data=False,wall_seconds=time.time()-began,unix=time.time()))

if __name__=='__main__':main()
