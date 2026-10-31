"""CPU-only reset/respawn/Bank migration checks; no policy training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
from types import SimpleNamespace
import time,numpy as np
import jax,jax.numpy as j
from scripts.population_run_state import atomic_json,sha
from scripts.bun_spawn_side import install,install_collector,side_counts,VERSION
from jax_bomb import je07_bun_env_r1_items as E

def main():
    started=time.time();E.prepare();E.configure_training('full=1',5,reward_profile='legacy',retention_replay=0)
    keys=jax.random.split(jax.random.PRNGKey(2026317500),256)
    old=jax.jit(jax.vmap(E._fresh))(keys)
    old_hashes=[np.array(x) for x in jax.tree.leaves(old)]
    install()
    lessons=j.eye(16)[j.arange(256)%16]
    states=jax.jit(jax.vmap(E._fresh))(keys,lessons)
    homes=np.floor(np.array(states.bun_spawn_pos)).astype(int)
    assert side_counts(homes)['both_own']==256
    np.testing.assert_array_equal(states.pos,states.bun_spawn_pos)
    # Actual environment respawn transition, both dead players one tick away.
    st=jax.tree.map(lambda x:x[0],states)
    st=st._replace(core=st.core._replace(alive=j.array([False,False]),pos=j.array([[9.5,12.5],[9.5,2.5]])),bun_respawn=j.array([1,1],j.int16))
    after,_,info=jax.jit(lambda s:E.step(s,j.array([[4,0],[4,0]]),jax.random.PRNGKey(9),False,True,require_stock_window=False))(st)
    np.testing.assert_array_equal(after.pos,st.bun_spawn_pos)
    assert np.asarray(after.alive).all()
    # Bank current positions must not teleport; only historical wrong respawn
    # homes are replaced with legal native homes, preserving causal trajectory.
    fake=SimpleNamespace(fresh=lambda k:(states,{'provenance_source':j.arange(256)}))
    install_collector(fake);restored,aux=fake.fresh(keys)
    np.testing.assert_array_equal(restored.pos,states.pos)
    assert side_counts(np.floor(np.asarray(restored.bun_spawn_pos)).astype(int))['both_own']==256
    # Random-brick A3SR resets remain within the same physical native groups.
    from scripts import bomb_threat_a3sr_spawn as S
    _,meta=jax.jit(jax.vmap(S.sample_jax))(keys[:32],jax.tree.map(lambda x:x[:32],states.core))
    random_core,_=jax.jit(jax.vmap(S.sample_jax))(keys[:32],jax.tree.map(lambda x:x[:32],states.core))
    assert side_counts(np.floor(np.asarray(random_core.pos)).astype(int))['both_own']==32
    # All pre-migration source data remained immutable.
    for a,b in zip(old_hashes,jax.tree.leaves(old)):np.testing.assert_array_equal(a,b)
    out=Path('runs/v2_ownside_selection_20261031')
    atomic_json(out/'JAX_FOCUSED_ACCEPTANCE.json',dict(status='PASS',version=VERSION,reset_cases=256,lessons=16,
        respawn_cases=2,random_brick_cases=32,Bank_current_position_exact=True,old_state_immutable=True,
        wrong_sides=0,wall_seconds=time.time()-started,source_sha256={f:sha(Path(f)) for f in ['scripts/bun_spawn_side.py','scripts/bun_spawn_side_train.py']},unix=time.time()))
    print('JAX own-side reset/respawn/Bank focused PASS',time.time()-started)
if __name__=='__main__':main()
