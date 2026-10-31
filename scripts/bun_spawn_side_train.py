"""Own-side successor worker. Never selected by R1 or an in-flight old chunk."""
import os,json
from pathlib import Path
from types import ModuleType

def worker_source(r3=False,custody=False):
    if r3:
        from scripts.r3_items_fast2_train import worker_source as parent
    elif custody:
        from scripts.bun_native_v2_custody_fast2_train import worker_source as parent
    else:
        from scripts.bun_native_v2_train import worker_source as parent
    s=parent()
    anchor="    M.configure(protocol['r1_attack'],protocol['r2_antifarm'])"
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+"\n    from scripts.bun_spawn_side import install_collector\n    install_collector(M)")
    # First migration is deliberately a fresh episode boundary, not an exact
    # old-env next step. Later own-side chunks retain exact full state.
    anchor="if not cps and protocol.get('r2_same_version',False):"
    assert s.count(anchor)==2
    s=s.replace(anchor,"if not cps and protocol.get('r2_same_version',False) and not protocol.get('spawn_side_migration',False):")
    s=s.replace('keys=jax.random.split(jax.random.PRNGKey(seed),n)',
        "keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),int(protocol['chunk_start_update'])^0x51DE),n)")
    s=s.replace("out/'R3_PERFORMANCE2_RESTORE.json'","out/'RESTORE.json'").replace("out/'PERFORMANCE2_RESTORE.json'","out/'RESTORE.json'")
    anchor='    params=jax.tree.map(j.asarray,params);adam=jax.tree.map(j.asarray,adam)'
    assert s.count(anchor)==1
    s=s.replace(anchor,"""    if not cps and protocol.get('spawn_side_migration',False):
        homes=np.asarray(st.bun_spawn_pos).reshape(-1,2,2)
        assert (homes[:,0,1]<7.5).all() and (homes[:,1,1]>=8.5).all()
        atomic_json(out/'SPAWN_SIDE_MIGRATION.json',dict(status='PASS',source=protocol['continuation_source'],
            spawn_side_version=protocol['spawn_side_version'],Actor_Critic_Adam_ring_exact=True,
            old_env_aux_RNG_retained=False,old_unfinished_episodes_rebuilt=n,discarded_updates=0,
            real_Bank_current_positions_preserved=True,wrong_respawn_homes=0,unix=time.time()))
"""+anchor)
    anchor="            update=index,formal_update=index,environment_steps=env_steps,optimizer_steps=optimizer_steps,metrics=row)"
    assert s.count(anchor)==1
    s=s.replace(anchor,"            update=index,formal_update=index,environment_steps=env_steps,optimizer_steps=optimizer_steps,metrics=row,spawn_side_version=protocol['spawn_side_version'])")
    anchor='        jax.block_until_ready(par);after=P.one(par);delta=P.norms(before,after)'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+"""
        homes=np.asarray(st.bun_spawn_pos).reshape(-1,2,2)
        assert (homes[:,0,1]<7.5).all() and (homes[:,1,1]>=8.5).all(),'wrong-side respawn in live collector'
""")
    return s

def main():
    import sys
    p=json.loads(Path(sys.argv[1]).read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    r3='r3_reward_version' in p;custody=p.get('r2_reward_version')=='custody-graded-conservative-v2'
    if r3:
        from scripts.r3_items_runtime import install
    else:
        from scripts.bun_native_v2_runtime import install
    install()
    if custody:
        from scripts.bun_native_v2_custody_fast2 import install as fast
        fast()
    from scripts.bun_spawn_side import install as own_side
    own_side()
    m=ModuleType('own_side_successor');m.__file__=__file__
    exec(compile(worker_source(r3,custody),__file__,'exec'),m.__dict__);m.main()

if __name__=='__main__':main()
