"""Bounded R3-sp worker; opt-in pool/monitor changes, original R3 untouched."""
import os
import json
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bun_spawn_side_train import worker_source as parent
    source = parent(True, True)
    anchor = "BASE=ROOT/'runs/r3_items_route_20261030'"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "BASE=ROOT/'runs/r3_selfplay_ramp_20261031'")
    anchor = "    from scripts.bun_spawn_side import install_collector"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "    from scripts.r3_selfplay_monitor import install as install_monitor\n    install_monitor(M)\n"+anchor)
    anchor = '    params=jax.tree.map(j.asarray,params);adam=jax.tree.map(j.asarray,adam)'
    assert source.count(anchor) == 1
    source = source.replace(anchor, """    from scripts.r3_items_observation import migrate
    pool=[]
    for identity in protocol['selfplay_fixed_pool']:
        cp=ROOT/identity['checkpoint']
        assert sha(cp/'policy.pt')==identity['policy_sha256']
        pool.append(migrate(load_actor(cp)))
    fixed=tuple(pool)
    atomic_json(out/'FIXED_POOL.json',dict(status='PASS',models=protocol['selfplay_fixed_pool'],
        assignment='fixed actor by immutable environment index modulo3; no PFSP',
        no_reward_change=True,unix=time.time()))
"""+anchor)
    anchor = "    probe_groups['bot']=np.array([0]) if first_kind<3 else np.array([],int)"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor+"\n    for identity in range(3):probe_groups['fixed'+str(identity)]=np.array([0]) if first_kind==5 and identity==0 else np.array([],int)")
    anchor = "    receipt=None\n"
    assert source.count(anchor) == 1
    # The baseline uses train RNG/states and never changes the starting state.
    # No gradients, optimizer steps, gate/dev/holdout data or reward edits.
    source = source.replace(anchor, """    baseline=BASE/'MONITOR_BASELINE.json'
    if not args.verify_restore and not baseline.exists():
        before_digest=digest((P.one(par),P.one(adm),st,ax,keys,ring))
        ss,aa,kk=st,ax,keys;totals={}
        for diagnostic_rollout in range(20):
            if paused():
                save(start,dict(user_paused=True,baseline_interrupted=True,
                    diagnostic_rollouts_discarded=diagnostic_rollout,formal_updates_discarded=0))
                return
            (ss,aa,kk),_,diagnostics=collect(par,ss,aa,kk,bs,ba,rep(ring[start%4]),rep(fixed))
            jax.block_until_ready(diagnostics)
            for name,value in jax.device_get(diagnostics).items():
                if name.startswith('r3_sp_'):
                    value=np.asarray(value).sum(0)
                    totals[name]=totals.get(name,np.zeros_like(value))+value
        assert digest((P.one(par),P.one(adm),st,ax,keys,ring))==before_digest
        atomic_json(baseline,dict(status='PASS',metrics={k:v.tolist() for k,v in totals.items()},
            parent=protocol['continuation_source'],train_only=True,updates=0,
            diagnostic_environment_steps=20*batch_size,starting_state_unchanged=True,
            source_sha256=protocol['source_sha256'],unix=time.time()))
"""+anchor)
    source=source.replace('Actor_Critic_Adam_ring_old_rows_exact=True,zero_new_input_rows=True',
        'Actor_Critic_Adam_ring_exact=True,no_input_schema_change=True')
    return source


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'], JAX_PLATFORMS='cuda',
        JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    from scripts.bun_spawn_side import install as ownside
    ownside()
    module=ModuleType('r3_sp_worker');module.__file__=__file__
    exec(compile(worker_source(), __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
