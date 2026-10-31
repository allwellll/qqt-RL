"""Isolated R3 on-policy worker, counted smoke, full-state atomic continuation."""
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bun_native_v2_custody_train import worker_source as parent
    s=parent().replace("BASE=ROOT/'runs/bun_native_v2_20261030'","BASE=ROOT/'runs/r3_items_route_20261030'")
    s=s.replace('bun_native_v2_custody_collector as M','r3_items_collector as M')
    anchor="    M.configure(protocol['r1_attack'],protocol['r2_antifarm'])"
    assert s.count(anchor)==1
    s=s.replace(anchor,"""    if not cps and not protocol.get('r2_same_version',False):
        from scripts.r3_items_observation import migrate
        # Exact retained old rows/moments/counters; only new rows are zero.
        params=migrate(params);adam=migrate(adam);ring=migrate(ring);fixed=migrate(fixed)
        atomic_json(out/'INPUT_MIGRATION.json',dict(status='PASS',old_rows_and_moments_retained=True,
            new_rows_zero=True,Adam_counters_retained=True,Actor_Critic_independent=True,
            actor_observation=[25,13,15],global_vector=37,source=protocol['continuation_source']))
"""+anchor)
    s=s.replace('Actor_Critic_Adam_ring_exact=True,old_env_aux_imported=False',
        'Actor_Critic_Adam_ring_old_rows_exact=True,zero_new_input_rows=True,old_env_aux_imported=False')
    s=s.replace("ROOT/'runs/iter_pbt_v1',ROOT/'runs/bomb_threat_v1'","ROOT/'runs/iter_pbt_v1',ROOT/'runs/bomb_threat_v1',ROOT/'runs/bun_native_v2_20261030'")
    s=s.replace("def paused():return any(","def paused():return (BASE/'RESOURCE_PAUSE.json').exists() or any(")
    s=s.replace("r2_reward_version=protocol['r2_reward_version'])","r2_reward_version=protocol['r2_reward_version'],r3_reward_version=protocol['r3_reward_version'])")
    anchor='        before=P.one(par);began=time.monotonic()'
    assert s.count(anchor)==1
    s=s.replace(anchor,"""        if (index-1)%40==0:
            request=ROOT/protocol['r3_item_coefficient_request']
            control=json.loads(request.read_text()) if request.exists() else {}
            if control.get('protocol_id')=='R3-ITEMS-ROUTE-CHAIN-20261030':
                factor=float(control['factor']);assert factor in (1.,.5)
                ledger=dict(ax['combat_mix']['bomb_threat'])
                ledger['r3']=dict(ledger['r3'],item_scale=j.full_like(ledger['r3']['item_scale'],factor))
                ledger['items']=dict(ledger['items'],scale=j.full_like(ledger['items']['scale'],factor))
                ax=dict(ax,combat_mix=dict(ax['combat_mix'],bomb_threat=ledger))
                atomic_json(out/'R3_ITEM_COEFFICIENT_APPLIED.json',dict(factor=factor,boundary=index-1,request_sha256=sha(request),old_pending_not_repriced=True))
"""+anchor)
    anchor="            assert row['exposure']['r1_item_pay']>0,'no matured item progress reward entered collector/GAE'"
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+"\n            assert sum(row['exposure']['r3_positive_pay'])>0,'R3 intervention never reached real collector/GAE'\n            assert min(row['update_l2'].values())>0,'no real Actor/Critic update'")
    return s


def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast import install as fast
    fast()
    m=ModuleType('r3_worker');m.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),m.__dict__);m.main()


if __name__=='__main__':main()
