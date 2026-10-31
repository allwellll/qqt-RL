"""R1 native-items migration; old reference workers and artifacts stay frozen."""
from pathlib import Path
from types import ModuleType
from scripts.bomb_threat_r1_train import worker_source as old_source


def worker_source():
    s=old_source().replace('bomb_threat_r1_collector as M','bomb_threat_r1_progress_collector as M')
    s=s.replace("BASE/'iter2/r1_attack_20261030/cache'","BASE/'iter2/r1_attack_20261030/items_v2/cache'")
    begin=s.index('    if source is not None and not cps:\n');end=s.index('    if args.verify_restore:\n',begin)
    s=s[:begin]+s[end:]
    s=s.replace("items=[v for v in bank() if v['record']['split']=='train'];bs,ba=merged(items,'states'),merged(items,'aux')","bs,ba=None,None")
    # Source old-state deserialization is isolated in the audited export process.
    anchor="        payload=pickle.load((cp/'state.pkl').open('rb'))"
    replacement='''        if not cps:
            exported=json.loads((ROOT/protocol['r1_parent_export']).read_text())
            assert exported['status']=='PASS' and exported['sha256']==sha(ROOT/exported['output'])
            payload=pickle.load((ROOT/exported['output']).open('rb'))
            payload=dict(payload,states=None,aux=None,keys=None)
        else:
            payload=pickle.load((cp/'state.pkl').open('rb'))'''
    assert s.count(anchor)==1;s=s.replace(anchor,replacement)
    anchor='    params=jax.tree.map(j.asarray,params);adam=jax.tree.map(j.asarray,adam)'
    migration='''    if not cps:
        preserved=digest((params,adam,ring,ring_cursor,fixed,warm_rows))
        keys=jax.random.split(jax.random.PRNGKey(seed),n)
        spawns=json.loads((ROOT/protocol['spawn_manifest']).read_text())['spawn_cells']
        designated=j.asarray(combat_scene_mask(n,protocol['round14_collector_seed']^0x806,.10))
        kinds=fixed_allocation(n,protocol['round14_collector_seed'],protocol['A1_weights'])
        seats=j.asarray(M.seat_allocation(n,protocol['round14_collector_seed'],.5))
        fresh,aux=jax.jit(M.fresh)(keys,bs,ba,designated,j.asarray(spawns),seats)
        aux=dict(aux,combat_mix=M.metadata(fresh,kinds,designated,seats,aux['r1_attribute_highwater']))
        st,ax,keys=shard(fresh,1),shard(aux,1),shard(keys,1)
        assert digest((params,adam,ring,ring_cursor,fixed,warm_rows))==preserved
        atomic_json(out/'SOURCE_HANDOFF.json',dict(status='PASS',source=protocol['continuation_source'],
            Actor_Critic_Adam_ring_exact=True,old_env_aux_imported=False,natural_episode_restart=True,
            new_bank='genuine native-item causal acquisitions both physical seats',new_reward_ledger_reset=protocol['r1_attack'],
            RNG_rekeyed=True,rule_version=protocol['rule_version'],critic_warmup_updates=0,formal_updates_discarded=0,unix=time.time()))
    # Initialize cached safety constants eagerly, outside manual pmap mesh.
    if protocol['r1_attack']:
        from scripts.bomb_threat_r1_progress_reward import native_safety
        native_safety()
'''
    assert s.count(anchor)==1;s=s.replace(anchor,migration+anchor)
    finite_anchor="        assert finite and delta['actor']>0 and delta['critic']>0,'nonfinite or zero real PPO update'"
    assert s.count(finite_anchor)==1
    s=s.replace(finite_anchor,finite_anchor+"\n        assert int(np.max(np.asarray(st.native_items.overflow)))==0,'native item queue capacity overflow'")
    # New reward includes a separate item coefficient; anti-hack R1 scaling
    # does not silently scale item weights or retroactively change pending pay.
    anchor='        before=P.one(par);began=time.monotonic()'
    addition='''        if protocol['r1_attack'] and (index-1)%40==0:
            item_request=ROOT/protocol['r1_item_coefficient_request']
            request=json.loads(item_request.read_text()) if item_request.exists() else {}
            factor=float(request.get('factor',1.));assert factor in (1.,.5)
            if request.get('protocol_id')=='A3SR-ATTACK-R1-ITEMS-V2':
                credit=dict(ax['combat_mix']['bomb_threat']);item=dict(credit['items'])
                item['scale']=j.full_like(item['scale'],factor);credit['items']=item
                ax=dict(ax,combat_mix=dict(ax['combat_mix'],bomb_threat=credit))
                atomic_json(out/'ITEM_COEFFICIENT_APPLIED.json',dict(factor=factor,boundary=index-1,request_sha256=sha(item_request),unix=time.time()))
'''
    assert s.count(anchor)==1;s=s.replace(anchor,addition+anchor)
    smoke_anchor="        assert min(row['exposure']['bank_seat_resets'])>0,'no genuine Bank reset on both physical seats'"
    smoke_check="\n        if protocol['r1_attack']:\n            assert sum(row['exposure']['r1_item_actual_growth'])>0,'no actual native growth exposure'\n            assert row['exposure']['r1_item_candidate']>0,'highwater progress intervention remained zero'\n            assert row['exposure']['r1_item_pay']>0,'no matured item progress reward entered collector/GAE'"
    assert s.count(smoke_anchor)==1;s=s.replace(smoke_anchor,smoke_anchor+smoke_check)
    return s


def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bomb_threat_r1_items_runtime import install
    install()
    module=ModuleType('r1_progress_native_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__);module.main()

if __name__=='__main__':main()
