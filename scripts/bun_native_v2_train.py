"""New native-v2 R2 fork, retaining donor Actor/Critic/Adam/ring exactly."""
from pathlib import Path
from types import ModuleType


def worker_source():
    from scripts.bomb_threat_r1_progress_train import worker_source as parent
    source=parent()
    source=source.replace("BASE=ROOT/'runs/bomb_threat_v1'","BASE=ROOT/'runs/bun_native_v2_20261030'")
    source=source.replace('bomb_threat_r1_progress_collector as M','bun_native_v2_collector as M')
    source=source.replace("BASE/'iter2/r1_attack_20261030/items_v2/cache'","BASE/'cache'")
    begin=source.index('        if not cps:\n            exported=')
    end=source.index("        params=payload['params']",begin)
    source=source[:begin]+"        payload=pickle.load((cp/'state.pkl').open('rb'))\n"+source[end:]
    anchor="    M.configure(protocol['r1_attack'])"
    assert source.count(anchor)==1
    source=source.replace(anchor,"    M.configure(protocol['r1_attack'],protocol['r2_antifarm'])")
    # Migration at an explicit episode reset; no old-rule in-flight env/aux.
    # Same-version chunk continuation retains complete environment and RNG.
    begin=source.index('    if not cps:\n        preserved=')
    end=source.index('    # Initialize cached safety',begin)
    replacement='''    if not cps and protocol.get('r2_same_version',False):
        assert protocol['rule_version']=='bun-native-v2'
        atomic_json(out/'SOURCE_HANDOFF.json',dict(status='PASS',source=protocol['continuation_source'],
            Actor_Critic_Adam_env_aux_RNG_ring_exact=True,rule_version='bun-native-v2',unix=time.time()))
    elif not cps:
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
            real_bank_restored=True,RNG_rekeyed=True,new_reward_ledger_reset=True,
            rule_version='bun-native-v2',antifarm=protocol['r2_antifarm'],
            critic_warmup_updates=0,formal_updates_discarded=0,unix=time.time()))
'''
    source=source[:begin]+replacement+source[end:]
    # All user global pause/reboot files take precedence over this new queue.
    old="    def paused():return any((b/f).exists() for b in (BASE,ROOT/'runs/iter_pbt_v1') for f in ('USER_PAUSE.json','USER_STOP.json'))"
    assert source.count(old)==1
    source=source.replace(old,"    def paused():return any((b/f).exists() for b in (BASE,ROOT/'runs/iter_pbt_v1',ROOT/'runs/bomb_threat_v1') for f in ('USER_PAUSE.json','USER_STOP.json','STOP_FOR_REBOOT.json'))")
    anchor="            protocol_sha256=sha(Path(args.protocol)))"
    source=source.replace(anchor,"            protocol_sha256=sha(Path(args.protocol)),rule_version='bun-native-v2',r2_antifarm=protocol['r2_antifarm'])")
    # Metadata must label actual inherited warmup, not claim new R2 warmup.
    source=source.replace('warmup_environment_steps=20*batch_size,','warmup_environment_steps=0, inherited_warmup_rows=len(warm_rows),')
    anchor='        before=P.one(par);began=time.monotonic()'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n        atomic_json(out/'CYCLE_STAGE.json',dict(stage='COLLECTOR',update=index,unix=time.time()))")
    anchor='        rollout=time.monotonic()-began'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n        atomic_json(out/'CYCLE_STAGE.json',dict(stage='OPTIMIZER',update=index,rollout_seconds=rollout,unix=time.time()))")
    anchor='        jax.block_until_ready(par);after=P.one(par);delta=P.norms(before,after)'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n        atomic_json(out/'CYCLE_STAGE.json',dict(stage='UPDATED_CHECKING_ATOMIC',update=index,cycle_seconds=time.monotonic()-began,unix=time.time()))")
    source=source.replace("int(protocol['chunk_start_update'])+2) or index%40", "int(protocol['chunk_start_update'])+2,int(protocol['chunk_start_update'])+3) or index%40")
    anchor='    if args.verify_restore:\n'
    handoff='''    if not cps and protocol.get('r2_same_version',False):
        expected=json.loads((source.parent.parent/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert probe(P.one(par),st,ax,keys,start)==expected['digest'],'v2 continuation next-step mismatch'
        assert digest(P.one(adm))==expected['Adam_digest']
        assert digest((st,ax,keys))==expected['full_environment_digest']
        assert digest(ring)==expected['ring_digest']
'''
    assert source.count(anchor)==1
    source=source.replace(anchor,handoff+anchor)
    # Native-v2 reward changes are not subject to the unrelated old R1 cap
    # component indices. Event shaping still retains its own per-episode cap.
    begin=source.index("        if spec['iter2_arm'] in ('A3S','A3SR') and index%40==0:")
    end=source.index("        with (out/'updates.jsonl')",begin)
    source=source[:begin]+source[end:]
    return source


def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bun_native_v2_runtime import install
    install()
    module=ModuleType('native_v2_R2_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),module.__dict__);module.main()


if __name__=='__main__':main()
