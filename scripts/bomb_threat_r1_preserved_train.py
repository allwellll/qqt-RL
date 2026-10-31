"""Slot03 legacy-preserving R1 reward version; same native-item physics."""
from pathlib import Path
from types import ModuleType
from scripts.bomb_threat_r1_progress_train import worker_source as original_source

def worker_source():
    s=original_source().replace('bomb_threat_r1_progress_collector as M','bomb_threat_r1_preserved_collector as M')
    s=s.replace("items_v2/cache'","items_v2/conservation_v3/cache'")
    s=s.replace('from scripts.bomb_threat_r1_progress_reward import native_safety','from scripts.bomb_threat_r1_preserved_reward import native_safety')
    begin=s.index('        if not cps:\n            exported=')
    end=s.index("        params=payload['params']",begin)
    s=s[:begin]+"        payload=pickle.load((cp/'state.pkl').open('rb'))\n"+s[end:]
    begin=s.index('    if not cps:\n        preserved=')
    end=s.index('    # Initialize cached safety',begin)
    migration="""    if not cps:
        from scripts import bomb_threat_r1_preserved_reward as PR
        previous=ax['combat_mix']['bomb_threat']
        from scripts import bomb_threat_r1_progress_collector as predecessor
        predecessor.configure(True)
        token=json.loads((source.parent.parent/f'NEXT_STEP_u{start:06d}.json').read_text())
        kind=int(np.asarray(ax['combat_mix']['kind'])[0,0])
        groups={name:np.array([0]) if kind==i else np.array([],int) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]}
        groups['bot']=np.array([0]) if kind<3 else np.array([],int)
        ss,aa,kk=jax.tree.map(lambda x:j.asarray(x[0,:1]),(st,ax,keys))
        chooser=jax.jit(lambda p,s,a,k,r:predecessor.choose(p,r,fixed,s,a,k,groups))
        stepper=jax.jit(lambda s,a,ac,k:predecessor.transition(s,a,ac,k,0.))
        nk,ke,ac,_=chooser(params,ss,aa,kk,ring[start%4])
        assert digest((nk,ke,ac,stepper(ss,aa,ac,ke)))==token['digest'],'predecessor next-step mismatch'
        assert digest(adam)==token['Adam_digest']
        assert digest((st,ax,keys))==token['full_environment_digest']
        assert digest(ring)==token['ring_digest']
        preserved=digest((params,adam,st,keys,ring,ring_cursor,fixed,warm_rows))
        credit=shard(PR.state(n),1)
        credit=dict(credit,pre_revision=previous,pre_revision_episode=j.ones((1,n),bool))
        ax=dict(ax,combat_mix=dict(ax['combat_mix'],bomb_threat=credit))
        assert digest((params,adam,st,keys,ring,ring_cursor,fixed,warm_rows))==preserved
        atomic_json(out/'SOURCE_HANDOFF.json',dict(status='PASS',source=protocol['continuation_source'],
            Actor_Critic_Adam_env_RNG_ring_exact=True,previous_full_credit_preserved=True,donor_next_step_exact_before_credit_migration=True,
            reward_change_only_after_each_natural_episode_boundary=True,critic_warmup_updates=0,unix=time.time()))
"""
    s=s[:begin]+migration+s[end:]
    anchor="            assert row['exposure']['r1_item_pay']>0,'no matured item progress reward entered collector/GAE'"
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+"\n            assert row['exposure']['r1_legacy_pay']>0,'new legacy payment stayed zero'\n            assert row['exposure']['r1_additions_pay']>0,'new additions payment stayed zero'\n            assert row['exposure']['r1_pre_revision_ticks']<262144,'all smoke exposure still prior-revision episodes'")
    return s

def main():
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')
    from scripts.bomb_threat_r1_items_runtime import install
    install()
    m=ModuleType('r1_preserved_worker');m.__file__=str(Path(__file__))
    exec(compile(worker_source(),__file__,'exec'),m.__dict__);m.main()

if __name__=='__main__':main()
