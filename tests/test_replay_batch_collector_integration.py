"""Real collector, GAE, enlarged native replay, PPO and state restore."""


def test_real_collector_ppo_and_restore_with_128_native_replay_slots(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax_bomb import jax_train as train
    from scripts import prestock_joint_replay as joint,je09_stage_h5c_dagger as legacy
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    from tests.test_replay_batch_recipe import candidate_namespace
    candidate=candidate_namespace()
    # Restore the actual Actor scoring function; only the batch size is changed.
    from scripts.return_error_replay import target_mass
    candidate['target_mass']=target_mass
    actor=train.init_net(jax.random.PRNGKey(42),'transformer',24,13,15,embed=16,depth=1)
    actor['heads']['wb']=(actor['heads']['wb'][0],jnp.array([1000.,-1000.,-1000.]))
    # Same production dispatch as formal, with a tiny native collector fixture.
    monkeypatch.setattr(legacy,'CAPACITY',4)
    monkeypatch.setattr(legacy,'ADDITIONS',2)
    calls=[]
    valid_counts=[]
    def aggregate(old,batch,key,generation):
        updated,replay,stats=candidate['aggregate'](old,batch,key,generation,actor)
        assert replay['valid'].shape==(128,)
        jax.debug.callback(lambda count:valid_counts.append(int(count)),replay['valid'].sum())
        calls.append(replay['valid'].shape)
        return updated,replay,stats
    monkeypatch.setattr(joint,'aggregate',aggregate)
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
    assert calls==[(128,)]
    assert valid_counts and all(count>0 for count in valid_counts)
