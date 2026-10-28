"""Real native learner collector -> production-sized longer FIFO -> PPO/restore."""


def test_real_collector_4096_fifo_and_unchanged_128_loss_slots(monkeypatch):
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    from jax_bomb import jax_train as train
    from scripts import prestock_joint_replay as joint,je09_stage_h5c_dagger as legacy
    from scripts.return_error_replay import aggregate as selected_aggregate
    from scripts.replay_horizon_recipe import patched
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    namespace={};exec(compile(patched(Path('scripts/je09_stage_h5c_dagger.py').read_text()),'isolated-horizon-default','exec'),namespace)
    monkeypatch.setattr(legacy,'CAPACITY',namespace['CAPACITY'])
    monkeypatch.setattr(legacy,'ADDITIONS',namespace['ADDITIONS'])
    prior_buffer=joint.buffer
    def production_buffer(**kwargs):
        kwargs.update(capacity=namespace['CAPACITY'],additions=namespace['ADDITIONS'])
        return prior_buffer(**kwargs)
    monkeypatch.setattr(joint,'buffer',production_buffer)
    actor=train.init_net(jax.random.PRNGKey(42),'transformer',24,13,15,embed=16,depth=1)
    actor['heads']['wb']=(actor['heads']['wb'][0],jnp.array([1000.,-1000.,-1000.]))
    observed=[]
    def aggregate(old,batch,key,generation):
        assert old['valid'].shape==(4096,) and old['additions'].shape==()
        # The isolated collector has 8 real rows; the production write budget
        # is 256. Reuse those observed rows in this test without new rollouts.
        expanded=jax.tree.map(lambda x:jnp.tile(x,(32,)+(1,)*(x.ndim-1)),batch)
        updated,replay,stats=selected_aggregate(old,expanded,key,generation,actor)
        assert updated['valid'].shape==(4096,) and replay['valid'].shape==(128,)
        jax.debug.callback(lambda n:observed.append(int(n)),replay['valid'].sum())
        return updated,replay,stats
    monkeypatch.setattr(joint,'aggregate',aggregate)
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
    assert observed and all(n>0 for n in observed)
