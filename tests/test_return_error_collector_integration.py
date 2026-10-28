"""Isolated native collector/GAE/current-fit replay/PPO/save-restore wiring."""
def test_real_collector_and_ppo_restore_with_current_fit_return_sampling(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax_bomb import jax_train as train
    from scripts import return_error_replay as candidate,prestock_joint_replay as joint
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    actor=train.init_net(jax.random.PRNGKey(42),'transformer',24,13,15,embed=16,depth=1)
    actor['heads']['wb']=(actor['heads']['wb'][0],jnp.array([1000.,-1000.,-1000.]))
    # Matches the underlying test's actual pre-update Actor. No disk edits.
    monkeypatch.setattr(joint,'aggregate',lambda old,batch,key,generation:
        candidate.aggregate(old,batch,key,generation,actor))
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
