"""CPU-only real production collector/PPO validation for future quotas."""


def test_real_collector_ppo_reset_restore_with_navigation_replay(monkeypatch):
    from tests.test_navigation_replay_recipe import implementation
    from scripts import prestock_joint_replay as joint
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    candidate=implementation()
    monkeypatch.setattr(joint,'aggregate',candidate.aggregate)
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
