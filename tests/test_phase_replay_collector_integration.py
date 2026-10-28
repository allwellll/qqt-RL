"""Isolated real collector/PPO test with the successor aggregate dispatch."""


def test_real_joint_collector_ppo_reset_and_restore_with_balanced_replay(monkeypatch):
    from scripts import phase_balanced_joint_replay as candidate
    from scripts import prestock_joint_replay as joint
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    # In-process dispatch only; repository/running sources are never patched.
    monkeypatch.setattr(joint,'aggregate',candidate.aggregate)
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
