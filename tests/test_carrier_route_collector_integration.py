"""Isolated CPU real collector/PPO check; no active source patch."""


def test_real_collector_ppo_reset_restore_with_physical_carrier_fields(monkeypatch):
    from scripts import je08_bun_potential as potential
    from scripts.carrier_route_fields import fields
    from scripts import phase_balanced_joint_replay as replay
    from scripts import prestock_joint_replay as joint
    from tests.test_carrier_route_recipe import candidate_refresh
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    candidate=candidate_refresh()
    monkeypatch.setattr(potential,'refresh',candidate['refresh'])
    monkeypatch.setattr(joint,'aggregate',replay.aggregate)
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
