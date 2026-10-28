"""Isolated candidate wiring through the real collector, GAE, FIFO and PPO."""
def test_candidate_real_collector_preserves_finite_update_and_restore(monkeypatch):
    from scripts import prestock_recipe
    from scripts.return_query_recipe import patched
    from tests.test_prestock_collector_integration import test_real_collector_shared_potential_joint_labels_reset_and_replay
    original=prestock_recipe.patched_sources
    def candidate(sources):
        result=original(sources)
        name='scripts/je09_stage_h6_ops.py'
        result[name]=patched(result[name])
        return result
    monkeypatch.setattr(prestock_recipe,'patched_sources',candidate)
    test_real_collector_shared_potential_joint_labels_reset_and_replay()
