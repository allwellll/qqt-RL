"""Qualification telemetry retains source/phase denominators and public schema."""
import numpy as np
import pytest
from scripts.dagger_rl_goal_guard_collector import summary, factory


def test_removed_item_overridden_route_is_not_counted_as_final_removal():
    flags=np.zeros((3,16));flags[0,0]=1;flags[1,15]=1;flags[2,14]=1
    out=summary(flags,np.array([-1,4,-1]),np.array([[1,1,0],[1,1,1],[0,0,0]]))
    assert out['total_ticks']==3
    assert out['rows']['natural/carry']['removed_goal0_route_rows']==1
    assert out['rows']['natural/carry']['removed_final_teacher_rows_after_item_priority']==0
    assert out['rows']['real_pickup_bank/empty_window']['removed_final_teacher_rows_after_item_priority']==1
    assert sum(x['ticks'] for x in out['rows'].values())==3


def test_alignment_failure_explicit():
    with pytest.raises(ValueError):summary(np.zeros((3,16)),np.zeros(2),np.zeros((3,3)))


def test_materialized_collector_anchor_and_public_schema():
    # Compilation is a technical construction check, without GPU or env steps.
    from scripts import dagger_rl_r10_train as template
    collector=factory(template,1,legacy_teacher=lambda s,a:None,stream='PPO',counters={})
    assert callable(collector)
