"""Retired reset scenes and all-scene events cannot become causal evidence."""
from scripts import je10_diagnostics as diagnostics


def test_only_valid_empty_handed_routes_are_sampling_diagnostic_targets():
    assert set(diagnostics.sampling_diagnostic_targets())=={'vulnerable_steal','defend_and_steal','route_break_capture'}


def test_native_full_events_use_role_filtered_transitions_not_all_scene_totals():
    row=dict(dagger=dict(native_transitions={'own_kill':1000},opponent_course={
        '0':dict(native_transitions=[1,2,3,4,5,6,7,8]),
        '1':dict(native_transitions=[2,3,4,5,6,7,8,9]),
        '2':dict(native_transitions=[0]*8)}))
    result=diagnostics.native_full_event_counts(row)
    assert result['own_kill']==5 and result['capture']==15
    assert result['inventory_or_native_steal']==11
