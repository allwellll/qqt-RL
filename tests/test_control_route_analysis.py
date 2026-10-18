"""Initialized carriers must not inflate a window-to-target funnel."""
from scripts.analyze_control_route_curve import route_rates


def test_targets_without_native_windows_do_not_enter_conversion_numerator():
    result=route_rates(dict(games=3,episodes=[
        dict(window_events=0,target_events=1),
        dict(window_events=1,target_events=1),
        dict(window_events=1,target_events=0)]))
    assert result['window_to_target']==dict(numerator=1,denominator=2,rate=.5)


def test_initialized_carrier_only_has_no_native_window_conversion_rate():
    result=route_rates(dict(games=1,episodes=[dict(window_events=0,target_events=1)]))
    assert result['window_to_target']==dict(numerator=0,denominator=0,rate=None)


def test_phase_coverage_keeps_bounded_selection_distinct_from_geometric_coverage():
    from scripts.analyze_control_route_curve import phase_coverage
    roles={'0':dict(eligible=110,query=3,candidate_a=10,candidate_b=100,
        candidate_carry_return=0,candidate_cage_delivery=0,path_a=2,path_b=1,
        carry_return=0,cage_delivery=0)}
    audit=dict(eligible_active=110,safety_accepted_drop=5,safety_accepted_stock=2,
        safety_accepted_return=0,safety_accepted_delivery=0,safety_accepted=7,first_deviation_query=3)
    result=phase_coverage(roles,audit)
    assert result['stock']['active']==100 and result['stock']['accepted']==2
    assert result['stock']['accepted_per_active']['rate']==.02
    assert result['stock']['selected'] is None and result['stock']['geometric'] is None
    assert result['return']['accepted_per_active']['rate'] is None


def test_unlogged_accepted_phase_coverage_remains_unknown():
    from scripts.analyze_control_route_curve import phase_coverage
    roles={'0':dict(eligible=10,query=1,candidate_a=0,candidate_b=10,
        candidate_carry_return=0,candidate_cage_delivery=0,path_a=0,path_b=1,
        carry_return=0,cage_delivery=0)}
    result=phase_coverage(roles,{})
    assert result['stock']['accepted'] is None
    assert result['stock']['accepted_per_active'] is None


def test_route_delta_uses_pooled_counts_not_summed_episode_rates():
    from scripts.analyze_control_route_curve import normalized_route_delta
    previous=route_rates(dict(games=2,oracle_matches=3,oracle_opportunities=4,
        progress_ticks=1,target_ticks=4,oracle_agreement=17.))
    current=route_rates(dict(games=2,oracle_matches=2,oracle_opportunities=4,
        progress_ticks=2,target_ticks=4,oracle_agreement=12.))
    delta=normalized_route_delta(current,previous)
    assert delta['sustained_oracle']==dict(numerator=-1,denominator=0,rate=-.25)
    assert delta['progress_per_target_tick']['rate']==.25


def test_route_delta_does_not_replace_missing_opportunities_with_zero():
    from scripts.analyze_control_route_curve import normalized_route_delta
    current=route_rates(dict(games=2,oracle_matches=1,oracle_opportunities=2))
    previous=route_rates(dict(games=2))
    assert normalized_route_delta(current,previous)['sustained_oracle']==dict(
        numerator=1,denominator=2,rate=None)
