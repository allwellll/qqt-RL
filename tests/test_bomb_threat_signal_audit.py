import pytest
from scripts.bomb_threat_signal_audit import summarize_window


def test_signal_window_keeps_counters_and_denominator_semantics_separate():
    row=dict(new_update=40,finite=True,exposure=dict(total_ticks=100,
        learner_bomb_requests=10,learner_bombs_placed=8,learner_bomb_requests_rejected=2,
        threat_events=2,topology_events=1,chain_candidates=4,chain_settled=3,
        chain_reward=.1,total_reward=20.,absolute_total_reward=30.))
    result=summarize_window([row])
    assert result['request_accounting_exact']
    assert result['actual_place_per_request']==.8
    assert result['first_safe_threat_events_per_actual_place']==.25
    assert result['payment_over_net_reward']==.005
    assert result['positive_payment_transitions']==3
    assert result['terminal_kill_delivery_reward'] is None
    assert 'event_id_to_payment_or_terminal_link' in result['missing_metrics']


def test_signal_window_rejects_inconsistent_requests_or_nonfinite_updates():
    with pytest.raises(ValueError,match='request/place/reject'):
        summarize_window([dict(new_update=1,finite=True,exposure=dict(
            learner_bomb_requests=3,learner_bombs_placed=1,learner_bomb_requests_rejected=1))])
    with pytest.raises(ValueError,match='nonfinite'):
        summarize_window([dict(new_update=1,finite=False,exposure={})])
