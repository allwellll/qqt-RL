import pytest

from scripts.bomb_threat_round7_audit import summarize


def row():
    return dict(new_update=40, finite=True, actual_environment_steps_this_update=100,
                exposure=dict(resets=2, bank_resets=1, captures=2, total_reward=50.,
                              native_parts=[.75, 24., 3., 1., -.1]))


COMPONENTS = ['transport_shape', 'transport_capture', 'control_entry_count',
              'control_residual_count', 'route_shaped_reward']


def test_paid_units_are_not_raw_event_or_causal_funnel_counts():
    result = summarize([row()], COMPONENTS)
    assert result['delivery_events'] == 2
    assert result['transport_capture_payment_units'] == 1
    assert result['first_eligible_stock_payment_units'] == 1
    assert result['attributed_native_control_entry_events'] == 3
    assert result['raw_kill_window_count'] is None
    assert result['raw_window_pickup_count'] is None
    assert result['same_window_pickup_delivery_count'] is None
    assert result['environment_ticks'] == 100
    assert result['bank_resets'] == 1


def test_audit_rejects_nonfinite_or_wrong_component_abi():
    data = row()
    data['finite'] = False
    with pytest.raises(ValueError, match='finite'):
        summarize([data], COMPONENTS)
    with pytest.raises(ValueError, match='width'):
        summarize([row()], COMPONENTS + ['unexpected'])
