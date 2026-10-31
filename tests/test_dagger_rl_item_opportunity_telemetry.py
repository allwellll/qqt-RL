import numpy as np
import pytest

from scripts.dagger_rl_item_opportunity_telemetry import summarize


def test_sources_phases_eligibility_and_selected_glue_are_not_conflated():
    flags = np.zeros((2, 2, 16))
    oracle = np.zeros((2, 2, 6))
    origins = np.array([[-1, 4], [-1, -1]])
    # Natural carrier with banana, positive release forecast, no selected release.
    flags[0, 0, [0, 1, 2]] = 1
    oracle[0, 0, [0, 2]] = 1
    # Bank carrier selected glue: ability2 is legal but not a banana action.
    flags[0, 1, [0, 2, 3]] = 1
    # Empty/free natural state with a nearby owned banana must not become a
    # native-control or genuine-carry opportunity merely from proximity.
    flags[1, 0, 14] = 1
    oracle[1, 0, 1] = 1
    # Empty native window, no inventory or plan.
    flags[1, 1, 15] = 1
    before = [x.copy() for x in [flags, oracle, origins]]
    x = summarize(flags, oracle, origins)
    assert x['total_ticks'] == 4 and x['phase_unclassified_ticks'] == 0
    natural = x['rows']['natural/carry']
    assert natural['static_teacher_precondition_ticks'] == 1
    assert natural['certified_release_label_ticks'] == 1
    assert natural['selected_banana_release_ticks'] == 0
    bank = x['rows']['real_pickup_bank/carry']
    assert bank['selected_any_item_release_ticks'] == 1
    assert bank['selected_banana_release_ticks'] == 0
    combat = x['rows']['natural/empty_combat']
    assert combat['nearby_owned_banana_ticks'] == 1
    assert combat['static_teacher_precondition_ticks'] == 0
    assert x['rows']['natural/empty_window']['ticks'] == 1
    for old, current in zip(before, [flags, oracle, origins]):
        np.testing.assert_array_equal(old, current)


def test_unknown_shapes_and_nonfinite_data_fail_instead_of_silent_denominators():
    with pytest.raises(ValueError):
        summarize(np.zeros((2, 15)), np.zeros((2, 6)), np.zeros(2))
    with pytest.raises(ValueError):
        summarize(np.zeros((2, 16)), np.zeros((2, 5)), np.zeros(2))
    f = np.zeros((2, 16)); f[0, 1] = np.nan
    with pytest.raises(ValueError):
        summarize(f, np.zeros((2, 6)), np.zeros(2))


def test_nonfinite_source_cannot_silently_disappear_from_both_source_denominators():
    with pytest.raises(ValueError):
        summarize(np.zeros((2, 16)), np.zeros((2, 6)), np.array([-1., np.nan]))


def test_overlapping_phases_are_rejected_instead_of_double_counting_ticks():
    flags = np.zeros((2, 16))
    flags[0, [0, 14]] = 1
    with pytest.raises(ValueError):
        summarize(flags, np.zeros((2, 6)), np.array([-1, -1]))
