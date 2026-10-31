"""A truncated diagnostic must bootstrap at its own next state."""
import numpy as np
from scripts.combat_item_bank import aligned_next_values


def test_cutoff_uses_observed_next_value_not_end_of_larger_chunk():
    values = np.arange(8., dtype=float)[:, None]
    actual = aligned_next_values(values, np.array([99.]), 5)
    np.testing.assert_array_equal(actual[:, 0], [1., 2., 3., 4., 5.])


def test_full_chunk_uses_tail_and_short_completed_trajectory_stays_aligned():
    values = np.arange(3., dtype=float)[:, None]
    actual = aligned_next_values(values, np.array([9.]), 5)
    np.testing.assert_array_equal(actual[:, 0], [1., 2., 9.])
