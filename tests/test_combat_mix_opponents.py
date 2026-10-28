import numpy as np
import pytest

from scripts.combat_mix_opponents import (
    DEFAULT_WEIGHTS, OPPONENTS, combat_scene_mask, fixed_allocation,
    merge_opponent_actions, validate_weights,
)
import jax.numpy as jnp


def test_opponent_allocation_is_deterministic_and_uses_declared_weights():
    first = fixed_allocation(10000, 20261027)
    second = fixed_allocation(10000, 20261027)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == set(range(len(OPPONENTS)))
    observed = np.bincount(first, minlength=len(OPPONENTS)) / len(first)
    assert np.all(np.abs(observed - DEFAULT_WEIGHTS) < 0.02)


@pytest.mark.parametrize("weights", [
    (0.1, 0.1), (0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0.1),
    (float("nan"), 0, 0, 0, 0, 0),
])
def test_invalid_pool_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_combat_scene_fraction_is_exact_reproducible_and_does_not_touch_other_rows():
    left = combat_scene_mask(100, 42, 0.25)
    right = combat_scene_mask(100, 42, 0.25)
    assert np.array_equal(left, right)
    assert left.sum() == 25
    assert combat_scene_mask(100, 42, 0.0).sum() == 0
    assert combat_scene_mask(100, 42, 1.0).all()


def test_mixed_opponent_dispatch_changes_only_opponent_seat_and_selects_declared_pool_row():
    original = jnp.asarray([[[1, 0], [4, 0]], [[2, 1], [4, 0]]])
    replacement = jnp.arange(2 * len(OPPONENTS) * 2).reshape(2, len(OPPONENTS), 2)
    kinds = jnp.asarray([1, 4])
    actual = merge_opponent_actions(original, replacement, kinds)
    assert np.array_equal(np.asarray(actual[:, 0]), np.asarray(original[:, 0]))
    assert np.array_equal(np.asarray(actual[0, 1]), np.asarray(replacement[0, 1]))
    assert np.array_equal(np.asarray(actual[1, 1]), np.asarray(replacement[1, 4]))
