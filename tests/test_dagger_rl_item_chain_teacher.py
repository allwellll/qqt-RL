"""RED/GREEN tests for certified item-chain teacher composition."""

import jax.numpy as j
import numpy as np

from scripts.dagger_rl_item_chain_teacher import compose_teacher


def _route():
    out = j.zeros((3, 5, 3), dtype=bool)
    # Movement and non-release abilities are ordinary route labels.
    out = out.at[:, 1, 0].set(True)
    out = out.at[:, 2, 1].set(True)
    # Ability 2 is the release action and must require a certified plan.
    return out.at[:, 3, 2].set(True)


def test_uncertified_item_chain_removes_only_release_from_route():
    route = _route()
    item = j.zeros_like(route)
    actual = compose_teacher(route, item)
    expected = route.at[:, :, 2].set(False)
    np.testing.assert_array_equal(actual, expected)


def test_certified_item_plan_has_priority_and_is_not_removed():
    route = _route()
    item = j.zeros_like(route).at[0, 4, 2].set(True)
    actual = compose_teacher(route, item)
    expected = j.zeros_like(route).at[0, 4, 2].set(True)
    expected = expected.at[1:, :, :2].set(route[1:, :, :2])
    np.testing.assert_array_equal(actual, expected)


def test_no_route_labels_are_preserved_when_no_item_plan_exists():
    route = _route().at[:, 3, 2].set(False)
    actual = compose_teacher(route, j.zeros_like(route))
    np.testing.assert_array_equal(actual, route)


def test_r18_wires_the_adapter_into_both_teacher_call_sites():
    import scripts.dagger_rl_r18_train as r18
    from scripts import dagger_rl_release_qualification as qualification

    r18.install_item_chain_teacher()
    assert qualification.combine is compose_teacher
    assert r18.L.C.O.combine is compose_teacher
