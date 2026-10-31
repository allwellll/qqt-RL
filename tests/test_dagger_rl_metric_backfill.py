import copy

import pytest

from scripts.dagger_rl_metric_backfill import compare_natural


def fixture():
    return {'results': {'natural_full2400': {'episodes': [
        {'seed': 1, 'delivery': 1, 'item_metrics': {'carry_banana': {
            'owned_self_contacts': 0, 'other_self_contacts': 1,
            'legal_releases': 1, 'release_probability_sum': .3}}}]}}}


def test_only_declared_passive_attribution_can_change():
    a = fixture(); b = copy.deepcopy(a)
    row = b['results']['natural_full2400']['episodes'][0]['item_metrics']['carry_banana']
    row.update(owned_self_contacts=1, other_self_contacts=0)
    assert len(compare_natural(a, b)) == 2
    row['legal_releases'] = 2
    with pytest.raises(AssertionError): compare_natural(a, b)


@pytest.mark.parametrize('key,value', [('seed', 2), ('delivery', 0)])
def test_episode_identity_and_outcomes_are_exact(key, value):
    a = fixture(); b = copy.deepcopy(a)
    b['results']['natural_full2400']['episodes'][0][key] = value
    with pytest.raises(AssertionError): compare_natural(a, b)


def test_same_named_field_outside_item_rows_is_not_exempt():
    a = fixture(); b = copy.deepcopy(a)
    a['owned_self_contacts'] = 0; b['owned_self_contacts'] = 1
    with pytest.raises(AssertionError): compare_natural(a, b)


def test_passive_integer_zero_can_be_replaced_by_real_net_distance():
    a = fixture(); b = copy.deepcopy(a)
    a['results']['natural_full2400']['episodes'][0]['item_metrics']['carry_banana']['owned_net_home_distance'] = 0
    b['results']['natural_full2400']['episodes'][0]['item_metrics']['carry_banana']['owned_net_home_distance'] = 2.5
    assert len(compare_natural(a, b)) == 1
