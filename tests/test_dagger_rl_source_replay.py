"""Source balancing changes only already-certified old item replay draws."""
import importlib
import pickle
import numpy as np
import pytest
from scripts.dagger_rl_memory import aggregate_rows


def rows(n=40):
    ids = np.arange(n, dtype=np.int32)
    item = ids % 2 == 0
    mask = np.zeros((n, 5, 3), bool)
    mask[:, 0, 0] = True
    mask[item, 0, 0] = False
    mask[item, 1, 2] = True
    return (ids[:, None], ids[:, None].astype(np.float32),
            np.ones((n, 5), bool), np.ones((n, 3), bool), mask, item)


def call(*args, **kwargs):
    mod = importlib.import_module('scripts.dagger_rl_source_replay')
    return mod.aggregate_with_sources(*args, **kwargs)


def test_control_replay_is_exact_legacy_and_sources_follow_arrival_ids():
    new = rows()
    source = np.where(np.arange(40) % 7 == 0, -1, 3).astype(np.int32)
    expected, replay, _ = aggregate_rows(
        new, None, seed=11, update=1, strategy='reservoir', capacity=30, replay_size=100)
    got, rb, stats = call(new, source, None, seed=11, update=1,
                          strategy='reservoir', balance_items=False,
                          capacity=30, replay_size=100)
    for k in ['rows', 'arrival_id', 'arrival_update']:
        for a, b in zip(got[k] if k == 'rows' else [got[k]],
                        expected[k] if k == 'rows' else [expected[k]]):
            np.testing.assert_array_equal(a, b)
    for a, b in zip(rb, replay):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(got['source'], source[got['arrival_id']])
    assert not stats['source_balance_active']


def test_treatment_balances_item_sources_but_keeps_route_draws_and_slots_exact():
    new = rows()
    source = np.full(40, 2, np.int32)
    source[:2] = -1
    _, control, _ = call(new, source, None, seed=12, update=1,
                         strategy='fifo', balance_items=False, replay_size=100)
    history, treated, stats = call(new, source, None, seed=12, update=1,
                                    strategy='fifo', balance_items=True, replay_size=100)
    isitem = control[-1]
    np.testing.assert_array_equal(treated[-1], isitem)
    for a, b in zip(control, treated):
        np.testing.assert_array_equal(a[~isitem], b[~isitem])
    assert stats['source_balance_active']
    assert stats['replay_item_natural_rows'] == 25
    assert stats['replay_item_bank_rows'] == 25
    assert np.isin(treated[0], new[0]).all()
    np.testing.assert_array_equal(history['source'], source)


def test_resume_is_exact_with_eviction_and_does_not_change_incoming_arrays():
    source = np.where(np.arange(40) % 3 == 0, -1, 1).astype(np.int32)
    new = rows()
    copies = tuple(a.copy() for a in new)
    history, _, _ = call(new, source, None, seed=13, update=1,
                          strategy='reservoir', balance_items=True, capacity=17)
    restored = pickle.loads(pickle.dumps(history))
    a, rb, stats = call(new, source, history, seed=13, update=2,
                        strategy='reservoir', balance_items=True, capacity=17)
    b, qb, other = call(new, source, restored, seed=13, update=2,
                        strategy='reservoir', balance_items=True, capacity=17)
    assert stats == other
    for x, y in zip(rb, qb):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a['source'], b['source'])
    ids = a['arrival_id'] % 40
    np.testing.assert_array_equal(a['source'], source[ids])
    for x, y in zip(new, copies):
        np.testing.assert_array_equal(x, y)


def test_single_source_falls_back_exactly_instead_of_inventing_opportunities():
    new = rows()
    src = np.full(40, -1, np.int32)
    _, control, _ = call(new, src, None, seed=14, update=1,
                         strategy='fifo', balance_items=False)
    _, treatment, stats = call(new, src, None, seed=14, update=1,
                               strategy='fifo', balance_items=True)
    assert not stats['source_balance_active']
    for a, b in zip(control, treatment):
        np.testing.assert_array_equal(a, b)


def test_missing_provenance_or_unaccepted_rows_are_rejected():
    new = rows()
    with pytest.raises(ValueError):
        call(new, np.zeros(39, np.int32), None, seed=1, update=1,
             strategy='fifo', balance_items=True)
    bad = list(new)
    bad[-2] = np.zeros_like(new[-2])
    with pytest.raises(ValueError):
        call(tuple(bad), np.zeros(40, np.int32), None, seed=1, update=1,
             strategy='fifo', balance_items=True)
