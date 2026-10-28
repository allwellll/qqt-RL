"""Sampling must preserve causal labels and exact resume without extra updates."""
import importlib
import numpy as np
import pytest


def append(*args, **kwargs):
    return importlib.import_module('scripts.dagger_rl_reservoir').append


def batch(ids):
    ids=np.asarray(ids,dtype=np.int64)
    return (ids, np.stack([ids, -ids],axis=-1), (ids%2==0))


def test_pre_capacity_is_identical_to_fifo_and_does_not_mutate_inputs():
    fn=append();old=batch(range(4));new=batch(range(4,7));copies=tuple(a.copy() for a in old+new)
    got,seen=fn(old,new,seen=4,capacity=8,seed=7)
    assert seen==7
    for value,expected in zip(got,batch(range(7))):np.testing.assert_array_equal(value,expected)
    for value,expected in zip(old+new,copies):np.testing.assert_array_equal(value,expected)


def test_resume_is_exact_with_declared_update_rng_and_keeps_label_rows_aligned():
    fn=append();pool=None;seen=0
    for u in range(1,5):pool,seen=fn(pool,batch(range((u-1)*20,u*20)),seen=seen,capacity=17,seed=u)
    import pickle
    restored=pickle.loads(pickle.dumps((pool,seen)))
    continued,n=fn(pool,batch(range(80,100)),seen=seen,capacity=17,seed=5)
    replayed,m=fn(restored[0],batch(range(80,100)),seen=restored[1],capacity=17,seed=5)
    assert n==m==100
    for a,b in zip(continued,replayed):np.testing.assert_array_equal(a,b)
    ids=continued[0]
    assert len(set(ids.tolist()))==17 and (ids>=0).all() and (ids<100).all()
    np.testing.assert_array_equal(continued[1],batch(ids)[1])
    np.testing.assert_array_equal(continued[2],batch(ids)[2])


def test_uniform_stream_retention_does_not_systematically_discard_early_opportunities():
    fn=append();counts=np.zeros(100,dtype=int)
    for seed in range(500):
        pool,seen=fn(None,batch(range(100)),seen=0,capacity=10,seed=seed)
        counts[pool[0]]+=1
    # Expected count50 per row; deterministic broad tolerance catches FIFO/head-only bias.
    assert (counts>20).all() and (counts<85).all()
    assert counts[:25].sum()>1000 and counts[-25:].sum()>1000


def test_invalid_seen_state_is_rejected_before_sampling():
    fn=append()
    with pytest.raises(ValueError):fn(batch(range(3)),batch([4]),seen=2,capacity=8,seed=1)
    with pytest.raises(ValueError):fn(None,batch([4]),seen=0,capacity=0,seed=1)
