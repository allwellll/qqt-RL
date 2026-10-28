"""Candidate eviction changes must preserve replay math and restored history."""
import importlib
import pickle
import numpy as np


def rows(ids):
    ids=np.asarray(ids,dtype=np.int64);n=len(ids)
    item=(ids%3==0);label=np.zeros((n,5,3),bool)
    label[np.arange(n),ids%5,np.where(item,2,0)]=True
    return ids[:,None],ids[:,None]*2,np.ones((n,5),bool),np.ones((n,3),bool),label,item


def test_fifo_control_retains_existing_sample_distribution_and_exact_rng():
    aggregate=importlib.import_module('scripts.dagger_rl_memory').aggregate_rows
    history=None;reference=None
    for u in range(1,9):
        new=rows(range((u-1)*7,u*7))
        history,replay,stats=aggregate(new,history,seed=41,update=u,strategy='fifo',capacity=17,replay_size=64)
        reference=tuple(np.concatenate([a,b])[-17:] for a,b in zip(reference,new)) if reference else new
        for a,b in zip(history['rows'],reference):np.testing.assert_array_equal(a,b)
        rng=np.random.default_rng(41*1000+u);item=reference[-1]
        ii,rr=np.flatnonzero(item),np.flatnonzero(~item)
        indices=np.concatenate([rng.choice(ii,32),rng.choice(rr,32)]);rng.shuffle(indices)
        for a,b in zip(replay,reference):np.testing.assert_array_equal(a,b[indices])
        assert stats['new_rows']==7 and stats['history_rows']==min(u*7,17)


def test_reservoir_and_fifo_match_before_capacity_and_resume_with_metadata():
    aggregate=importlib.import_module('scripts.dagger_rl_memory').aggregate_rows
    histories={k:None for k in ['fifo','reservoir']}
    for u in range(1,7):
        new=rows(range((u-1)*5,u*5));outputs={}
        for kind in histories:
            restored=pickle.loads(pickle.dumps(histories[kind]))
            h,r,s=aggregate(new,histories[kind],seed=52,update=u,strategy=kind,capacity=17,replay_size=64)
            hh,rr,ss=aggregate(new,restored,seed=52,update=u,strategy=kind,capacity=17,replay_size=64)
            assert s==ss and h['seen']==u*5
            for a,b in zip(r,rr):np.testing.assert_array_equal(a,b)
            for a,b in zip(h['rows'],hh['rows']):np.testing.assert_array_equal(a,b)
            assert len(h['arrival_id'])==len(h['rows'][0])
            np.testing.assert_array_equal(h['rows'][0][:,0],h['arrival_id'])
            histories[kind]=h;outputs[kind]=r
        if u<=3:
            for a,b in zip(outputs['fifo'],outputs['reservoir']):np.testing.assert_array_equal(a,b)
    assert not np.array_equal(histories['fifo']['arrival_id'],histories['reservoir']['arrival_id'])


def test_empty_opportunity_uses_existing_on_policy_fallback_without_fabricated_label():
    aggregate=importlib.import_module('scripts.dagger_rl_memory').aggregate_rows
    empty=rows([]);fallback=list(rows(range(8)));fallback[-2]=np.zeros_like(fallback[-2]);fallback[-1]=np.zeros_like(fallback[-1])
    h,replay,stats=aggregate(empty,None,seed=1,update=1,strategy='reservoir',capacity=17,replay_size=64,fallback=tuple(fallback))
    assert h['seen']==0 and stats['history_rows']==0 and not replay[-2].any()
