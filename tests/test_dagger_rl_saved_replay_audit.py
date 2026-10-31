import importlib
import numpy as np


def test_saved_audit_uses_the_actual_source_treatment_and_retained_ids():
    audit = importlib.import_module('scripts.dagger_rl_saved_replay_audit')
    from scripts.dagger_rl_source_replay import aggregate_with_sources
    n = 100
    mask = np.zeros((n,5,3),bool); mask[:,0,0] = True
    mask[0,:,0] = False; mask[0,0,2] = True
    item = np.arange(n) < 50
    sources = np.zeros(n,np.int32); sources[0] = -1
    rows = (np.arange(n)[:,None],np.ones((n,2)),np.ones((n,5),bool),
            np.ones((n,3),bool),mask,item)
    history,_,_ = aggregate_with_sources(rows,sources,None,seed=2,update=1,
        strategy='reservoir',balance_items=False)
    empty = tuple(x[:0] for x in rows)
    for balanced in [False, True]:
        _, replay, stats = aggregate_with_sources(empty,sources[:0],history,
            seed=2,update=3,strategy='reservoir',balance_items=balanced)
        indices,actual = audit.replay_indices(history,seed=2,update=3,balance_items=balanced)
        np.testing.assert_array_equal(indices,replay[0][:,0])
        assert actual['replay_unique_rows'] == stats['replay_unique_rows']
        count = int((indices==0).sum())
        if balanced:assert count == 2048
        else:assert count < 200
