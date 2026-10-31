import numpy as np
from scripts.campaign_gpu_analysis import compare_witness


def test_actual_history_metadata_and_numeric_witness():
    a={'history':{'strategy':'reservoir','rows':np.array([1,2])},'loss':np.array([.25])}
    assert compare_witness(a,a)['passed']
    b={**a,'history':{**a['history'],'strategy':'fifo'}}
    assert not compare_witness(a,b)['passed']
    assert not compare_witness(a,{**a,'loss':np.array([np.nan])})['passed']


def test_discrete_parity_is_not_relaxed_by_float_tolerance():
    assert not compare_witness({'actions':np.array([1])},{'actions':np.array([2])})['passed']
