import numpy as np
from scripts.dagger_rl_phase_update_metrics import summarize, phase_groups


def test_phase_precedence_and_padding_partition():
    o=np.zeros((6,24,13,15));g=np.zeros((6,24));g[:,9:11]=1
    g[1,10]=0;g[1,20]=1
    g[2,13:15]=1;g[2,10]=1
    o[3,17,2,2]=1
    g[4,9]=2
    groups=phase_groups(o,g,np.array([1,1,1,1,1,0],bool))
    assert [groups[k].sum() for k in ['combat','stock_window','drop','carry','inactive','other']]==[1,1,1,1,2,0]
    assert np.stack(list(groups.values())).sum(0).tolist()==[1]*6


def test_alignment_uses_actual_advantage_and_denominators():
    result=summarize(np.array([2.,-1.,0.]),np.array([.5,-.5,4.]),np.array([1,1,0],bool))
    assert result['rows']==2
    assert result['advantage_logp_delta_product_mean']==.75
    assert result['nonzero_advantage_rows']==2
    assert result['positive_alignment_fraction']==1
    assert result['positive_advantage_fraction']==.5


def test_empty_phase_reports_missing_instead_of_zero_success():
    result=summarize(np.ones(3),np.ones(3),np.zeros(3,bool))
    assert result['rows']==0
    assert result['advantage_mean'] is None
    assert result['positive_alignment_fraction'] is None
