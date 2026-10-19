import hashlib
import numpy as np
from scripts.learner_trace_sampling import observation_state_hash


class OnlySelectedRow:
    def __init__(self,row):self.row=row;self.indices=[]
    def __array__(self,*args,**kwargs):raise AssertionError('full learner batch copied to host')
    def __getitem__(self,index):self.indices.append(index);return self.row


def test_hash_selects_device_tick_environment_before_host_copy():
    obs=np.arange(24*13*15,dtype=np.float32).reshape(24,13,15)
    gv=np.arange(24,dtype=np.float32)
    observations=OnlySelectedRow(obs);globals_=OnlySelectedRow(gv)
    index=(3,27,400)
    actual=observation_state_hash(observations,globals_,index)
    assert actual==hashlib.sha256(obs.tobytes()+gv.tobytes()).hexdigest()
    assert observations.indices==globals_.indices==[index]


def test_verifier_accounting_separates_selection_safety_and_episode_quota():
    from scripts.learner_trace_sampling import verifier_audit_counts
    context=np.zeros((1,3,103),np.float32)
    context[...,92:101]=[[[1,1,1,1,0,0,1,0,0],
                         [1,1,1,0,0,0,1,0,0],
                         [1,1,0,0,1,0,0,0,0]]]
    counts=verifier_audit_counts(context)
    assert counts['eligible_active']==3 and counts['geometrically_reachable']==3
    assert counts['safety_accepted']==2 and counts['first_deviation_query']==1
    assert counts['reachable_rejected']==1 and counts['safety_accepted_stock']==2


def test_executed_sources_remove_weak_and_assist_overrides_from_league():
    from scripts.learner_trace_sampling import executed_opponent_exposure
    context=np.zeros((1,7,1,106),np.float32)
    context[0,:,0,92:95]=[[3,0,1],[3,8,1],[3,2,1],[4,0,1],[5,8,1],[6,2,1],[3,0,0]]
    result=executed_opponent_exposure(context)
    assert result['valid_decision_ticks']==6
    assert result['current']==result['ring']==result['fixed_league']==1
    assert result['weak_courier']==result['weak_guard']==result['carry_assist']==1
    assert result['fraction_of_valid']['current']==1/6
    assert sum(result['counts'].values())==6


def test_verifier_audit_offset_survives_added_host_provenance_columns():
    from scripts.learner_trace_sampling import verifier_audit_counts
    context=np.zeros((1,3,106),np.float32)
    context[...,95:104]=[[[1,1,1,1,0,0,1,0,0],
                         [1,1,1,0,0,0,1,0,0],
                         [1,1,0,0,1,0,0,0,0]]]
    result=verifier_audit_counts(context)
    assert result['eligible_active']==3 and result['first_deviation_query']==1
