"""Actual bounded replay payload provenance, without touching learner data."""
import hashlib
import numpy as np
from scripts.replay_exposure import summarize


def payload():
    return dict(obs=np.arange(8*2*2,dtype=np.float32).reshape(2,4,1,2,2),
        gv=np.arange(16,dtype=np.float32).reshape(2,4,2),
        legal=np.ones((2,4,5),bool),ability_legal=np.ones((2,4,3),bool),
        joint_target=np.ones((2,4,5,3),bool),valid=np.ones((2,4),bool),
        kind=np.full((2,4),3,np.int32),preparation=np.zeros((2,4),bool),
        opening=np.zeros((2,4),bool),round=np.array([[5,5,4,4],[5,5,4,4]],np.int32))


def test_actual_rows_keep_duplicates_half_slots_and_hashes_without_mutation():
    data=payload();data['valid'][1,3]=False
    data['obs'][0,2]=data['obs'][0,0];data['gv'][0,2]=data['gv'][0,0]
    before={k:v.copy() for k,v in data.items()};result=summarize(data,5,'a'*64)
    assert result['generation']==5 and result['source_actor_digest']=='a'*64
    assert result['optimizer_steps']==0 and result['controls_training'] is False
    assert result['counts']['current']['returning']==4
    assert result['counts']['history']['returning']==3
    assert len(result['rows'])==7
    rows=result['rows'];assert rows[0]['half']=='current' and rows[2]['half']=='history'
    assert rows[0]['observation_sha256']==rows[2]['observation_sha256']
    assert rows[0]['observation_sha256']==hashlib.sha256(data['obs'][0,0].tobytes()+data['gv'][0,0].tobytes()).hexdigest()
    for name,value in data.items():np.testing.assert_array_equal(value,before[name])


def test_current_history_are_actual_slots_even_when_history_falls_back_to_current():
    data=payload();data['round'].fill(1);data['kind'][0,0]=2;data['preparation'][0,1]=True
    result=summarize(data,1,'b'*64)
    assert result['counts']['current']['returning']==2
    assert result['counts']['history']['returning']==4
    assert [r['half'] for r in result['rows']].count('history')==4
    assert all(r['query_generation']==1 for r in result['rows'])


def test_label_hash_distinguishes_legal_joint_sets_at_same_observation():
    data=payload();data['obs'][0,1]=data['obs'][0,0];data['gv'][0,1]=data['gv'][0,0]
    data['joint_target'][0,1,3,:]=False
    rows=summarize(data,5,'c'*64)['rows']
    assert rows[0]['observation_sha256']==rows[1]['observation_sha256']
    assert rows[0]['label_sha256']!=rows[1]['label_sha256']
