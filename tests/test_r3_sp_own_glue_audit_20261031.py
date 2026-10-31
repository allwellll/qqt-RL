import pytest
from scripts.r3_sp_own_glue_audit_20261031 import summarize


def episode(seed,seat=0,winner=0,contacts=None):
    return dict(seed=seed,seat=seat,winner=winner,deaths=[2,1],self_deaths=[0,0],
                r3={'contacts':contacts or []})


def test_episode_denominator_and_exact_event_filter():
    contact=dict(type=2,owner=0,player=0,carried=1,tick=20)
    excluded=[dict(contact,type=1),dict(contact,owner=1),dict(contact,player=1),dict(contact,carried=-1)]
    groups,rows=summarize([episode(1,contacts=[contact,dict(contact,tick=40)]),
                          episode(2,winner=-1,contacts=excluded),episode(3,winner=1),episode(4,seat=1)])
    assert groups['carrying_own_glue']['games']==1
    assert groups['carrying_own_glue']['contacts']==2
    assert groups['carrying_own_glue']['wins']==1
    assert groups['no_carrying_own_glue']['games']==2
    assert groups['no_carrying_own_glue']['draws']==1
    assert groups['no_carrying_own_glue']['losses']==1
    assert groups['no_carrying_own_glue']['deaths']==4
    assert len(rows)==3


def test_duplicate_episodes_rejected_and_empty_group_not_zero_filled():
    with pytest.raises(AssertionError):summarize([episode(1),episode(1)])
    groups,_=summarize([episode(1)])
    assert groups['carrying_own_glue']['deaths_per_game'] is None
