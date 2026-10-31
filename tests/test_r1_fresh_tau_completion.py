import copy
import pytest
from scripts import r1_fresh_tau_completion as C
from scripts.bomb_threat_eval import summarize


def row(seed=10,seat=0,winner=None,placed=None):
    return dict(seed=seed,seat=seat,winner=winner,placed=placed or [0,0],
                causal_kills=[0,0],cage_kills=[0,0],deaths=[0,0],self_deaths=[0,0],
                requests=[0,0],stored=[[1,0],[0,1]])


def test_pair_matches_seed_and_seat_and_rejects_duplicates():
    a=[row(winner=0),row(seed=11,seat=1,winner=0)]
    b=[row(seed=11,seat=1,winner=1),row(winner=None)]
    result=C.checked_pair(a,b)
    assert [r['score_difference'] for r in result]==[1,-2]
    with pytest.raises(AssertionError):C.checked_pair(a+[copy.deepcopy(a[0])],b)
    with pytest.raises(AssertionError):C.checked_pair(a,[row(seed=12),b[0]])


def test_zero_bomb_denominators_and_seats():
    rows=[row(placed=[0,2]),row(seed=11,placed=[0,0]),row(seed=12,seat=1,placed=[1,2])]
    out=summarize(rows)
    assert out['zero_bomb_games']==1 and out['structural_zero_bomb_games']==1
    assert out['raw_zero_bomb_games']==2 and out['opponent_bomb_opportunity_games']==2
    assert out['by_seat']['P0']['games']==2 and out['by_seat']['P1']['games']==1


def test_throughput_does_not_compare_different_run():
    before={'a':dict(run='old',env_per_second=20),'b':dict(run='same',env_per_second=10)}
    now={'a':dict(run='new',env_per_second=1),'b':dict(run='same',env_per_second=9)}
    assert C.comparison(now,before)=={'b':.9}
