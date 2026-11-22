import pytest
from scripts.export_dcu_kill_replays import statistics


def game(seat,winner,metrics,full=True):
    return dict(spec=dict(version='current',opponent='DUEL_latest',seat=seat),
                summary=dict(full_game=full,ticks=1800,winner=winner,metrics=metrics))


def test_statistics_follow_model_seat_and_distinguish_draws():
    rows=statistics([
        game(0,0,[[5,1,0,0,0,1],[1,5,1,0,0,0]]),
        game(1,0,[[4,2,0,0,0,1],[2,4,2,0,0,0]]),
        game(1,None,[[3,3,1,0,0,.5],[3,3,1,0,0,.5]]),
    ])
    assert len(rows)==1
    row=rows[0]
    assert (row['wins'],row['draws'],row['losses'])==(1,1,1)
    assert (row['kills'],row['deaths'],row['self_deaths'])==(10,8,3)
    assert row['score_rate']==.5


def test_short_games_are_not_counted_as_complete_results():
    with pytest.raises(ValueError,match='Incomplete'):
        statistics([game(0,None,[[0]*6,[0]*6],full=False)])
