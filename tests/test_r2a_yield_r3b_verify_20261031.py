from scripts import r2a_yield_r3b_verify_20261031 as task


def test_draw_kill_difference_excludes_decisive_games_and_other_seat():
    def episode(seat,winner,kills,contacts=()):
        return dict(seat=seat,winner=winner,kills=[dict(attacker=k) for k in kills],r3=dict(contacts=list(contacts)))
    rows=[episode(1,-1,[1,1,0]),episode(1,1,[1]*8),episode(0,-1,[0]*5)]
    result=task.extra_metrics(rows,1)
    assert result['draw_games']==1
    assert result['draw_causal_kill_difference']==1


def test_carrying_glue_requires_own_placement_self_contact_and_package():
    contacts=[dict(type=2,owner=1,player=1,carried=0),
              dict(type=2,owner=1,player=1,carried=-1),
              dict(type=2,owner=0,player=1,carried=0),
              dict(type=1,owner=1,player=1,carried=0)]
    result=task.extra_metrics([dict(seat=1,winner=None,kills=[],r3=dict(contacts=contacts))],1)
    assert result['carrying_own_glue_contacts']==1
