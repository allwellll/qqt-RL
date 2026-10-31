from scripts.bomb_threat_decision2040 import evaluate_points,boundary_recipe


def test_two_future_points_previous8_strict_and_no_historical_trigger():
    rows=[dict(update=i*40,train_unix=i,checkpoint=f'cp{i}',P0=dict(bombs_per_game=20.,causal_kills=16,win_minus_loss=12)) for i in range(1,11)]
    for r in rows[-2:]:r['P0']['bombs_per_game']=49
    v=evaluate_points(rows,8)
    assert v['status']=='REDUCE_THREAT_AT_NEXT_FULL_CHUNK' and len(v['trigger_pairs'])==1
    assert evaluate_points(rows,10)['status']=='RED_OBSERVE'
    rows[-1]['P0']['causal_kills']=17
    assert not evaluate_points(rows,8)['trigger_pairs']
    assert evaluate_points(rows[:7],0)['points'][-1]['qualifies'] is None


def test_boundary_preserves_recipe_without_trigger(tmp_path,monkeypatch):
    from scripts import bomb_threat_decision2040 as D
    monkeypatch.setattr(D,'OUT',tmp_path)
    r=dict(iter2_arm='A3',lr=1.,entropy_coefficient=.005)
    assert D.boundary_recipe(1,r,{})==r


def test_threat_only_reduction_preserves_other_event_payments():
    import numpy as np
    import jax.numpy as j
    from scripts.bomb_threat_threat25_train import threat_settlement
    from scripts.bomb_threat_iter2_recipe import settlement
    from scripts.bomb_threat_reward import state
    e=dict(enemy_cell=j.array([1]),old_cell=j.array([1]),active=j.array([True]),bad=j.array([False]),terminal=j.array([False]),placed=j.array([True]),safe=j.array([True]),threat=j.array([True]),chain=j.array([True]),old_id=j.array([1]),owned_removed=j.ones((1,13,15),bool),improved=j.array([True]),settled=j.array([True]))
    _,a,_=settlement(5.,.6)(e,state(1));_,b,_=threat_settlement(5.,.6)(e,state(1))
    np.testing.assert_allclose(a-b,[.05],atol=1e-7)
    e['threat']=j.array([False])
    _,a,_=settlement(5.,.6)(e,state(1));_,b,_=threat_settlement(5.,.6)(e,state(1))
    np.testing.assert_array_equal(a,b)
