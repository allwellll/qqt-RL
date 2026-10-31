from scripts.bomb_threat_slot04_guard2140 import evaluate_points, transition_scope


def point(u,w=0,z=12):
    return dict(update=u,P0=dict(win_minus_loss=w,zero_bomb_games=z),P1=dict(zero_bomb_games=0))


def test_three_consecutive_from880_inclusive_no_missing_bridge():
    assert evaluate_points([point(800),point(840),point(880)])['trigger'] is False
    assert evaluate_points([point(880),point(920),point(960)])['trigger'] is True
    assert evaluate_points([point(880),point(960),point(1000)])['trigger'] is False
    assert evaluate_points([point(880),point(920,w=1),point(960)])['trigger'] is False
    assert evaluate_points([point(880),point(920,z=11),point(960)])['trigger'] is False


def test_only_slot04_a3s_complete_chunk_can_switch():
    assert transition_scope(4,'A3S',960,True)
    assert not transition_scope(7,'A3S',960,True)
    assert not transition_scope(4,'A3S',920,False)
    assert not transition_scope(4,'A3',960,True)


def test_fullstate_seat_drain_never_relabels_live_episode():
    import jax.numpy as j
    import numpy as np
    from scripts.bomb_threat_a3s_to_p0_collector import reset_completed
    states=dict(carry=j.array([1,0]),position=j.array([[3,4],[7,8]]))
    aux=dict(seat=j.array([0,1]),pending=j.array([.2,.3]),rng=j.array([19,23]))
    fresh=dict(carry=j.array([-1,-1]),position=j.zeros((2,2),j.int32))
    fa=dict(seat=j.zeros(2,j.int32),pending=j.zeros(2),rng=j.array([29,31]))
    a,b=reset_completed(states,aux,fresh,fa,j.array([False,False]))
    for k in states:np.testing.assert_array_equal(a[k],states[k])
    for k in aux:np.testing.assert_array_equal(b[k],aux[k])
    a,b=reset_completed(states,aux,fresh,fa,j.array([False,True]))
    np.testing.assert_array_equal(b['seat'],[0,0]);np.testing.assert_allclose(b['pending'],[.2,0])
    np.testing.assert_array_equal(a['carry'],[1,-1]);np.testing.assert_array_equal(b['rng'],[19,31])


def test_other_slot_never_refreshes_or_forks(monkeypatch):
    from scripts import bomb_threat_slot04_guard2140 as G
    monkeypatch.setattr(G,'refresh',lambda: (_ for _ in ()).throw(AssertionError('must not run for slot07')))
    assert G.apply_switch(7,dict(iter2_arm='A3S'),{},99) is None
    assert G.apply_switch(4,dict(iter2_arm='A3'),{},99) is None
