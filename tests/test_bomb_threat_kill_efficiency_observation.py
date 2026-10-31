from scripts.bomb_threat_fresh256_observation import kill_efficiency,low_rate_triplets


def test_actual_placements_pooled_controls_and_missing_are_explicit():
    c=dict(actual_bombs=100,causal_kills=5)
    refs={'0':dict(actual_bombs=100,causal_kills=10),'3':dict(actual_bombs=300,causal_kills=30)}
    v=kill_efficiency(c,refs)
    assert v['kills_per100_actual_bombs']==5 and v['pooled_control_per100']==10
    assert v['below70percent'] is True
    assert kill_efficiency(c,{'0':refs['0']})['below70percent'] is None
    assert kill_efficiency(dict(actual_bombs=0,causal_kills=0),refs)['below70percent'] is None
    c['causal_kills']=7
    assert kill_efficiency(c,refs)['below70percent'] is False


def test_three_known_consecutive_points_only():
    pts=[dict(update=u,below70percent=True) for u in (40,80,120)]
    assert len(low_rate_triplets(pts))==1
    pts[1]['below70percent']=None
    assert not low_rate_triplets(pts)
    pts[1].update(update=60,below70percent=True)
    assert not low_rate_triplets(pts)
