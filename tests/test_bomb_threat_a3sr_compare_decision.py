from scripts.bomb_threat_a3sr_compare_decision import compare


def model(w):
    p=dict(games=128,win_minus_loss=w,causal_kills=20,deaths=5,self_deaths=0,
           zero_bomb_games=3,draw_fraction=.5,funnel=dict(window_stock_delivery=3))
    return dict(games=256,by_seat={'P1':p})


def values(scores):
    return {tier:{'u800':model(10),'a3sr_u640':model(w),'a3sr_u1280':dict(games=0)}
            for tier,w in zip(('hard','normal','easy'),scores)}


def test_p1_sum_must_improve_and_each_tier_within_minus5():
    r=compare(values([5,15,11]))['a3sr_u640']
    assert r['status']=='REPORT_ELIGIBLE'
    assert r['observable_dominance'] is False  # report-eligible does not imply every metric is better
    assert compare(values([4,20,20]))['a3sr_u640']['status']=='NOT_ELIGIBLE'
    assert compare(values([10,10,10]))['a3sr_u640']['status']=='NOT_ELIGIBLE'
    assert compare(values([10,10,11]))['a3sr_u1280']['status']=='MISSING'
