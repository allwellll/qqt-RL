from scripts.r4_carrier_attribution import recompute


def test_delayed_carrier_kill_and_recovery_do_not_use_instantaneous_flag():
    e=dict(r4_carrier=dict(events=[dict(type='own_stock_taken',learner=0,tick=1),
        dict(type='carrier_death',learner=0,tick=8)],carrier_causal_kills=[0,0]),
        kills=[dict(attacker=0,victim_carried=0,tick=8)],
        antifarm=dict(events=[dict(type='delivery',origin=0,player=0,tick=12)]))
    v=recompute(e,0)
    assert v['carrier_causal_kills_funnel']==1 and v['carrier_causal_kills_sim_flag']==0
    assert v['completed_antifarm_cycles_funnel']==1
    e['antifarm']['events'].insert(0,dict(type='delivery',origin=0,player=1,tick=9))
    assert recompute(e,0)['completed_antifarm_cycles_funnel']==0
