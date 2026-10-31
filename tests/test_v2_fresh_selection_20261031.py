from scripts.v2_fresh_selection_20261031 import select


def result(p1=(10,10,10),p0=10,selfd=0):
    return {t:{'P1':dict(win_minus_loss=v,self_deaths=selfd),
               'P0':dict(win_minus_loss=p0,self_deaths=0)}
            for t,v in zip(('hunter_hard','hunter_normal','hunter_easy'),p1)}


def test_three_tiers_and_fixed_guards_exclude_large_unsafe_peak():
    rows={'baseline':result(),'ok':result((11,11,11)),
          'one_bad_tier':result((4,100,100)),'unsafe':result((100,100,100),selfd=1),
          'p0_bad':result((100,100,100),p0=6)}
    out=select(rows,{'candidate_order':['one_bad_tier','unsafe','p0_bad','ok']})
    assert out['winner']=='ok'
    assert not out['comparison']['one_bad_tier']['guards']['each_P1']
    assert not out['comparison']['unsafe']['guards']['self_deaths']
    assert not out['comparison']['p0_bad']['guards']['P0']


def test_tie_keeps_baseline_and_coop_does_not_change_main_selection():
    rows={'baseline':result(),'candidate':result()}
    rows['candidate']['coop_hard']={'P1':dict(win_minus_loss=128,self_deaths=0)}
    out=select(rows,{'candidate_order':['candidate']})
    assert out['winner']=='baseline' and not out['eligible_for_conditional_publish']
