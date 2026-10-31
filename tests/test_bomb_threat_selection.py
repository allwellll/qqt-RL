from scripts.bomb_threat_selection import hard_gate_failures


def sample(**kwargs):
    return {'greedy':dict(games=64,zero_bomb_games=3,bombs_per_game=15,
        cage_kill_fraction=.1,draw_fraction=.5,self_deaths=1,**kwargs)}


def test_promotion_boundaries_and_unknown_reward_are_explicit():
    n=sample();p=sample();n["greedy"]["draw_fraction"]=.49
    assert hard_gate_failures(n,p,{'reward_share_gate_pass':True},True)==[]
    assert hard_gate_failures(n,p,{},True)==['NATIVE_REWARD_SHARE_MISSING_OR_FAILED']
    bad=sample();bad['greedy'].update(zero_bomb_games=4,bombs_per_game=14.99,cage_kill_fraction=.201)
    errors=hard_gate_failures(bad,p,{'reward_share_gate_pass':True},True)
    assert {'ARGMAX_ZERO_BOMBS_GT_3','ARGMAX_BOMBS_LT_15','CAGE_KILL_REGRESSION'}<=set(errors)


def test_baselines_and_matching_parent_required_without_invented_training_gate():
    errors=hard_gate_failures(sample(),None,{'reward_share_gate_pass':True},False)
    assert 'THREE_BASELINES_INCOMPLETE' in errors and 'MATCHED_PARENT_REFERENCE_MISSING' in errors


def parent_metrics():
    return {'greedy': dict(games=64, zero_bomb_games=14, bombs_per_game=18.203125,
        cage_kill_fraction=1/30, draw_fraction=56/64, self_deaths=0)}


def test_two_gate_levels_do_not_conflate_pbt_with_best():
    from scripts.bomb_threat_selection import relative_gate_failures
    p=parent_metrics()
    n={'greedy':dict(p['greedy'], bombs_per_game=p['greedy']['bombs_per_game']*.8)}
    reward={'reward_share_gate_pass':False, 'threat_reward_fraction_total':.3}
    assert relative_gate_failures(n,p,reward,True)==[]
    assert hard_gate_failures(n,p,reward,True)
    n['greedy']['bombs_per_game']-=.001
    assert 'ARGMAX_BOMBS_LT_PARENT_80_PERCENT' in relative_gate_failures(n,p,reward,True)


def test_relative_gate_boundaries_and_missing_data_fail_closed():
    from scripts.bomb_threat_selection import relative_gate_failures
    p=parent_metrics();n={'greedy':dict(p['greedy'])}
    n['greedy'].update(cage_kill_fraction=1/30+.1)
    assert relative_gate_failures(n,p,{'threat_reward_fraction_total':.3},True)==[]
    n['greedy'].update(zero_bomb_games=15,draw_fraction=57/64,self_deaths=1,cage_kill_fraction=.134)
    errors=relative_gate_failures(n,p,{'threat_reward_fraction_total':.3001},True)
    assert {'ARGMAX_ZERO_BOMBS_GT_PARENT','DRAW_REGRESSION','SELF_DEATH_REGRESSION',
            'CAGE_KILL_REGRESSION','RELATIVE_NATIVE_REWARD_SHARE_MISSING_OR_FAILED'}<=set(errors)
    assert 'RELATIVE_NATIVE_REWARD_SHARE_MISSING_OR_FAILED' in relative_gate_failures(p,p,{},True)
    assert 'PARENT_ARGMAX_DENOMINATOR_NOT_64' in relative_gate_failures(p,{'greedy':{'games':32}}, {},True)


def test_absolute_draw_must_be_strictly_lower_under_amendment():
    p=sample();n=sample()
    assert 'DRAW_NOT_LOWER_THAN_PARENT' in hard_gate_failures(n,p,{'reward_share_gate_pass':True},True)


def test_two_consecutive_points_require_bombs_up_and_both_kills_and_wins_not_up():
    from scripts.bomb_threat_selection import shaping_trend
    def point(b,k,w):return {'bombs_per_game':b,'causal_kills':k,'win':w}
    p=point(10,5,2)
    assert not shaping_trend([point(11,5,2)],p)['pause_required']
    assert shaping_trend([point(11,5,2),point(12,4,1)],p)['pause_required']
    assert not shaping_trend([point(11,5,2),point(12,5,3)],p)['pause_required']
