from copy import deepcopy
from scripts.bomb_threat_gates_v2 import evaluate

def evidence():
    p=dict(games=64,self_deaths=0,cage_kill_fraction=.03,zero_bomb_games=14,bombs_per_game=25.,draw_fraction=.8,win_minus_loss=2,causal_kills=30,deaths=15)
    g=dict(p,zero_bomb_games=7,bombs_per_game=20.,win_minus_loss=10,deaths=20)
    refs={k:{'greedy':dict(p,win_minus_loss=v)} for k,v in [('reference_transport1536',-3),('reference_selfplay3072',-9)]}
    return {'greedy':g,'categorical':{'games':64}},{'greedy':p},{'threat_reward_fraction_total':.3},refs

def test_absolute_boundaries_and_no_automatic_promotion():
    n,p,r,refs=evidence();e=evaluate(n,p,r,'holdout',refs)
    assert e['absolute']['status']=='PASS';assert e['best']['status']=='INCOMPLETE'
    n['greedy']['win_minus_loss']=9
    assert 'WIN_MINUS_LOSS_LT_PARENT_PLUS_8' in evaluate(n,p,r,'holdout',refs)['absolute']['failures']

def test_relative_boundaries_argmax_only_and_missing_is_incomplete():
    n,p,r,refs=evidence();n['greedy'].update(zero_bomb_games=17,self_deaths=2,cage_kill_fraction=.13,draw_fraction=.85)
    n['categorical']={'games':0,'zero_bomb_games':64}
    assert evaluate(n,p,r,'dev')['relative']['status']=='PASS'
    n['greedy']['zero_bomb_games']=18
    assert evaluate(n,p,r,'dev')['relative']['status']=='FAIL'
    e=evaluate(n,p,{},'dev')['relative']
    assert e['status']=='INCOMPLETE' and e['failures'] and e['missing']==['NATIVE_REWARD_SHARE_MISSING']
    assert evaluate(n,p,r,'dev')['absolute']['missing']==['HOLDOUT_ARGMAX_AND_CATEGORICAL_64_REQUIRED']

def test_all_clauses_listed_and_no_missing_comparison_pass():
    n,p,r,refs=evidence();n['greedy'].update(zero_bomb_games=64,bombs_per_game=0,draw_fraction=1,self_deaths=3,cage_kill_fraction=.2)
    assert len(evaluate(n,p,r,'holdout',refs)['relative']['failures'])==5
    assert evaluate(n,None,r,'dev')['relative']['status']=='INCOMPLETE'
