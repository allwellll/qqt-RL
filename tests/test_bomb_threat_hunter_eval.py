import json

import numpy as np
import pytest

from scripts import bomb_threat_hunter_eval as H


def test_plan_reuses_JS_seed_seat_and_cyclic_frozen_births():
    spawns=dict(seed=2026122900,spawn_cells=[[[1,1],[11,13]],[[1,13],[11,1]]])
    plan=H.make_plan({'split':'dev','iter2_arm':'A1'},spawns)
    assert plan['games_per_mode']==256 and plan['horizon']==2400
    assert plan['seed_start']==spawns['seed'] and len(plan['modes'])==3
    rows=H.initial_spec(plan,spawns)
    assert len(rows)==768
    assert rows[0]['spawn_cells']==rows[2]['spawn_cells']
    assert rows[1]['seat']==1 and rows[256]['seed']==rows[0]['seed']
    assert len(H.make_plan({},spawns)['modes'])==2


def test_first_terminal_only_and_exact_seed_denominators():
    plan=dict(games_per_mode=2,seed_start=100,modes=[{'name':'greedy','tau':0}],horizon=2400)
    rows=[dict(seed=100+i,seat=i%2,mode='greedy',done=True,ticks=2400,winner=0,
        placed=[1,2],requests=[2,3],causal_kills=[1,2],deaths=[0,1],self_deaths=[0,0],
        capture=[0,0],stock_steals=[0,0],loose_pickups=[0,0],transfers=[0,0],stored=[[3,0],[0,3]]) for i in range(2)]
    value=H.summarize(rows,plan)['greedy']
    assert value['games']==2 and value['win']==1 and value['loss']==1
    assert value['bombs_per_game']==1.5 and value['deaths']==1
    rows[0]['ticks']=2401
    with pytest.raises(AssertionError):H.summarize(rows,plan)
    rows[0]['ticks']=2400;rows[1]['seed']=100
    with pytest.raises(AssertionError):H.summarize(rows,plan)


def test_JS_latency_uses_immutable_first_completion_not_regenerated_gate(tmp_path):
    cp=tmp_path/'formal/model/u000040';cp.mkdir(parents=True)
    (cp/'COMMITTED.json').write_text(json.dumps({'metrics':{'unix':100}}))
    out=tmp_path/'eval/job';out.mkdir(parents=True)
    (out/'NATURAL_COMPLETE.json').write_text(json.dumps({'unix':160}))
    (out/'GATE_METRICS.json').write_text(json.dumps({'unix':10000}))
    req=tmp_path/'eval_requests/job.json';req.parent.mkdir()
    req.write_text(json.dumps({'checkpoint':'formal/model/u000040','split':'dev'}))
    value=H.latency_report(tmp_path,tmp_path,now=300)
    assert value['completed']['count']==1 and value['completed']['P50_seconds']==60
