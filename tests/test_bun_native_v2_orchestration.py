import json
from pathlib import Path
from scripts import bun_native_v2_owner as O
from scripts.bun_native_v2_eval_owner import summary,gate


def test_gate_waits_actual_diagnostic_and_real_collector(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'OUT',tmp_path)
    missing=O.gates()
    assert 'DIAGNOSTIC_COMPLETE.json' in missing and 'COLLECTOR_WIRING.json' in missing
    assert 'parity/event_v3/ACCEPTANCE.json' in missing


def test_evaluation_draw_denominator_own_recovery_not_active_delivery():
    e=dict(seat=1,winner=None,kills=[],placed=[1,0],self_deaths=[0,0],deaths=[0,0],
        antifarm=dict(stock_pickups=[0,1],deliveries=[0,1],enemy_cage_entries=[0,1],alive_ticks=[10,10],
        actionable_ticks=[10,10],near_home_ticks=[0,0],events=[dict(type='delivery',player=1,origin=1)]))
    s=summary([e])['P1']
    assert s['draws']==1 and s['games']==1 and s['unilateral_zero_bomb']==1
    assert s['active_deliveries']==0 and s['all_origin_deposits']==1
    assert s['farm_fraction'] is None


def test_no_kills_is_missing_farm_data_not_automatic_pass():
    s=dict(games=64,win_minus_loss=0,unilateral_zero_bomb=0,zero_bomb=0,self_deaths=0,active_deliveries=1,farm_fraction=None)
    doc=dict(metrics=dict(holdout={m:{p:dict(s) for p in ('P0','P1')} for m in ('greedy','categorical')}))
    result=gate(doc,doc)
    assert result['status']=='NOT_QUALIFIED'
    assert all(not x['farm_guard'] for x in result['comparisons'].values())


def test_P1_all_zero_bomb_guard_plus3_not_only_unilateral():
    s=dict(games=64,win_minus_loss=0,unilateral_zero_bomb=0,zero_bomb=10,
           self_deaths=0,active_deliveries=1,farm_fraction=.2,enemy_stock_pickups=10,carrier_kills=2,wins=20)
    parent=dict(metrics=dict(holdout={m:{p:dict(s) for p in ('P0','P1')} for m in ('greedy','categorical')}))
    candidate=json.loads(json.dumps(parent))
    for mode in ('greedy','categorical'):
        candidate['metrics']['holdout'][mode]['P1']['zero_bomb']=13
    assert gate(candidate,parent)['status']=='PASS'
    candidate['metrics']['holdout']['greedy']['P1']['zero_bomb']=14
    result=gate(candidate,parent)
    assert result['status']=='NOT_QUALIFIED'
    assert not result['comparisons']['greedy_P1']['P1_zero_bomb_guard']
    assert result['comparisons']['greedy_P1']['trend_guard']


def test_enemy_delivery_loss_requires_terminal_tick_enemy_origin():
    base=dict(seat=1,winner=0,ticks=20,kills=[],placed=[1,0],self_deaths=[0,0],deaths=[0,0],
        antifarm=dict(stock_pickups=[1,0],deliveries=[1,0],enemy_cage_entries=[1,0],alive_ticks=[20,10],
            actionable_ticks=[20,10],near_home_ticks=[0,4],events=[]))
    for event,expected in [(dict(type='delivery',player=0,origin=1,tick=20),1),
                            (dict(type='delivery',player=0,origin=1,tick=19),0),
                            (dict(type='delivery',player=0,origin=0,tick=20),0)]:
        base['antifarm']['events']=[event]
        result=summary([base])['P1']
        assert result['opponent_delivery_losses']==expected
        assert result['near_home_alive_fraction']==.4


def test_native_owner_does_not_restart_failed_phase_and_can_adopt_live_workers():
    source=Path('scripts/bun_native_v2_owner.py').read_text()
    assert "processes(entry.get('worker_module',MODULE),entry['run'])" in source
    assert 'worker exited before phase acceptance' in source
    assert 'USER_PAUSED' in source
    assert 'R2_DONOR_LOCK.json' in source


def test_custody_antifarm_requires_all_three_signals_and_active_delivery():
    s=dict(games=64,win_minus_loss=0,unilateral_zero_bomb=0,zero_bomb=0,self_deaths=0,
        active_deliveries=1,farm_fraction=.2,enemy_stock_pickups=10,carrier_kills=2,wins=20)
    parent=dict(metrics=dict(holdout={m:{p:dict(s) for p in ('P0','P1')} for m in ('greedy','categorical')}))
    candidate=json.loads(json.dumps(parent));x=candidate['metrics']['holdout']['greedy']['P0']
    x.update(enemy_stock_pickups=16,carrier_kills=3)
    assert gate(candidate,parent)['comparisons']['greedy_P0']['custody_farming_detected']
    x['wins']=21
    assert not gate(candidate,parent)['comparisons']['greedy_P0']['custody_farming_detected']
    x['active_deliveries']=0
    assert not gate(candidate,parent)['comparisons']['greedy_P0']['active_delivery_guard']
