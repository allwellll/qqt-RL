import copy
import pytest
from scripts.r3_selfplay_gate_20261031 import decide
from scripts.r3_selfplay_train_20261031 import worker_source


def result(wins=80, losses=12):
    seats={'P0':dict(games=128,wins=90,losses=10,win_minus_loss=80),
           'P1':dict(games=128,wins=wins,losses=losses,win_minus_loss=wins-losses)}
    return dict(status='COMPLETE',games=6144,models={'r3':{'r3':True},'base':{'r3':False}},
        selection={'comparison':{'r3':{'P1_sum':200}}},
        metrics={'r3':{'hunter_hard':copy.deepcopy(seats),'coop_hard':copy.deepcopy(seats)}})


def test_complete_and_both_hard_conditions_required():
    r=result();assert decide(r,['r3'])['status']=='PASS'
    r['metrics']['r3']['coop_hard']['P1'].update(wins=60,win_minus_loss=48)
    g=decide(r,['r3']);assert g['status']=='NOT_ELIGIBLE'
    assert g['conditions']['coop_hard']['win_rate_gap']>0
    for status in ('RUNNING','INVALIDATED'):
        r=result();r['status']=status
        with pytest.raises(AssertionError):decide(r,['r3'])


def test_boundary_negative_and_seat_gates():
    assert decide(result(71,19),['r3'])['status']=='PASS'
    assert decide(result(70,19),['r3'])['status']=='NOT_ELIGIBLE'
    assert decide(result(80,20),['r3'])['status']=='NOT_ELIGIBLE'
    r=result();r['metrics']['r3']['hunter_hard']['P0']['win_minus_loss']=0
    assert decide(r,['r3'])['status']=='NOT_ELIGIBLE'


def test_best_parent_fixed_primary_and_tie_order():
    r=result();r['models']['second']={'r3':True}
    r['metrics']['second']=copy.deepcopy(r['metrics']['r3'])
    r['selection']['comparison']['second']={'P1_sum':200}
    assert decide(r,['r3','second'])['parent']=='r3'
    r['selection']['comparison']['second']['P1_sum']=201
    assert decide(r,['r3','second'])['parent']=='second'


def test_generated_worker_preserves_counted_smoke_restore_and_onpolicy():
    source=worker_source();compile(source,'<R3-sp>','exec')
    assert source.count('install_monitor(M)')==1
    assert 'starting_state_unchanged=True' in source
    assert 'diagnostic_environment_steps=20*batch_size' in source
    assert 'full_env_aux_RNG_exact=True' in source
    assert 'compute_gae' not in source  # inherited collector computes true on-policy GAE
    assert 'selfplay_fixed_pool' in source


def test_owner_waits_without_gpu_or_registration(tmp_path, monkeypatch):
    import json
    from scripts import r3_selfplay_owner_20261031 as owner
    monkeypatch.setattr(owner,'OUT',tmp_path)
    monkeypatch.setattr(owner,'paused',lambda:False)
    monkeypatch.setattr(owner,'evaluate',lambda:dict(status='WAIT_OWNSIDE_6144',training_started=False))
    def stop(_):raise StopIteration()
    monkeypatch.setattr(owner.time,'sleep',stop)
    with pytest.raises(StopIteration):owner.main()
    assert json.loads((tmp_path/'STATUS.json').read_text())['training_started'] is False
    assert not (tmp_path/'REGISTRATION.json').exists()


def test_passive_and_delivery_monitor_equal_budget_and_half_boundary():
    import numpy as np
    from scripts.r3_selfplay_owner_20261031 import monitor_rows
    values={k:np.full((3,2),v) for k,v in dict(r3_sp_episodes=100.,r3_sp_draws=10.,
        r3_sp_passive=0.,r3_sp_active_deliveries=40.,r3_sp_fixed_games=100.,
        r3_sp_own_zero_bomb=2.,r3_sp_both_zero_bomb=1.,r3_sp_custody_loops=3.,r3_sp_unresolved=0.,
        r3_sp_fixed_wins=60.,r3_sp_fixed_draws=20.,r3_sp_fixed_losses=20.,r3_sp_fixed_unresolved=0.).items()}
    baseline={'metrics':{k:v.tolist() for k,v in values.items()}}
    rows=[{'exposure':{k:(v/20).tolist() for k,v in values.items()}} for _ in range(20)]
    measured=monitor_rows(rows,baseline)
    assert measured['status']=='OBSERVING'
    assert np.isclose(measured['seats']['P0']['own_zero_bomb'],6)
    assert np.isclose(measured['seats']['P1']['by_pool']['fixed']['custody_loops'],3)
    unknown_rows=copy.deepcopy(rows)
    for row in unknown_rows:
        row['exposure']['r3_sp_fixed_unresolved']=(np.full((3,2),20.)/20).tolist()
        row['exposure']['r3_sp_fixed_losses']=np.zeros((3,2)).tolist()
    fixed=monitor_rows(unknown_rows,baseline)['fixed']['0']['P0']
    assert np.isclose(fixed['resolved_games'],80.)
    assert np.isclose(fixed['score'],.875)
    for row in rows:row['exposure']['r3_sp_active_deliveries']=(values['r3_sp_active_deliveries']/40).tolist()
    m=monitor_rows(rows,baseline);assert m['seats']['P0']['flags']['delivery_halved']
    assert m['status']=='ROLLBACK'


def test_resource_guard_prefilled_baseline_and_disjoint_windows(tmp_path, monkeypatch):
    from scripts import r3_selfplay_owner_20261031 as owner
    from scripts.population_run_state import atomic_json
    monkeypatch.setattr(owner,'OUT',tmp_path)
    monkeypatch.setattr(owner,'notify',lambda *args,**kwargs:None)
    atomic_json(tmp_path/'RESOURCE_GUARD.json',dict(
        baseline={'reference':dict(run='control',update=100,env_per_second=100.)},
        strikes={},last_update={}))
    for update,expected in ((105,1),(105,1),(110,2),(115,3)):
        monkeypatch.setattr(owner,'reference_snapshot',lambda:dict(
            reference=dict(run='control',update=update,env_per_second=90.)))
        owner.resource_guard()
        measured=owner.read(tmp_path/'RESOURCE_GUARD.json')
        assert measured['strikes']['reference']==expected
        assert (tmp_path/'RESOURCE_PAUSE.json').exists()==(expected==3)


@pytest.mark.parametrize('key,ratio,trigger',[
    ('runs/bomb_threat_v1:5',.91,False),
    ('runs/bomb_threat_v1:0',.85,False),
    ('runs/bomb_threat_v1:7',.849,True),
    ('runs/r3_items_route_20261030:a',.949,True),
    ('runs/bun_native_v2_20261030:b',.95,False),
    ('runs/bun_native_v2_20261030:c',.949,True),
])
def test_amended_guard_threshold_and_three_disjoint_windows(tmp_path,monkeypatch,key,ratio,trigger):
    from scripts import r3_selfplay_owner_20261031 as owner
    from scripts.population_run_state import atomic_json
    monkeypatch.setattr(owner,'OUT',tmp_path)
    monkeypatch.setattr(owner,'notify',lambda *a,**kw:None)
    atomic_json(tmp_path/'RESOURCE_GUARD.json',dict(baseline={key:dict(run='same',update=100,env_per_second=100.)},strikes={},last_update={}))
    for update in (104,105,105,109,110,114,115):
        monkeypatch.setattr(owner,'reference_snapshot',lambda: {key:dict(run='same',update=update,env_per_second=100*ratio)})
        owner.resource_guard()
        assert (tmp_path/'RESOURCE_PAUSE.json').exists()==(trigger and update==115)
    assert owner.read(tmp_path/'RESOURCE_GUARD.json')['comparisons'][key]['minimum_ratio']==owner.reference_minimum_ratio(key)


def test_reference_throughput_is_total_steps_over_total_walltime_and_phase_matched(tmp_path,monkeypatch):
    from scripts import r3_selfplay_owner_20261031 as owner
    from scripts.population_run_state import atomic_json
    rows=[dict(new_update=u,cycle_seconds=29. if u%2 else 60.,actual_environment_steps_this_update=262144) for u in range(101,109)]
    base=owner.reference_window_snapshot(rows)
    assert base['env_per_second']==5*262144/(3*60+2*29)
    assert base['phase_baselines']['1']['env_per_second']==5*262144/(2*60+3*29)
    monkeypatch.setattr(owner,'OUT',tmp_path);monkeypatch.setattr(owner,'notify',lambda *a,**k:None)
    key='runs/r3_items_route_20261030:b'
    atomic_json(tmp_path/'RESOURCE_GUARD.json',dict(baseline={key:dict(run='same',**base)},strikes={},last_update={}))
    for end in (113,118,123):
        now=owner.reference_window_snapshot([dict(r,new_update=r['new_update']+end-108,
            cycle_seconds=29. if (r['new_update']+end-108)%2 else 60.) for r in rows])
        monkeypatch.setattr(owner,'reference_snapshot',lambda:{key:dict(run='same',**now)})
        owner.resource_guard()
        assert owner.read(tmp_path/'RESOURCE_GUARD.json')['comparisons'][key]['ratio']==1.
        assert not (tmp_path/'RESOURCE_PAUSE.json').exists()
