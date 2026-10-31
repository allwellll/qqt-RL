import copy
import json
import pytest
from scripts import bomb_threat_train_monitor as M
from test_bomb_threat_train_monitor import row


def test_alert_severity_and_no_historical_window_stall():
    windows=[M.aggregate_window('run',[row(i)],i) for i in (1,2)]
    windows[-1]['unix_last']=2000
    alerts=M.detect_alerts(windows,active=True,now=2001)
    assert not any(a['code']=='NO_UPDATE_15_MINUTES' for a in alerts)
    assert all(a['severity']=='DATA_GAP' for a in alerts if a['code'].endswith('UNAVAILABLE'))
    assert all(a['severity']=='ANOMALY' for a in alerts if a['code']=='LOW_ENV_PER_SECOND')


def test_recorded_probability_weighted_by_real_ticks():
    r=row();r['exposure'].update(policy_entropy_sum=150.,bomb_probability_sum=25.,bomb_argmax_count=10.,policy_diagnostic_ticks=100.)
    out=M.aggregate_window('run',[r],1)
    assert out['policy_entropy']==1.5 and out['bomb_action_probability']==.25
    assert out['bomb_argmax_fraction']==.1


def test_scope_only_latest_active_window_is_current(tmp_path, monkeypatch):
    windows=[M.aggregate_window('run',[row(i)],i) for i in (1,2)]
    alerts=[M._alert('run',i,'BOMB_PROBABILITY_UNAVAILABLE','missing') for i in (1,2)]
    monkeypatch.setattr(M,'collect',lambda root:(windows,alerts))
    monkeypatch.setattr(M,'active_run_names',lambda:{'run'})
    receipt=M.refresh(tmp_path)
    assert [w['scope'] for w in windows]==['HISTORICAL','CURRENT']
    assert receipt['alert_counts_by_scope_and_severity']=={'HISTORICAL/DATA_GAP':1,'CURRENT/DATA_GAP':1}


def test_latency_requires_matching_policy_and_complete_js_dual64(tmp_path, monkeypatch):
    from scripts import bomb_threat_steer1120_report as S
    monkeypatch.setattr(S,'ROOT',tmp_path)
    base=tmp_path/'runs/bomb_threat_v1';cp=base/'formal/run/u000040'
    cp.mkdir(parents=True);(base/'eval_requests').mkdir()
    (cp/'COMMITTED.json').write_text(json.dumps(dict(policy_sha256='x',unix=100)))
    for name,policy,modes in [('valid','x',{'greedy':{'games':64},'categorical':{'games':64}}),
                               ('partial','x',{'greedy':{'games':64}}),
                               ('mismatch','y',{'greedy':{'games':64},'categorical':{'games':64}})]:
        (base/'eval_requests'/f'{name}.json').write_text(json.dumps(dict(checkpoint=str(cp.relative_to(tmp_path)))))
        out=base/'eval'/name;out.mkdir(parents=True)
        (out/'NATURAL_COMPLETE.json').write_text(json.dumps(dict(policy_sha256=policy,unix=140,natural_js_full2400=modes)))
    receipt=S.js_latency(base)
    assert receipt['completed']==1 and receipt['p50_seconds']==40
    assert receipt['max_seconds']==40 and receipt['incomplete_dual64_markers']==['partial']


def test_a3_early_trigger_requires_three_consecutive_cadence_points_and_matched_controls():
    from scripts.bomb_threat_iter2_analysis import a3_early_observation
    arms=[dict(update=u,train_unix=u,metrics=dict(loss=6,win_minus_loss=0)) for u in (200,240,280)]
    controls=[dict(slot=s,arm=None,train_unix=u,metrics=dict(win_minus_loss=10),marker=str((s,u))) for u in (200,240,280) for s in ('0','3','6')]
    d=a3_early_observation(arms,controls)
    assert d['status']=='RESTORE_ORIGINAL_RECIPE'
    assert a3_early_observation(arms[:2],controls)['status'].startswith('PENDING')
    arms[1]['metrics']['loss']=5
    assert a3_early_observation(arms,controls)['status']=='CONTINUE'
    arms[1]['metrics']['loss']=6
    assert a3_early_observation(arms,[r for r in controls if r['slot']!='3'])['status'].startswith('PENDING')


def test_passive_diagnostics_dont_change_sampler_output():
    import jax
    import jax.numpy as j
    import numpy as np
    from scripts.bomb_threat_passive_diagnostics import policy_sums
    lm=jax.nn.log_softmax(j.array([[1.,2.,3.]]));lb=jax.nn.log_softmax(j.array([[1.,3.,2.]]))
    before=np.asarray(jax.random.categorical(jax.random.PRNGKey(1),lb))
    d=policy_sums(lm,lb)
    np.testing.assert_array_equal(before,jax.random.categorical(jax.random.PRNGKey(1),lb))
    assert float(d['bomb_argmax_count'])==1.
    assert float(d['bomb_probability_sum'])==pytest.approx(float(j.exp(lb)[0,1]))
    assert float(d['policy_entropy_sum'])>0


def test_new_protocol_diagnostics_only_after_pass_and_existing_protocol_stays_frozen(tmp_path,monkeypatch):
    from scripts import bomb_threat_iter2_schedule as S
    from scripts.population_run_state import sha
    base=tmp_path/'runs/bomb_threat_v1';(base/'iter2').mkdir(parents=True)
    monkeypatch.setattr(S,'BASE',base);monkeypatch.setattr(S,'ROOT',tmp_path)
    config=dict(common={},policy_source={'checkpoint':'parent'},source_sha256={},initialization_decision='decision',spawn_manifest='spawns',A1_weights=[],fixed_actor_source={})
    (base/'iter2/TRAINING_PROTOCOL.json').write_text(json.dumps(config))
    proof=base/'iter2/parity.json';proof.write_text(json.dumps({'status':'PASS'}))
    files=['scripts/bomb_threat_passive_diagnostics.py','scripts/bomb_threat_collector_diagnostics_v1.py','scripts/bomb_threat_iter2_diagnostics_train.py']
    for f in files:
        p=tmp_path/f;p.parent.mkdir(exist_ok=True,parents=True);p.write_text('frozen')
    old=S.make_protocol(1,1,None,{'iter2_arm':'A3'});old_bytes=old.read_bytes()
    (base/'iter2/DIAGNOSTICS_ACTIVATION.json').write_text(json.dumps(dict(status='READY_AT_NEXT_CHUNK',parity_receipt=str(proof.relative_to(tmp_path)),parity_sha256=sha(proof),source_sha256={f:sha(tmp_path/f) for f in files})))
    new=S.make_protocol(1,2,None,{'iter2_arm':'A3'})
    assert json.loads(new.read_text())['worker_module']=='scripts.bomb_threat_iter2_diagnostics_train'
    assert S.make_protocol(1,1,None,{'iter2_arm':'A3'}).read_bytes()==old_bytes


def test_table_data_gap_not_red_and_anomaly_is_red():
    from scripts import build_model_metrics_table as T
    h,_=T.table([dict(window=1,data_gaps='UNAVAILABLE',anomalies='LOW_ENV_PER_SECOND')],[('data_gaps','缺失'),('anomalies','异常')])
    assert '<td style="">UNAVAILABLE</td>' in h
    assert '<td style="background:#f6a0a0">LOW_ENV_PER_SECOND</td>' in h
