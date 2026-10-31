import json
from scripts import build_model_metrics_table as M


def test_missing_settlement_is_unknown_not_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'OUT',tmp_path)
    e=dict(seed=100,team=0,ticks=2400,causal_kill=2,surviving_kill=1,death=1,self_death=0,delivery=0)
    (tmp_path/'candidate_greedy_000.json').write_text(json.dumps({'results':{'natural_full2400':{'episodes':[e]}}}))
    rows=M.legacy_rows()
    assert len(rows)==1 and rows[0]['win'] is None and rows[0]['loss'] is None
    assert rows[0]['bombs_per_game'] is None and rows[0]['causal_kills']==2


def test_modes_aggregate_shards_without_mixing(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'OUT',tmp_path)
    for mode,seed in [('greedy',100),('categorical',200)]:
        for i in (0,8):
            e=dict(seed=seed+i,team=0,ticks=2400,causal_kill=2,surviving_kill=1,death=1,self_death=0,delivery=0)
            p=tmp_path/f'candidate_{mode}_{i:03d}.json'
            p.write_text(json.dumps({'results':{'natural_full2400':{'episodes':[e]}}}))
            (tmp_path/(p.name+'.outcome.json')).write_text(json.dumps({'episodes':[dict(done=True,ticks=2400,winner=0,bombs=[10,0])]}))
    rows=M.legacy_rows()
    assert len(rows)==2
    for r in rows:assert r['games']==2 and r['win']==2 and r['bombs_per_game']==10
    assert 'bombs_per_game' not in M.LOW


def test_amend2_missing_is_not_fail_and_old_gate_has_separate_columns(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'NEW',tmp_path);monkeypatch.setattr(M,'OUT',tmp_path/'old')
    out=tmp_path/'eval/candidate';out.mkdir(parents=True)
    (out/'NATURAL_COMPLETE.json').write_text(json.dumps({'measurement_version':2,'split':'dev','seed':1,'natural_js_full2400':{'greedy':{'games':64}}}))
    (out/'GATE_METRICS.json').write_text(json.dumps({'absolute_gate_status':'INCOMPLETE','pbt_gate_status':'PASS',
        'previous_best_gate':'FAIL','previous_pbt_gate':'FAIL','best_gate_missing':['HOLDOUT_REQUIRED'],'best_eligible':False,'pbt_selection_eligible':True}))
    row=M.rows()[0]
    assert row['best_gate']=='INCOMPLETE' and row['pbt_gate']=='PASS'
    assert row['previous_best_gate']=='FAIL' and row['previous_pbt_gate']=='FAIL'
    assert 'ARGMAX_ZERO_BOMBS_GT_3' not in row['flags']


def test_iter2_half_temperature_is_separate_and_never_gate_input(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'NEW',tmp_path);monkeypatch.setattr(M,'OUT',tmp_path/'old')
    out=tmp_path/'eval/candidate';out.mkdir(parents=True)
    (out/'NATURAL_COMPLETE.json').write_text(json.dumps(dict(measurement_version=2,split='dev',seed=1,
        policy_sha256='policy',iter2_arm='A2',natural_js_full2400={'greedy':{'games':64}})))
    half=dict(policy_sha256='policy',tau=.5,games=64,metrics={'games':64,'win':25},gate_input=False)
    (out/'TEMPERATURE_HALF_COMPLETE.json').write_text(json.dumps(half))
    row=[r for r in M.rows() if r['mode']=='categorical_tau0.5'][0]
    assert row['win']==25 and row['best_gate']==row['pbt_gate']=='NOT_APPLICABLE'
    half['policy_sha256']='other';(out/'TEMPERATURE_HALF_COMPLETE.json').write_text(json.dumps(half))
    assert len(M.rows())==1
