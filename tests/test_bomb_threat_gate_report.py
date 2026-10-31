import json
from scripts import bomb_threat_gate_report as G
from scripts import bomb_threat_eval as E
from scripts.population_run_state import sha


def test_recompute_draw_difference_preserves_immutable_natural_marker(tmp_path):
    files={}
    for mode in ('greedy','categorical'):
        for start in range(0,64,8):
            primary=[];side=[]
            for i in range(start,start+8):
                seat=i%2;kills=[1,3] if seat else [3,1]
                primary.append(dict(seed=100+i,team=seat,ticks=2400,causal_kill=3))
                side.append(dict(seed=100+i,seat=seat,ticks=2400,done=True,winner=-1,
                    causal_kills=kills,cage_kills=[0,0],placed=[1,1],requests=[1,1],
                    deaths=[0,0],self_deaths=[0,0],stored=[[0,0],[0,0]]))
            p=tmp_path/f'natural_{mode}_{start:03d}.json'
            p.write_text(json.dumps({'results':{'natural_full2400':{'episodes':primary}}}))
            s=tmp_path/(p.name+'.outcome.json');s.write_text(json.dumps({'measurement_version':2,'episodes':side}))
            files[p.name]=sha(p);files[s.name]=sha(s)
    p=tmp_path/'NATURAL_COMPLETE.json'
    p.write_text(json.dumps(dict(measurement_version=2,rule_version='stock-window-v1',files=files,
        natural_js_full2400={'greedy':{'delivery':9},'categorical':{'delivery':8}})))
    digest=sha(p);marker,natural=G.recompute_natural(tmp_path)
    assert sha(p)==digest
    assert natural['greedy']['delivery']==9
    assert natural['greedy']['draw_kill_difference']==128
    # Altered evidence must be rejected, rather than silently recounted.
    s.write_text('{}')
    import pytest
    with pytest.raises(AssertionError):G.recompute_natural(tmp_path)


def test_reward_ceiling_is_absolute_only_and_legacy_is_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(E,'BASE',tmp_path)
    run=tmp_path/'formal/example';run.mkdir(parents=True)
    parts=[0.]*90;parts[2]=1.
    (run/'updates.jsonl').write_text(json.dumps(dict(new_update=2,exposure=dict(
        threat_events=1,chain_reward=2,total_reward=10,absolute_total_reward=10,native_parts=parts,resets=1)))+'\n')
    value=G.reward_monitor(run/'u000002',2)
    assert value['threat_reward_fraction_total']==.2
    assert value['relative_reward_share_gate_pass'] is True
    assert value['reward_share_gate_pass'] is False
    assert G.reward_monitor(tmp_path/'legacy/u000002',2)['threat_reward_fraction_total'] is None


def test_json_web_reference_has_no_native_checkpoint_and_cannot_crash_queue(tmp_path,monkeypatch):
    monkeypatch.setattr(G,'BASE',tmp_path)
    p=tmp_path/'eval/reference_transport1536/NATURAL_COMPLETE.json';p.parent.mkdir(parents=True);p.write_text('{}')
    summary=p.parent/'SUMMARY.json';summary.write_text('{"historical":true}')
    complete=p.parent/'COMPLETE.json';complete.write_text(json.dumps({'summary_sha256':sha(summary)}))
    old_summary_sha=sha(summary);old_natural_sha=sha(p)
    marker=dict(checkpoint=None,role='web_reference',model_json='model.json',policy_sha256='abc',split='holdout',seed=123)
    natural={'greedy':dict(games=64,win_minus_loss=1,causal_kills=3,deaths=1)}
    monkeypatch.setattr(G,'recompute_natural',lambda out:(marker,natural))
    result=G.refresh_all()['reference_transport1536']
    assert result['checkpoint'] is None
    assert not result['best_eligible'] and not result['pbt_selection_eligible']
    assert 'REFERENCE_ONLY' in result['pbt_gate_failures']
    assert sha(p)==old_natural_sha
    assert sha(p.parent/'SUMMARY.pre_gate_amend_20261029_0240.json')==old_summary_sha
    assert json.loads(complete.read_text())['summary_sha256']==sha(summary)
    assert json.loads(summary.read_text())['selection_eligible'] is False


def test_dev_absolute_pass_cannot_promote_without_independent_holdout():
    assert G.promotion_failures([], 'dev') == ['HOLDOUT_REQUIRED_FOR_BEST']
    assert G.promotion_failures([], 'holdout') == []
    assert G.promotion_failures(['zero_bombs'], 'holdout') == ['zero_bombs']


def test_holdout_request_is_exact_checkpoint_and_never_duplicate(tmp_path,monkeypatch):
    monkeypatch.setattr(G,'BASE',tmp_path)
    marker={'checkpoint':'runs/bomb_threat_v1/formal/bt_slot00_c000009/u000800','split':'dev'}
    assert G.request_holdout('dev_name',marker,False) is None
    path=G.request_holdout('dev_name',marker,True)
    data=json.loads(path.read_text())
    assert data['checkpoint']==marker['checkpoint'] and data['split']=='holdout'
    digest=sha(path)
    assert G.request_holdout('dev_name',marker,True)==path and sha(path)==digest


def test_explicit_monitor_priority_survives_refresh_and_preserves_order(tmp_path,monkeypatch):
    monkeypatch.setattr(G,'BASE',tmp_path)
    requests=tmp_path/'eval_requests';requests.mkdir()
    for name in ('slot03_holdout','slot07_holdout'):
        (requests/(name+'.json')).write_text('{}')
    decision=tmp_path/'MONITOR_20261030_0200.json'
    decision.write_text(json.dumps(dict(status='AUTHORIZED',priority_names=[
        'slot03_holdout','slot07_holdout','slot03_holdout','missing','../escape'])))
    assert G.monitor_priority()==['slot03_holdout','slot07_holdout']
    # Refresh has no eligible automatic candidates; the explicit priority remains.
    G.refresh_all()
    result=json.loads((tmp_path/'GATE_AMEND2_HOLDOUT_PRIORITY.json').read_text())
    assert result['names']==['slot03_holdout','slot07_holdout']
    decision.write_text(json.dumps(dict(status='REVOKED',priority_names=['slot03_holdout'])))
    assert G.monitor_priority()==[]
