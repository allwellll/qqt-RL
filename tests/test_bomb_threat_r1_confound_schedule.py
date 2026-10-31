from pathlib import Path
from scripts import bomb_threat_r1_confound_schedule as S
from scripts.bomb_threat_r1_candidate_confirm import gate


def test_switch_never_touches_active_or_wrong_slot(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'BASE',tmp_path);monkeypatch.setattr(S,'OUT',tmp_path/'revision')
    entry=dict(run='run',r1_items_applied=True)
    assert S.apply_switch(0,entry,8) is None
    assert S.apply_switch(3,entry,8) is None
    (tmp_path/'revision').mkdir();(tmp_path/'revision/READY.json').write_text('{"status":"PASS"}')
    assert S.apply_switch(3,entry,8) is None # no COMPLETE means live/between updates
    assert S.apply_switch(3,dict(entry,r1_preserved_legacy=True),8) is None


def test_fallback_not_for_healthy_or_complete_restore(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'BASE',tmp_path)
    entry=dict(run='run',r1_items_applied=True)
    assert S.fallback(3,entry,8) is None
    p=tmp_path/'formal/run';p.mkdir(parents=True);(p/'TECHNICAL_FAILURE.json').write_text('{"exit_code":1}')
    (p/'RESTORE.json').write_text('{"passed":true}')
    assert S.fallback(3,entry,8) is None


def test_candidate_gate_requires_all_three_full_conditions(tmp_path,monkeypatch):
    import json
    from scripts import bomb_threat_r1_candidate_confirm as C
    monkeypatch.setattr(C,'WATCH',tmp_path)
    tiers={t:dict(games=256,by_seat={'P1':{'win_minus_loss':v}}) for t,v in [('hard',29),('normal',50),('easy',66)]}
    b={t:dict(by_seat={'P1':{'win_minus_loss':v}}) for t,v in [('hard',13),('normal',26),('easy',52)]}
    p=tmp_path/'RESULTS.json';p.write_text(json.dumps(dict(fresh={'0':tiers},baseline=b)))
    assert C.gate()['status']=='PASS' and C.gate()['P1_sum']==145
    tiers['hard']['by_seat']['P1']['win_minus_loss']=7;p.write_text(json.dumps(dict(fresh={'0':tiers},baseline=b)))
    assert C.gate()['status']=='FAIL' # aggregate alone cannot mask tier regression
    tiers['easy']['games']=248;p.write_text(json.dumps(dict(fresh={'0':tiers},baseline=b)))
    assert C.gate()['status']=='WAITING_EASY_COMPLETE'


def test_generated_fullstate_worker_never_rekeys_or_reinitializes():
    from scripts.bomb_threat_r1_preserved_train import worker_source
    s=worker_source();compile(s,'new-worker','exec')
    assert 'parent_model_adam' not in s and "exported=json.loads" not in s
    assert 'previous_full_credit_preserved=True' in s
    assert "credit=dict(credit,pre_revision=previous" in s
    assert "jax.random.split(jax.random.PRNGKey(seed),n)" not in s[s.index('    if not cps:\n        from scripts import bomb_threat_r1_preserved_reward'):s.index('    # Initialize cached safety')]
    assert "row['exposure']['r1_legacy_pay']>0" in s
