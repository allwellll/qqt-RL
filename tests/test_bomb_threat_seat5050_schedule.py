import json
from scripts import bomb_threat_seat5050_schedule as S


def test_switch_requires_approval_and_target_slot_and_complete_chunk(tmp_path,monkeypatch):
    base=tmp_path/'runs/bomb_threat_v1';(base/'a3s').mkdir(parents=True)
    monkeypatch.setattr(S,'BASE',base)
    assert S.apply_switch(4,dict(run='old',iter2_arm='A3'),{},1) is None
    (base/'a3s/AUTHORIZATION.json').write_text(json.dumps(dict(status='AUTHORIZED_READY')))
    assert S.apply_switch(1,dict(run='control',iter2_arm='A3'),{},1) is None
    assert S.apply_switch(4,dict(run='old',iter2_arm='A3'),{},1) is None
    assert S.apply_switch(7,dict(run='new',iter2_arm='A3S'),{},1) is None


def test_terminal_caps640_and_requests_both_splits_without_launching(tmp_path,monkeypatch):
    from scripts import bomb_threat_seat5050_verdict as V
    monkeypatch.setattr(V,'refresh',lambda *args:dict(status='PENDING'))
    base=tmp_path/'runs/bomb_threat_v1';base.mkdir(parents=True)
    monkeypatch.setattr(S,'BASE',base)
    e=dict(run='bt_slot04_c000019',iter2_arm='A3S')
    assert not S.terminal(4,e,dict(update=320))
    done=dict(update=640,checkpoint='runs/bomb_threat_v1/formal/bt_slot04_c000019/u000640')
    assert S.terminal(4,e,done)
    paths=list((base/'eval_requests').glob('*.json'));assert len(paths)==2
    assert {json.loads(p.read_text())['split'] for p in paths}=={'dev','holdout'}
    assert not S.terminal(1,dict(iter2_arm='A3'),done)


def test_verdict_requires_both_splits_and_preserves_all_three_conditions():
    from scripts.bomb_threat_seat5050_verdict import compare
    def met(p0,p1,zero):return dict(by_seat={s:dict(games=32,win=w,loss=0,zero_bomb_games=z) for s,w,z in [('P0',p0,0),('P1',p1,zero)]})
    c={k:met(10,8,4) for k in ('dev','holdout')};r={k:met(13,3,4) for k in ('dev','holdout')}
    assert compare(c,r)['status']=='PASS'
    assert compare({'dev':c['dev']},r)['status']=='PENDING'
    c['holdout']=met(9,8,4);assert compare(c,r)['status']=='FAIL'
    c['holdout']=met(10,7,4);assert compare(c,r)['status']=='FAIL'
    c['holdout']=met(10,8,5);assert compare(c,r)['status']=='FAIL'


def test_passive_acceptance_handles_retained_state_removed_after_chunk(tmp_path):
    from scripts.bomb_threat_seat5050_acceptance import retained_atomic
    initial=tmp_path/'old';active=tmp_path/'active'
    (initial/'u000320').mkdir(parents=True)
    (initial/'u000320/COMMITTED.json').write_text('{}')
    assert retained_atomic(initial,active) is None


def test_authorized_extension_allows640_and960_but_caps1280(tmp_path,monkeypatch):
    from scripts.population_run_state import sha
    b=tmp_path/'base';(b/'a3s/extension1280').mkdir(parents=True)
    i=tmp_path/'decision.md';i.write_text('extend exactly640 updates')
    monkeypatch.setattr(S,'BASE',b)
    assert S.budget_limit()==640
    (b/'a3s/extension1280/AUTHORIZATION.json').write_text(json.dumps(dict(status='AUTHORIZED',instruction=str(i),instruction_sha256=sha(i),formal_updates_per_seed=1280)))
    assert S.budget_limit()==1280
    assert not S.terminal(4,dict(iter2_arm='A3S'),dict(update=640))
    assert not S.terminal(7,dict(iter2_arm='A3S'),dict(update=960))
