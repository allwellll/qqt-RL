from scripts import bomb_threat_round16 as R
from scripts.population_run_state import atomic_json


def test_only_authorized_boundary_can_continue(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'AUTH',tmp_path/'auth.json')
    atomic_json(R.AUTH,dict(boundary_run='old'))
    assert R.apply_switch(7,dict(run='old'),{},29) is None
    assert R.apply_switch(5,dict(run='other'),{},29) is None
    assert R.apply_switch(5,dict(run='old',round16_applied=True),{},29) is None
    assert not R.terminal(dict(round14_applied=True),dict(update=1280))


def test_exact_self_continuation_no_old_cap_or_rekey(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'ROOT',tmp_path)
    monkeypatch.setattr(R,'BASE',tmp_path/'runs/bomb_threat_v1')
    monkeypatch.setattr(R,'AUTH',R.BASE/'auth.json')
    source=dict(checkpoint='runs/bomb_threat_v1/formal/old/u001280',update=1280,
                state_sha256='state',policy_sha256='policy')
    recipe=dict(iter2_arm='A3SR',lr=7.5e-5,learner_seat_fraction=.5)
    atomic_json(R.BASE/'protocols/old.json',dict(arms={'old':recipe},source_sha256={}))
    auth=dict(source=source,target_seed=2026112905,collector_seed=2026112905)
    p=R.make_protocol(auth,29);d=R.read(p)
    assert d['arms'][p.stem]==recipe and d['continuation_source']==source
    assert d['chunk_start_update']==1280 and d['chunk_updates']==320
    assert d['environment_steps_per_run']==1600*262144
    assert d['round14_rekey'] is False and d['round14_collector_seed']==2026112905
    assert R.make_protocol(auth,29)==p


def test_holdout_waits_for_real_fresh_completion(tmp_path,monkeypatch):
    from scripts import bomb_threat_eval_owner as E
    monkeypatch.setattr(E,'ROOT',tmp_path)
    r=dict(after_complete='runs/fresh/COMPLETE.json')
    assert not E.round16_ready(r)
    atomic_json(tmp_path/r['after_complete'],dict(status='FAIL',collection_status='COMPLETE'))
    assert E.round16_ready(r)  # collection complete, not a demand for a passing old candidate
    assert E.round16_ready({})
    assert not E.round16_ready(dict(after_complete='../outside'))


def test_two_point_decline_requires_every_metric_both_points(tmp_path,monkeypatch):
    from scripts import bomb_threat_round16_report as E
    monkeypatch.setattr(E,'OUT',tmp_path/'round16')
    controls=[dict(win_minus_loss=9,causal_kills=24,bombs_per_game=18)]*4
    monkeypatch.setattr(E,'recent',lambda slot:controls)
    rows={2120:dict(win_minus_loss=8,causal_kills=12,bombs_per_game=15),
          2160:dict(win_minus_loss=18,causal_kills=19,bombs_per_game=17)}
    monkeypatch.setattr(E,'argmax',lambda name:rows[int(name.rsplit('_u',1)[1])])
    result=E.decision()
    assert result['status']=='NOT_TRIGGERED'
    assert result['strict_below'][1]['win_minus_loss'] is False
    # Once fully observed, the prescribed comparison is frozen, not reselected later.
    rows[2160]['win_minus_loss']=0
    assert E.decision()['status']=='NOT_TRIGGERED'


def test_gpu2_request_not_published_before_fresh_marker(tmp_path,monkeypatch):
    from scripts import bomb_threat_round16_report as E
    monkeypatch.setattr(E,'BASE',tmp_path/'runs/bomb_threat_v1')
    monkeypatch.setattr(E,'OUT',E.BASE/'round16')
    p=E.OUT/'pending_requests/target_holdout.json'
    atomic_json(p,dict(name=p.stem,checkpoint='runs/model/u001280',split='holdout'))
    assert E.release_requests() is False
    assert not (E.BASE/'eval_requests'/p.name).exists()
    atomic_json(E.BASE/'fresh256_a3sr_u640/COMPLETE.json',dict(status='FAIL',collection_status='COMPLETE'))
    assert E.release_requests() is True
    assert E.read(E.BASE/'eval_requests'/p.name)==E.read(p)
    assert E.release_requests() is True
    assert E.read(E.BASE/'GATE_AMEND2_HOLDOUT_PRIORITY.json')['names']==['target_holdout']
