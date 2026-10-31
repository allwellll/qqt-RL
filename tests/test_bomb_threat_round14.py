import pytest
from scripts import bomb_threat_round14 as R
from scripts.population_run_state import atomic_json
from scripts.bomb_threat_a3sr_resume_train import worker_source


def test_resume_parity_precedes_optional_rng_change():
    s=worker_source()
    compile(s,'resume-worker','exec')
    assert s.index("'source next-step parity mismatch'") < s.index("if protocol.get('round14_rekey')")
    assert s.index("if protocol.get('round14_rekey')") < s.index('    if args.verify_restore:')
    assert "protocol['round14_collector_seed']" in s
    assert "bomb_threat_a3sr_collector as M" in s
    assert "start+length<=640" not in s


def test_no_authorization_or_wrong_slot_cannot_switch(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'AUTH',tmp_path/'AUTHORIZATION.json')
    assert R.apply_switch(5,{'run':R.DONOR},{},27) is None
    atomic_json(R.AUTH,{'status':'AUTHORIZED'})
    assert R.apply_switch(4,{'run':R.DONOR},{},27) is None


def test_budget_and_recipe_are_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'ROOT',tmp_path)
    monkeypatch.setattr(R,'BASE',tmp_path/'runs/bomb_threat_v1')
    monkeypatch.setattr(R,'AUTH',tmp_path/'AUTHORIZATION.json')
    atomic_json(R.AUTH,{'status':'AUTHORIZED','seed_by_slot':{'5':2026112905,'7':2026112907}})
    recipe={'iter2_arm':'A3SR','lr':7.5e-5}
    atomic_json(R.BASE/'protocols'/f'{R.DONOR}.json',{'arms':{R.DONOR:recipe},'source_sha256':{}})
    for f in ('scripts/bomb_threat_round14.py','scripts/bomb_threat_a3sr_resume_train.py'):
        p=tmp_path/f;p.parent.mkdir(exist_ok=True);p.write_text('fixture')
    source={'update':960,'checkpoint':'retained','state_sha256':'state','policy_sha256':'policy'}
    p=R.make_protocol(5,28,source,recipe,640)
    value=R.read(p)
    assert value['chunk_start_update']+value['chunk_updates']==1280
    assert value['continuation_source']==source
    assert value['round14_rekey'] is False
    with pytest.raises(AssertionError):R.make_protocol(5,29,dict(source,update=1280),recipe)
    with pytest.raises(AssertionError):R.make_protocol(5,29,source,dict(recipe,lr=1e-4))
