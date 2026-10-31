import pytest
from scripts import bomb_threat_round15 as R
from scripts.population_run_state import atomic_json


def episodes(wins=29, losses=0, zeros=2):
    return [dict(seed=2026122900+i, seat=i%2, done=True,
                 winner=i%2 if i<wins else 1-i%2 if i<wins+losses else None,
                 placed=[0,1] if i<zeros and i%2==0 else [1,0] if i<zeros else [1,1])
            for i in range(64)]


def test_admission_uses_loss_and_unilateral_zero():
    assert R.admission(episodes())['eligible']
    score=R.admission(episodes(25,4,5))
    assert score['win_minus_loss']==21 and score['unilateral_zero']==5
    assert not score['eligible']
    assert not R.admission(episodes(29,5,1))['eligible']
    with pytest.raises(AssertionError):R.admission(episodes()[:-1])


def test_non_target_slot_and_already_copied_are_untouched(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'AUTH',tmp_path/'auth.json')
    atomic_json(R.AUTH,dict(slot07_boundary_run='old'))
    assert R.apply_switch(5,dict(run='old'),{},1) is None
    assert R.apply_switch(7,dict(run='other'),{},1) is None
    assert R.apply_switch(7,dict(run='old',round15_applied=True),{},1) is None


def test_new_chunk_preserves_recipe_and_exact_source(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'ROOT',tmp_path)
    monkeypatch.setattr(R,'BASE',tmp_path/'runs/bomb_threat_v1')
    monkeypatch.setattr(R,'AUTH',R.BASE/'auth.json')
    recipe=dict(iter2_arm='A3SR',lr=7.5e-5,learner_seat_fraction=.5)
    source=dict(checkpoint='runs/bomb_threat_v1/formal/donor/u001120',update=1120,
                state_sha256='state',policy_sha256='policy')
    atomic_json(R.BASE/'protocols/donor.json',dict(arms={'donor':recipe},source_sha256={}))
    auth=dict(donor=source,target_seed=2026112907,collector_seed=2026112905,
              chunk_updates=320,rng='parity then rekey',copied=['Actor','Critic','Adam','env','aux','ring'])
    path=R.make_protocol(auth,29);p=R.read(path)
    assert p['continuation_source']==source and p['chunk_start_update']==1120
    assert p['chunk_updates']==320 and p['environment_steps_per_run']==1440*262144
    assert p['arms'][path.stem]==recipe and p['round14_rekey'] is True
    assert p['round14_collector_seed']==2026112905
    assert p['worker_module']=='scripts.bomb_threat_a3sr_resume_train'
    assert R.make_protocol(auth,29)==path
    with pytest.raises(AssertionError):R.make_protocol(dict(auth,donor=dict(source,update=1160)),29)


def test_terminal_does_not_apply_old1280_cap():
    assert R.terminal(dict(round14_applied=True),dict(update=1280)) is False


def test_future_pin_survives_until_checkpoint_exists(tmp_path,monkeypatch):
    from scripts import checkpoint_retention as retention
    monkeypatch.setattr(R,'BASE',tmp_path/'runs/bomb_threat_v1')
    name='runs/bomb_threat_v1/formal/donor/u001160'
    R.pin([name], 'protect before creation')
    R.pin([name], 'idempotent')
    assert len(R.read(R.BASE/'STATE_RETENTION_PINS.json')['checkpoints'])==1
    cp=tmp_path/name;cp.mkdir(parents=True)
    (cp/'state.pkl').write_bytes(b'complete state fixture')
    plan=retention.build_plan(tmp_path,[],set())
    assert name+'/state.pkl' in plan['protected']


def test_latest_admitted_source_can_advance_only_before_handoff(tmp_path,monkeypatch):
    from scripts.population_run_state import sha
    from scripts import iter_pbt_owner
    monkeypatch.setattr(iter_pbt_owner,'notify',lambda *a,**kw:None)
    monkeypatch.setattr(R,'ROOT',tmp_path)
    monkeypatch.setattr(R,'BASE',tmp_path/'runs/bomb_threat_v1')
    monkeypatch.setattr(R,'OUT',R.BASE/'round15')
    monkeypatch.setattr(R,'AUTH',R.OUT/'AUTHORIZATION.json')
    auth=dict(donor_run='donor',candidates=[])
    def candidate(update,wins=29):
        cp=R.BASE/'formal/donor'/f'u{update:06d}';cp.mkdir(parents=True)
        (cp/'state.pkl').write_bytes(b'complete state');(cp/'policy.pt').write_bytes(b'actor')
        name=str(cp.relative_to(tmp_path));auth['candidates'].append(name)
        cm=dict(checkpoint=name,update=update,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),
                metrics=dict(finite=True,update_l2=dict(actor=1,critic=1)))
        atomic_json(cp/'COMMITTED.json',cm)
        out=R.BASE/'eval'/f'donor_u{update:06d}';(out/'model').mkdir(parents=True)
        (out/'model/model.json').write_bytes(b'export fixture')
        rows=episodes(wins=wins)
        for i in range(0,64,8):
            p=out/f'natural_greedy_{i:03d}.json'
            atomic_json(p,dict(model_json_sha256=sha(out/'model/model.json'),assist=False,
                results=dict(natural_full2400=dict(episodes=[dict(seed=r['seed'],team=r['seat'],ticks=2400) for r in rows[i:i+8]]))))
            atomic_json(p.with_name(p.name+'.outcome.json'),dict(rule_version='stock-window-v1',teacher=False,
                episodes=[dict(r,ticks=2400) for r in rows[i:i+8]]))
    candidate(1080)
    assert R.select_donor(auth)['update']==1080
    candidate(1120)
    assert R.select_donor(auth)['update']==1120
    candidate(1160,wins=20)
    assert R.select_donor(auth)['update']==1120
    atomic_json(R.OUT/'SLOT07_HANDOFF.json',dict(status='PROTOCOL_READY'))
    candidate(1200)
    assert R.select_donor(auth)['update']==1120
