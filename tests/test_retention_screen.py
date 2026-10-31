import pytest
from scripts import retention_screen as screen


@pytest.fixture(autouse=True)
def isolated_user_pause_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, 'ROOT', tmp_path)


def test_gpu_contract_never_accepts_other_compute_clients_or_activity():
    empty=[(i,0,0) for i in range(8)]
    assert screen.resource_idle(empty,[])
    assert not screen.resource_idle(empty,['unrelated'])
    assert not screen.resource_idle([(i,0,20) for i in range(8)],[])
    assert not screen.resource_idle(empty[:7],[])


def test_design_is_atomic_paired_same_parent_without_old_relay():
    design=screen.declaration()
    assert len(design['runs'])==6
    for seed in design['train_seeds']:
        pair=[r for r in design['runs'] if r['seed']==seed]
        assert [r['arm'] for r in pair]==['A0','A1']
        assert len({r['parent'] for r in pair})==1
        assert len({r['namespace'] for r in pair})==2
    assert design['points']==[0,8,16,24,32]
    assert design['generations']==32 and design['auto_relay'] is False


def test_nonzero_requires_every_actor_and_critic_replica():
    row=dict(finite=True,nonzero_update=True,generation=1,
        actor_update_l2=[.1]*8,critic_update_l2=[.2]*8)
    screen.validate_update(row)
    for field,value in [('actor_update_l2',[0.]*8),('critic_update_l2',[float('nan')]*8)]:
        with pytest.raises(ValueError):screen.validate_update(dict(row,**{field:value}))


def test_declared_eval_seeds_are_disjoint_and_no_synthetic_training():
    d=screen.declaration()
    assert not set(d['route_dev_seeds'])&set(d['route_test_seeds'])
    assert d['synthetic_training_carry'] is False
    assert d['return_multiplier']==2


def test_user_semantic_hold_blocks_owner_and_direct_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(screen,'BASE',tmp_path)
    (tmp_path/'SEMANTIC_AUDIT_HOLD.json').write_text('{}')
    with pytest.raises(RuntimeError,match='semantic audit hold'):screen.owner()
    with pytest.raises(RuntimeError,match='semantic audit hold'):
        screen.spawn_worker(tmp_path/'unused')


def test_excluded_run_cannot_be_retried(tmp_path,monkeypatch):
    monkeypatch.setattr(screen,'BASE',tmp_path)
    run=tmp_path/'excluded';run.mkdir()
    (run/'EXCLUDED_PENDING_SEMANTIC_AUDIT.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Excluded'):
        screen.spawn_worker(run)


def test_explicit_user_resume_marker_allows_only_new_unexcluded_namespace(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(screen,'BASE',tmp_path)
    marker=tmp_path/'SEMANTIC_AUDIT_HOLD.json'
    marker.write_text(json.dumps(dict(automatic_training_allowed=True,audit_complete=False)))
    with pytest.raises(RuntimeError):screen.require_training_authorized()
    marker.write_text(json.dumps(dict(automatic_training_allowed=True,audit_complete=True,
        status='AUDIT_COMPLETE_USER_AUTHORIZED_RESUME')))
    screen.require_training_authorized(tmp_path/'fresh')
    excluded=tmp_path/'excluded';excluded.mkdir();(excluded/'EXCLUDED_INVALID.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Excluded'):screen.require_training_authorized(excluded)
