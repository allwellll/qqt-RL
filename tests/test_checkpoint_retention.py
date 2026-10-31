import json
import pickle
from pathlib import Path

import pytest

from scripts import checkpoint_retention as R
from scripts.checkpoint_policy_artifact import verify_policy, load_actor
from scripts.population_run_state import sha


def cp(root, run, u, real=True, scope='bomb_threat_v1'):
    p=root/'runs'/scope/'formal'/run/f'u{u:06d}';p.mkdir(parents=True)
    (p/'state.pkl').write_bytes(b'optimizer/env/RNG')
    (p/'policy.pt').write_bytes(pickle.dumps({'weight':[1.,2.]}))
    marker=dict(passed=True,update=u,state_sha256=sha(p/'state.pkl'),policy_sha256=sha(p/'policy.pt'),
        metrics=dict(finite=real,update_l2=dict(actor=1 if real else 0,critic=1 if real else 0)))
    (p/'COMMITTED.json').write_text(json.dumps(marker));return p


def protocol(root,name,source):
    b=root/'runs/bomb_threat_v1';(b/'protocols').mkdir(parents=True,exist_ok=True)
    d=dict(namespace='runs/bomb_threat_v1/formal',arms={name:{}},sources={'1':{'checkpoint':str(source.relative_to(root))}})
    (b/'protocols'/f'{name}.json').write_text(json.dumps(d))
    (b/'SLOTS.json').write_text(json.dumps({'0':{'run':name}}))


def test_latest_and_unconsumed_source_then_release_source(tmp_path):
    src=cp(tmp_path,'parent',40);fresh=cp(tmp_path,'live',40,False);protocol(tmp_path,'live',src)
    p=R.build_plan(tmp_path,[],set());assert not p['candidates']
    latest=cp(tmp_path,'live',80)
    p=R.build_plan(tmp_path,[],set());assert set(r['path'] for r in p['candidates'])=={str((x/'state.pkl').relative_to(tmp_path)) for x in (src,fresh)}
    assert str((latest/'state.pkl').relative_to(tmp_path)) in p['protected']


def test_round14_donor_and_replaced_slot_persist(tmp_path):
    donor=cp(tmp_path,'donor',640);old=cp(tmp_path,'slot07old',1280)
    live=cp(tmp_path,'live',680);protocol(tmp_path,'live',donor)
    auth=tmp_path/'runs/bomb_threat_v1/iter2/a3sr/round14/AUTHORIZATION.json'
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps(dict(status='AUTHORIZED',donor={'checkpoint':str(donor.relative_to(tmp_path))},slot07_boundary_run='slot07old')))
    p=R.build_plan(tmp_path,[],set())
    for c in (donor,old,live):assert str((c/'state.pkl').relative_to(tmp_path)) in p['protected']
    assert not p['candidates']


def test_prune_only_state_and_recheck_newly_protected_file(tmp_path):
    old=cp(tmp_path,'retired',40,scope='iter_pbt_v1');other=tmp_path/'runs/untouched/state.pkl';other.parent.mkdir(parents=True);other.write_bytes(b'keep')
    plan=R.build_plan(tmp_path,[],set());key=plan['candidates'][0]['path']
    held=lambda root:R.build_plan(root,[],{key})
    assert R.apply_plan(plan,tmp_path,held)['deleted_count']==0
    result=R.apply_plan(plan,tmp_path,lambda root:R.build_plan(root,[],set()))
    assert result['deleted_count']==1 and other.exists()
    assert (old/'policy.pt').exists() and (old/'COMMITTED.json').exists()


def test_policy_verifies_without_state_and_rejects_corruption(tmp_path):
    p=cp(tmp_path,'retired',40);(p/'state.pkl').unlink()
    assert load_actor(p)=={'weight':[1.,2.]};assert verify_policy(p)['update']==40
    (p/'policy.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Policy artifact'):verify_policy(p)


def test_reject_wrong_filename_and_symlink(tmp_path):
    p=cp(tmp_path,'retired',40);plan=R.build_plan(tmp_path,[],set());plan['candidates'][0]['path']=str((p/'policy.pt').relative_to(tmp_path))
    with pytest.raises(ValueError):R.apply_plan(plan,tmp_path,lambda root:R.build_plan(root,[],set()))
    (p/'state.pkl').unlink();(p/'state.pkl').symlink_to(p/'policy.pt')
    with pytest.raises(ValueError,match='symbolic'):R.build_plan(tmp_path,[],set())


def test_owner_recovery_preserves_remaining_budget_and_latest_atomic(tmp_path):
    from scripts.bomb_threat_owner import retention_recovery_source
    src=cp(tmp_path,'old',40);latest=cp(tmp_path,'live',80)
    (latest.parent/'NEXT_STEP_u000080.json').write_text('{}')
    cm=json.loads((latest/'COMMITTED.json').read_text());cm['checkpoint']=str(latest.relative_to(tmp_path))
    (latest/'COMMITTED.json').write_text(json.dumps(cm))
    proto=dict(sources={'1':{'checkpoint':str(src.relative_to(tmp_path))}},environment_steps_per_run=360*262144)
    recovered,remaining=retention_recovery_source(proto,latest.parent,tmp_path)
    assert remaining==280 and recovered['update']==80  # no source-delete/import race
    (src/'state.pkl').unlink()
    recovered,remaining=retention_recovery_source(proto,latest.parent,tmp_path)
    assert remaining==280 and recovered['update']==80
    (latest.parent/'NEXT_STEP_u000080.json').unlink()
    with pytest.raises(ValueError,match='next-step'):retention_recovery_source(proto,latest.parent,tmp_path)
