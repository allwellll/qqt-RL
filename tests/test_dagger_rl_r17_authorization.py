"""Bounded successor cannot bypass complete evidence, parent identity or stop."""
import pytest
from scripts import dagger_rl_r17_run as M
from scripts.population_run_state import atomic_json, sha


def fixture(tmp_path,monkeypatch):
    b=tmp_path/'new';prior=tmp_path/'prior';b.mkdir();prior.mkdir()
    monkeypatch.setattr(M,'R',tmp_path);monkeypatch.setattr(M,'B',b);monkeypatch.setattr(M,'PRIOR',prior)
    parent=b/'inputs/parent.pkl';parent.parent.mkdir();parent.write_bytes(b'common parent Adam')
    pause=tmp_path/'runs/retention_20261021/USER_GPU_PAUSE.json';atomic_json(pause,dict(paused=True))
    guard=tmp_path/'scripts/dagger_rl_route_goal_eligibility.py';guard.parent.mkdir();guard.write_text('goal_kind')
    p=dict(arms=['sequential_anchor'],paired_seeds=[2026102661,2026102662,2026102663],
        updates=32,switch_update=16,total_environment_steps=1572864,total_optimizer_steps=3072,
        goal_kind_guard=True,whole_legacy_active_guard=False,parent_input_sha256=sha(parent),goal_guard_source_sha256=sha(guard))
    atomic_json(prior/'PROTOCOL.json',p)
    for n in ['GPU_COMPLETE.json','COMPLETE_EVIDENCE_CONSUMED.json','SUPPLEMENTAL_EVIDENCE_CONSUMED.json',
              'TRAIN_PROJECTION_AUDIT.json','GOAL0_REPLAY_WITNESS_v2.json','REAL_GOAL_GUARD_CHECK.json',
              'GOAL_KIND_GUARD_PREPARATION.json','FINAL_COMPARISON.json']:
        atomic_json(prior/n,dict(passed=True,kind_only=True))
    atomic_json(prior/'FIXED_DEV_DIRECTION_AUDIT_v2.json',dict(completed=19))
    atomic_json(b/'MATCHED_CONTROLS.json',dict(controls=[]))
    def register():
        atomic_json(b/'PROTOCOL.json',p)
        atomic_json(b/'SCOPED_GPU_AUTHORIZATION.json',dict(authorized_by='explicit_user_instruction',
            old_continuous_resume=False,protocol_sha256=sha(b/'PROTOCOL.json'),old_global_pause_sha256=sha(pause)))
        atomic_json(b/'SCIENTIFIC_SELECTION.json',dict(execute_goal_kind=True,protocol_sha256=sha(b/'PROTOCOL.json'),
            R16_comparison_sha256=sha(prior/'FINAL_COMPARISON.json'),
            R16_supplement_sha256=sha(prior/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json'),
            matched_controls_sha256=sha(b/'MATCHED_CONTROLS.json')))
    register();M.authorize()
    return b,prior,p,register


def test_parent_tampering_rejected(tmp_path,monkeypatch):
    b,_,_,_=fixture(tmp_path,monkeypatch)
    (b/'inputs/parent.pkl').write_bytes(b'wrong')
    with pytest.raises(AssertionError):M.authorize()


@pytest.mark.parametrize('field,value',[('whole_legacy_active_guard',True),('goal_kind_guard',False),
    ('arms',['joint_anchor']),('total_environment_steps',1572865),('switch_update',17)])
def test_wrong_intervention_or_budget_rejected(tmp_path,monkeypatch,field,value):
    _,_,p,register=fixture(tmp_path,monkeypatch);p[field]=value;register()
    with pytest.raises(AssertionError):M.authorize()


def test_prior_stop_also_blocks_successor(tmp_path,monkeypatch):
    _,prior,_,_=fixture(tmp_path,monkeypatch);atomic_json(prior/'USER_STOP.json',dict(paused=True))
    with pytest.raises(M.UserStopped):M.authorize()


def test_incomplete_fixed_evidence_rejected(tmp_path,monkeypatch):
    _,prior,_,_=fixture(tmp_path,monkeypatch)
    atomic_json(prior/'FIXED_DEV_DIRECTION_AUDIT_v2.json',dict(completed=18))
    with pytest.raises(AssertionError):M.authorize()
