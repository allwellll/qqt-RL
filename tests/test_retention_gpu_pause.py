"""Global user GPU hold overrides all local experiment authorizations."""
import json
import pytest
from scripts import retention_screen as engine
from scripts import retention_phase_screen as phase


@pytest.fixture
def paused(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'ROOT', tmp_path)
    monkeypatch.setattr(engine, 'BASE', tmp_path/'r4')
    engine.BASE.mkdir()
    (engine.BASE/'SEMANTIC_AUDIT_HOLD.json').write_text(json.dumps(dict(
        automatic_training_allowed=True, audit_complete=True,
        status='AUDIT_COMPLETE_USER_AUTHORIZED_RESUME')))
    marker=tmp_path/'runs/retention_20261021/USER_GPU_PAUSE.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(decision='pause_all_gpu_training_until_explicit_user_resume')))
    return marker


def test_global_pause_blocks_authorized_run_and_idle_wait(paused):
    with pytest.raises(RuntimeError, match='user GPU pause'):
        engine.require_training_authorized()
    with pytest.raises(RuntimeError, match='user GPU pause'):
        engine.wait_idle()


def test_global_pause_blocks_smoke_before_namespace_creation(paused, monkeypatch):
    run=engine.ROOT/'fresh'
    with pytest.raises(RuntimeError, match='user GPU pause'):
        engine.smoke(dict(namespace='fresh', arm='A0'))
    assert not run.exists()


def test_global_pause_blocks_r5_prepare_before_any_side_effect(paused, monkeypatch):
    monkeypatch.setattr(phase, 'BASE', engine.ROOT/'r5')
    with pytest.raises(RuntimeError, match='user GPU pause'):
        phase.prepare()
    assert not phase.BASE.exists()


@pytest.mark.parametrize('content', ['{}', '{invalid', '{"decision":"resume"}'])
def test_ambiguous_or_damaged_pause_marker_cannot_authorize_gpu(paused, content):
    paused.write_text(content)
    with pytest.raises(RuntimeError, match='user GPU pause'):
        engine.require_training_authorized()


def test_explicit_superseding_user_authorization_preserves_pause_evidence(paused):
    receipt=paused.parent/'USER_GPU_RESUME_20261021.json'
    receipt.write_text(json.dumps(dict(decision='resume_gpu_experiments',
        authorized_by='explicit_user_instruction',supersedes=str(paused.relative_to(engine.ROOT)),
        authorization_contract='.hermes/r5_authorized_resume_plan.md')))
    marker=json.loads(paused.read_text())
    marker.update(status='SUPERSEDED_BY_EXPLICIT_USER_RESUME',
        automatic_training_allowed=True,resume_receipt=str(receipt.relative_to(engine.ROOT)))
    paused.write_text(json.dumps(marker))
    engine.require_training_authorized()
    assert paused.exists() and receipt.exists()
    receipt.write_text('{}')
    with pytest.raises(RuntimeError,match='user GPU pause'):
        engine.require_training_authorized()
