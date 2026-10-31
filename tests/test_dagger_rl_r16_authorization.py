"""Deny corrupted parent/unknown arms/budget drift before any GPU entry."""
import json
from pathlib import Path
import pytest
from scripts import dagger_rl_r16_run as M
from scripts.population_run_state import atomic_json, sha


def fixture(tmp_path, monkeypatch):
    root = tmp_path
    base = root/'new'
    base.mkdir()
    monkeypatch.setattr(M, 'R', root)
    monkeypatch.setattr(M, 'B', base)
    pause = root/'runs/retention_20261021/USER_GPU_PAUSE.json'
    atomic_json(pause, dict(paused=True))
    parent = base/'inputs/parent.pkl'
    parent.parent.mkdir()
    parent.write_bytes(b'clean common parent and Adam')
    prior = root/'runs/dagger_rl_20261024/r15_carry_mask'
    for name in ['COMPLETE_EVIDENCE_CONSUMED.json', 'SUPPLEMENTAL_EVIDENCE_CONSUMED.json',
                 'ZERO_STEP_REMAINING_AUDIT.json', 'GPU_COMPLETE.json', 'COMPLETE_EVIDENCE_SYNTHESIS.json']:
        atomic_json(prior/name, dict(passed=True))
    protocol = dict(arms=['ppo_anchor', 'sequential_anchor', 'joint_anchor'],
                    paired_seeds=[2026102661, 2026102662, 2026102663], updates=32, switch_update=16,
                    total_environment_steps=4718592, total_optimizer_steps=9216,
                    parent_input_sha256=sha(parent))

    def register():
        atomic_json(base/'PROTOCOL.json', protocol)
        atomic_json(base/'SCOPED_GPU_AUTHORIZATION.json', dict(
            authorized_by='explicit_user_instruction', old_continuous_resume=False,
            protocol_sha256=sha(base/'PROTOCOL.json'), old_global_pause_sha256=sha(pause)))
        atomic_json(base/'SCIENTIFIC_SELECTION.json', dict(
            execute_retention_schedule=True, R15_complete_evidence_sha256=sha(prior/'COMPLETE_EVIDENCE_SYNTHESIS.json'),
            R15_supplement_sha256=sha(prior/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json'),
            protocol_sha256=sha(base/'PROTOCOL.json')))
    register()
    M.authorize()
    return base, protocol, register


def test_parent_file_tampering_fails_before_GPU_setup(tmp_path, monkeypatch):
    base, _, _ = fixture(tmp_path, monkeypatch)
    (base/'inputs/parent.pkl').write_bytes(b'degraded or mismatched parent')
    with pytest.raises(AssertionError):
        M.authorize()


def test_unknown_same_size_arm_list_is_rejected(tmp_path, monkeypatch):
    _, p, register = fixture(tmp_path, monkeypatch)
    p['arms'][1] = 'unknown'
    register()
    with pytest.raises(AssertionError):
        M.authorize()


def test_changed_fixed_budget_is_rejected(tmp_path, monkeypatch):
    _, p, register = fixture(tmp_path, monkeypatch)
    p['total_environment_steps'] += 16384
    register()
    with pytest.raises(AssertionError):
        M.authorize()


def test_user_stop_overrides_valid_scoped_resume(tmp_path, monkeypatch):
    base, _, _ = fixture(tmp_path, monkeypatch)
    atomic_json(base/'USER_STOP.json', dict(paused=True))
    with pytest.raises(M.UserStopped):
        M.authorize()
