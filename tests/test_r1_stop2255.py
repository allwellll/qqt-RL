import json
from unittest.mock import patch
import pytest
from scripts import bomb_threat_r1_stop2255 as m


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'BASE', tmp_path / 'base')
    monkeypatch.setattr(m, 'OUT', tmp_path / 'stop')
    (tmp_path / 'decision.md').write_text('freeze current chunk')
    m.atomic_json(m.OUT / 'slot04.json', dict(status='AUTHORIZED',
        boundary_run='current', mode='COMPLETE_CURRENT_CHUNK', instruction='decision.md',
        instruction_sha256=m.sha(tmp_path / 'decision.md')))
    return tmp_path


def test_inflight_and_recovery_allowed(gate):
    assert not m.consume(4, dict(run='current'), {'current': 1})
    assert not m.consume(4, dict(run='current'), {})


def test_all_successor_paths_blocked(gate):
    assert m.consume(4, dict(run='next'), {})
    assert m.consume(4, None, {})


def test_complete_requires_verified_full_checkpoint(gate):
    cp=gate / 'checkpoint'; cp.mkdir()
    (cp / 'state.pkl').write_bytes(b'full state')
    (cp / 'policy.pt').write_bytes(b'policy')
    done=dict(passed=True, update=4240, checkpoint='checkpoint',
        state_sha256=m.sha(cp / 'state.pkl'), policy_sha256=m.sha(cp / 'policy.pt'))
    m.atomic_json(cp / 'COMMITTED.json', done)
    m.atomic_json(m.BASE / 'formal/current/COMPLETE.json', done)
    m.atomic_json(m.BASE / 'protocols/current.json', dict(chunk_start_update=3600,chunk_updates=640))
    with patch.object(m, 'pin'), patch('scripts.iter_pbt_owner.notify'):
        assert m.consume(4, dict(run='current'), {})
        assert (m.OUT / 'slot04_STOPPED.json').exists()
        (m.OUT / 'slot04_STOPPED.json').unlink()
        (cp / 'state.pkl').write_bytes(b'corrupt')
        with pytest.raises(AssertionError):m.consume(4, dict(run='current'), {})


def test_failed_run_retired(gate):
    p=m.OUT / 'slot04.json'; r=json.loads(p.read_text());r['mode']='RETIRE_FAILED'
    m.atomic_json(p,r)
    assert m.consume(4, dict(run='current'), {})


def test_retirement_never_blocks_other_slots(gate):
    assert m.consume(4, dict(run='next'), {})
    receipt=json.loads((m.OUT/'slot04_CONSUMED.json').read_text())
    assert receipt['block_this_slot_successors']
    assert not receipt['block_all_other_dispatch']
    for slot in (0,5,7):
        assert not m.consume(slot, dict(run='current'), {})
