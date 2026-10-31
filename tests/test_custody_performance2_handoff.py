import json
import pytest
from scripts.custody_performance2_switch import latest
from scripts.population_run_state import sha


def saved(folder, *, last_update=12, paused=True):
    cp = folder / 'u000012'
    cp.mkdir()
    (cp / 'state.pkl').write_bytes(b'complete-state-adam-env-aux-rng')
    (cp / 'policy.pt').write_bytes(b'policy')
    receipt = dict(update=12, checkpoint='neutral/u000012', metrics=dict(user_paused=paused),
        state_sha256=sha(cp / 'state.pkl'), policy_sha256=sha(cp / 'policy.pt'))
    (cp / 'COMMITTED.json').write_text(json.dumps(receipt))
    (folder / 'NEXT_STEP_u000012.json').write_text('{}')
    (folder / 'updates.jsonl').write_text(json.dumps(dict(new_update=last_update)) + '\n')
    return cp, receipt


def test_handoff_requires_every_effective_update_saved(tmp_path):
    saved(tmp_path, last_update=13)
    with pytest.raises(AssertionError, match='latest effective update not saved'):
        latest(tmp_path)


def test_handoff_rejects_corrupt_atomic_state(tmp_path):
    cp, _ = saved(tmp_path)
    (cp / 'state.pkl').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):
        latest(tmp_path)


def test_handoff_only_accepts_cooperative_atomic_receipt(tmp_path):
    cp, receipt = saved(tmp_path, paused=False)
    with pytest.raises(AssertionError, match='not a cooperative atomic pause'):
        latest(tmp_path)
    receipt['metrics']['user_paused'] = True
    (cp / 'COMMITTED.json').write_text(json.dumps(receipt))
    assert latest(tmp_path) == receipt
