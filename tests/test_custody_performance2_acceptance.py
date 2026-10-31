import json
import pytest
from scripts.custody_performance2_acceptance import update_rows, warm_rows, verify_file, first_atomic
from scripts.population_run_state import sha


def test_post_control_excludes_crossing_cycle_and_cold_updates():
    history = [dict(new_update=i, unix=100+i*10, cycle_seconds=12) for i in range(1, 11)]
    # Activation155: update6 ends160 but started148, and cannot be counted.
    assert [r['new_update'] for r in warm_rows(history, 0, activated=155)] == [7, 8, 9, 10]
    assert [r['new_update'] for r in warm_rows(history, 7)] == [10]


def test_pending_tail_does_not_hide_completed_corruption(tmp_path):
    path = tmp_path/'updates.jsonl'
    path.write_text('{"new_update": 1}\n{"new_')
    assert update_rows(path) == [dict(new_update=1)]
    path.write_text('{"new_update": 1}\n{"new_\n')
    with pytest.raises(json.JSONDecodeError):
        update_rows(path)


def test_cached_hash_rechecks_changed_checkpoint(tmp_path):
    path = tmp_path/'state.pkl'
    path.write_bytes(b'complete')
    expected = sha(path)
    verify_file(path, expected)
    verify_file(path, expected)
    path.write_bytes(b'corrupt-state')
    with pytest.raises(AssertionError):
        verify_file(path, expected)


def test_first_update_requires_recovery_token_and_nonzero_optimizer(tmp_path):
    restore = dict.fromkeys(('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'),True)
    (tmp_path/'RESTORE.json').write_text(json.dumps(restore))
    cp=tmp_path/'u000011';cp.mkdir()
    for name in ('state.pkl','policy.pt'):(cp/name).write_bytes(b'full-state')
    committed=dict(state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'))
    (cp/'COMMITTED.json').write_text(json.dumps(committed))
    row=dict(new_update=11,finite=True,update_l2=dict(actor=.2,critic=.3),actual_environment_steps_this_update=262144)
    (tmp_path/'updates.jsonl').write_text(json.dumps(row)+'\n')
    assert first_atomic(tmp_path,10,'RESTORE.json') is None
    (tmp_path/'NEXT_STEP_u000011.json').write_text('{}')
    assert first_atomic(tmp_path,10,'RESTORE.json')['finite']
    row['update_l2']['actor']=0
    (tmp_path/'updates.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(AssertionError):first_atomic(tmp_path,10,'RESTORE.json')
