import json
import pytest
from scripts import bomb_threat_r1_continue_schedule as S
from scripts.bomb_threat_r1_continue_train import worker_source


@pytest.mark.parametrize('preserved', [False, True])
def test_same_version_worker_has_no_migration_and_checks_donor(preserved):
    s = worker_source(preserved)
    compile(s, 'continuation', 'exec')
    assert 'exported=json.loads' not in s
    assert 'pre_revision=previous' not in s
    assert "same_version_next_step_exact=True" in s
    assert "digest((st,ax,keys))==expected['full_environment_digest']" in s
    assert "digest(P.one(adm))==expected['Adam_digest']" in s
    assert "row['exposure']['bank_seat_resets']" in s
    # All reinitialization belongs only to the unreachable no-source branch.
    body = s[s.index('    M.configure'):s.index('    params=jax.tree.map')]
    assert 'PRNGKey' not in body and 'M.fresh' not in body


def test_stop_precedes_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'BASE', tmp_path)
    monkeypatch.setattr(S, 'OUT', tmp_path / 'authorization')
    (tmp_path / 'USER_STOP.json').write_text('{}')
    assert S.apply_continue(0, {'r1_items_applied': True}, {}, 1) is None


def test_protocol_preserves_recipe_source_seal_and_seed(tmp_path, monkeypatch):
    root = tmp_path
    base = root / 'runs/bomb_threat_v1'
    monkeypatch.setattr(S, 'ROOT', root)
    monkeypatch.setattr(S, 'BASE', base)
    monkeypatch.setattr(S, 'OUT', base / 'continuation')
    monkeypatch.setattr(S, 'ITEMS', base / 'items')
    monkeypatch.setattr(S, 'pin', lambda *args: None)
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    d = root / S.DECISION
    d.parent.mkdir(); d.write_text('authorization')
    put(S.OUT / 'AUTHORIZATION.json', dict(status='AUTHORIZED', decision_sha256=S.sha(d)))
    worker = root / 'scripts/bomb_threat_r1_continue_train.py'
    worker.parent.mkdir();worker.write_text('worker')
    cp = base / 'formal/old/u002960'
    cp.mkdir(parents=True)
    (cp / 'state.pkl').write_bytes(b'fullstate')
    (cp / 'policy.pt').write_bytes(b'policy')
    done = dict(passed=True, update=2960, checkpoint=str(cp.relative_to(root)),
                state_sha256=S.sha(cp / 'state.pkl'), policy_sha256=S.sha(cp / 'policy.pt'))
    put(cp / 'COMMITTED.json', done)
    done = dict(done, complete=True)
    put(cp.parent / 'COMPLETE.json', done)
    put(cp.parent / 'NEXT_STEP_u002960.json', dict(digest='frozen'))
    recipe = dict(lr=7.5e-5, opponent_weights=[.05,.35,.30,.15,.10,.05])
    prior = dict(arms={'old':recipe}, r1_seed=123, source_sha256={},
                 chunk_start_update=2320, chunk_updates=640, reward_version='unchanged',
                 r1_preserved_legacy=True)
    put(base / 'protocols/old.json', prior)
    entry = dict(run='old', r1_items_applied=True, r1_preserved_legacy=True)
    result = S.apply_continue(3, entry, done, 38)
    p = S.read(base / 'protocols/bt_slot03_c000038.json')
    assert p['arms'][result['run']] == recipe and p['r1_seed'] == 123
    assert p['continuation_source'] == done
    assert p['chunk_start_update'] == 2960 and p['chunk_updates'] == 640
    assert p['environment_steps_per_run'] == 3600 * 262144
    assert p['reward_version'] == 'unchanged' and p['r1_preserved_legacy']
    assert not p['pbt_lineage']['RNG_rekeyed']
    # Restart at the same pending handoff cannot silently overwrite a protocol.
    again = S.apply_continue(3, entry, done, 38)
    assert again == result
    (base / 'STOP_FOR_REBOOT.json').write_text('{}')
    assert S.apply_continue(3, entry, done, 39) is None
