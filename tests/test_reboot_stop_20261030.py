import pytest

from scripts import reboot_stop_20261030 as S


def test_scope_excludes_other_worktree_and_combat_mix():
    assert S.classify(['python', '-m', 'scripts.combat_mix_js_owner']) is None
    assert S.classify(['python', '-m', 'scripts.bomb_threat_owner'], False) is None
    assert S.classify(['node', 'other_project.js']) is None
    assert S.classify(['python', '-m', 'scripts.unrelated']) is None
    assert S.classify(['python', '-m', 'scripts.bomb_threat_r1_progress_train']) == 'training_worker'
    assert S.classify(['python', '-m', 'scripts.bomb_threat_r1_preserved_train']) == 'training_worker'
    assert S.classify(['node', 'scripts/je05_web_hunter.js', '--out', 'runs/bomb_threat_v1/eval/x/y.json']) == 'JS_evaluator'


def test_no_training_signal_and_revalidate_identity(monkeypatch):
    p = dict(pid=111, starttime='42', args=['python', '-m', 'scripts.bomb_threat_r1_progress_train'], role='training_worker')
    calls = []
    monkeypatch.setattr(S, 'processes', lambda: [p])
    monkeypatch.setattr(S.os, 'kill', lambda *args: calls.append(args))
    with pytest.raises(AssertionError, match='atomic pause'):
        S.terminate_identified(p)
    assert not calls
    stale = dict(p, role='maintenance', starttime='41')
    assert S.terminate_identified(stale) is None and not calls


def test_reboot_stop_requires_verified_publication(monkeypatch):
    monkeypatch.setattr(S, 'read', lambda *a: dict(status='INCOMPLETE'))
    with pytest.raises(AssertionError, match='must precede shutdown'):
        S.stop()
