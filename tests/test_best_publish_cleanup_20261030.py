import json
from pathlib import Path

from scripts import best_model_cleanup_20261030 as C
from scripts.best_publish_selection_20261030 import select


def tier(p1=10, p0=10, self_deaths=0):
    return dict(by_seat={'P0': {'win_minus_loss': p0}, 'P1': {'win_minus_loss': p1}}, self_deaths=self_deaths)


def test_selection_rejects_one_bad_tier_p0_and_self_damage():
    base = {t: tier() for t in ('hard', 'normal', 'easy')}
    for bad in ('tier', 'P0', 'self'):
        candidate = {t: tier(20) for t in base}
        if bad == 'tier':
            candidate['hard'] = tier(4)
        if bad == 'P0':
            candidate['hard'] = tier(20, -1)
        if bad == 'self':
            candidate['hard'] = tier(20, self_deaths=1)
        d = select(dict(baseline=base, candidate=candidate))
        assert d['winner'] == 'baseline' and not d['publish']
    assert select(dict(baseline=base, candidate={t: tier(11) for t in base}))['publish']
    assert not select(dict(baseline=base, candidate=base))['publish']


def test_export_requires_full_complete_modes_and_policy_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'ROOT', tmp_path)
    monkeypatch.setattr(C, 'B', tmp_path / 'runs/bomb_threat_v1')
    group = C.B / 'eval/example'
    cp = C.B / 'formal/run/u000040'
    cp.mkdir(parents=True)
    (group / 'model').mkdir(parents=True)
    (cp / 'policy.pt').write_bytes(b'original policy')
    h = C.sha(cp / 'policy.pt')
    (cp / 'COMMITTED.json').write_text(json.dumps(dict(policy_sha256=h)))
    (group / 'SUMMARY.json').write_text(json.dumps(dict(checkpoint=str(cp.relative_to(tmp_path)), policy_sha256=h)))
    complete = dict(passed=True, summary_sha256=C.sha(group / 'SUMMARY.json'), files={})
    for mode in ('greedy', 'categorical'):
        for start in range(0, 64, 8):
            name = f'natural_{mode}_{start:03d}.json.outcome.json'
            (group / name).write_text(json.dumps(dict(episodes=[dict(done=True) for _ in range(8)])))
            complete['files'][name] = C.sha(group / name)
    (group / 'COMPLETE.json').write_text(json.dumps(complete))
    (group / 'model/model.json').write_text(json.dumps(dict(meta=dict(checkpoint_sha256=h))))
    assert C.completed_export_group(group, {})['policy_sha256'] == h
    (group / 'natural_categorical_056.json.outcome.json').write_text(json.dumps(dict(episodes=[dict(done=False)] * 8)))
    assert C.completed_export_group(group, {}) is None
    (cp / 'policy.pt').write_bytes(b'corrupted')
    assert C.completed_export_group(group, {}) is None


def test_open_audit_blocks_on_lsof_error(monkeypatch):
    import subprocess
    import pytest
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess([], 1, '', 'permission denied'))
    with pytest.raises(RuntimeError, match='audit incomplete'):
        C.open_audit(['runs/bomb_threat_v1/formal/a/u000040/state.pkl'])
