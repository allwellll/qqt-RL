from tests.external_artifacts import requires_artifacts
"""Queue and process identity checks without starting any GPU jobs."""
import ast
from pathlib import Path
import json


@requires_artifacts('runs/combat_mix_v1/PROTOCOL.json', 'runs/combat_mix_v1/USER_PBT_QUEUE_OVERRIDE.json')
def test_latest_authorization_removes_only_unstarted_pair_completion_runs():
    p=json.loads(Path('runs/combat_mix_v1/PROTOCOL.json').read_text())
    assert len(p['arms'])==5
    override=json.loads(Path('runs/combat_mix_v1/USER_PBT_QUEUE_OVERRIDE.json').read_text())
    expected={f'{arm}_s{seed}' for seed in p['seeds'] for arm in p['arms']}
    expected-=set(override['cancelled_not_launched'])
    assert len(p['queue_order'])==len(set(p['queue_order']))
    assert set(p['queue_order'])==expected
    assert set(override['already_running_to_fixed_u320'])<=expected
    for name in p['queue_order']:
        assert Path('runs/combat_mix_v1/protocols',name+'.json').exists()


@requires_artifacts('runs/combat_mix_v1/AUX_WATCHDOG.py')
def test_aux_watchdog_process_names_match_every_lease_and_module():
    path=Path('runs/combat_mix_v1/AUX_WATCHDOG.py')
    tree=ast.parse(path.read_text());values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('jobs','leases'):
            values[node.targets[0].id]=ast.literal_eval(node.value)
    assert set(x[0] for x in values['jobs'])==set(values['leases'])
    for needle,args,_ in values['jobs']:
        assert needle in args


def test_gpu_owner_requires_atomic_terminal_hash_and_honors_pause_not_just_live_process(tmp_path,monkeypatch):
    source=Path('scripts/combat_mix_owner.py').read_text()
    assert "sha(cp/'state.pkl')==marker['state_sha256']" in source
    assert "sha(cp/'policy.pt')==marker['policy_sha256']" in source
    assert "holds=paused()" in source and 'if not pending or holds:' in source
    assert "stage='two_update_smoke'" in source and "stage='fresh_restore'" in source
    from scripts import combat_mix_owner as owner
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    monkeypatch.setattr(owner,'BASE',tmp_path/'runs/combat_mix_v1')
    owner.BASE.mkdir(parents=True)
    assert owner.paused()==[]
    (owner.BASE/'USER_PAUSE.json').write_text('{}')
    assert owner.paused()==['runs/combat_mix_v1/USER_PAUSE.json']


def test_notification_accepts_terminal_receipt_timestamp(tmp_path,monkeypatch):
    from scripts import combat_mix_owner as owner
    monkeypatch.setattr(owner,'BASE',tmp_path)
    class Result:returncode=0
    monkeypatch.setattr(owner.subprocess,'run',lambda *a,**k:Result())
    owner.notify('terminal','fixed run complete',dict(unix=1,passed=True))
    result=json.loads((tmp_path/'notifications/terminal.json').read_text())
    assert result['passed'] and result['unix']>1
