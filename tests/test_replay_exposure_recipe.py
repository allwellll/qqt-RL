"""Passive provenance only, installed after the fixed parent releases GPUs."""
from pathlib import Path
import json
import pytest
from scripts.replay_exposure_recipe import install,patched_worker


def test_candidate_records_only_selected_replay_and_cannot_change_optimizer():
    old=Path('scripts/je10_worker.py').read_text();new=patched_worker(old)
    assert "summarize_replay_exposure(dagger_replay,g,row['dagger']['source_policy_digest'])" in new
    assert "DAGGER_REPLAY_EXPOSURE_" in new
    assert 'diagnostic_error' in new
    assert patched_worker(new)==new
    assert new.count('params,adam,ema,loss=learn(')==old.count('params,adam,ema,loss=learn(')==1
    assert 'new_buffer,dagger_replay,dagger_stats=aggregate(' in new


def test_passive_hook_preserves_update_checker_across_two_generations(monkeypatch):
    from scripts import replay_exposure
    calls=[]
    monkeypatch.setattr(replay_exposure,'summarize',
        lambda replay,generation,digest:dict(generation=generation,digest=digest))
    patched=patched_worker(Path('scripts/je10_worker.py').read_text())
    start=patched.index('        # Passive selected DAgger payload provenance.')
    end=patched.index("        assert row['dagger']['noneligible_label']==0",start)
    hook=patched[start:end]
    source='''def exercise():
    summarize=lambda params,previous: calls.append((params,previous)) or (1.,True)
    RUN=Path('.')
    row={'dagger':{'source_policy_digest':'raw-policy'}}
    dagger_replay=object()
    for g in [1,2]:
        norms,finite=summarize('params','previous')
'''+hook
    written=[]
    namespace=dict(Path=Path,calls=calls,atomic_json=lambda path,data:written.append(data))
    exec(compile(source,'passive-hook-two-generations','exec'),namespace)
    namespace['exercise']()
    assert calls==[('params','previous'),('params','previous')]
    assert [row['generation'] for row in written]==[1,2]


def test_diagnostic_and_failure_marker_write_errors_cannot_stop_updates(monkeypatch):
    from scripts import replay_exposure
    monkeypatch.setattr(replay_exposure,'summarize',lambda *args: {})
    patched=patched_worker(Path('scripts/je10_worker.py').read_text())
    start=patched.index('        # Passive selected DAgger payload provenance.')
    end=patched.index("        assert row['dagger']['noneligible_label']==0",start)
    source='''def exercise():
    summarize=lambda *args: calls.append('update') or (1.,True)
    RUN=Path('.')
    row={'dagger':{'source_policy_digest':'raw-policy'}}
    dagger_replay=object()
    for g in [1,2]:
        norms,finite=summarize('params','previous')
'''+patched[start:end]
    def fail_write(*args):raise OSError('diagnostic directory unavailable')
    calls=[]
    namespace=dict(Path=Path,calls=calls,atomic_json=fail_write)
    exec(compile(source,'passive-hook-write-failure','exec'),namespace)
    namespace['exercise']()
    assert calls==['update','update']


def test_install_requires_complete_parent_and_eight_card_release(tmp_path):
    root=tmp_path/'repo';(root/'scripts').mkdir(parents=True)
    worker=root/'scripts/je10_worker.py';worker.write_text(Path('scripts/je10_worker.py').read_text())
    base=root/'runs/new';parent=base/'formal/arm/gen000128';parent.mkdir(parents=True)
    terminal=parent.parent/'TRAINER_EXIT.json';release=base/'GPU_RELEASE_RECEIPT.json'
    terminal.write_text(json.dumps(dict(generation=127,passed=True,atomic_stop=False)))
    release.write_text(json.dumps(dict(released=True,compute_clients=0,devices=list(range(8)))))
    old=worker.read_text()
    with pytest.raises(AssertionError):install(root,parent)
    assert worker.read_text()==old
    terminal.write_text(json.dumps(dict(generation=128,passed=True,atomic_stop=False)))
    release.write_text(json.dumps(dict(released=False,compute_clients=1,devices=list(range(8)))))
    with pytest.raises(AssertionError):install(root,parent)
    assert worker.read_text()==old
    release.write_text(json.dumps(dict(released=True,compute_clients=0,devices=list(range(8)))))
    result=install(root,parent)
    assert result['passive_only'] and result['optimizer_steps_added']==0
    assert result['labels_queries_loss_rng_unchanged']
    assert worker.read_text()==patched_worker(old)


def test_nested_relay_plan_and_fallback_preserve_passive_provenance():
    import inspect
    from scripts.je10_supervise import choose_recipe,predeclare_continuity,supervise
    plan=dict(combat_fraction=.35,bridge_fraction=.2,return_dagger_weight=True)
    proposal=dict(recipe='return_dagger_weight',replay_exposure=True,
        reason='Need actual row exposure to distinguish FIFO presence from loss use',
        changed_factor='No new numerical factor; passive row provenance only')
    decision=choose_recipe(plan,[],proposal)
    assert decision['replay_exposure']
    assert choose_recipe(dict(plan,replay_exposure=True),[],None)['replay_exposure']
    source=inspect.getsource(supervise)
    assert "decision.get('proposal',{}).get('replay_exposure')" in source
    assert source.index('replay_exposure=install_replay_exposure(')<source.index('c.prepare(')
    assert "plan['replay_exposure']=replay_exposure" in source
    assert "declaration['replay_exposure']=replay_exposure" in source
    assert "'scripts/replay_exposure.py','scripts/replay_exposure_recipe.py'" in source
    assert 'replay_exposure' in inspect.getsource(predeclare_continuity)
