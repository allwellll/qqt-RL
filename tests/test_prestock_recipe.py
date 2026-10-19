"""Future terminal-only wiring must share targets and explicitly carry labels."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=['scripts/je10_ops.py','scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h6_ops.py',
       'scripts/je09_route_pbrs.py','scripts/je10_worker.py','scripts/je10_campaign.py']


def test_all_production_patches_compile_are_idempotent_and_leave_live_sources_untouched():
    from scripts.prestock_recipe import patched_sources
    before={f:(ROOT/f).read_text() for f in FILES}
    changed=patched_sources(before)
    assert set(changed)==set(before)
    for f,source in changed.items():
        ast.parse(source)  # Already-installed recipes must remain idempotent.
    assert patched_sources(changed)==changed
    assert before=={f:(ROOT/f).read_text() for f in FILES}


def test_collector_keeps_original_offsets_and_replay_optimizer_hooks_are_explicit():
    from scripts.prestock_recipe import patched_sources
    changed=patched_sources({f:(ROOT/f).read_text() for f in FILES})
    base=changed[FILES[1]];ops=changed[FILES[0]];h6=changed[FILES[2]];pbrs=changed[FILES[3]]
    assert 'route_labels[:, :12]' in base and 'route_labels[:, 12:]' in base
    assert base.index('context,trace,provenance')<base.index('route_labels[:, 12:]')
    assert base.index('update(states,aux')<base.index('obs=train.both_perspectives(states)')
    assert 'dagger.buffer=joint_replay.buffer' in ops
    assert ('dagger.aggregate=joint_replay.aggregate' in ops or
            'dagger.aggregate=phase_balanced.aggregate' in ops)
    assert 'dagger.replay_loss=joint_replay.replay_loss' in ops
    assert 'joint_mismatch' in h6 and 'first_deviations' in h6
    assert 'prestock_potential(states,aux[\'prestock\']' in pbrs
    assert 'prestock_potential(next_states,newaux[\'prestock\']' in pbrs
    assert 'newaux[\'prestock\']=prestock_update(next_states' in pbrs
    assert "worker.STATE_SCHEMA=read(Path(run)/'plan.json')['state_schema']" in changed['scripts/je10_campaign.py']


def test_install_rejects_nonterminal_or_unreleased_predecessor(tmp_path):
    from scripts.prestock_recipe import install
    import json,pytest
    base=tmp_path/'runs'/'parent';arm=base/'formal'/'arm';parent=arm/'gen000128';parent.mkdir(parents=True)
    for terminal,release in [(dict(generation=127,passed=True,atomic_stop=False),dict(released=True,compute_clients=0)),
        (dict(generation=128,passed=True,atomic_stop=False),dict(released=False,compute_clients=1))]:
        (arm/'TRAINER_EXIT.json').write_text(json.dumps(terminal));(base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(release))
        with pytest.raises(AssertionError):install(tmp_path,parent)


def test_terminal_restore_uses_run_plan_schema_and_is_added_to_successor_snapshot():
    from scripts.prestock_recipe import patched_sources
    files=FILES+['scripts/terminal_checkpoint_restore.py']
    before={f:(ROOT/f).read_text() for f in files}
    before['scripts/terminal_checkpoint_restore.py']=before['scripts/terminal_checkpoint_restore.py'].replace(
        "    verify_state_schema(proof,json.loads((base/'plan.json').read_text()))",
        "    assert proof['state_schema']=='je10-control-route-v38'")
    changed=patched_sources(before)
    assert "verify_state_schema(proof,json.loads((base/'plan.json').read_text()))" in changed['scripts/terminal_checkpoint_restore.py']
