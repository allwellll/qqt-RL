"""A future replay recipe must not mutate healthy run sources."""
import json
from pathlib import Path
import pytest


def source():return Path('scripts/je10_ops.py').read_text()


def test_patch_changes_only_aggregate_dispatch_and_is_idempotent():
    from scripts.phase_balanced_replay_recipe import patched
    installed='    from scripts import phase_balanced_joint_replay as phase_balanced\n    dagger.aggregate=phase_balanced.aggregate'
    # Exercise the historical input even when the live source is installed.
    old=source().replace(installed,'    dagger.aggregate=joint_replay.aggregate')
    new=patched(old)
    assert 'dagger.aggregate=phase_balanced.aggregate' in new
    assert 'dagger.replay_loss=joint_replay.replay_loss' in new
    assert patched(new)==new
    import ast
    a=ast.dump(ast.parse(old.replace('    dagger.aggregate=joint_replay.aggregate','')))
    b=ast.dump(ast.parse(new.replace('    from scripts import phase_balanced_joint_replay as phase_balanced\n    dagger.aggregate=phase_balanced.aggregate','')))
    assert a==b


def create_root(tmp_path,generation=128,released=True,atomic_stop=False):
    root=tmp_path/'root';path=root/'scripts/je10_ops.py';path.parent.mkdir(parents=True);path.write_text(source())
    (path.parent/'je10_worker.py').write_text(Path('scripts/je10_worker.py').read_text())
    parent=root/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=atomic_stop)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1)))
    return root,parent,path


@pytest.mark.parametrize('generation,released,stop',[(127,True,False),(128,False,False),(128,True,True)])
def test_install_rejects_nonterminal_busy_or_interrupted_parent(tmp_path,generation,released,stop):
    from scripts.phase_balanced_replay_recipe import install
    root,parent,path=create_root(tmp_path,generation,released,stop);old=path.read_text()
    with pytest.raises(AssertionError):install(root,parent)
    assert path.read_text()==old


def test_install_after_fixed_terminal_and_release_only(tmp_path):
    from scripts.phase_balanced_replay_recipe import install
    root,parent,path=create_root(tmp_path);result=install(root,parent)
    assert result['replay_allocation_only'] and not result['reward_coefficient_changed']
    assert not result['query_budget_changed'] and not result['actor_observation_changed']
    assert not result['buffer_additions_changed'] and not result['buffer_capacity_changed']
    assert result['current_history_half_preserved']
    assert 'dagger.aggregate=phase_balanced.aggregate' in path.read_text()


def test_worker_diagnostic_patch_only_adds_selected_critic_backed_gae():
    from scripts.phase_balanced_replay_recipe import patched_diagnostics
    old="""def record(batch, mismatch, failures):
            for dev,tick,env_index in np.argwhere(mismatch)[:16]:
                failures.append(dict(
                    hash_scope='learner observation and global state; native full-state hash requires Review replay',
                ))
"""
    new=patched_diagnostics(old)
    assert '**selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index)))' in new
    assert patched_diagnostics(new)==new
    import ast
    compile(new,'worker-diagnostic','exec')
    added='            from scripts.phase_balanced_joint_replay import selected_ppo_diagnostic\n'
    entry='                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),\n'
    assert new.replace(added,'').replace(entry,'')==old


def test_normalized_worker_already_has_collector_evidence_without_repatch():
    from scripts.phase_balanced_replay_recipe import patched_diagnostics
    current=Path('scripts/je10_worker.py').read_text()
    assert '**collector_evidence,' in current and '**normalized_diagnostic(' in current
    assert patched_diagnostics(current)==current
