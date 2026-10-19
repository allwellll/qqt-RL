"""Successor-only installer for joint DAgger replay stage allocation."""
import ast
import json
from pathlib import Path


def patched(source):
    old='    dagger.aggregate=joint_replay.aggregate'
    new='    from scripts import phase_balanced_joint_replay as phase_balanced\n    dagger.aggregate=phase_balanced.aggregate'
    if new in source:return source
    assert source.count(old)==1,'unrecognized joint replay dispatch'
    result=source.replace(old,new)
    ast.parse(result)
    return result


def patched_diagnostics(source):
    # The normalized successor already records the same collector keys once.
    if '**collector_evidence,' in source and '**normalized_diagnostic(' in source:
        return source
    anchor='            for dev,tick,env_index in np.argwhere(mismatch)[:16]:'
    import_line='            from scripts.phase_balanced_joint_replay import selected_ppo_diagnostic\n'
    old="                    hash_scope='learner observation and global state; native full-state hash requires Review replay',"
    new=old+'\n                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),'
    if new in source:return source
    assert source.count(anchor)==source.count(old)==1,'unrecognized host diagnostic layout'
    result=source.replace(anchor,import_line+anchor).replace(old,new)
    ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/je10_ops.py';old=path.read_text();new=patched(old)
    worker=root/'scripts/je10_worker.py';old_worker=worker.read_text();new_worker=patched_diagnostics(old_worker)
    if old!=new:path.write_text(new)
    if old_worker!=new_worker:worker.write_text(new_worker)
    return dict(changed_factor='Joint replay stage allocation within unchanged current/history halves',
        replay_allocation_only=True,reward_coefficient_changed=False,query_budget_changed=False,
        actor_observation_changed=False,buffer_additions_changed=False,buffer_capacity_changed=False,
        current_history_half_preserved=True,total_replay_per_device=64,
        per_half=dict(ordinary=20,ordinary_return=12,ordinary_drop=4,ordinary_ready_stock=4,navigation=10,opening=2),
        missing_group_policy='Redistribute within the same current/history half; ordinary-only choice exact legacy')
