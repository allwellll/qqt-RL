"""Uninstalled terminal-only candidate for verified return movement KL."""
import ast
import json
from pathlib import Path


def patched(source):
    anchor='            penalty+=jnp.mean(jnp.where(combat,strong,weak)*(kl[:,0]+kl[:,1]))'
    replacement="""            from scripts.native_return_retention import movement_scale,retained_kl
            scale=movement_scale(context[ii[:half]],s[:half],m[:half],b[:half])
            penalty+=retained_kl(kl[:,0],kl[:,1],jnp.where(combat,strong,weak),scale)"""
    if source.count(replacement)==1 and anchor not in source:
        return source
    assert source.count(anchor)==1,'unrecognized old-teacher PPO retention call'
    result=source.replace(anchor,replacement)
    ast.parse(result)
    return result


def patched_worker(source):
    anchor="        row['dagger']['learner_generation']=g-1"
    replacement=anchor+"""
        # Passive certified-return retention exposure, never controls training.
        if p.get('native_return_retention'):
            try:
                from scripts.native_return_retention import exposure_summary
                row['native_return_retention_exposure']=exposure_summary(ctx,np.asarray(batch[1]),
                    np.asarray(batch[6][0]),np.asarray(batch[6][1]))
            except Exception as diagnostic_error:
                row['native_return_retention_exposure']=dict(controls_training=False,
                    diagnostic_error=type(diagnostic_error).__name__)
"""
    if source.count(replacement)==1:return source
    assert source.count(anchor)==1,'unrecognized worker passive evidence location'
    result=source.replace(anchor,replacement)
    ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['passed'] and terminal['generation']==128 and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0 and set(release['devices'])==set(range(8))
    path=root/'scripts/je09_stage_h5c_ops.py'
    worker=root/'scripts/je10_worker.py'
    before=path.read_text();after=patched(before)
    worker_before=worker.read_text();worker_after=patched_worker(worker_before)
    if before!=after:path.write_text(after)
    if worker_before!=worker_after:worker.write_text(worker_after)
    return dict(enabled=True,
        changed_factor='Old teacher movement KL omitted only for certified safe native enemy-origin return rows',
        bomb_kl_unchanged=True,combat_danger_retention_unchanged=True,
        native_custody_and_current_legal_joint_certificate_required=True,
        denominator='Original PPO actor half, unchanged; zero movement contributions are not renormalized',
        parent_retention_unchanged=True,other_losses_reward_queries_fifo_unchanged=True,
        return_dagger_multiplier=2.,passive_actual_actor_rollout_exposure=True,capability_gate=False,
        causal_scope='Sequential successor candidate; same-input gradient evidence is not a full-update causal claim')
