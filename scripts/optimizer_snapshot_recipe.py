"""Terminal-only passive actual optimizer input evidence installation."""
import ast
import json
from pathlib import Path


def patched(source):
    old='        roll_seconds=time.time()-start;t=time.time()\n'
    block='''        if g in p.get('optimizer_input_snapshot',{}).get('points',[]):
            try:
                from scripts.optimizer_input_snapshot import capture
                capture(RUN,g,live['params'],live['adam'],batch,key,dagger_replay,mb)
            except Exception as error:
                atomic_json(RUN/'diagnostics'/f'OPTIMIZER_INPUT_FAILURE_{g:06d}.json',dict(error_type=type(error).__name__,controls_training=False,training_continues=True))
'''
    if block in source:return source
    assert source.count(old)==1,'unrecognized pre-update snapshot location'
    result=source.replace(old,block+old);ast.parse(result);return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text());release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert terminal['passed'] and terminal['generation']==128 and not terminal['atomic_stop']
    assert release['released'] and release['compute_clients']==0 and set(release['devices'])==set(range(8))
    target=root/'scripts/je09_stage_h5c_worker.py';before=target.read_text();after=patched(before)
    if before!=after:target.write_text(after)
    return dict(enabled=True,points=[4,16,64,128],first_optimizer_step_only=True,
        numerical_factor_changed=False,sampling_labels_rng_loss_optimizer_unchanged=True,
        records='Exact recorded raw first Actor/Critic permutation inputs, full raw GAE, pre-update raw parameters/Adam and native replay',
        snapshot_failure_blocks_training=False,capability_gate=False)
