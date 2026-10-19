"""Terminal-only installation of passive selected replay provenance."""
import ast
import json
from pathlib import Path


def patched_worker(source):
    marker='        # Passive selected DAgger payload provenance.'
    if marker in source:return source
    anchor="        assert row['dagger']['noneligible_label']==0"
    assert source.count(anchor)==1,'unrecognized DAgger statistics boundary'
    extra="""        # Passive selected DAgger payload provenance.
        try:
            from scripts.replay_exposure import summarize as summarize_replay_exposure
            atomic_json(RUN/'diagnostics'/f'DAGGER_REPLAY_EXPOSURE_{g:06d}.json',
                summarize_replay_exposure(dagger_replay,g,row['dagger']['source_policy_digest']))
        except Exception as diagnostic_error:
            try:
                atomic_json(RUN/'diagnostics'/f'DAGGER_REPLAY_EXPOSURE_FAILURE_{g:06d}.json',
                    dict(diagnostic_error=type(diagnostic_error).__name__,controls_training=False,optimizer_steps_added=0))
            except Exception:
                print('Passive replay provenance write failed; training continues.',flush=True)
"""
    result=source.replace(anchor,extra+anchor);ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0 and set(release['devices'])==set(range(8))
    worker=root/'scripts/je10_worker.py';before=worker.read_text();after=patched_worker(before)
    if after!=before:worker.write_text(after)
    return dict(enabled=True,passive_only=True,optimizer_steps_added=0,
        labels_queries_loss_rng_unchanged=True,
        payload='Only already selected current/history replay rows; hashes, labels and source generation',
        failure_policy='Diagnostic failure marker; healthy fixed training continues',
        scope='No numerical recipe change; exposes actual row-level replay coverage before optimizer reuse')
