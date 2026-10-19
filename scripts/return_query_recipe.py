"""Terminal-only installer for a second verified distinct-cell return query."""
import ast
import json
from pathlib import Path


def patched(source):
    old_import='    from scripts.prestock_fragment_queries import joint_mismatch,first_deviations'
    new_import=('    from scripts.prestock_fragment_queries import joint_mismatch\n'
                '    from scripts.return_query_diversity import first_deviations')
    old_call='    first = first_deviations(mismatch,fragment,done,c[...,110]>0,c[...,111].astype(jnp.int32),c[...,113]>0)'
    new_call=('    cells=jnp.argmax(obs[...,0,:,:].reshape(obs.shape[:2]+(-1,)),axis=-1)\n'
              '    first = first_deviations(mismatch,fragment,done,c[...,110]>0,c[...,111].astype(jnp.int32),c[...,113]>0,cells)')
    if new_import in source and new_call in source:return source
    assert source.count(old_import)==source.count(old_call)==1,'unrecognized return query selection'
    source=source.replace(old_import,new_import).replace(old_call,new_call)
    ast.parse(source)
    return source


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    paths={root/'scripts/je09_stage_h6_ops.py':patched,
           root/'scripts/je10_worker.py':patched_worker}
    changes={path:(path.read_text(),patch(path.read_text())) for path,patch in paths.items()}
    for path,(before,after) in changes.items():
        if before!=after:path.write_text(after)
    return dict(enabled=True,changed_factor='Second verified ordinary return mismatch at a distinct observed cell',
        max_return_queries=2,min_tick_gap=8,quota_scope='rollout episode chunk; native terminal resets',
        native_verification_unchanged=True,baseline_queries_preserved=True,
        host_query_counts=True,
        other_phase_and_preparation_quotas_unchanged=True,return_multiplier=2.,
        fifo_capacity_and_sampling_unchanged=True,reward_ppo_network_mixture_unchanged=True,
        causal_scope='Sequential continuation without same-parent control; ability attribution remains a hypothesis')


def patched_worker(source):
    anchor="            row['dagger']['verifier_audit']=verifier_audit_counts(ctx)"
    addition=(anchor+'\n            from scripts.return_query_diversity import observed_query_counts\n'
              "            row['dagger']['return_query_diversity']=observed_query_counts(ctx)")
    if addition in source:return source
    assert source.count(anchor)==1,'unrecognized host verifier telemetry'
    source=source.replace(anchor,addition)
    ast.parse(source)
    return source
