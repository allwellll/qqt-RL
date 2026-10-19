"""Successor-only host telemetry; all numerical training operations stay intact."""
import ast
import json
from pathlib import Path


def patched_worker(source):
    marker='# Successor-only normalized PPO advantage telemetry.'
    if marker in source:
        # Repair a legacy replay installer reintroducing keys already unpacked
        # by collector_evidence; this is host evidence only.
        if '**collector_evidence,' in source:
            source=source.replace('                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),\n','')
        ast.parse(source)
        return source
    anchor='    summarize=jax.pmap(summaries)'
    replacement=anchor+'\n    '+marker+'\n    from scripts.ppo_advantage_diagnostics import replica_moments, normalized_diagnostic\n    advantage_device_moments=jax.pmap(replica_moments)'
    loop='            for dev,tick,env_index in np.argwhere(mismatch)[:16]:'
    new_loop='            advantage_moments=np.asarray(advantage_device_moments(batch[4])) if mismatch.any() else None\n'+loop+'\n                collector_evidence=selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index)))'
    old='                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),'
    new="                    **collector_evidence,\n                    **normalized_diagnostic(collector_evidence['collector_gae_advantage'],advantage_moments[int(dev)]),"
    for before,after in [(anchor,replacement),(loop,new_loop),(old,new)]:
        assert source.count(before)==1,'unrecognized PPO telemetry layout'
        source=source.replace(before,after)
    ast.parse(source)
    return source


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/je10_worker.py';old=path.read_text();new=patched_worker(old)
    if old!=new:path.write_text(new)
    return dict(numerical_recipe_changed=False,host_scalars_per_device=2,
        scope='Capped selected deviations: exact per-replica PPO normalization before ratio/clipping; not Q or full gradient',
        actor_input_changed=False,ppo_or_gae_changed=False,random_keys_changed=False,
        reward_or_supervision_coefficient_changed=False,checkpoint_schema_changed=False)
