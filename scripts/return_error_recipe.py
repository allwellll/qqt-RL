"""Candidate terminal-only installer for current-fit return replay sampling."""
import ast
import json
from pathlib import Path


def patched_worker(source):
    old='    aggregate=jax.pmap(dagger.aggregate)'
    new='    from scripts import return_error_replay\n    aggregate=jax.pmap(return_error_replay.aggregate)'
    call="new_buffer,dagger_replay,dagger_stats=aggregate(live['dagger_buffer'],batch,key,jnp.full(count,g,jnp.int32))"
    replacement=call[:-1]+",live['params']['actor'])"
    if new in source and replacement in source:return source
    assert source.count(old)==source.count(call)==1,'unrecognized aggregate dispatch'
    result=source.replace(old,new).replace(call,replacement)
    ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/je10_worker.py';before=path.read_text();after=patched_worker(before)
    if before!=after:path.write_text(after)
    return dict(enabled=True,changed_factor='Error-weighted sampling only inside existing native return replay slots',
        epsilon=.05,weight='epsilon + 1 - current Actor accepted legal joint target probability',
        return_multiplier=2.,other_phase_draws_unchanged=True,
        current_history_and_phase_slot_counts_unchanged=True,
        fifo_schema_capacity_additions_unchanged=True,queries_and_labels_unchanged=True,
        reward_ppo_network_observation_unchanged=True,extra_actor_forward_no_gradient=True,
        causal_scope='Sequential continuation without same-parent control; tests sampling/fitting hypothesis, not established ability causality')
