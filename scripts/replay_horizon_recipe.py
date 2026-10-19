"""Terminal-only extension of the native DAgger FIFO history window."""
import ast
import json
from pathlib import Path


def patched(source):
    old='CAPACITY=2048\n';new='CAPACITY=4096\n'
    if source.count(new)==1 and old not in source:return source
    assert source.count(old)==1 and new not in source,'unrecognized native FIFO capacity'
    result=source.replace(old,new);ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['passed'] and terminal['generation']==128 and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0 and set(release['devices'])==set(range(8))
    path=root/'scripts/je09_stage_h5c_dagger.py'
    before=path.read_text();after=patched(before)
    if before!=after:path.write_text(after)
    return dict(enabled=True,changed_factor='Native joint FIFO capacity per device 2048 to 4096; additions remain 256',
        capacity_per_device=4096,previous_capacity_per_device=2048,
        new_per_generation_per_device=256,nominal_fifo_generations=16,
        previous_nominal_fifo_generations=8,current_slots=64,history_slots=64,
        return_loss_multiplier=2.,error_epsilon=.05,
        current_history_and_phase_allocation_unchanged=True,
        oracle_queries_labels_custody_reward_loss_network_unchanged=True,
        scope='All native joint phases share the same FIFO; historical retention is extended for every phase, not only return. Fresh FIFO required, no synthetic custody or imported labels.',
        causal_scope='Sequential fixed-budget continuation; observed expiry and fit decline motivate the test but do not prove causality.',
        capability_gate=False)
