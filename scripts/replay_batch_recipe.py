"""Terminal-only candidate for a larger native joint replay batch."""
import ast
import json
from pathlib import Path


def patched_replay(source):
    old='k1,k2=jax.random.split(key);size=32 if capacity==legacy.CAPACITY else 2'
    new='k1,k2=jax.random.split(key);size=64 if capacity==legacy.CAPACITY else 2'
    if source.count(new)==1 and old not in source:return source
    assert source.count(old)==1,'unrecognized native replay batch dispatch'
    result=source.replace(old,new)
    ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0 and set(release['devices'])==set(range(8))
    path=root/'scripts/return_error_replay.py'
    before=path.read_text();after=patched_replay(before)
    if before!=after:path.write_text(after)
    return dict(enabled=True,current_slots=64,history_slots=64,prior_half_slots=32,
        changed_factor='Native joint replay batch per device 64 to 128; phase and current/history proportions unchanged',
        ordinary_slots_per_half=32,return_slots_when_all_phases_present=24,
        navigation_slots_per_half=30,opening_slots_per_half=2,
        slot_counts_scope='All ordinary, acquisition, return, navigation and opening groups present; unchanged missing-group redistribution and half fallback otherwise',
        error_epsilon=.05,return_loss_multiplier=2.,loss_mean_normalization_unchanged=True,
        query_fifo_labels_reward_ppo_network_observation_unchanged=True,
        fixed_budget_unchanged=True,capability_gate=False,
        causal_scope='Sequential continuation; batch coverage hypothesis, no same-parent causal control')
