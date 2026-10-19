"""Successor-only replay quota rebalance; no running worker mutation."""
import ast
import json
from pathlib import Path


def patched(source):
    replacements=[('max(1,num//16)','max(1,num//32)'),
        ('jnp.where(has_navigation,num*5//8,num-opening_slots)','jnp.where(has_navigation,num//2,num-opening_slots)'),
        ('ordinary_slots*3//5','ordinary_slots*3//4')]
    if all(new in source and old not in source for old,new in replacements):return source
    for old,new in replacements:
        assert source.count(old)==1,'unrecognized phase replay quota'
        source=source.replace(old,new)
    source=source.replace('Opening gets two of 32 slots','Opening gets one of 32 slots')
    source=source.replace('ordinary routes retain five eighths of the slots','ordinary routes retain half the slots')
    source=source.replace('Reserve three fifths of ordinary slots','Reserve three quarters of ordinary slots')
    ast.parse(source)
    return source


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/phase_balanced_joint_replay.py';old=path.read_text();new=patched(old)
    if old!=new:path.write_text(new)
    return dict(changed_factor='Replay quota allocation only, preserving return while reallocating opening/acquisition slots to preparation navigation',
        per_half=dict(ordinary=16,returning=12,drop=2,stock_ready=2,navigation=15,opening=1),
        current_history_half_preserved=True,total_replay_per_device=64,
        reward_coefficient_changed=False,query_budget_changed=False,buffer_storage_changed=False,
        actor_observation_changed=False,missing_group_policy='Same-half redistribution; no preparation remains exact old uniform')
