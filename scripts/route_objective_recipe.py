"""Prepared target-consistency installer; no active worker mutation."""
import ast
import json
from pathlib import Path


def patched_pbrs(source):
    marker='# Attributed own-recovery target alignment, successor-only recipe.'
    if marker in source:return source
    assert 'sampled_distance' in source,'shared continuous field is required first'
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='phase_values']
    assert len(functions)==1,'unrecognized target-selection source'
    node=functions[0]
    assert [a.arg for a in node.args.args]==['states','route','drop_state']
    lines=source.splitlines(keepends=True)
    original=''.join(lines[node.lineno-1:node.end_lineno])
    renamed=original.replace('def phase_values(', 'def _phase_values_before_recovery_alignment(',1)
    wrapper='''

def phase_values(states, route, drop_state):
    # Attributed own-recovery target alignment, successor-only recipe.
    from scripts.route_objective_consistency import align_recovery
    values = _phase_values_before_recovery_alignment(states, route, drop_state)
    return align_recovery(states, route, drop_state, values)
'''
    result=''.join(lines[:node.lineno-1])+renamed+wrapper+''.join(lines[node.end_lineno:])
    ast.parse(result)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/je09_route_pbrs.py';old=path.read_text();new=patched_pbrs(old)
    if new!=old:path.write_text(new)
    return dict(changed_factor='PBRS target identity aligned with existing attributed own-recovery teacher',
        additional_env_steps=0,reward_coefficient_changed=False,query_budget_changed=False,
        actor_observation_changed=False,new_auxiliary_state=False)
