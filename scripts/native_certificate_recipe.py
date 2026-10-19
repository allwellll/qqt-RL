"""Prepared successor-only patch; never imported by the current GPU worker."""
import ast
import json
from pathlib import Path


def patched_verifier(source):
    if '# Explicit bounded joint teacher.' in source:
        ast.parse(source)
        return source
    marker='        # Native-step progress certificate, successor-only recipe.\n'
    if marker in source:return source
    anchor='        safe = temporal_safe & one_step_safe\n'
    gate='(labels[:, 5] * accepted)[:, None]'
    assert source.count(anchor)==source.count(gate)==1,'unrecognized strict verifier source'
    insertion='''        # Native-step progress certificate, successor-only recipe.
        # Reuse all five existing counterfactuals, retaining safety and slots.
        from scripts.native_route_certificate import certify_native_positions
        native_kind=labels[ids,6].astype(jnp.int32)
        native_channel=jnp.where((native_kind==1)|(native_kind==4),sample_seat,
            jnp.where(native_kind==2,2+sample_seat,4+sample_seat))
        native_distance=aux['dagger']['distance'][ids,native_channel]
        native_blocked=aux['dagger']['blocked'][ids]
        native_before=sample_states.pos[sample_rows,sample_seat]
        native_after=nxt.pos[jnp.arange(sample_count*5),jnp.repeat(sample_seat,5)].reshape(sample_count,5,2)
        native_cell=jnp.clip(jnp.floor(native_before).astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
        native_goal=native_distance[sample_rows,native_cell[:,0],native_cell[:,1]]==0
        next_blocked=(nxt.wall|nxt.brick|nxt.core.pushable).reshape(sample_count,5,13,15)
        stable_topology=jnp.all(next_blocked==native_blocked[:,None],axis=(-2,-1))
        native_targets,native_valid=certify_native_positions(native_before,native_after,
            native_distance,native_blocked,legal[ids],safe,native_goal,stable_topology)
        proposed=jnp.zeros_like(proposed).at[ids].max(native_targets*selected_slots[:,None])
'''
    patched=source.replace(anchor,anchor+insertion).replace(gate,'(labels[:, 7] * accepted)[:, None]')
    ast.parse(patched)
    return patched


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/je09_stage_h6_ops.py'
    source=path.read_text();patched=patched_verifier(source)
    if patched!=source:path.write_text(patched)
    return dict(native_one_step_certificate=True,changed_factor='strict route oracle native-step progress set',
                reuses_native_counterfactuals=True,additional_env_steps=0,query_budget_changed=False)
