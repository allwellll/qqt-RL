"""Install return-only DAgger weighting after a fixed predecessor releases GPUs."""
import ast
import json
from pathlib import Path


def patched(source):
    old="""    from scripts.prestock_joint_oracle import joint_set_loss
    move,ability,_,_=train.net_forward(params,'transformer',replay['obs'],replay['gv'])
    return joint_set_loss(move/legacy.TEMPERATURE,ability/legacy.TEMPERATURE,
        replay['legal'],replay['ability_legal'],replay['joint_target'],replay['valid'],legacy.COEFFICIENT)[0]"""
    new="""    from scripts.return_dagger_weight import weighted_joint_loss
    move,ability,_,_=train.net_forward(params,'transformer',replay['obs'],replay['gv'])
    returning=(replay['kind']==3)&~replay['preparation']
    return weighted_joint_loss(move/legacy.TEMPERATURE,ability/legacy.TEMPERATURE,
        replay['legal'],replay['ability_legal'],replay['joint_target'],replay['valid'],
        returning,2.,legacy.COEFFICIENT)[0]"""
    if new in source:return source
    assert source.count(old)==1,'unrecognized joint replay loss'
    source=source.replace(old,new);ast.parse(source)
    return source


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/prestock_joint_replay.py';old=path.read_text();new=patched(old)
    if new!=old:path.write_text(new)
    return dict(changed_factor='Verified native return DAgger contribution multiplier only',
        return_multiplier=2.,base_coefficient=.2,other_phase_contributions_unchanged=True,
        denominator='All valid accepted legal joint labels, unchanged',
        query_budget_changed=False,replay_selection_changed=False,
        reward_changed=False,ppo_or_gae_changed=False,actor_input_changed=False,
        causal_scope='Sequential continuation without same-parent control; no claim of causal ability attribution')
