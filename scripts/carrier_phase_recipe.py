"""Successor-only PBRS/oracle phase patch, prepared without live source edits."""
import ast
import json
from pathlib import Path


def patched_ops(source):
    marker='    # Carrier phase recovery hysteresis, successor-only recipe.\n'
    if marker in source:return source
    replacements={
        'PARENT_PHASE_VALUES=pbrs.phase_values':
        'PARENT_PHASE_VALUES=pbrs.phase_values\n'
        'from scripts import je09_return_expert as return_teacher\n'
        'PARENT_RETURN_GATE=return_teacher.gate',
        "    aux['route_pbrs']=pbrs.state(n)":
        "    aux['route_pbrs']=pbrs.state(n)\n"+marker+
        "    aux['carrier_phase_latched']=jnp.zeros((n,2),jnp.bool_)\n"
        "    aux['routing']['carrier_route_allowed']=jnp.zeros((n,2),jnp.bool_)",
        '    reward,new,parts,rm=pbrs.apply_reward(states,after,info,aux,reward,new,parts,ROUTE_COEFFICIENT)':
        '    from scripts.carrier_phase_hysteresis import transition_gates\n'
        "    aux,new=transition_gates(states,after,aux,new,info['h_route_terminal'])\n"
        '    reward,new,parts,rm=pbrs.apply_reward(states,after,info,aux,reward,new,parts,ROUTE_COEFFICIENT)',
        '    allowed=oracle.route_allowed(carrying,states.alive[:,::-1],states.trapped[:,::-1],separation)':
        "    allowed=route['carrier_route_allowed']",
        '\ndef install():':
        '\ndef carrier_return_gate(states,aux,seat,window=120):\n'
        '    from scripts.carrier_phase_hysteresis import current_gate\n'
        '    return PARENT_RETURN_GATE(states,aux,seat,window)&current_gate(states,aux)[jnp.arange(seat.size),seat]\n'
        '\n\ndef install():',
        '    teacher.action_set=oracle.action_set':
        '    teacher.gate=carrier_return_gate\n'
        '    teacher.action_set=oracle.action_set',
    }
    for before,after in replacements.items():
        assert source.count(before)==1,'unrecognized phase ops source'
        source=source.replace(before,after)
    ast.parse(source)
    return source


def patched_oracle(source):
    marker='    # Shared carrier phase memory with PBRS; no Actor takeover.\n'
    if marker in source:return source
    before='    allowed=route_allowed(~empty,states.alive[r,enemy],states.trapped[r,enemy],enemy_distance)'
    assert source.count(before)==1,'unrecognized route oracle source'
    after=marker+'    from scripts.carrier_phase_hysteresis import current_gate\n'+\
        '    allowed=current_gate(states,aux)[r,seat]'
    source=source.replace(before,after);ast.parse(source)
    return source


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    paths=[(root/'scripts/je10_ops.py',patched_ops),
           (root/'scripts/je10_route_oracle.py',patched_oracle)]
    prepared=[(path,path.read_text(),patch(path.read_text())) for path,patch in paths]
    for path,old,new in prepared:
        if old!=new:path.write_text(new)
    return dict(carrier_phase_hysteresis=True,changed_factor='carrier phase recovery hysteresis',
        enter_distance=4.,resume_distance=5.,actor_observation_changed=False,
        additional_env_steps=0,reward_coefficient_changed=False,query_budget_changed=False)
