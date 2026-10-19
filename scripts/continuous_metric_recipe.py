"""Prepared shared continuous-field recipe; never patches an active worker."""
import ast
import json
from pathlib import Path


def _replace(source,arguments,replacement):
    marker='    # Shared topology-continuous field, successor-only recipe.\n'
    if marker in source:return source
    try:tree=ast.parse(source)
    except SyntaxError as error:raise AssertionError('unrecognized continuous field source') from error
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='continuous_distance']
    assert len(functions)==1,'unrecognized continuous field source'
    node=functions[0]
    assert [arg.arg for arg in node.args.args]==arguments,'unrecognized continuous field signature'
    lines=source.splitlines(keepends=True)
    source=''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:])
    ast.parse(source)
    return source


def patched_oracle(source):
    return _replace(source,['pos','distance','blocked'],'''def continuous_distance(pos,distance,blocked):
    # Shared topology-continuous field, successor-only recipe.
    from scripts.continuous_route_metric import continuous_distance as shared
    return shared(pos,distance,blocked)
''')


def patched_pbrs(source):
    return _replace(source,['states','route','phase','_target_identity'],'''def continuous_distance(states,route,phase,_target_identity=None):
    # Shared topology-continuous field, successor-only recipe.
    from scripts.continuous_route_metric import sampled_distance
    n=states.pos.shape[0]
    teams=jnp.broadcast_to(jnp.arange(2)[None],(n,2))
    channels=jnp.where(phase>=PHASE_CARRY,teams+2,teams)
    pos=states.pos.reshape(n,2,2)
    main=sampled_distance(pos,route['distance'],states.wall,channels)
    stock=sampled_distance(pos,route['stock_distance'],states.wall,teams)
    return jnp.where(phase==PHASE_STOCK,stock,main)
''')


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    paths=[(root/'scripts/je09_route_pbrs.py',patched_pbrs),
           (root/'scripts/je10_route_oracle.py',patched_oracle)]
    prepared=[(path,path.read_text(),patch(path.read_text())) for path,patch in paths]
    for path,old,new in prepared:
        if old!=new:path.write_text(new)
    return dict(continuous_route_metric=True,changed_factor='continuous topology field shared by PBRS and route oracle',
        construction='shared local-component corners and cell-center triangular interpolation',
        native_alignment_cost=.125,goal_scope='grid centers remain zero; goal wait uses existing native contact gate',
        exact_continuous_geodesic=False,additional_env_steps=0,
        reward_coefficient_changed=False,query_budget_changed=False,actor_observation_changed=False)
