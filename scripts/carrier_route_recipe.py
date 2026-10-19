"""Candidate successor-only physical carrier home field installer."""
import ast
import json
from pathlib import Path


def patched_potential(source):
    old='        result = shortest(next_states.wall[:, None], next_states.brick[:, None], allgoals)'
    new='        from scripts.carrier_route_fields import fields\n        result = fields(next_states.wall, next_states.brick, allgoals)'
    if new in source:return source
    assert source.count(old)==1,'unrecognized route cache rebuild'
    source=source.replace(old,new);ast.parse(source)
    return source


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    path=root/'scripts/je08_bun_potential.py';old=path.read_text();new=patched_potential(old)
    if old!=new:path.write_text(new)
    return dict(changed_factor='Shared home route field treats present brick as impassable for native carriers; drop channels remain exact weighted planning',
        home_channels=[2,3],drop_channels_unchanged=True,cache_schema_changed=False,
        actor_input_changed=False,ppo_or_gae_changed=False,
        coefficients_changed=False,query_budget_changed=False,
        causal_scope='One shared field mechanism changes its legacy carry/bridge and PBRS consumers together; existing native DAgger already uses independently brick-blocked distances and stays unchanged; sequential continuation without same-parent control',
        limitations='Future enemy explosions and pushable movement still need native verification; this is current wall/brick connectivity, not a temporal safety certificate')
