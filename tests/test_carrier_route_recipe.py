"""Successor-only field dispatch, with unchanged cache/schema/reset behavior."""
import ast
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def candidate_refresh():
    from scripts.carrier_route_recipe import patched_potential
    source=patched_potential(Path('scripts/je08_bun_potential.py').read_text())
    space={'__name__':'carrier_fields_isolated_candidate'};exec(compile(source,'<carrier_fields_candidate>','exec'),space)
    return space


def test_patch_is_idempotent_and_keeps_cache_schema():
    from scripts.carrier_route_recipe import patched_potential
    source=Path('scripts/je08_bun_potential.py').read_text()
    source=source.replace('        from scripts.carrier_route_fields import fields\n        result = fields(next_states.wall, next_states.brick, allgoals)',
        '        result = shortest(next_states.wall[:, None], next_states.brick[:, None], allgoals)')
    new=patched_potential(source)
    ast.parse(new);assert new!=source and patched_potential(new)==new
    a=ast.parse(source);b=ast.parse(new)
    for name in ['routing_state','phase_reward','phase_state']:
        find=lambda tree:next(ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert find(a)==find(b)


def test_dirty_wall_opening_and_clean_cache_restore_keep_exact_fields():
    from scripts.carrier_route_fields import fields
    from scripts.je08_bun_potential import INF
    candidate=candidate_refresh();h,w=7,9
    wall=jnp.zeros((1,h,w),bool).at[:,0,:].set(True).at[:,-1,:].set(True)
    wall=wall.at[:,:,0].set(True).at[:,:,-1].set(True)
    brick=jnp.zeros_like(wall).at[:,1:-1,4].set(True)
    pos=jnp.array([[[2.5,1.5],[4.5,7.5]]])
    state=SimpleNamespace(wall=wall,brick=brick,pos=pos)
    route=candidate['routing_state'](1,h,w)
    goals=jnp.zeros((2,h,w),bool).at[0,2,7].set(True).at[1,4,1].set(True)
    info=dict(drop=jnp.zeros((1,2),bool),drop_origin=jnp.zeros((1,2),int),drop_cells=jnp.zeros((1,2,2),int))
    drop=dict(nonce=jnp.zeros((1,2),int))
    updated,dirty=candidate['refresh'](state,state,info,route,drop,goals)
    assert bool(dirty[0]) and int(updated['distance'][0,2,2,1])==INF
    restored=pickle.loads(pickle.dumps(updated))
    clean,dirty=candidate['refresh'](state,state,info,restored,drop,goals)
    assert not bool(dirty[0])
    for a,b in zip(jax.tree.leaves(updated),jax.tree.leaves(clean)):np.testing.assert_array_equal(a,b)
    opened=SimpleNamespace(wall=wall,brick=brick.at[0,3,4].set(False),pos=pos)
    result,dirty=candidate['refresh'](state,opened,info,clean,drop,goals)
    assert bool(dirty[0]) and int(result['distance'][0,2,2,1])<INF


@pytest.mark.parametrize('generation,released',[(127,True),(128,False)])
def test_installer_rejects_nonterminal_or_busy_devices(tmp_path,generation,released):
    from scripts.carrier_route_recipe import install
    run=tmp_path/'runs/parent';parent=run/'formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=False)))
    (run/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1)))
    source=tmp_path/'scripts/je08_bun_potential.py';source.parent.mkdir();source.write_text(Path('scripts/je08_bun_potential.py').read_text());before=source.read_text()
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert source.read_text()==before
