"""Prepared target alignment must not modify any live training recipe."""
import ast
import json
from pathlib import Path

import pytest


def test_alignment_patch_is_paired_with_continuous_field_and_idempotent():
    from scripts.route_objective_recipe import patched_pbrs
    source=Path('scripts/je09_route_pbrs.py').read_text()
    patched=patched_pbrs(source)
    ast.parse(patched)
    assert patched_pbrs(patched)==patched
    assert patched.count('env.step(')==source.count('env.step(')
    assert 'sampled_distance' in patched
    assert 'align_recovery(states, route, drop_state, values)' in patched
    with pytest.raises(AssertionError):patched_pbrs('unrecognized source')


def test_installer_requires_fixed_terminal_and_eight_gpu_release(tmp_path):
    from scripts.route_objective_recipe import install
    source=tmp_path/'scripts/je09_route_pbrs.py';source.parent.mkdir()
    source.write_text(Path('scripts/je09_route_pbrs.py').read_text());original=source.read_text()
    base=tmp_path/'run';parent=base/'formal/arm/gen000128';parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):install(tmp_path,parent)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=127,passed=True,atomic_stop=False)))
    with pytest.raises(AssertionError):install(tmp_path,parent)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=128,passed=True,atomic_stop=False)))
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=False,compute_clients=1)))
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert source.read_text()==original
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=True,compute_clients=0)))
    receipt=install(tmp_path,parent)
    assert receipt['additional_env_steps']==0 and not receipt['reward_coefficient_changed']


def test_prepared_native_pbrs_reward_has_aligned_target_and_original_reset_schema(monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from test_route_objective_consistency import native_case
    from scripts.route_objective_recipe import patched_pbrs
    from scripts import je09_route_pbrs as pbrs,je10_ops as ops
    from jax_bomb import je07_bun_env as env
    prepared={};exec(compile(patched_pbrs(Path('scripts/je09_route_pbrs.py').read_text()),'<prepared recovery PBRS>','exec'),prepared)
    state,aux,route=native_case()
    # Recovery alignment is already installed in the published baseline.
    assert int(ops.safe_phase_values(state,route,aux['drop'])[0][0,0])==2
    monkeypatch.setattr(ops,'PARENT_PHASE_VALUES',prepared['phase_values'])
    monkeypatch.setattr(pbrs,'phase_values',ops.safe_phase_values)
    phase=ops.safe_phase_values(state,route,aux['drop'])[0]
    assert int(phase[0,0])==2
    aux['routing']={k:v for k,v in route.items() if k!='stock_distance'}
    joint=jnp.array([[[2,0],[4,0]]])
    after,_,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        state,joint,jax.random.split(jax.random.PRNGKey(24),1))
    info=dict(info,h_joint_actions=joint,blocked_own_route=jnp.zeros((1,2),bool))
    for terminal in [False,True]:
        reward,new,parts,metrics=ops.combined_reward(state,after,
            dict(info,h_route_terminal=jnp.array([terminal])),aux,jnp.ones(1,bool))
        assert jax.tree.structure(new)==jax.tree.structure(aux)
        assert np.isfinite(np.asarray(reward)).all() and np.isfinite(np.asarray(parts)).all()
        assert int(metrics['route_phase'][0,0])==2
        if terminal:np.testing.assert_array_equal(new['route_pbrs']['phi'],0)


def test_prepared_own_carrier_preserves_native_threat_gate_and_terminal_reward(monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from test_route_objective_consistency import native_case
    from scripts.route_objective_recipe import patched_pbrs
    from scripts import je09_route_pbrs as pbrs,je10_ops as ops
    from jax_bomb import je07_bun_env as env
    prepared={};exec(compile(patched_pbrs(Path('scripts/je09_route_pbrs.py').read_text()),'<prepared own carrier>','exec'),prepared)
    state,aux,route=native_case()
    state=state._replace(bun_carried=state.bun_carried.at[0,0].set(0))
    aux['routing']={k:v for k,v in route.items() if k!='stock_distance'}
    monkeypatch.setattr(ops,'PARENT_PHASE_VALUES',prepared['phase_values'])
    monkeypatch.setattr(pbrs,'phase_values',ops.safe_phase_values)
    assert int(ops.safe_phase_values(state,route,aux['drop'])[0][0,0])==3
    near=state._replace(core=state.core._replace(alive=jnp.ones_like(state.alive),
        pos=state.pos.at[0,1].set(state.pos[0,0]+jnp.array([0.,2.]))),
        bun_respawn=jnp.zeros_like(state.bun_respawn))
    assert int(ops.safe_phase_values(near,route,aux['drop'])[0][0,0])==0
    joint=jnp.array([[[0,0],[4,0]]])
    after,_,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        state,joint,jax.random.split(jax.random.PRNGKey(27),1))
    info=dict(info,h_joint_actions=joint,blocked_own_route=jnp.zeros((1,2),bool),h_route_terminal=jnp.ones(1,bool))
    reward,new,parts,metrics=ops.combined_reward(state,after,info,aux,jnp.ones(1,bool))
    assert int(metrics['route_phase'][0,0])==3
    assert jax.tree.structure(new)==jax.tree.structure(aux)
    np.testing.assert_array_equal(new['route_pbrs']['phi'],0)
    assert np.isfinite(np.asarray(reward)).all() and np.isfinite(np.asarray(parts)).all()
