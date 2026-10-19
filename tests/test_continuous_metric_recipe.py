"""Continuous field recipe is installed only after the predecessor releases GPUs."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def test_patch_pairs_stock_drop_home_pbrs_and_oracle_without_new_env_steps():
    from scripts.continuous_metric_recipe import patched_pbrs,patched_oracle
    from scripts.je08_bun_potential import cpu_shortest
    from scripts import je09_route_pbrs as pbrs
    level=next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x['qqt_id']==806)
    wall=np.asarray(level['wall'],bool).reshape(13,15);goals=np.zeros_like(wall)
    row,column=level['bun_bases'][0];goals[row:row+3,column:column+3]=True
    d=jnp.array(cpu_shortest(wall,np.zeros_like(wall),goals))[None]
    states=SimpleNamespace(pos=jnp.array([[[5.99999,9.82],[6.00001,9.82]]]),
        wall=jnp.array(wall)[None],brick=jnp.zeros((1,13,15),bool))
    route=dict(distance=jnp.broadcast_to(d[:,None],(1,4,13,15)),stock_distance=jnp.broadcast_to(d[:,None],(1,2,13,15)))
    old=pbrs.continuous_distance(states,route,jnp.full((1,2),3))
    # The installed field is already fixed; reinstallation must preserve it.
    assert abs(float(old[0,0]-old[0,1]))<.0001
    adapted={};exec(compile(patched_pbrs(Path('scripts/je09_route_pbrs.py').read_text()),'<prepared PBRS>','exec'),adapted)
    oracle={};exec(compile(patched_oracle(Path('scripts/je10_route_oracle.py').read_text()),'<prepared oracle>','exec'),oracle)
    for phase in [1,2,3,4]:
        result=adapted['continuous_distance'](states,route,jnp.full((1,2),phase))
        expected=oracle['continuous_distance'](states.pos.reshape(2,2),jnp.repeat(d,2,0),jnp.repeat(states.wall,2,0))
        np.testing.assert_allclose(result[0],expected,atol=1e-6)
        assert abs(float(result[0,0]-result[0,1]))<.0001
    for source,patch in [(Path('scripts/je09_route_pbrs.py').read_text(),patched_pbrs),
                         (Path('scripts/je10_route_oracle.py').read_text(),patched_oracle)]:
        updated=patch(source);ast.parse(updated)
        assert patch(updated)==updated
        assert source.count('env.step(')==updated.count('env.step(')
    with pytest.raises(AssertionError):patched_pbrs('unexpected source')
    with pytest.raises(AssertionError):patched_oracle('unexpected source')


def test_continuous_metric_followup_keeps_scene_mix_and_native_certificate_only():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.20,oracle_certificate={'native_one_step_certificate':True})
    proposal=dict(recipe='continuous_route_metric',reason='native Review and JAX RED boundary jump',
        changed_factor='continuous topology field shared by PBRS and route oracle')
    result=choose_recipe(plan,[],proposal)
    assert result['continuous_route_metric'] and result['native_one_step_certificate']
    assert result['combat_fraction']==.35 and result['bridge_fraction']==.20 and result['route_mass']==.1
    assert not result.get('carrier_phase_hysteresis',False) and result['blocks_training'] is False


def test_install_refuses_nonterminal_or_busy_predecessor(tmp_path):
    from scripts.continuous_metric_recipe import install
    for name in ['je09_route_pbrs.py','je10_route_oracle.py']:
        path=tmp_path/'scripts'/name;path.parent.mkdir(exist_ok=True);path.write_text(Path('scripts',name).read_text())
    original={p:p.read_text() for p in (tmp_path/'scripts').glob('*.py')}
    base=tmp_path/'run';parent=base/'formal/arm/gen000128';parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):install(tmp_path,parent)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=128,passed=True,atomic_stop=False)))
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=False,compute_clients=1)))
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert all(p.read_text()==text for p,text in original.items())
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=True,compute_clients=0)))
    assert install(tmp_path,parent)['changed_factor']=='continuous topology field shared by PBRS and route oracle'


def test_native_pbrs_transition_retains_finite_reward_state_and_terminal_zero(monkeypatch):
    import jax
    from scripts.continuous_metric_recipe import patched_pbrs
    from scripts import je09_route_pbrs as pbrs,je09_bun_scenes as scenes,je10_ops as ops
    from jax_bomb import je07_bun_env as env
    adapted={};exec(compile(patched_pbrs(Path('scripts/je09_route_pbrs.py').read_text()),'<prepared PBRS>','exec'),adapted)
    monkeypatch.setattr(pbrs,'continuous_distance',adapted['continuous_distance'])
    state=scenes.init_batch(jax.random.PRNGKey(32),2,jax.nn.one_hot(3,10))
    aux=ops.reward_state(2);joint=jnp.zeros((2,2,2),int).at[:,:,0].set(4)
    after,done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        state,joint,jax.random.split(jax.random.PRNGKey(33),2))
    info=dict(info,h_route_terminal=jnp.array([True,False]),h_joint_actions=joint,blocked_own_route=jnp.zeros((2,2),bool))
    reward,new,parts,metrics=ops.combined_reward(state,after,info,aux,jnp.ones(2,bool))
    assert jax.tree.structure(new)==jax.tree.structure(aux)
    assert np.isfinite(np.asarray(reward)).all() and np.isfinite(np.asarray(parts)).all()
    np.testing.assert_array_equal(new['route_pbrs']['phi'][0],0.)
