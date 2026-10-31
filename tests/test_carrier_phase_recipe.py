"""Prepared phase recipe must pair PBRS and learner-state oracle gates."""
import ast
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_patch_is_idempotent_and_preserves_learner_actions_and_coefficients():
    from scripts.carrier_phase_recipe import patched_ops,patched_oracle
    ops=Path('scripts/je10_ops.py').read_text()
    oracle=Path('scripts/je10_route_oracle.py').read_text()
    for source,patch in [(ops,patched_ops),(oracle,patched_oracle)]:
        updated=patch(source);ast.parse(updated)
        assert patch(updated)==updated
        assert updated.count('env.step(')==source.count('env.step(')
    assert 'ROUTE_COEFFICIENT=.5' in patched_ops(ops)
    assert 'dagger.COEFFICIENT=.2' in patched_ops(ops)
    with pytest.raises(AssertionError):patched_ops('unrecognized')
    with pytest.raises(AssertionError):patched_oracle('unrecognized')


def test_pbrs_and_oracle_agree_on_latched_carrier_and_both_native_control_edges(monkeypatch):
    from scripts.carrier_phase_recipe import patched_ops,patched_oracle
    from scripts import je09_bun_scenes as scenes
    ops={};exec(compile(patched_ops(Path('scripts/je10_ops.py').read_text()),'<prepared ops>','exec'),ops)
    oracle={};exec(compile(patched_oracle(Path('scripts/je10_route_oracle.py').read_text()),'<prepared oracle>','exec'),oracle)
    state=scenes.init_batch(jax.random.PRNGKey(14),2,jax.nn.one_hot(3,10))
    state=state._replace(core=state.core._replace(
        pos=jnp.array([[[3.5,3.5],[3.5,7.6]],[[3.5,3.5],[3.5,7.6]]]),
        alive=jnp.ones((2,2),bool)),trapped=jnp.zeros((2,2),jnp.int16),
        bun_carried=jnp.array([[1,-1],[1,-1]],state.bun_carried.dtype))
    aux=ops['reward_state'](2)
    aux['carrier_phase_latched']=jnp.array([[True,False],[False,False]])
    aux['dagger']['custody']=jnp.ones((2,2),bool)
    from scripts.carrier_phase_hysteresis import transition_gates
    _,aux=transition_gates(state,state,aux,aux)
    # Isolate phase permission from unreachable geometry; both choose legal down.
    monkeypatch.setitem(ops,'PARENT_PHASE_VALUES',lambda *a:(jnp.full((2,2),3),jnp.ones((2,2)),
        -jnp.ones((2,2)),jnp.ones((2,2)),jnp.ones((2,2))))
    monkeypatch.setitem(ops,'PARENT_RETURN_GATE',lambda *a:jnp.ones(2,bool))
    monkeypatch.setitem(oracle,'action_set',lambda *a:(jnp.tile(jnp.array([[0.,1.,0.,0.,0.]]),(2,1)),jnp.ones(2,bool)))
    def check(expected):
        phase=ops['safe_phase_values'](state,aux['routing'],aux['drop'])[0][:,0]
        labels=oracle['labels'](state,aux,jnp.ones((2,5),bool),jnp.zeros(2,int),jnp.zeros((2,13,15)))
        np.testing.assert_array_equal(phase>0,expected)
        np.testing.assert_array_equal(labels[:,7]>0,expected)
        legacy_return=ops['carrier_return_gate'](state,aux,jnp.zeros(2,int),120)
        np.testing.assert_array_equal(legacy_return,expected)
    check([False,True])
    for changes in [dict(core=state.core._replace(alive=state.alive.at[:,1].set(False))),
                    dict(trapped=state.trapped.at[:,1].set(20))]:
        controlled=state._replace(**changes)
        _,new=transition_gates(state,controlled,aux,aux)
        old_state,old_aux=state,aux;state,aux=controlled,new
        check([True,True]);state,aux=old_state,old_aux


def test_installer_refuses_live_predecessor_and_does_not_change_sources(tmp_path):
    from scripts.carrier_phase_recipe import install
    for name in ['je10_ops.py','je10_route_oracle.py']:
        path=tmp_path/'scripts'/name;path.parent.mkdir(exist_ok=True)
        path.write_text(Path('scripts',name).read_text())
    original={p:p.read_text() for p in (tmp_path/'scripts').glob('*.py')}
    base=tmp_path/'run';parent=base/'formal/arm/gen000128';parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):install(tmp_path,parent)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=128,passed=True,atomic_stop=False)))
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=False,compute_clients=1)))
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert all(p.read_text()==text for p,text in original.items())
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=True,compute_clients=0)))
    assert install(tmp_path,parent)['changed_factor']=='carrier phase recovery hysteresis'


def test_native_transition_preserves_reset_tree_schema_and_pairs_pbrs_gates():
    from scripts.carrier_phase_recipe import patched_ops
    from scripts import je09_bun_scenes as scenes
    from jax_bomb import je07_bun_env as env
    ops={};exec(compile(patched_ops(Path('scripts/je10_ops.py').read_text()),'<prepared ops>','exec'),ops)
    state=scenes.init_batch(jax.random.PRNGKey(21),2,jax.nn.one_hot(3,10))
    aux=ops['reward_state'](2)
    joint=jnp.zeros((2,2,2),jnp.int32).at[:,:,0].set(4)
    following,done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        state,joint,jax.random.split(jax.random.PRNGKey(22),2))
    info=dict(info,h_route_terminal=done,h_joint_actions=joint,
        blocked_own_route=jnp.zeros((2,2),bool))
    reward,new,parts,metrics=ops['combined_reward'](state,following,info,aux,jnp.ones(2,bool))
    assert jax.tree.structure(new)==jax.tree.structure(aux)
    assert np.isfinite(np.asarray(reward)).all() and np.isfinite(np.asarray(parts)).all()
    from scripts.carrier_phase_hysteresis import current_gate
    np.testing.assert_array_equal(new['routing']['carrier_route_allowed'],current_gate(following,new))
    # Non-terminal reward state is checkpointable with exactly its reset schema.
    reset=jax.tree.map(lambda a,b:jnp.where(jnp.array([[True],[False]]).reshape(
        (2,)+(1,)*(a.ndim-1)),b,a),new,ops['reward_state'](2))
    assert not bool(reset['carrier_phase_latched'][0].any())


def test_predeclared_phase_followup_keeps_scene_mix_native_certificate_and_no_gate():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.20,oracle_certificate={'native_one_step_certificate':True})
    proposal=dict(recipe='carrier_phase_hysteresis',reason='fixed Review period-two carrier phase oscillation',
        changed_factor='carrier phase recovery hysteresis')
    chosen=choose_recipe(plan,[],proposal)
    assert chosen['combat_fraction']==.35 and chosen['bridge_fraction']==.20
    assert chosen['route_mass']==.10
    assert chosen['carrier_phase_hysteresis'] and chosen['native_one_step_certificate']
    assert chosen['blocks_training'] is False
