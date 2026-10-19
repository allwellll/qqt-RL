"""The candidate changes a single supervised contribution after release."""
import ast
import json
from pathlib import Path
import pytest


def fixture(tmp_path,generation=128,released=True,stop=False):
    parent=tmp_path/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=stop)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1)))
    source=tmp_path/'scripts/prestock_joint_replay.py';source.parent.mkdir()
    baseline=Path('scripts/prestock_joint_replay.py').read_text()
    start=baseline.index('def replay_loss(')
    baseline=baseline[:start]+'''def replay_loss(params,replay):
    from jax_bomb import jax_train as train
    from scripts.prestock_joint_oracle import joint_set_loss
    move,ability,_,_=train.net_forward(params,'transformer',replay['obs'],replay['gv'])
    return joint_set_loss(move/legacy.TEMPERATURE,ability/legacy.TEMPERATURE,
        replay['legal'],replay['ability_legal'],replay['joint_target'],replay['valid'],legacy.COEFFICIENT)[0]
'''
    source.write_text(baseline)
    return parent,source


@pytest.mark.parametrize('generation,released,stop',[(127,True,False),(128,False,False),(128,True,True)])
def test_no_install_into_healthy_live_worker(tmp_path,generation,released,stop):
    from scripts.return_dagger_recipe import install
    parent,source=fixture(tmp_path,generation,released,stop);before=source.read_text()
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert source.read_text()==before


def test_patch_changes_only_replay_loss_and_is_idempotent(tmp_path):
    from scripts.return_dagger_recipe import install,patched
    parent,source=fixture(tmp_path);before=source.read_text();receipt=install(tmp_path,parent);after=source.read_text()
    assert before!=after and patched(after)==after
    trees=[ast.parse(x) for x in [before,after]]
    functions=[{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef)} for t in trees]
    assert {k for k in functions[0] if functions[0][k]!=functions[1][k]}=={'replay_loss'}
    assert receipt['return_multiplier']==2. and receipt['other_phase_contributions_unchanged']
    assert not receipt['query_budget_changed'] and not receipt['replay_selection_changed']
    install(tmp_path,parent);assert source.read_text()==after


def test_successor_preserves_all_inherited_factors_with_return_weight():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.20,
        carrier_route_fields={'enabled':True},navigation_replay={'enabled':True},
        normalized_advantage_diagnostic={'enabled':True})
    proposal=dict(recipe='return_dagger_weight',reason='fixed return execution and FIFO fit',
        changed_factor='return-only verified DAgger contribution x2')
    result=choose_recipe(plan,[],proposal)
    assert result['return_dagger_weight'] and result['carrier_route_fields']
    assert result['navigation_replay'] and result['normalized_advantage_diagnostic']
    assert result['combat_fraction']==.35 and result['bridge_fraction']==.20
    assert result['route_mass']==.1 and result['blocks_training'] is False


def test_dispatch_installs_and_seals_candidate_before_preparing_successor():
    import inspect
    from scripts.je10_supervise import supervise
    source=inspect.getsource(supervise)
    assert source.index('return_weight=install_return_weight(')<source.index('c.prepare(')
    assert "plan['return_dagger_weight']=return_weight" in source
    assert "declaration['return_dagger_weight']=return_weight" in source
    assert "'scripts/return_dagger_weight.py','scripts/return_dagger_recipe.py'" in source


def test_installed_replay_loss_uses_native_kind_and_preparation_mask(monkeypatch):
    import types
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_bomb import jax_train as train
    from scripts import je09_stage_h5c_dagger as legacy
    from scripts.return_dagger_recipe import patched
    from scripts.return_dagger_weight import weighted_joint_loss
    space=types.ModuleType('isolated_return_weight_replay')
    exec(compile(patched(Path('scripts/prestock_joint_replay.py').read_text()),'<candidate>','exec'),space.__dict__)
    move=jnp.array([[2.,1.,0.,-1.,0.],[0.,2.,1.,-1.,0.],[1.,0.,2.,0.,-1.]])
    ability=jnp.array([[1.,0.,-1.],[0.,1.,2.],[2.,0.,1.]])
    monkeypatch.setattr(train,'net_forward',lambda p,*args:(p,ability,None,None))
    monkeypatch.setattr(legacy,'COEFFICIENT',.2)
    target=jnp.zeros((3,5,3),bool).at[:,0,0].set(True)
    replay=dict(obs=jnp.zeros((3,1)),gv=jnp.zeros((3,1)),
        legal=jnp.ones((3,5),bool),ability_legal=jnp.ones((3,3),bool),
        joint_target=target,valid=jnp.ones(3,bool),kind=jnp.array([3,3,2]),
        preparation=jnp.array([False,True,False]))
    expected=lambda m:weighted_joint_loss(m,ability,replay['legal'],replay['ability_legal'],target,replay['valid'],jnp.array([True,False,False]),2.,.2)[0]
    actual=lambda m:space.replay_loss(m,replay)
    np.testing.assert_allclose(jax.jit(actual)(move),expected(move),rtol=1e-7)
    np.testing.assert_allclose(jax.grad(actual)(move),jax.grad(expected)(move),atol=1e-7)
