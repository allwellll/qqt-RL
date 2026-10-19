from scripts.je10_supervise import choose_recipe


def test_zero_route_support_gets_one_mixture_change_without_capability_gate():
    p={'combat_fraction':.4,'bridge_fraction':.25}
    r=choose_recipe(p,[])
    assert r['combat_fraction']==.35 and r['bridge_fraction']==.20
    assert r['route_mass']>.0999
    assert r['blocks_training'] is False


def test_retention_regression_allocates_combat_without_removing_route_support():
    p={'combat_fraction':.35,'bridge_fraction':.20}
    baseline={'hunter_js':{'hard':{'safe_win':3}},'tactical_python':{'safe_win':17},'route':{}}
    latest={'hunter_js':{'hard':{'safe_win':1}},'tactical_python':{'safe_win':16},'route':{}}
    r=choose_recipe(p,[baseline,latest])
    assert r['combat_fraction']>=.35 and r['route_mass']>.09
    assert r['blocks_training'] is False


def test_selected_mixture_reaches_every_native_route_initialization():
    import numpy as np
    import jax.numpy as jnp
    from scripts import je09_stage_h6_ops as sampler
    recipe=choose_recipe({'combat_fraction':.4,'bridge_fraction':.25},[])
    weights=np.asarray(sampler.curriculum_weights(jnp.zeros((10,96)),recipe['combat_fraction'],recipe['bridge_fraction']))
    assert np.all(weights[[1,5,7]]>0)
    np.testing.assert_array_equal(weights[[2,3,4,6]],0)
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-6)
    np.testing.assert_allclose(weights[[0,8,9]],[.35,.35,.20],atol=1e-6)


def test_target_alignment_followup_preserves_inherited_recipe_and_fixed_mix():
    plan=dict(combat_fraction=.35,bridge_fraction=.20,
        oracle_certificate={'enabled':True},continuous_metric={'enabled':True})
    proposal=dict(recipe='route_objective_consistency',reason='native own-recovery target conflict',
        changed_factor='PBRS own-recovery target alignment')
    chosen=choose_recipe(plan,[],proposal)
    assert chosen['route_objective_consistency']
    assert chosen['native_one_step_certificate'] and chosen['continuous_route_metric']
    assert chosen['combat_fraction']==.35 and chosen['bridge_fraction']==.20
    assert chosen['route_mass']==.10 and chosen['blocks_training'] is False
    inherited=choose_recipe(dict(plan,route_objective_alignment={'enabled':True}),[])
    assert inherited['route_objective_consistency']


def test_target_alignment_helpers_are_installed_and_sealed_before_prepare():
    import inspect
    from scripts.je10_supervise import supervise
    source=inspect.getsource(supervise)
    assert "from scripts.route_objective_recipe import install as install_objective" in source
    assert source.index('objective=install_objective(')<source.index('c.prepare(')
    assert "plan['route_objective_alignment']=objective" in source
    assert "declaration['route_objective_alignment']=objective" in source
    assert "'scripts/route_objective_consistency.py','scripts/route_objective_recipe.py'" in source
