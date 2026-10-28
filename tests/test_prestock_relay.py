"""Pre-stock followup is an immediate technical-only fixed-budget relay."""
import inspect


def test_preparation_recipe_inherits_prior_mechanisms_and_preserves_mix_without_gate():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.20,oracle_certificate={'enabled':True},
        continuous_metric={'enabled':True},route_objective_alignment={'enabled':True})
    proposal=dict(recipe='prestock_preparation',reason='Native empty-hand opening required before custody',
        changed_factor='shared pre-stock objective and joint supervision')
    chosen=choose_recipe(plan,[],proposal)
    assert chosen['prestock_preparation'] and chosen['route_objective_consistency']
    assert chosen['native_one_step_certificate'] and chosen['continuous_route_metric']
    assert [chosen[k] for k in ['combat_fraction','bridge_fraction','natural_full_fraction','route_mass']]==[.35,.20,.35,.10]
    assert not chosen['blocks_training']
    inherited=choose_recipe(dict(plan,prestock_preparation={'enabled':True}),[],
        dict(recipe='prestock_preparation',reason='fixed continuation',changed_factor='no new numerical factor'))
    assert inherited['prestock_preparation']


def test_terminal_installer_runs_before_prepare_and_extended_schema_helpers_are_sealed():
    from scripts.je10_supervise import supervise
    source=inspect.getsource(supervise)
    assert source.index('prestock=install_prestock(')<source.index('c.prepare(')
    assert "plan['prestock_preparation']=prestock" in source
    assert "plan['state_schema']='je10-control-route-v38-prestock-v1'" in source
    assert "declaration['prestock_preparation']=prestock" in source
    assert "evaluation['sources']['scripts/audit_joint_dagger.py']" in source
    for name in ['prestock_training_state','prestock_verified_labels','prestock_native_teacher',
                 'prestock_joint_replay','prestock_fragment_queries','frontier_native_witness','route_frontier']:
        assert f'scripts/{name}.py' in source


def test_inherited_native_certificate_installer_recognizes_joint_verifier():
    from pathlib import Path
    from scripts.prestock_recipe import patched_sources
    from scripts.native_certificate_recipe import patched_verifier
    files=['scripts/je10_ops.py','scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h6_ops.py',
        'scripts/je09_route_pbrs.py','scripts/je10_worker.py']
    source=patched_sources({name:Path(name).read_text() for name in files})['scripts/je09_stage_h6_ops.py']
    assert patched_verifier(source)==source
