"""Future field alignment retains the current immutable recipe and relay."""
from pathlib import Path


def test_choose_carrier_field_preserves_all_existing_recipe_flags():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.2,prestock_preparation={'enabled':True},
        phase_balanced_replay={'enabled':True},navigation_replay={'enabled':True},
        oracle_certificate={'enabled':True},continuous_metric={'enabled':True},
        route_objective_alignment={'enabled':True})
    proposal=dict(recipe='carrier_route_fields',reason='reward field crosses present bricks while native carrier cannot place',changed_factor='shared carrier home field only')
    d=choose_recipe(plan,[],proposal)
    for name in ['carrier_route_fields','navigation_replay','phase_balanced_replay',
                 'prestock_preparation','native_one_step_certificate','continuous_route_metric','route_objective_consistency']:
        assert d[name]
    assert d['combat_fraction']==.35 and d['bridge_fraction']==.2 and d['route_mass']==.1
    assert not d['blocks_training']


def test_continuation_preserves_carrier_field_and_optional_diagnostic(tmp_path):
    from scripts.je10_supervise import predeclare_continuity,choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.2,carrier_route_fields={'enabled':True},
        navigation_replay={'enabled':True},normalized_advantage_diagnostic={'enabled':True})
    declaration=dict(round_number=9,evaluation=dict(points=[0,4,8,16,32,64,96,128]),
        terminal_relay=dict(successor='runs/je10_bun_20261018/control_route_v38_r10'))
    p=predeclare_continuity(tmp_path,plan,declaration)
    assert p['recipe']=='carrier_route_fields'
    d=choose_recipe(plan,[],p)
    assert d['carrier_route_fields'] and d['navigation_replay'] and d['normalized_advantage_diagnostic']
    assert p['fixed_budget']['generations']==128 and not p['evaluation_blocks_training']


def test_future_install_and_helper_sealing_happen_only_in_start():
    s=Path('scripts/je10_supervise.py').read_text()
    assert 'carrier_fields=install_carrier_fields' in s
    assert s.index('carrier_fields=install_carrier_fields')>s.index('navigation=install_navigation')
    for helper in ['carrier_route_fields.py','carrier_route_recipe.py','ppo_advantage_diagnostics.py','ppo_advantage_diagnostic_recipe.py']:
        assert "'scripts/"+helper+"'" in s
    assert "plan['carrier_route_fields']=carrier_fields" in s
