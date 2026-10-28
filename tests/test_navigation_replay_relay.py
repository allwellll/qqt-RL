"""Future navigation quota recipe must preserve continuity and prior factors."""
from pathlib import Path


def test_choose_navigation_quota_keeps_existing_recipe_factors():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.2,prestock_preparation={'enabled':True},
        phase_balanced_replay={'enabled':True},oracle_certificate={'enabled':True},
        continuous_metric={'enabled':True},route_objective_alignment={'enabled':True})
    proposal=dict(recipe='navigation_replay',reason='navigation fit remains weak while opening fit rises',changed_factor='replay quotas only')
    d=choose_recipe(plan,[],proposal)
    assert d['navigation_replay'] and d['phase_balanced_replay'] and d['prestock_preparation']
    assert d['native_one_step_certificate'] and d['continuous_route_metric'] and d['route_objective_consistency']
    assert d['combat_fraction']==.35 and d['bridge_fraction']==.2 and d['route_mass']==.1
    assert not d['blocks_training']


def test_fallback_preserves_navigation_quota_without_score_dependency(tmp_path):
    from scripts.je10_supervise import predeclare_continuity,choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.2,prestock_preparation={'enabled':True},
        phase_balanced_replay={'enabled':True},navigation_replay={'enabled':True})
    declaration=dict(round_number=8,evaluation=dict(points=[0,4,8,16,32,64,96,128]),
        terminal_relay=dict(successor='runs/je10_bun_20261018/control_route_v38_r09'))
    p=predeclare_continuity(tmp_path,plan,declaration)
    assert p['recipe']=='navigation_replay' and choose_recipe(plan,[],p)['navigation_replay']
    assert p['fixed_budget']['generations']==128 and not p['evaluation_blocks_training']


def test_start_installs_and_seals_navigation_after_existing_phase_recipe():
    s=Path('scripts/je10_supervise.py').read_text()
    assert 'navigation=install_navigation' in s
    assert s.index('navigation=install_navigation')>s.index('phase_replay=install_phase_replay')
    assert "plan['navigation_replay']=navigation" in s
    assert "'scripts/navigation_replay_recipe.py'" in s
