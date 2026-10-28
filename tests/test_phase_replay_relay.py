"""Numerical replay candidate selection retains all prior recipe factors."""
from pathlib import Path
import json


def test_choose_replay_factor_keeps_mixture_and_preparation():
    from scripts.je10_supervise import choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.2,prestock_preparation={'enabled':True},
        oracle_certificate={'enabled':True},continuous_metric={'enabled':True},route_objective_alignment={'enabled':True})
    proposal=dict(recipe='phase_balanced_replay',reason='saved FIFO stage dilution',changed_factor='replay allocation only')
    chosen=choose_recipe(plan,[],proposal)
    assert chosen['phase_balanced_replay'] and chosen['prestock_preparation']
    assert chosen['route_objective_consistency'] and chosen['continuous_route_metric'] and chosen['native_one_step_certificate']
    assert chosen['combat_fraction']==.35 and chosen['bridge_fraction']==.2 and chosen['route_mass']==.1
    assert not chosen['blocks_training']


def test_fallback_preserves_new_dispatch_with_fixed_seed_budget(tmp_path):
    from scripts.je10_supervise import predeclare_continuity,choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.2,prestock_preparation={'enabled':True},phase_balanced_replay={'enabled':True})
    declaration=dict(round_number=7,evaluation=dict(points=[0,4,8,16,32,64,96,128]),
        terminal_relay=dict(successor='runs/je10_bun_20261018/control_route_v38_r08'))
    proposal=predeclare_continuity(tmp_path,plan,declaration)
    assert proposal['recipe']=='phase_balanced_replay'
    assert proposal['changed_factor'].startswith('No new numerical factor')
    assert proposal['fixed_budget']['generations']==128 and not proposal['evaluation_blocks_training']
    assert choose_recipe(plan,[],proposal)['phase_balanced_replay']


def test_start_installs_and_seals_replay_helper_after_preparation():
    source=Path('scripts/je10_supervise.py').read_text()
    assert "from scripts.phase_balanced_replay_recipe import install as install_phase_replay" in source
    assert source.index('phase_replay=install_phase_replay')>source.index('prestock=install_prestock')
    assert "plan['phase_balanced_replay']=phase_replay" in source
    assert "'scripts/phase_balanced_joint_replay.py'" in source
    assert "'scripts/phase_balanced_replay_recipe.py'" in source
