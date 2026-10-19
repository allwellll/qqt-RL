"""Every concrete run declares a same-recipe fixed successor before training."""
import json


def test_predeclare_preserves_specific_proposal_and_never_imposes_gate(tmp_path):
    from scripts.je10_supervise import predeclare_continuity
    existing=dict(recipe='prestock_preparation',changed_factor='specific tested mechanism')
    (tmp_path/'NEXT_RECIPE_PROPOSAL.json').write_text(json.dumps(existing))
    predeclare_continuity(tmp_path,{},dict(round_number=6))
    assert json.loads((tmp_path/'NEXT_RECIPE_PROPOSAL.json').read_text())==existing


def test_concrete_preparation_continuation_has_fixed_seed_budget_protocol_and_same_mix(tmp_path):
    from scripts.je10_supervise import predeclare_continuity,choose_recipe
    plan=dict(combat_fraction=.35,bridge_fraction=.20,prestock_preparation={'enabled':True})
    declaration=dict(round_number=6,evaluation=dict(points=[0,4,8,16,32,64,96,128]),
        terminal_relay=dict(successor='runs/je10_bun_20261018/control_route_v38_r07'))
    result=predeclare_continuity(tmp_path,plan,declaration)
    assert result['recipe']=='prestock_preparation'
    assert result['fixed_budget']==dict(devices=8,generations=128,fresh_ticks=268435456)
    assert not result['capability_gate'] and not result['evaluation_blocks_training']
    assert result['seeds']['train']!=result['seeds']['technical']
    assert result['parent'].endswith('control_route_v38_r06/formal/bun392_bridge_pfsp8/gen000128')
    chosen=choose_recipe(plan,[],result)
    assert chosen['combat_fraction']==.35 and chosen['bridge_fraction']==.20 and chosen['prestock_preparation']
