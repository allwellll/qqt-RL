"""Install query diversity only after the fixed predecessor releases GPUs."""
import ast
import inspect
import json
from pathlib import Path
import pytest


def fixture(tmp_path, generation=128, released=True, stopped=False):
    parent=tmp_path/'runs/prior/formal/arm/gen000128'
    parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(
        generation=generation,passed=True,atomic_stop=stopped)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(
        released=released,compute_clients=0 if released else 1)))
    source=tmp_path/'scripts/je09_stage_h6_ops.py'
    source.parent.mkdir()
    baseline=Path('scripts/je09_stage_h6_ops.py').read_text()
    # The test remains reproducible after the next run installs the candidate.
    baseline=baseline.replace(
        '    from scripts.prestock_fragment_queries import joint_mismatch\n'
        '    from scripts.return_query_diversity import first_deviations',
        '    from scripts.prestock_fragment_queries import joint_mismatch,first_deviations')
    baseline=baseline.replace(
        '    cells=jnp.argmax(obs[...,0,:,:].reshape(obs.shape[:2]+(-1,)),axis=-1)\n'
        '    first = first_deviations(mismatch,fragment,done,c[...,110]>0,c[...,111].astype(jnp.int32),c[...,113]>0,cells)',
        '    first = first_deviations(mismatch,fragment,done,c[...,110]>0,c[...,111].astype(jnp.int32),c[...,113]>0)')
    source.write_text(baseline)
    (source.parent/'je10_worker.py').write_text(Path('scripts/je10_worker.py').read_text())
    return parent,source


@pytest.mark.parametrize('generation,released,stopped',[
    (127,True,False),(128,False,False),(128,True,True)])
def test_live_or_invalid_predecessor_cannot_install(tmp_path,generation,released,stopped):
    from scripts.return_query_recipe import install
    parent,source=fixture(tmp_path,generation,released,stopped)
    before=source.read_text()
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert source.read_text()==before


def test_only_collector_query_selection_changes_and_is_idempotent(tmp_path):
    from scripts.return_query_recipe import install,patched
    parent,source=fixture(tmp_path)
    before=source.read_text();receipt=install(tmp_path,parent);after=source.read_text()
    assert before!=after and patched(after)==after
    functions=lambda text:{n.name:ast.dump(n) for n in ast.parse(text).body
        if isinstance(n,ast.FunctionDef)}
    a,b=functions(before),functions(after)
    assert {name for name in a if a[name]!=b[name]}=={'collect'}
    assert "obs[...,0,:,:].reshape(obs.shape[:2]+(-1,))" in after
    assert receipt['max_return_queries']==2 and receipt['min_tick_gap']==8
    assert receipt['native_verification_unchanged'] and receipt['return_multiplier']==2.
    install(tmp_path,parent);assert source.read_text()==after


def test_nested_proposal_survives_old_owner_and_candidate_is_inherited():
    from scripts.je10_supervise import choose_recipe,predeclare_continuity
    plan=dict(combat_fraction=.35,bridge_fraction=.20,return_dagger_weight={'enabled':True})
    proposal=dict(recipe='return_dagger_weight',return_query_diversity=True,
        reason='later verified distinct-cell return corrections',changed_factor='return quota only')
    decision=choose_recipe(plan,[],proposal)
    assert decision['proposal']['return_query_diversity']
    assert decision['return_dagger_weight'] and not decision['blocks_training']
    inherited=choose_recipe(dict(plan,return_query_diversity={'enabled':True}),[],proposal)
    assert inherited['return_query_diversity']
    assert 'return_query_diversity' in inspect.getsource(predeclare_continuity)


def test_successor_dispatches_and_seals_before_prepare():
    from scripts.je10_supervise import supervise
    source=inspect.getsource(supervise)
    assert source.index('query_diversity=install_return_queries(')<source.index('c.prepare(')
    assert "plan['return_query_diversity']=query_diversity" in source
    assert "declaration['return_query_diversity']=query_diversity" in source
    assert "'scripts/return_query_diversity.py','scripts/return_query_recipe.py'" in source
