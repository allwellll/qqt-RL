import ast
import inspect
import json
from pathlib import Path
import pytest


def fixture(tmp_path,generation=128,released=True,stopped=False):
    parent=tmp_path/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=stopped)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1)))
    path=tmp_path/'scripts/je10_worker.py';path.parent.mkdir()
    source=Path('scripts/je10_worker.py').read_text()
    # Build the prior dispatch even after the candidate has been installed.
    source=source.replace('    from scripts import return_error_replay\n    aggregate=jax.pmap(return_error_replay.aggregate)',
        '    aggregate=jax.pmap(dagger.aggregate)')
    source=source.replace("jnp.full(count,g,jnp.int32),live['params']['actor'])",
        'jnp.full(count,g,jnp.int32))')
    path.write_text(source)
    return parent,path


@pytest.mark.parametrize('generation,released,stopped',[(127,True,False),(128,False,False),(128,True,True)])
def test_never_install_into_live_or_invalid_run(tmp_path,generation,released,stopped):
    from scripts.return_error_recipe import install
    parent,path=fixture(tmp_path,generation,released,stopped);before=path.read_text()
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert path.read_text()==before


def test_only_worker_aggregation_receives_current_raw_actor(tmp_path):
    from scripts.return_error_recipe import install,patched_worker
    parent,path=fixture(tmp_path);before=path.read_text();receipt=install(tmp_path,parent);after=path.read_text()
    assert before!=after and patched_worker(after)==after
    functions=lambda s:{n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(before),functions(after)
    assert {k for k in a if a[k]!=b[k]}=={'worker'}
    assert "jnp.full(count,g,jnp.int32),live['params']['actor'])" in after
    assert receipt['return_multiplier']==2. and receipt['other_phase_draws_unchanged']
    install(tmp_path,parent);assert path.read_text()==after


def test_nested_old_owner_proposal_and_fresh_followups_preserve_error_replay():
    from scripts.je10_supervise import choose_recipe,predeclare_continuity
    plan=dict(combat_fraction=.35,bridge_fraction=.2,return_dagger_weight={'enabled':True},
        return_query_diversity={'enabled':True})
    proposal=dict(recipe='return_dagger_weight',return_error_replay=True,
        reason='specific native return states underfit despite replay exposure',
        changed_factor='return replay sampling weights only')
    selected=choose_recipe(plan,[],proposal)
    assert selected['proposal']['return_error_replay'] and selected['return_error_replay']
    assert selected['return_query_diversity'] and selected['return_dagger_weight']
    inherited=choose_recipe(dict(plan,return_error_replay={'enabled':True}),[],None)
    assert inherited['return_error_replay'] and not inherited['blocks_training']
    assert 'return_error_replay' in inspect.getsource(predeclare_continuity)


def test_terminal_dispatch_seals_candidate_before_smoke():
    from scripts.je10_supervise import supervise
    source=inspect.getsource(supervise)
    assert source.index('error_replay=install_return_error(')<source.index('c.prepare(')
    assert "decision.get('proposal',{}).get('return_error_replay')" in source
    assert "plan['return_error_replay']=error_replay" in source
    assert "declaration['return_error_replay']=error_replay" in source
    assert "'scripts/return_error_replay.py','scripts/return_error_recipe.py'" in source
