"""Future-only installer: never mutate the current numerical worker."""
import ast
import json
from pathlib import Path

import pytest


def baseline():
    source=Path('scripts/je09_stage_h5c_ops.py').read_text()
    installed="""            from scripts.native_return_retention import movement_scale,retained_kl
            scale=movement_scale(context[ii[:half]],s[:half],m[:half],b[:half])
            penalty+=retained_kl(kl[:,0],kl[:,1],jnp.where(combat,strong,weak),scale)"""
    return source.replace(installed,
        '            penalty+=jnp.mean(jnp.where(combat,strong,weak)*(kl[:,0]+kl[:,1]))')


def test_candidate_patch_only_changes_old_teacher_movement_penalty():
    from scripts.native_return_retention_recipe import patched
    before=baseline();after=patched(before)
    assert after!=before
    assert patched(after)==after
    trees=[ast.parse(s) for s in [before,after]]
    nodes=[{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef)} for t in trees]
    assert {k for k in nodes[0] if nodes[0][k]!=nodes[1][k]}=={'ppo'}
    assert 'retained_kl(kl[:,0],kl[:,1]' in after
    assert "daggerloss=dagger.replay_loss(p,dagger_replay)" in after
    assert 'parent_kl=parent_retention_kl(' in after
    with pytest.raises(AssertionError):patched('def ppo(): pass')


@pytest.mark.parametrize('generation,released,stop',[
    (127,True,False),(128,False,False),(128,True,True),(128,True,False),
])
def test_install_requires_normal_terminal_and_eight_cards_release(tmp_path,generation,released,stop):
    from scripts.native_return_retention_recipe import install,patched
    parent=tmp_path/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(
        passed=True,generation=generation,atomic_stop=stop)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(
        released=released,compute_clients=0 if released else 1,devices=list(range(8)))))
    source=tmp_path/'scripts/je09_stage_h5c_ops.py';source.parent.mkdir();source.write_text(baseline())
    worker=tmp_path/'scripts/je10_worker.py';worker.write_text(Path('scripts/je10_worker.py').read_text())
    before=source.read_text()
    if generation!=128 or not released or stop:
        with pytest.raises(AssertionError):install(tmp_path,parent)
        assert source.read_text()==before
    else:
        receipt=install(tmp_path,parent)
        assert source.read_text()==patched(before)
        assert receipt['bomb_kl_unchanged'] and receipt['combat_danger_retention_unchanged']
        assert not receipt['capability_gate']
        install(tmp_path,parent);assert source.read_text()==patched(before)


def test_successor_can_inherit_nested_candidate_and_predeclare_continuity(tmp_path):
    import inspect
    from scripts.je10_supervise import choose_recipe,predeclare_continuity,supervise
    plan=dict(combat_fraction=.35,bridge_fraction=.2,return_dagger_weight=True,
        replay_batch=True,native_return_retention=True)
    proposal=dict(recipe='return_dagger_weight',native_return_retention=True,
        reason='Fixed native return fitting and retention objective evidence',
        changed_factor='Certified safe native return movement KL only')
    chosen=choose_recipe(plan,[],proposal)
    assert chosen['native_return_retention'] and chosen['replay_batch']
    assert choose_recipe(plan,[],None)['native_return_retention']
    declaration=dict(round_number=18,namespace='runs/prior',evaluation={},
        terminal_relay=dict(successor='runs/following'))
    assert predeclare_continuity(tmp_path,plan,declaration)['native_return_retention']
    source=inspect.getsource(supervise)
    assert "decision.get('proposal',{}).get('native_return_retention')" in source
    assert source.index('native_retention=install_native_retention(')<source.index('c.prepare(')
    assert "plan['native_return_retention']=native_retention" in source
    assert "declaration['native_return_retention']=native_retention" in source
    assert "'scripts/native_return_retention.py','scripts/native_return_retention_recipe.py'" in source


def test_passive_worker_exposure_installs_once_without_changing_learn_call():
    from scripts.native_return_retention_recipe import patched_worker
    source=Path('scripts/je10_worker.py').read_text()
    # Reconstruct baseline even after the future terminal-only installation.
    lines=source.splitlines(keepends=True)
    nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="p.get('native_return_retention')"]
    for node in sorted(nodes,key=lambda n:n.lineno,reverse=True):
        del lines[node.lineno-1:node.end_lineno]
    source=''.join(lines)
    after=patched_worker(source)
    assert after!=source and patched_worker(after)==after
    assert 'exposure_summary(ctx,np.asarray(batch[1])' in after
    assert 'controls_training' in after
    before_tree=ast.parse(source);after_tree=ast.parse(after)
    calls=lambda tree:[ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id in ['learn','roll','aggregate']]
    assert calls(before_tree)==calls(after_tree)
