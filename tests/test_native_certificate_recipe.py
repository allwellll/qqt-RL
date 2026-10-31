"""Native progress labels are installed only at a fixed run boundary."""
import ast
from pathlib import Path
import pytest
from scripts.native_certificate_recipe import patched_verifier


def test_patch_reuses_existing_native_endpoints_and_preserves_safety_and_query_budget():
    source=Path('scripts/je09_stage_h6_ops.py').read_text()
    patched=patched_verifier(source)
    ast.parse(patched)
    assert patched.count('env.step(')==source.count('env.step(')
    assert 'certify_native_positions(' in patched
    assert 'labels[:, 7] * accepted' in patched
    assert 'slots_per_fragment=slots' in patched
    assert 'safe = temporal_safe & one_step_safe' in patched
    assert patched_verifier(patched)==patched


def test_patch_rejects_unrecognized_training_source():
    with pytest.raises(AssertionError):patched_verifier('unexpected source')


def test_native_certificate_followup_changes_oracle_only_and_keeps_training_running():
    from scripts.je10_supervise import choose_recipe
    plan={'combat_fraction':.35,'bridge_fraction':.20}
    proposal={'recipe':'native_one_step_certificate','reason':'fixed Review endpoint conflict',
              'changed_factor':'strict route oracle native-step progress set'}
    recipe=choose_recipe(plan,[],proposal)
    assert recipe['combat_fraction']==.35 and recipe['bridge_fraction']==.20
    assert recipe['route_mass']==.10
    assert recipe['changed_factor']==proposal['changed_factor']
    assert recipe['native_one_step_certificate'] and recipe['blocks_training'] is False


def test_install_requires_terminal_and_verified_gpu_release(tmp_path):
    import json
    from scripts.native_certificate_recipe import install
    source=Path('scripts/je09_stage_h6_ops.py').read_text()
    target=tmp_path/'scripts/je09_stage_h6_ops.py';target.parent.mkdir();target.write_text(source)
    base=tmp_path/'run';parent=base/'formal/arm/gen000128';parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):install(tmp_path,parent)
    assert target.read_text()==source
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps({'generation':128,'passed':True,'atomic_stop':False}))
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps({'released':False,'compute_clients':1}))
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert target.read_text()==source
    (base/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps({'released':True,'compute_clients':0}))
    assert install(tmp_path,parent)['additional_env_steps']==0
    assert 'certify_native_positions(' in target.read_text()
