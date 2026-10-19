"""Future telemetry installation cannot alter an active numerical recipe."""
import ast
import json
from pathlib import Path
import pytest


def fixture(tmp_path,generation=128,released=True):
    run=tmp_path/'runs/parent';parent=run/'formal/arm/gen000128'
    parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(generation=generation,passed=True,atomic_stop=False)))
    (run/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=0 if released else 1)))
    worker=tmp_path/'scripts/je10_worker.py';worker.parent.mkdir()
    worker.write_text(Path('scripts/je10_worker.py').read_text())
    return parent,worker


def test_future_patch_is_idempotent_and_keeps_rollout_update_and_losses():
    from scripts.ppo_advantage_diagnostic_recipe import patched_worker
    installed=Path('scripts/je10_worker.py').read_text()
    # Reconstruct the pre-install host block, keeping all numerical code exact.
    source=installed.replace('    # Successor-only normalized PPO advantage telemetry.\n    from scripts.ppo_advantage_diagnostics import replica_moments, normalized_diagnostic\n    advantage_device_moments=jax.pmap(replica_moments)\n','')
    source=source.replace('            advantage_moments=np.asarray(advantage_device_moments(batch[4])) if mismatch.any() else None\n','')
    source=source.replace('                collector_evidence=selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index)))\n','')
    source=source.replace("                    **collector_evidence,\n                    **normalized_diagnostic(collector_evidence['collector_gae_advantage'],advantage_moments[int(dev)]),",'                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),')
    patched=patched_worker(source)
    assert patched!=source and patched_worker(patched)==patched
    assert patched_worker(installed)==installed
    ast.parse(patched)
    # Existing collect/PPO/summarize and state-update operations stay verbatim.
    for line in source.splitlines():
        if any(text in line for text in ['states,aux,key,batch,stats=roll(', 'params,adam,ema,loss=learn(', 'norms,finite=summarize(', "live.update("]):
            assert line in patched.splitlines()
    assert patched.count('advantage_device_moments=jax.pmap(replica_moments)')==1
    assert 'np.asarray(advantage_device_moments(batch[4]))' in patched
    assert '**normalized_diagnostic(collector_evidence' in patched


@pytest.mark.parametrize('generation,released',[(127,True),(128,False)])
def test_installer_requires_fixed_terminal_and_gpu_release(tmp_path,generation,released):
    from scripts.ppo_advantage_diagnostic_recipe import install
    parent,worker=fixture(tmp_path,generation,released);source=worker.read_text()
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert worker.read_text()==source


def test_install_writes_only_diagnostics_after_terminal(tmp_path):
    from scripts.ppo_advantage_diagnostic_recipe import install
    parent,worker=fixture(tmp_path);first=install(tmp_path,parent);source=worker.read_text()
    assert install(tmp_path,parent)==first
    assert worker.read_text()==source
    assert first['numerical_recipe_changed'] is False
    assert first['host_scalars_per_device']==2
