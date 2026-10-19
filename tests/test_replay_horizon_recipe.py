"""A longer causal native FIFO must retain history without changing row budgets."""
from pathlib import Path
import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_patch_changes_only_capacity_and_fresh_joint_default():
    from scripts.replay_horizon_recipe import patched
    source=Path('scripts/je09_stage_h5c_dagger.py').read_text()
    changed=patched(source)
    assert changed==source.replace('CAPACITY=2048\n','CAPACITY=4096\n')
    assert patched(changed)==changed
    with pytest.raises(AssertionError):patched('CAPACITY=1024\n')
    namespace={};exec(compile(changed,'isolated-longer-fifo','exec'),namespace)
    joint=Path('scripts/prestock_joint_replay.py').read_text()
    joint=joint.replace('from scripts import je09_stage_h5c_dagger as legacy','')
    copy={'legacy':SimpleNamespace(**namespace)};exec(compile(joint,'isolated-joint-default','exec'),copy)
    state=copy['buffer'](obs_shape=(1,1,1),global_size=2)
    assert state['valid'].shape==(4096,) and int(state['additions'])==256
    assert not bool(state['valid'].any())
    assert not bool(state['joint_target'].any())


@pytest.mark.parametrize('generation,released,atomic,clients',[(127,True,False,0),(128,False,False,1),(128,True,True,0),(128,True,False,1),(128,True,False,0)])
def test_install_requires_normal_terminal_and_all_gpu_release(tmp_path,generation,released,atomic,clients):
    from scripts.replay_horizon_recipe import install,patched
    target=tmp_path/'scripts/je09_stage_h5c_dagger.py';target.parent.mkdir()
    source=Path('scripts/je09_stage_h5c_dagger.py').read_text();target.write_text(source)
    parent=tmp_path/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    (parent.parent/'TRAINER_EXIT.json').write_text(json.dumps(dict(passed=True,generation=generation,atomic_stop=atomic)))
    (parent.parents[2]/'GPU_RELEASE_RECEIPT.json').write_text(json.dumps(dict(released=released,compute_clients=clients,devices=list(range(8)))))
    if generation!=128 or not released or atomic or clients:
        with pytest.raises(AssertionError):install(tmp_path,parent)
        assert target.read_text()==source
    else:
        result=install(tmp_path,parent);assert target.read_text()==patched(source)
        assert result['capacity_per_device']==4096
        assert result['new_per_generation_per_device']==256
        assert result['nominal_fifo_generations']==16
        assert result['current_slots']==result['history_slots']==64
        assert result['return_loss_multiplier']==2.
        install(tmp_path,parent);assert target.read_text()==patched(source)


def test_real_aggregator_retains_early_generation_at_g12_and_expires_at_g17(monkeypatch):
    from scripts.replay_horizon_recipe import patched
    from scripts import je09_stage_h5c_dagger as legacy,return_error_replay as replay
    from scripts.prestock_joint_replay import buffer
    from test_replay_batch_recipe import mixed_batch
    patched(Path('scripts/je09_stage_h5c_dagger.py').read_text())
    monkeypatch.setattr(replay,'target_mass',lambda actor,data:jnp.full(data['valid'].shape,.5))
    results={}
    for capacity in [2048,4096]:
        monkeypatch.setattr(legacy,'CAPACITY',capacity)
        aggregate=jax.jit(replay.aggregate)
        state=buffer(capacity=capacity,additions=256,obs_shape=(1,1,1),global_size=2)
        selected=[]
        for generation in range(1,18):
            batch=list(mixed_batch(512));batch[0]=batch[0]+generation*1000
            state,payload,stats=aggregate(state,tuple(batch),jax.random.PRNGKey(generation),jnp.int32(generation),{})
            assert int(state['cursor'])==(generation*256)%capacity
            assert payload['valid'].shape==(128,)
            assert bool(payload['valid'].all())
            assert int(stats['replay_current_returning'])==int(stats['replay_history_returning'])==24
            assert bool((payload['round'][:64]==generation).all())
            selected.append(np.asarray(payload['obs'][:64]))
            if generation==12:
                resident=np.asarray(state['round'])[np.asarray(state['valid'])]
                assert resident.min()==(5 if capacity==2048 else 1)
                results[capacity]=np.asarray(state['obs'])[np.asarray(state['valid'])]
            if generation==16 and capacity==4096:
                assert int(state['round'][state['valid']].min())==1
            if generation==17 and capacity==4096:
                assert int(state['round'][state['valid']].min())==2
        results[str(capacity)+'_current']=selected
    for a,b in zip(results['2048_current'],results['4096_current']):np.testing.assert_array_equal(a,b)
    assert not (results[2048]<5000).any()
    assert (results[4096]<2000).any()


def test_terminal_relay_carries_horizon_flag_and_correct_actual_metadata():
    import inspect
    from scripts.je10_supervise import choose_recipe,predeclare_continuity,supervise
    plan=dict(combat_fraction=.35,bridge_fraction=.2,return_dagger_weight=True,replay_horizon=True)
    proposal=dict(recipe='return_dagger_weight',replay_horizon=True,reason='Fixed actual payload expired after eight generations',changed_factor='Native FIFO2048 to4096 only')
    assert choose_recipe(plan,[],proposal)['replay_horizon']
    assert choose_recipe(plan,[],None)['replay_horizon']
    source=inspect.getsource(supervise)
    assert "decision.get('proposal',{}).get('replay_horizon')" in source
    assert source.index('horizon=install_replay_horizon(')<source.index('c.prepare(')
    assert "plan['replay_horizon']=horizon" in source
    assert "declaration['replay_horizon']=horizon" in source
    assert "plan['dagger']['capacity_per_device']=horizon['capacity_per_device']" in source
    assert "plan['dagger']['replay_per_microbatch']=128" in source
    assert "'scripts/replay_horizon_recipe.py'" in source
    assert 'replay_horizon' in inspect.getsource(predeclare_continuity)
