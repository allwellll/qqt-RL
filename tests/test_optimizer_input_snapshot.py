"""Passive production-input capture must preserve the training computation."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def tiny_batch(devices=2,steps=4,envs=4):
    count=devices*steps*envs
    obs=jnp.arange(count*3,dtype=jnp.float32).reshape(devices,steps,envs,3)
    gv=obs+100;acts=jnp.zeros((devices,steps,envs,2),jnp.int32)
    old=jnp.arange(count,dtype=jnp.float32).reshape(devices,steps,envs)/10
    adv=old*old;ret=adv+3;mm=jnp.ones((devices,steps,envs,5),bool);am=jnp.ones((devices,steps,envs,3),bool)
    context=obs+200
    return obs,gv,acts,old,adv,ret,(mm,am),context


def test_first_microbatch_preserves_actor_critic_permutations_and_full_gae_normalization():
    from scripts.optimizer_input_snapshot import first_microbatch
    batch=tiny_batch();keys=jax.random.split(jax.random.PRNGKey(18),2)
    result=first_microbatch(batch,keys,8)
    assert result['actor_rows_per_device']==4
    for device in range(2):
        ka,kc=jax.random.split(keys[device]);a=np.asarray(jax.random.permutation(ka,16))[:4];c=np.asarray(jax.random.permutation(kc,16))[:4];ids=np.r_[a,c]
        np.testing.assert_array_equal(result['indices'][device],ids)
        for original,compact in zip(jax.tree.leaves(batch),jax.tree.leaves(result['batch'])):
            np.testing.assert_array_equal(np.asarray(compact)[device],np.asarray(original)[device].reshape((16,)+original.shape[3:])[ids])
        adv=batch[4][device];expected=(adv-adv.mean())/(adv.std()+1e-8)
        np.testing.assert_allclose(result['normalized_advantage'][device],np.asarray(expected).ravel()[ids],rtol=1e-6,atol=1e-6)
        assert not np.isclose(np.asarray(result['normalized_advantage'][device,:4]).mean(),0), 'Do not renormalize the selected subset'


def test_capture_has_no_rng_batch_or_parameter_mutation_and_atomic_pickle_restore(tmp_path):
    from scripts.optimizer_input_snapshot import capture
    batch=tiny_batch();keys=jax.random.split(jax.random.PRNGKey(22),2)
    params={'actor':jnp.tile(jnp.arange(6,dtype=jnp.float32),(2,1)),'critic':jnp.ones((2,3))}
    adam={'count':jnp.ones(2,jnp.int32),'mu':params,'nu':params}
    replay={'valid':jnp.ones((2,4),bool),'round':jnp.ones((2,4),jnp.int32)}
    initial=[np.array(x) for x in jax.tree.leaves((params,adam,batch,keys,replay))]
    marker=capture(tmp_path,4,params,adam,batch,keys,replay,8)
    assert marker['passed'] and marker['controls_training'] is False
    assert marker['optimizer_steps_added']==marker['new_games']==0
    assert marker['actor_rows_per_device']==4 and marker['devices']==2
    with (tmp_path/marker['payload']).open('rb') as stream:saved=pickle.load(stream)
    for k in params:np.testing.assert_array_equal(saved['params'][k],params[k][0])
    np.testing.assert_array_equal(saved['keys'],keys)
    np.testing.assert_array_equal(saved['full_raw_advantages'],batch[4])
    for a,c in zip(initial,jax.tree.leaves((params,adam,batch,keys,replay))):np.testing.assert_array_equal(a,c)
    assert not list(tmp_path.rglob('*.tmp'))
    assert capture(tmp_path,4,params,adam,batch,keys,replay,8)==marker


def test_invalid_minibatch_and_replica_disagreement_fail_closed(tmp_path):
    from scripts.optimizer_input_snapshot import first_microbatch,capture
    batch=tiny_batch();keys=jax.random.split(jax.random.PRNGKey(20),2)
    for size in [0,3,14,40]:
        with pytest.raises(ValueError):first_microbatch(batch,keys,size)
    with pytest.raises(ValueError,match='replica'):
        capture(tmp_path,4,{'actor':jnp.array([[1.],[2.]])},{'count':jnp.zeros(2)},batch,keys,{'valid':jnp.ones((2,4),bool)},8)


def test_terminal_only_installer_never_patches_active_worker(tmp_path):
    import json
    from pathlib import Path
    from scripts.optimizer_snapshot_recipe import install,patched
    target=tmp_path/'scripts/je09_stage_h5c_worker.py';target.parent.mkdir()
    source=Path('scripts/je09_stage_h5c_worker.py').read_text();target.write_text(source)
    parent=tmp_path/'runs/prior/formal/arm/gen000128';parent.mkdir(parents=True)
    terminal=parent.parent/'TRAINER_EXIT.json';release=parent.parents[2]/'GPU_RELEASE_RECEIPT.json'
    terminal.write_text(json.dumps(dict(passed=True,generation=127,atomic_stop=False)))
    release.write_text(json.dumps(dict(released=True,compute_clients=0,devices=list(range(8)))))
    with pytest.raises(AssertionError):install(tmp_path,parent)
    assert target.read_text()==source
    terminal.write_text(json.dumps(dict(passed=True,generation=128,atomic_stop=False)))
    metadata=install(tmp_path,parent)
    assert target.read_text()==patched(source) and patched(patched(source))==patched(source)
    assert metadata['numerical_factor_changed'] is False
    assert metadata['points']==[4,16,64,128]
    assert patched(source).index('capture(')<patched(source).index('params,adam,ema,loss=learn(')
