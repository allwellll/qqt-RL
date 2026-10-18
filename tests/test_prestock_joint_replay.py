"""Joint replay preserves legacy selection and carries indivisible labels."""
import jax
import jax.numpy as jnp
import numpy as np
from scripts import je09_stage_h5c_dagger as legacy


def batch():
    n=4;obs=jnp.arange(n,dtype=jnp.float32).reshape(1,n,1,1,1)
    gv=jnp.stack([jnp.arange(n),jnp.arange(n)+10],-1)[None].astype(jnp.float32)
    mm=jnp.ones((1,n,5),bool);am=jnp.array([[[True,True,False]]*n])
    ctx=jnp.zeros((1,n,110)).at[:,:,48].set(1).at[:,:,53].set(1)
    ctx=ctx.at[:,:,54].set(3).at[:,:,55].set(1)
    pair=jnp.zeros((1,n,5,3),bool).at[:,:,:,0].set(ctx[:,:,48:53]>0)
    pair=pair.at[:,:,:,1].set(ctx[:,:,48:53]>0)
    ctx=ctx.at[:,:,95:110].set(pair.reshape(1,n,15))
    return (obs,gv,None,None,None,None,(mm,am),ctx)


def test_legacy_fifo_and_sampling_are_bitwise_unchanged_for_marginal_labels():
    from scripts.prestock_joint_replay import buffer,aggregate
    old=legacy.buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    extended=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=batch()
    for generation in [1,2,3]:
        key=jax.random.PRNGKey(generation)
        old,expected,stats=legacy.aggregate(old,data,key,jnp.int32(generation))
        extended,actual,joint_stats=jax.jit(aggregate)(extended,data,key,jnp.int32(generation))
        for name in old:np.testing.assert_array_equal(extended[name],old[name])
        for name in expected:np.testing.assert_array_equal(actual[name],expected[name])
        for name in stats:np.testing.assert_array_equal(joint_stats[name],stats[name])


def test_joint_pairs_remain_paired_to_observation_through_fifo_and_history():
    from scripts.prestock_joint_replay import buffer,aggregate
    old=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=list(batch());ctx=data[-1]
    pair=jnp.zeros((1,4,5,3),bool)
    for i in range(4):pair=pair.at[0,i,i,1].set(True).at[0,i,4,0].set(True)
    data[-1]=ctx.at[:,:,95:110].set(pair.reshape(1,4,15));data=tuple(data)
    for generation in [1,2,3]:
        old,replay,_=aggregate(old,data,jax.random.PRNGKey(generation),jnp.int32(generation))
        for i in np.flatnonzero(np.asarray(replay['valid'])):
            identity=int(replay['obs'][i,0,0,0])
            np.testing.assert_array_equal(np.argwhere(replay['joint_target'][i]),[[identity,1],[4,0]])
            np.testing.assert_array_equal(replay['ability_legal'][i],[True,True,False])


def test_illegal_only_joint_sets_never_enter_replay_as_valid_labels():
    from scripts.prestock_joint_replay import buffer,aggregate
    old=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=list(batch());pair=jnp.zeros((1,4,5,3),bool).at[:,:,0,2].set(True)
    data[-1]=data[-1].at[:,:,95:110].set(pair.reshape(1,4,15))
    updated,replay,stats=jax.jit(aggregate)(old,tuple(data),jax.random.PRNGKey(1),jnp.int32(1))
    assert not bool(updated['valid'].any()) and not bool(replay['valid'].any())
    assert float(stats['query'])==0 and float(stats['covered'])==0


def test_runtime_buffer_hook_does_not_recurse(monkeypatch):
    from scripts import prestock_joint_replay as joint
    monkeypatch.setattr(legacy,'buffer',joint.buffer)
    result=joint.buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    assert result['joint_target'].shape==(4,5,3)


def test_preparation_metadata_stays_paired_through_replay_and_restore():
    from scripts.prestock_joint_replay import buffer,aggregate
    import pickle
    old=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=list(batch());metadata=jnp.stack([jnp.ones(4),jnp.arange(4)+9,jnp.full(4,2),jnp.arange(4)%2],-1)[None]
    data[-1]=jnp.concatenate([data[-1],metadata],-1)
    for generation in [1,2,3]:
        old,replay,_=aggregate(old,tuple(data),jax.random.PRNGKey(generation),jnp.int32(generation))
        old=jax.tree.map(jnp.asarray,pickle.loads(pickle.dumps(jax.device_get(old))))
        for i in np.flatnonzero(np.asarray(replay['valid'])):
            identity=int(replay['obs'][i,0,0,0])
            assert bool(replay['preparation'][i]) and int(replay['revision'][i])==identity+9
            assert int(replay['objective'][i])==2 and bool(replay['opening'][i])==bool(identity%2)
