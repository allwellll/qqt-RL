"""Successor-only replay allocation must preserve rare stages and FIFO authority."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np


def fixture():
    # Ninety navigation rows must not erase four ordinary return rows or one
    # rare native-verified opening. The ID lets tests follow paired payloads.
    n=100
    kind=jnp.full(n,2).at[90:94].set(3).at[94:97].set(1).at[97:99].set(2)
    prep=jnp.arange(n)<90;opening=jnp.arange(n)==99
    return dict(valid=jnp.ones(n,bool),kind=kind,preparation=prep|opening,opening=opening,
        identity=jnp.arange(n),round=jnp.zeros(n,jnp.int32))


def test_navigation_dominance_does_not_erase_ordinary_or_rare_opening():
    from scripts.phase_balanced_joint_replay import choose
    data=fixture();got=jax.jit(lambda d,k:choose(d,k,32))(data,jax.random.PRNGKey(4))
    ids=np.asarray(got['identity']);assert len(ids)==32
    assert int(np.asarray(got['opening']).sum())==1
    assert int((~np.asarray(got['preparation'])).sum())==16
    assert int((np.asarray(got['preparation'])&~np.asarray(got['opening'])).sum())==15
    for kind in [1,2,3]:assert np.any((np.asarray(got['kind'])==kind)&~np.asarray(got['preparation']))
    assert int(((np.asarray(got['kind'])==3)&~np.asarray(got['preparation'])).sum())==12
    for kind in [1,2]:assert int(((np.asarray(got['kind'])==kind)&~np.asarray(got['preparation'])).sum())==2
    np.testing.assert_array_equal(got['kind'],data['kind'][ids])


def test_no_preparation_is_bitwise_legacy_uniform_choice():
    from scripts.phase_balanced_joint_replay import choose
    d=fixture();d['preparation']=jnp.zeros(100,bool);d['opening']=jnp.zeros(100,bool)
    d['valid']=jnp.arange(100)%3!=0;k=jax.random.PRNGKey(5)
    mass=d['valid'].astype(jnp.float32);ids=jax.random.choice(k,100,(32,),p=mass/mass.sum())
    got=choose(d,k,32)
    for name,value in d.items():np.testing.assert_array_equal(got[name],value[ids])


def test_absent_groups_and_invalid_rows_never_mint_training_labels():
    from scripts.phase_balanced_joint_replay import choose
    d=fixture();d['valid']=jnp.arange(100)>=90
    got=choose(d,jax.random.PRNGKey(6),32)
    assert bool(got['valid'].all()) and int(got['opening'].sum())==1
    assert int((~got['preparation']).sum())==31
    d['valid']=jnp.arange(100)==99;got=choose(d,jax.random.PRNGKey(7),32)
    np.testing.assert_array_equal(got['identity'],jnp.full(32,99))
    d['valid']=jnp.zeros(100,bool);got=choose(d,jax.random.PRNGKey(8),32)
    assert not bool(got['valid'].any())


def test_saved_payload_and_next_sampling_are_exact():
    from scripts.phase_balanced_joint_replay import choose
    d=fixture();copy=pickle.loads(pickle.dumps(d));k=jax.random.PRNGKey(9)
    a=choose(d,k,32);b=choose(copy,k,32)
    for name in a:np.testing.assert_array_equal(a[name],b[name])


def test_aggregate_keeps_fifo_and_current_history_authority():
    from scripts import phase_balanced_joint_replay as candidate
    from scripts import prestock_joint_replay as uniform
    from test_prestock_joint_replay import batch
    old=uniform.buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=list(batch());metadata=jnp.array([[[1,4,2,0],[0,0,0,0],[1,5,2,1],[0,0,0,0]]])
    data[-1]=jnp.concatenate([data[-1],metadata],-1);data=tuple(data)
    for generation in [1,2,3]:
        key=jax.random.PRNGKey(generation)
        expected,_,prior=uniform.aggregate(old,data,key,jnp.int32(generation))
        actual,replay,stats=jax.jit(candidate.aggregate)(old,data,key,jnp.int32(generation))
        for name in expected:np.testing.assert_array_equal(actual[name],expected[name])
        for name in prior:
            if name not in ['history_fraction','new_fraction']:np.testing.assert_array_equal(stats[name],prior[name])
        assert replay['valid'].shape==(4,)
        assert bool((replay['round'][:2]==generation).all())
        if generation==1:assert bool((replay['round'][2:]==generation).all())
        else:assert bool((replay['round'][2:]<generation).all())
        restored=pickle.loads(pickle.dumps(old))
        repeated=candidate.aggregate(restored,data,key,jnp.int32(generation))
        for a,b in zip(jax.tree.leaves((actual,replay,stats)),jax.tree.leaves(repeated)):np.testing.assert_array_equal(a,b)
        old=actual


def test_aggregate_ordinary_replay_and_statistics_are_exact():
    from scripts import phase_balanced_joint_replay as candidate
    from scripts import prestock_joint_replay as uniform
    from test_prestock_joint_replay import batch
    old=uniform.buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    for generation in [1,2,3]:
        key=jax.random.PRNGKey(generation);expected=uniform.aggregate(old,batch(),key,jnp.int32(generation))
        actual=candidate.aggregate(old,batch(),key,jnp.int32(generation))
        for a,b in zip(jax.tree.leaves(expected[:2]),jax.tree.leaves(actual[:2])):np.testing.assert_array_equal(a,b)
        for name in expected[2]:np.testing.assert_array_equal(actual[2][name],expected[2][name])
        old=actual[0]


def test_actual_replay_telemetry_matches_each_current_history_half():
    from scripts import phase_balanced_joint_replay as candidate
    from scripts import prestock_joint_replay as uniform
    from test_prestock_joint_replay import batch
    old=uniform.buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    data=list(batch());metadata=jnp.array([[[1,4,2,0],[0,0,0,0],[1,5,2,1],[0,0,0,0]]])
    data[-1]=jnp.concatenate([data[-1],metadata],-1);data=tuple(data)
    for generation in [1,2,3]:
        updated,replay,stats=candidate.aggregate(old,data,jax.random.PRNGKey(generation),jnp.int32(generation))
        for half,selection in [('current',slice(0,2)),('history',slice(2,4))]:
            valid=replay['valid'][selection];prep=replay['preparation'][selection]
            opening=replay['opening'][selection];kind=replay['kind'][selection]
            expected=dict(valid=valid,ordinary=valid&~prep,returning=valid&~prep&(kind==3),
                drop=valid&~prep&((kind==1)|(kind==4)),stock_ready=valid&~prep&(kind==2),
                navigation=valid&prep&~opening,opening=valid&opening)
            for name,mask in expected.items():
                assert int(stats['replay_'+half+'_'+name])==int(mask.sum())
            assert int(stats['replay_'+half+'_valid'])==sum(int(stats['replay_'+half+'_'+k]) for k in ['ordinary','navigation','opening'])
        old=updated


def test_selected_collector_advantage_keeps_critic_and_value_target_provenance():
    from scripts.phase_balanced_joint_replay import selected_ppo_diagnostic
    from test_learner_trace_sampling import OnlySelectedRow
    fields=[OnlySelectedRow(np.float32(v)) for v in [-.7,-.25,1.5]]
    index=(2,8,12);batch=(None,None,None,*fields)
    got=selected_ppo_diagnostic(batch,index)
    assert got['collector_gae_advantage']==-.25
    assert got['collector_value_target']==1.5
    assert got['collector_value_estimate']==1.75
    assert np.isclose(got['behavior_action_log_probability'],-.7)
    assert all(field.indices==[index] for field in fields)
