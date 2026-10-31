import pickle
import jax
import jax.numpy as jnp
import numpy as np
from scripts import retention_core_replay as core


def rows(n=12):
    from scripts.prestock_joint_replay import buffer
    d=buffer(capacity=n,additions=2,obs_shape=(1,1,1),global_size=24)
    d={k:v for k,v in d.items() if k not in ('cursor','additions')}
    d.update(obs=jnp.arange(n,dtype=jnp.float32).reshape(n,1,1,1),
        gv=jnp.zeros((n,24)).at[:,9].set(1).at[:,13:15].set(1),
        valid=jnp.ones(n,bool),kind=jnp.full(n,3),round=jnp.ones(n,jnp.int32),
        legal=jnp.ones((n,5),bool),ability_legal=jnp.ones((n,3),bool),
        joint_target=jnp.zeros((n,5,3),bool).at[:,0,0].set(True),
        scene=jnp.array([8,9]*((n+1)//2))[:n],custody=jnp.ones(n,bool))
    return d


def test_unproved_synthetic_own_bun_and_illegal_labels_are_rejected():
    d=rows();d['custody']=d['custody'].at[0].set(False)
    d['scene']=d['scene'].at[1].set(2)
    d['gv']=d['gv'].at[2,14].set(0)
    d['joint_target']=d['joint_target'].at[3].set(False)
    bank=core.empty(d,per_stratum=4)
    bank=jax.jit(core.merge)(bank,d)
    ids=np.asarray(bank['obs'][bank['valid']]).reshape(-1)
    assert not set(ids)&{0,1,2,3}


def test_duplicate_content_ignores_clock_and_generation_and_preserves_joint_pair():
    d=rows(4);bank=core.empty(d,per_stratum=4)
    bank=core.merge(bank,d)
    duplicate=dict(d,gv=d['gv'].at[:,0].set(.9),round=d['round']+20)
    again=core.merge(bank,duplicate)
    assert int(again['valid'].sum())==4
    for name in ['obs','joint_target','round']:np.testing.assert_array_equal(bank[name],again[name])


def test_stratum_quota_cannot_be_flooded_and_selection_is_not_fifo():
    d=rows(20);d['gv']=d['gv'].at[10:,10].set(1)
    bank=core.merge(core.empty(d,per_stratum=2),d)
    assert int(bank['valid'].sum())==8
    for group in range(4):assert int(((core.stratum(bank)==group)&bank['valid']).sum())==2
    repeated=dict(d,round=d['round']+100)
    after=core.merge(bank,repeated)
    np.testing.assert_array_equal(bank['obs'],after['obs'])
    assert int(after['round'][after['valid']].max())==1


def test_only_half_of_existing_history_return_slots_change_and_restore_is_exact():
    d=rows(8);bank=core.merge(core.empty(d,per_stratum=2),d)
    replay={k:v for k,v in rows(8).items() if k not in ['scene','custody']}
    replay['obs']=replay['obs']+100
    replay['kind']=replay['kind'].at[4].set(1)
    out,count=core.replace_history(replay,bank,jax.random.PRNGKey(42),history_start=4)
    np.testing.assert_array_equal(out['obs'][:5],replay['obs'][:5])
    assert int(count)==2
    assert set(out)==set(replay)
    restored=pickle.loads(pickle.dumps(jax.device_get(bank)))
    again,_=jax.jit(core.replace_history,static_argnames='history_start')(replay,restored,jax.random.PRNGKey(42),history_start=4)
    for k in out:np.testing.assert_array_equal(out[k],again[k])


def test_empty_core_is_bitwise_baseline():
    d=rows(8);bank=core.empty(d,per_stratum=2)
    replay={k:v for k,v in d.items() if k not in ['scene','custody']}
    out,count=core.replace_history(replay,bank,jax.random.PRNGKey(1),4)
    assert int(count)==0
    for k in out:np.testing.assert_array_equal(out[k],replay[k])


def test_collector_integration_fifo_exact_and_core_proof_survives_serialization(monkeypatch):
    from scripts import return_error_replay as baseline
    from scripts.prestock_joint_replay import buffer
    from test_prestock_joint_replay import batch
    monkeypatch.setattr(baseline,'target_mass',lambda actor,d:jnp.full(d['valid'].size,.5))
    data=list(batch());data[1]=jnp.zeros((1,4,24)).at[:,:,9].set(1).at[:,:,13:15].set(1)
    data[-1]=data[-1].at[:,:,0].set(8).at[:,:,58].set(1)
    data=tuple(data)
    fifo=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=24)
    template={k:v for k,v in fifo.items() if k not in ['cursor','additions']}
    template.update(scene=jnp.zeros(4,jnp.int32),custody=jnp.zeros(4,bool))
    bank=core.empty(template,per_stratum=2);extended=dict(fifo,retention_core=bank)
    for generation in [1,2,3]:
        key=jax.random.PRNGKey(generation)
        expected,baseline_replay,_=core._BASE_AGGREGATE(fifo,data,key,jnp.int32(generation),{})
        updated,replay,stats=jax.jit(core.aggregate)(extended,data,key,jnp.int32(generation),{})
        for name in expected:np.testing.assert_array_equal(expected[name],updated[name])
        np.testing.assert_array_equal(replay['obs'][:2],baseline_replay['obs'][:2])
        assert int(updated['retention_core']['valid'].sum())>0
        assert float(stats['core_selected_joint_mass_sum'])==.5*int(stats['core_replaced_history_return'])
        assert bool(jnp.all(jnp.where(updated['retention_core']['valid'],updated['retention_core']['custody'],True)))
        recovered=pickle.loads(pickle.dumps(jax.device_get(extended)))
        again=jax.jit(core.aggregate)(recovered,data,key,jnp.int32(generation),{})
        for a,b in zip(jax.tree.leaves((updated,replay,stats)),jax.tree.leaves(again)):np.testing.assert_array_equal(a,b)
        fifo=expected;extended=updated
