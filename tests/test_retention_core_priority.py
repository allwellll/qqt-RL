"""Counterfactual sampler invariants; import never installs a training recipe."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from scripts import retention_core_replay as core
from scripts import retention_core_priority as priority
from test_retention_core_replay import rows


def bank():
    d=rows(16)
    d['gv']=d['gv'].at[8:,10].set(1)
    return d


def test_layers_keep_equal_probability_and_every_eligible_row_has_support():
    d=bank();mass=jnp.linspace(0,1,16)
    w=priority.selection_weights(d,mass);group=core.stratum(d)
    for layer in range(4):np.testing.assert_allclose(w[group==layer].sum(),.25,rtol=1e-6)
    assert np.all(np.asarray(w)>0)
    assert float(w[0])>float(w[2])


def test_unproved_retired_illegal_dead_or_own_bun_rows_have_zero_support():
    d=bank();d['custody']=d['custody'].at[0].set(False)
    d['scene']=d['scene'].at[1].set(2)
    d['joint_target']=d['joint_target'].at[2].set(False)
    d['gv']=d['gv'].at[3,9].set(0).at[4,14].set(0)
    w=np.asarray(priority.selection_weights(d,jnp.zeros(16)))
    np.testing.assert_array_equal(w[:5],0)
    assert np.all(w[5:]>0)


def test_priority_cannot_backpropagate_through_fit_or_sampling_probability():
    d=bank();mass=jnp.linspace(0,1,16)
    grad=jax.grad(lambda m:(priority.selection_weights(d,m)*jnp.arange(16)).sum())(mass)
    np.testing.assert_array_equal(grad,np.zeros(16))


def test_equal_within_each_layer_is_exact_baseline_despite_different_layer_fit():
    d=bank();replay={k:v for k,v in d.items() if k not in ['scene','custody']}
    replay['obs']=replay['obs']+1000
    mass=jnp.array([.1,.3,.7,.9])[core.stratum(d)]
    key=jax.random.PRNGKey(342)
    expected,n=core.replace_history(replay,d,key,8)
    actual,m=priority.replace_history(replay,d,key,8,mass)
    assert int(n)==int(m)
    for k in actual:np.testing.assert_array_equal(actual[k],expected[k])


def test_empty_bank_is_exact_baseline_without_inventing_custody():
    d=bank();empty=core.empty(d,per_stratum=4)
    replay={k:v for k,v in d.items() if k not in ['scene','custody']}
    actual,n=priority.replace_history(replay,empty,jax.random.PRNGKey(3),8,jnp.zeros(16))
    assert int(n)==0
    for k in actual:np.testing.assert_array_equal(actual[k],replay[k])


def test_only_existing_half_history_return_slots_change_and_harder_rows_are_drawn():
    d=bank();replay={k:v for k,v in rows(4096).items() if k not in ['scene','custody']}
    replay['obs']=replay['obs']+1000
    replay['kind']=replay['kind'].at[2048:2304].set(2)
    replay['preparation']=replay['preparation'].at[2304:2560].set(True)
    mass=jnp.where(jnp.arange(16)%4<2,0.,1.)
    key=jax.random.PRNGKey(788)
    actual,n=priority.replace_history(replay,d,key,2048,mass)
    returning=replay['valid']&(replay['kind']==3)&~replay['preparation']&(jnp.arange(4096)>=2048)
    slots=np.asarray(returning&(jnp.cumsum(returning)%2==1))
    assert int(n)==int(slots.sum())==768
    for k in actual:np.testing.assert_array_equal(actual[k][~slots],replay[k][~slots])
    selected=np.asarray(actual['obs'])[slots].reshape(-1).astype(int)
    assert set(selected)<=set(range(16))
    assert np.mean(np.asarray(mass)[selected]<.5)>.85
    recovered=pickle.loads(pickle.dumps(jax.device_get(d)))
    again,m=jax.jit(priority.replace_history,static_argnames='history_start')(replay,recovered,key,2048,mass)
    assert int(n)==int(m)
    for k in actual:np.testing.assert_array_equal(actual[k],again[k])


def test_actual_collector_writer_is_unchanged_and_final_statistics_are_recomputed(monkeypatch):
    from scripts.prestock_joint_replay import buffer
    from test_prestock_joint_replay import batch
    from scripts import return_error_replay as baseline
    monkeypatch.setattr(baseline,'target_mass',lambda actor,d:jnp.clip(d['obs'].reshape((d['valid'].size,-1))[:,0]/16,0,1))
    data=list(batch());data[1]=jnp.zeros((1,4,24)).at[:,:,9].set(1).at[:,:,13:15].set(1)
    data[-1]=data[-1].at[:,:,0].set(8).at[:,:,58].set(1);data=tuple(data)
    fifo=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=24)
    extended=dict(fifo,retention_core=bank());key=jax.random.PRNGKey(1)
    expected,uniform,oldstats=core.aggregate(extended,data,key,jnp.int32(2),{})
    updated,replay,stats=jax.jit(priority.aggregate)(extended,data,key,jnp.int32(2),{})
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(updated)):np.testing.assert_array_equal(a,b)
    for k in replay:np.testing.assert_array_equal(replay[k][:2],uniform[k][:2])
    ret=replay['valid']&(replay['kind']==3)&~replay['preparation']&(jnp.arange(4)>=2)
    mass=baseline.target_mass({},replay)
    np.testing.assert_allclose(stats['replay_history_return_target_mass_sum'],jnp.where(ret,mass,0).sum())
    slots=ret&(jnp.cumsum(ret)%2==1)
    np.testing.assert_allclose(stats['core_selected_joint_mass_sum'],jnp.where(slots,mass,0).sum())
    assert int(stats['core_replaced_history_return'])==int(oldstats['core_replaced_history_return'])
