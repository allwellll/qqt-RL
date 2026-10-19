import jax
import jax.numpy as jnp
import numpy as np
from scripts.phase_balanced_joint_replay import choose as baseline
from scripts.return_error_replay import replace_return_draws


def data():
    n=16
    return dict(valid=jnp.ones(n,bool),kind=jnp.array([3]*8+[1]*4+[2]*4),
        preparation=jnp.array([False]*12+[True]*4),opening=jnp.zeros(n,bool),
        row=jnp.arange(n),obs=jnp.arange(n)[:,None],gv=jnp.zeros((n,1)))


def test_hard_native_returns_get_more_of_existing_return_slots():
    d=data();scores=jnp.array([.99]*4+[.01]*4+[.5]*8)
    selected=replace_return_draws(d,jax.random.PRNGKey(42),scores,3200)
    returning=(selected['kind']==3)&~selected['preparation']&selected['valid']
    rows=np.asarray(selected['row'][returning])
    assert (rows>=4).mean()>.85
    assert (rows<4).any(), 'positive epsilon preserves all valid return support'


def test_other_phases_and_number_of_return_slots_are_exactly_preserved():
    d=data();key=jax.random.PRNGKey(9);scores=jnp.linspace(0,1,16)
    original=baseline(d,key,320);selected=replace_return_draws(d,key,scores,320)
    returning=(original['kind']==3)&~original['preparation']&original['valid']
    np.testing.assert_array_equal(selected['row'][~returning],original['row'][~returning])
    for name in ['kind','valid','preparation','opening']:
        np.testing.assert_array_equal(selected[name],original[name])


def test_equal_errors_keep_baseline_draws_exactly_under_jit():
    d=data();key=jax.random.PRNGKey(12)
    original=baseline(d,key,32)
    selected=jax.jit(replace_return_draws,static_argnames='num')(d,key,jnp.full(16,.7),num=32)
    for name in d:np.testing.assert_array_equal(selected[name],original[name])


def test_no_return_and_invalid_rows_keep_original_authority():
    d=data();d['valid']=d['valid'].at[:8].set(False)
    original=baseline(d,jax.random.PRNGKey(11),32)
    selected=replace_return_draws(d,jax.random.PRNGKey(11),jnp.zeros(16),32)
    for name in d:np.testing.assert_array_equal(selected[name],original[name])


def test_without_preparation_only_original_return_positions_can_change():
    d=data();d['preparation']=jnp.zeros(16,bool);key=jax.random.PRNGKey(19)
    original=baseline(d,key,320);selected=replace_return_draws(d,key,jnp.linspace(0,1,16),320)
    np.testing.assert_array_equal(selected['row'][original['kind']!=3],original['row'][original['kind']!=3])
    np.testing.assert_array_equal(selected['kind'],original['kind'])


def test_current_joint_mass_uses_legal_pairs_and_has_no_actor_gradient(monkeypatch):
    from scripts import return_error_replay as candidate
    move=jnp.array([[1.,0.,-1.,0.,0.],[0.,2.,1.,0.,-1.]])
    ability=jnp.array([[1.,0.,-1.],[0.,1.,0.]])
    d=dict(obs=jnp.zeros((2,1)),gv=jnp.zeros((2,1)),legal=jnp.ones((2,5),bool),
        ability_legal=jnp.array([[True,True,False],[True,False,True]]),
        joint_target=jnp.zeros((2,5,3),bool).at[0,0,0].set(True).at[1,1,1].set(True))
    monkeypatch.setattr(candidate,'net_forward',lambda a,*args:(a,ability,None,None))
    mass=candidate.target_mass(move,d)
    expected=jax.nn.softmax(move[0])[0]*jax.nn.softmax(ability[0,:2])[0]
    np.testing.assert_allclose(mass,[expected,0.],rtol=1e-6)
    np.testing.assert_array_equal(jax.grad(lambda a:candidate.target_mass(a,d).sum())(move),jnp.zeros_like(move))


def test_fifo_authority_fallback_statistics_and_restore_stay_exact(monkeypatch):
    import pickle
    from scripts import return_error_replay as candidate,phase_balanced_joint_replay as phase
    from scripts.prestock_joint_replay import buffer
    from test_prestock_joint_replay import batch
    monkeypatch.setattr(candidate,'target_mass',lambda actor,d:1./(1.+d['obs'][:,0,0,0]))
    old=buffer(capacity=4,additions=2,obs_shape=(1,1,1),global_size=2)
    for generation in [1,2,3]:
        key=jax.random.PRNGKey(generation)
        expected,prior_replay,prior_stats=phase.aggregate(old,batch(),key,jnp.int32(generation))
        actual,replay,stats=jax.jit(candidate.aggregate)(old,batch(),key,jnp.int32(generation),{})
        for name in actual:np.testing.assert_array_equal(actual[name],expected[name])
        for name in prior_stats:np.testing.assert_array_equal(stats[name],prior_stats[name])
        assert set(replay)==set(prior_replay), 'no cached error field enters state or loss'
        for half,sl in [('current',slice(0,2)),('history',slice(2,4))]:
            mass=1./(1.+replay['obs'][sl,0,0,0]);valid=replay['valid'][sl]
            np.testing.assert_allclose(stats['replay_'+half+'_return_target_mass_sum'],jnp.where(valid,mass,0).sum())
            assert int(stats['replay_'+half+'_returning'])==int(valid.sum())
        assert bool((replay['round'][:2]==generation).all())
        if generation==1:assert bool((replay['round'][2:]==generation).all())
        else:assert bool((replay['round'][2:]<generation).all())
        restored=pickle.loads(pickle.dumps(jax.device_get(old)))
        repeated=jax.jit(candidate.aggregate)(restored,batch(),key,jnp.int32(generation),{})
        for a,b in zip(jax.tree.leaves((actual,replay,stats)),jax.tree.leaves(repeated)):
            np.testing.assert_array_equal(a,b)
        old=actual
