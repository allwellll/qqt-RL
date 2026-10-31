"""Counterfactuals must remove exact components and preserve the shared anchor."""
import importlib
import jax
import jax.numpy as j
import numpy as np
import pytest
import optax


def test_joint_is_exact_identity_and_unknown_branch_rejected():
    m = importlib.import_module('scripts.dagger_rl_update_mechanism')
    value = j.array(-0.)
    metrics = j.arange(8, dtype=j.float32)
    assert m.branch_loss(value, metrics, 'joint_anchor') is value
    with pytest.raises(ValueError):m.branch_loss(value,metrics,'random')


def test_component_removal_keeps_independent_critic_out_of_dagger_gradient():
    m = importlib.import_module('scripts.dagger_rl_update_mechanism')
    # Independent symbolic heads check the selector; the real frozen batch
    # and actual Adam freeze are verified by the executed GPU diagnosis.
    def loss(params, branch):
        actor,critic=params
        ppo=actor**2;value=critic**2;entropy=actor*.3
        sup=(actor-2)**2;anchor=(actor+1)**2
        base=ppo+.5*value-.005*entropy+.2*sup+.05*anchor
        return m.branch_loss(base,j.array([ppo,value,entropy,sup,0.,0.,anchor,0.]),branch)
    params=j.array([.4,.8])
    joint=jax.grad(lambda p:loss(p,'joint_anchor'))(params)
    ppo=jax.grad(lambda p:loss(p,'ppo_anchor'))(params)
    dagger=jax.grad(lambda p:loss(p,'dagger_anchor'))(params)
    np.testing.assert_allclose(joint[1],ppo[1])
    assert dagger[1]==0 and dagger[0]!=0
    np.testing.assert_allclose(joint[0]-ppo[0],.4*(params[0]-2),rtol=1e-6)


def test_random_counterfactual_matches_each_module_norm_with_fixed_draw():
    m=importlib.import_module('scripts.dagger_rl_update_mechanism')
    before={'actor':{'w':np.zeros((8,9),np.float32)},'critic':{'w':np.zeros(17,np.float32)}}
    after={'actor':{'w':np.ones((8,9),np.float32)*.1},'critic':{'w':np.ones(17,np.float32)*.3}}
    first=m.random_displacement(before,after,seed=31)
    second=m.random_displacement(before,after,seed=31)
    for module in before:
        np.testing.assert_array_equal(first[module]['w'],second[module]['w'])
        np.testing.assert_allclose(np.linalg.norm(first[module]['w']),np.linalg.norm(after[module]['w']),rtol=1e-6)
    assert not np.allclose(first['actor']['w'],after['actor']['w'])


def test_executed_dagger_update_freezes_critic_params_and_nonzero_moments():
    m = importlib.import_module('scripts.dagger_rl_update_mechanism')
    from scripts.carrier_mechanism_train import preserve_critic
    params = {'actor': {'heads': {'wb': j.array([.4, -.2])}},
              'critic': {'w': j.array([.8])}}
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5))
    adam = opt.init(params)
    # A saved critic momentum must not drift even though its gradient is zero.
    changes, adam = opt.update(jax.tree.map(j.ones_like, params), adam, params)
    def objective(p, batch, replay, mode):
        actor=p['actor']['heads']['wb']; critic=p['critic']['w']
        ppo=(actor**2).sum(); value=(critic**2).sum()
        sup=((actor-2)**2).sum(); anchor=((actor+1)**2).sum()
        base=ppo+.5*value+.2*sup+.05*anchor
        return base, j.array([ppo,value,0.,sup,1.,1.,anchor,1.])
    replicas = jax.local_device_count()
    rep=lambda tree:jax.tree.map(lambda a:j.stack([a]*replicas),tree)
    rows=j.ones((replicas, 1, 32))
    batch=tuple(rows for _ in range(5))
    replay=(j.ones((replicas,32)),)
    keys=jax.random.split(jax.random.PRNGKey(3),replicas)
    fn=m.make_update(objective,opt,'dagger_anchor',preserve_critic)
    (new,na),loss=fn(rep(params),rep(adam),batch,replay,keys)
    one=lambda tree:jax.tree.map(lambda a:np.asarray(a)[0],tree)
    new,na=one(new),one(na)
    np.testing.assert_array_equal(new['critic']['w'],params['critic']['w'])
    for field in ['mu','nu']:
        np.testing.assert_array_equal(getattr(na[1][0],field)['critic']['w'],
                                      getattr(adam[1][0],field)['critic']['w'])
    assert int(na[1][0].count)==int(adam[1][0].count)+1
    assert not np.array_equal(new['actor']['heads']['wb'],params['actor']['heads']['wb'])
    assert np.isfinite(np.asarray(loss)).all()


def test_recovery_loads_verified_update_without_optimizer_or_environment(tmp_path):
    import pickle,json,hashlib
    m=importlib.import_module('scripts.dagger_rl_update_mechanism')
    from scripts.terminal_checkpoint_restore import digest
    source={'params':{'actor':np.array([1.],np.float32)},'adam':(np.array(7),)}
    path=tmp_path/'state.pkl';path.write_bytes(pickle.dumps(source))
    marker={'state_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'params_digest':digest(source['params']),'Adam_digest':digest(source['adam'])}
    (tmp_path/'COMMITTED.json').write_text(json.dumps(marker))
    value,receipt=m.load_committed_update(tmp_path,expected=source)
    assert digest(value)==digest(source) and receipt==marker
    with pytest.raises(AssertionError):m.load_committed_update(tmp_path,expected=dict(source,params={'actor':np.array([2.])}))
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(AssertionError):m.load_committed_update(tmp_path,expected=source)
