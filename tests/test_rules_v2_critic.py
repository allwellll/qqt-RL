"""Critic-only interventions must not change Actor inputs or initial policy."""
import copy
import numpy as np
import jax
import jax.numpy as j
import optax
import pytest

from jax_bomb import jax_net as net
from scripts import retention_phase_conditioning as phase
from scripts import rules_v2_critic as c


def fixture(conditioned=False):
    p=net.init_transformer(jax.random.PRNGKey(7),24,13,15,embed=8,depth=1,heads=4)
    if conditioned:p=phase.migrate(p)
    obs=j.zeros((2,24,13,15));gv=j.zeros((2,24)).at[:,9:11].set(1.)
    aux=dict(transport={k:j.zeros((2,2)) for k in ('balance','pending_capture','episode_gross')},
             route_pbrs=dict(phi=j.zeros((2,2)),offset=j.zeros((2,2)),phase=j.zeros((2,2),j.int8)),
             control_credit=dict(balance=j.zeros((2,2)),gross=j.zeros((2,2))),
             paid=j.zeros((2,2)),shape_paid=j.zeros((2,2)),terminal=j.zeros(2,bool))
    return dict(actor=p,critic=copy.deepcopy(p)),obs,gv,aux


@pytest.mark.parametrize('conditioned',[False,True])
def test_initial_actor_inputs_policy_and_value_exact(conditioned):
    p,o,g,ax=fixture(conditioned);baseline=c.forward(p,o,g,None)
    for arm in c.ARMS:
        q=c.migrate(p,arm)
        assert jax.tree.structure(q['actor'])==jax.tree.structure(p['actor'])
        for a,b in zip(jax.tree.leaves(q['actor']),jax.tree.leaves(p['actor'])):np.testing.assert_array_equal(a,b)
        encoded=c.ledger_features(ax)
        result=c.forward(q,o,g,encoded)
        for a,b in zip(baseline,result):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(c.actor_state(q,o,g),phase.conditioned_state(p['actor'],o,g))


@pytest.mark.parametrize('conditioned',[False,True])
def test_nonzero_actor_inputs_match_collector_forward_eager_and_jit(conditioned):
    from scripts.je03_bun_stage1_20261011 import split_forward
    p,o,g,ax=fixture(conditioned)
    o=jax.random.uniform(jax.random.PRNGKey(83),o.shape)
    g=g.at[:,0].set(.37).at[:,14].set(1.)
    def original(params,obs,state):
        return split_forward(params,'transformer',obs,state,
            lambda q,a,x,s:phase.forward(q,x,s))
    ledger=c.ledger_features(ax)
    # Identical compilation boundaries in each pair, never eager-vs-JIT.
    for compiled in (False,True):
        baseline=(jax.jit(original) if compiled else original)(p,o,g)
        call=jax.jit(c.forward) if compiled else c.forward
        for arm in c.ARMS:
            q=c.migrate(p,arm);actual=call(q,o,g,ledger)
            for i in (0,1):np.testing.assert_array_equal(actual[i],baseline[i])


def test_ledger_enters_only_critic_and_aux_gradient_never_actor():
    p,o,g,ax=fixture(True);q=c.migrate(p,'A2_ledger_plus_delivery_head')
    x=c.ledger_features(ax).at[:,0].set(2.)
    q['critic']['ledger_input_w']=j.ones_like(q['critic']['ledger_input_w'])*.2
    q['critic']['delivery_head']=(j.ones((8,1))*.03,j.zeros(1))
    ref=c.forward(q,o,g,j.zeros_like(x));changed=c.forward(q,o,g,x)
    np.testing.assert_array_equal(ref[0],changed[0]);np.testing.assert_array_equal(ref[1],changed[1])
    assert not np.allclose(ref[3],changed[3])
    def loss(params):
        logits=c.delivery_logits(params,o,g,x)
        return c.delivery_loss(logits,j.array([1.,0.]),j.array([True,True]))
    gradient=jax.grad(loss)(q)
    assert all(np.count_nonzero(a)==0 for a in jax.tree.leaves(gradient['actor']))
    assert sum(float(j.sum(j.abs(a))) for a in jax.tree.leaves(gradient['critic']))>0


def test_ledger_schema_seat_normalization_and_missing_is_error():
    _,_,_,ax=fixture();ax['transport']['pending_capture']=j.array([[24.,48.],[72.,96.]])
    x=c.ledger_features(ax,seat=1)
    assert x.shape==(2,len(c.LEDGER_FIELDS))
    np.testing.assert_allclose(x[:,1],np.arcsinh([2.,4.]),rtol=1e-6)
    del ax['route_pbrs']['phi']
    with pytest.raises(KeyError):c.ledger_features(ax)


def test_delivery_targets_cannot_cross_drop_recovery_or_censored_tail():
    ids=np.array([1,1,2,2,3,3]);carrying=np.ones(6,bool)
    labels,mask=c.custody_targets(ids,carrying,delivered=[0,0,0,1,0,0],
        lost=[0,1,0,0,0,0],terminal=[0]*6)
    np.testing.assert_array_equal(labels,[0,0,1,1,0,0])
    np.testing.assert_array_equal(mask,[1,1,1,1,0,0])
    # No known labels means zero loss, not an invented negative label.
    assert float(c.delivery_loss(j.array([3.,-1.]),j.array([j.nan,j.nan]),j.array([False,False])))==0.


def test_adam_extension_preserves_old_moments_count_and_actor():
    p,_,_,_=fixture();opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
    state=opt.init(p);grads=jax.tree.map(j.ones_like,p)
    _,state=opt.update(grads,state,p)
    q=c.migrate(p,'A2_ledger_plus_delivery_head');new=c.migrate_adam(state,q)
    old_adam,new_adam=state[1][0],new[1][0]
    assert int(new_adam.count)==int(old_adam.count)==1
    for name in ('mu','nu'):
        old=getattr(old_adam,name);current=getattr(new_adam,name)
        for role in ('actor','critic'):
            for key in old[role]:
                for a,b in zip(jax.tree.leaves(old[role][key]),jax.tree.leaves(current[role][key])):np.testing.assert_array_equal(a,b)
        assert np.count_nonzero(current['critic']['ledger_input_w'])==0


def test_ppo_auxiliary_enters_loss_and_optimizer_without_actor_contamination():
    p,o,g,ax=fixture(True);q=c.migrate(p,c.ARMS[2]);ledger=c.ledger_features(ax)
    mv,ab,_,_=c.forward(q,o,g,ledger)
    old=jax.nn.log_softmax(mv)[:,0]+jax.nn.log_softmax(ab)[:,0]
    batch=dict(obs=o,state=g,ledger=ledger,actions=j.zeros((2,2),j.int32),
        old_logp=old,advantage=j.array([1.,-1.]),returns=j.array([1.,0.]),
        move_mask=j.ones_like(mv,bool),ability_mask=j.ones_like(ab,bool),
        carrying=j.ones(2,bool),delivery_known=j.ones(2,bool),delivery_target=j.ones(2))
    full,metrics=c.ppo_objective(q,batch)
    without=dict(batch,carrying=j.zeros(2,bool))
    base,_=c.ppo_objective(q,without)
    np.testing.assert_allclose(full-base,.1*np.log(2),rtol=1e-5)
    grad=jax.grad(lambda x:c.ppo_objective(x,batch)[0])(q)
    reference=jax.grad(lambda x:c.ppo_objective(x,without)[0])(q)
    for a,b in zip(jax.tree.leaves(grad['actor']),jax.tree.leaves(reference['actor'])):np.testing.assert_array_equal(a,b)
    assert float(j.linalg.norm(grad['critic']['delivery_head'][1]))>0
    opt=optax.adam(7.5e-5);changes,_=opt.update(grad,opt.init(q),q)
    after=optax.apply_updates(q,changes)
    assert float(j.linalg.norm(after['critic']['delivery_head'][1]))>0
