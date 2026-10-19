"""Candidate-only retention boundaries; never install into a live worker."""
import os

os.environ.setdefault('JAXBOMB_RULE', 'bun')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.native_return_retention import movement_scale, retained_kl


def batch(n=1):
    context=jnp.zeros((n,125)).at[:,0].set(8).at[:,54].set(3)
    context=context.at[:,55].set(1).at[:,58].set(1).at[:,104].set(1)
    global_state=jnp.zeros((n,24)).at[:,9].set(1).at[:,13:15].set(1)
    return context,global_state,jnp.ones((n,5),bool),jnp.ones((n,3),bool)


def test_verified_safe_native_return_releases_movement_but_keeps_bomb_retention():
    c,g,m,a=batch()
    scale=movement_scale(c,g,m,a)
    np.testing.assert_array_equal(scale,[0.])
    move=jnp.array([2.]);bomb=jnp.array([3.]);weight=jnp.array([.025])
    assert float(retained_kl(move,bomb,weight,scale))==pytest.approx(.075)
    dm,db=jax.grad(lambda x,y:retained_kl(x,y,weight,scale),argnums=(0,1))(move,bomb)
    np.testing.assert_array_equal(dm,[0.]);np.testing.assert_allclose(db,[.025])


@pytest.mark.parametrize('field,value',[
    (0,0),(0,2),(0,3),(0,4),(0,6), # combat and retired carrier scenes
    (1,1), # actual danger exposure
    (47,1), # native terminal transition
    (54,1),(54,2),(54,4), # other native objectives
    (55,0),(58,0), # no active route or no native custody proof
    (110,1), # empty-hand preparation is a different objective
])
def test_other_or_unproved_states_keep_both_retention_terms(field,value):
    c,g,m,a=batch();c=c.at[:,field].set(value)
    np.testing.assert_array_equal(movement_scale(c,g,m,a),[1.])


@pytest.mark.parametrize('field,value',[(9,0),(9,1.5),(13,0),(14,0)])
def test_dead_trapped_empty_and_own_origin_do_not_release_retention(field,value):
    c,g,m,a=batch();g=g.at[:,field].set(value)
    np.testing.assert_array_equal(movement_scale(c,g,m,a),[1.])


@pytest.mark.parametrize('invalid',['empty','illegal_move','illegal_ability'])
def test_native_joint_certificate_must_accept_a_current_legal_pair(invalid):
    c,g,m,a=batch()
    if invalid=='empty':c=c.at[:,95:110].set(0)
    if invalid=='illegal_move':m=m.at[:,3].set(False)
    if invalid=='illegal_ability':a=a.at[:,0].set(False)
    np.testing.assert_array_equal(movement_scale(c,g,m,a),[1.])


def test_unverified_and_legacy_layout_are_exact_baseline_not_renormalized():
    c,g,m,a=batch(3);c=c.at[1,58].set(0).at[2,1].set(1)
    scale=movement_scale(c,g,m,a)
    np.testing.assert_array_equal(scale,[0.,1.,1.])
    move=jnp.array([2.,4.,8.]);bomb=jnp.array([3.,6.,9.]);weight=jnp.array([.025,.1,.025])
    expected=np.mean(np.array([.025*3,.1*10,.025*17]))
    assert float(retained_kl(move,bomb,weight,scale))==pytest.approx(expected)
    for width in [2,48,60,110]:
        np.testing.assert_array_equal(movement_scale(c[:,:width],g,m,a),np.ones(3))
    ordinary=c.at[:,58].set(0)
    baseline=jnp.mean(weight*(move+bomb))
    assert float(retained_kl(move,bomb,weight,movement_scale(ordinary,g,m,a)))==float(baseline)


def test_jit_and_sample_order_preserve_rowwise_certificate_pairing():
    c,g,m,a=batch(4);c=c.at[1,58].set(0).at[2,1].set(1)
    g=g.at[3,14].set(0)
    fn=jax.jit(movement_scale);expected=fn(c,g,m,a)
    order=jnp.array([3,0,2,1])
    np.testing.assert_array_equal(fn(c[order],g[order],m[order],a[order]),expected[order])


def test_passive_exposure_matches_actual_ppo_certificate_exception():
    from scripts.native_return_retention import exposure_summary
    c,g,m,a=batch(8)
    c=c.at[1,58].set(0).at[2,1].set(1).at[3,47].set(1)
    g=g.at[4,14].set(0)
    m=m.at[5,3].set(False)
    c=c.at[6,110].set(1)
    shape=(2,2,2)
    result=exposure_summary(np.asarray(c).reshape(*shape,-1),np.asarray(g).reshape(*shape,-1),
        np.asarray(m).reshape(*shape,-1),np.asarray(a).reshape(*shape,-1))
    expected=np.asarray(movement_scale(c,g,m,a))==0
    assert result['actor_rollout_rows']==8
    assert result['certified_return_movement_exempt_rows']==int(expected.sum())==2
    assert result['exempt_rows_by_device']==[1,1]
    assert result['original_denominator_rows']==8
    assert not result['controls_training']
