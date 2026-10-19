"""Return supervision must strengthen only verified native carrier labels."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np


def data():
    move=jnp.array([[2.,-1.,0.,1.,-.5],[-1.,1.,2.,0.,.5],[0.,1.,-1.,2.,.5]])
    ability=jnp.array([[0.,2.,-1.],[1.,-1.,2.],[2.,0.,1.]])
    mm=jnp.ones((3,5),bool);am=jnp.ones((3,3),bool)
    target=jnp.zeros((3,5,3),bool).at[0,3,0].set(True).at[1,1,0].set(True)
    target=target.at[1,1,2].set(True).at[2,0,0].set(True)
    return move,ability,mm,am,target,jnp.ones(3,bool)


def test_factor_two_adds_exactly_one_return_contribution_and_gradient():
    from scripts.return_dagger_weight import weighted_joint_loss
    from scripts.prestock_joint_oracle import joint_set_loss
    d=data();returning=jnp.array([False,True,False])
    base=lambda m:joint_set_loss(m,*d[1:],.2)[0]
    extra=lambda m:joint_set_loss(m,*d[1:5],returning,.2)[0]/3
    candidate=lambda m:weighted_joint_loss(m,*d[1:],returning,2.,.2)[0]
    np.testing.assert_allclose(candidate(d[0]),base(d[0])+extra(d[0]),rtol=1e-6)
    gradient=jax.grad(candidate)(d[0])-jax.grad(base)(d[0])
    np.testing.assert_allclose(gradient,jax.grad(extra)(d[0]),atol=2e-8)
    np.testing.assert_array_equal(np.asarray(gradient)[[0,2]],0.)


def test_unweighted_no_return_and_restore_are_exact():
    from scripts.return_dagger_weight import weighted_joint_loss
    from scripts.prestock_joint_oracle import joint_set_loss
    d=data();r=jnp.array([False,True,False]);baseline=joint_set_loss(*d,.2)
    for factor,returning in [(1.,r),(2.,jnp.zeros(3,bool))]:
        result=jax.jit(lambda *x:weighted_joint_loss(*x,returning,factor,.2))(*d)
        for a,b in zip(baseline,result):np.testing.assert_allclose(a,b,rtol=1e-7)
    first=weighted_joint_loss(*d,r,2.,.2)
    restored=weighted_joint_loss(*pickle.loads(pickle.dumps(d)),r,2.,.2)
    for a,b in zip(first,restored):np.testing.assert_array_equal(a,b)


def test_illegal_pairs_and_inactive_return_rows_cannot_add_loss_or_gradient():
    from scripts.return_dagger_weight import weighted_joint_loss
    from scripts.prestock_joint_oracle import joint_set_loss
    d=list(data());d[3]=d[3].at[1,0].set(False).at[1,2].set(False)
    returning=jnp.array([False,True,False])
    for active in [d[5],d[5].at[1].set(False),jnp.zeros(3,bool)]:
        d[5]=active
        base=lambda m:joint_set_loss(m,*d[1:],.2)[0]
        actual=lambda m:weighted_joint_loss(m,*d[1:],returning,2.,.2)[0]
        np.testing.assert_allclose(actual(d[0]),base(d[0]),rtol=1e-7)
        np.testing.assert_array_equal(jax.grad(actual)(d[0]),jax.grad(base)(d[0]))


def test_joint_ties_keep_event_mass_instead_of_uniform_cross_entropy():
    from scripts.return_dagger_weight import weighted_joint_loss
    move=jnp.array([[4.,0.,-2.,1.,-1.]])
    ability=jnp.array([[1.,0.,-1.]])
    target=jnp.zeros((1,5,3),bool).at[0,0,0].set(True).at[0,0,1].set(True)
    value,_=weighted_joint_loss(move,ability,jnp.ones((1,5),bool),jnp.ones((1,3),bool),target,jnp.ones(1,bool),jnp.ones(1,bool),2.,.2)
    expected=-.4*jnp.log(jax.nn.softmax(move)[0,0]*jax.nn.softmax(ability)[0,:2].sum())
    np.testing.assert_allclose(value,expected,rtol=1e-6)
