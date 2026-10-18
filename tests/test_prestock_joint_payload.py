"""A unified joint replay must preserve old movement loss and authority."""
import jax
import jax.numpy as jnp
import numpy as np


def test_legacy_move_teacher_has_identical_loss_and_zero_ability_gradient():
    from scripts.prestock_joint_payload import merge
    from scripts.prestock_joint_oracle import joint_set_loss
    from scripts.je10_route_oracle import supervised_loss
    move=jnp.array([[.2,-.3,1.,2.,-.5]])
    ability=jnp.array([[3.,-2.,1.]])
    mm=jnp.array([[True,True,True,False,True]])
    am=jnp.array([[True,False,True]])
    target=jnp.array([[1.,0.,1.,0.,0.]])
    joint,active=merge(target,jnp.array([True]),jnp.array([3]),mm,am,
        jnp.array([False]),jnp.zeros((1,5,3),bool))
    def loss(m,a):return joint_set_loss(m,a,mm,am,joint,active,coefficient=.2)[0]
    value=loss(move,ability)
    reference=lambda m:supervised_loss(m,mm,target,jnp.array([True]),coefficient=.2)[0]
    np.testing.assert_allclose(value,reference(move),rtol=1e-6)
    gradient=jax.grad(loss,argnums=(0,1))(move,ability)
    np.testing.assert_allclose(gradient[0],jax.grad(reference)(move),atol=1e-7)
    np.testing.assert_allclose(gradient[1],0,atol=1e-7)


def test_unverified_preparation_never_falls_back_to_premature_stock_pickup():
    from scripts.prestock_joint_payload import merge
    target=jnp.array([[0.,0.,0.,1.,0.]])
    mm=jnp.ones((1,5),bool);am=jnp.ones((1,3),bool)
    joint,active=jax.jit(merge)(target,jnp.array([True]),jnp.array([2]),mm,am,
        jnp.array([True]),jnp.zeros((1,5,3),bool))
    assert not bool(active[0]) and not bool(joint.any())
    # An illegal-only opening is also rejected, never replaced by stock.
    joint,active=merge(target,jnp.array([True]),jnp.array([2]),mm,am.at[:,1].set(False),
        jnp.array([True]),jnp.zeros((1,5,3),bool).at[0,4,1].set(True))
    assert not bool(active[0]) and not bool(joint.any())


def test_preparation_can_replace_only_stock_and_retains_joint_pairs():
    from scripts.prestock_joint_payload import merge
    n=4;target=jnp.zeros((n,5)).at[:,3].set(1)
    prep=jnp.zeros((n,5,3),bool).at[:,0,1].set(True).at[:,1,0].set(True)
    joint,active=merge(target,jnp.ones(n,bool),jnp.array([1,2,3,4]),
        jnp.ones((n,5),bool),jnp.ones((n,3),bool),jnp.ones(n,bool),prep)
    assert bool(active.all())
    np.testing.assert_array_equal(np.argwhere(joint[1]),[[0,1],[1,0]])
    for row in [0,2,3]:
        np.testing.assert_array_equal(np.argwhere(joint[row]),[[3,0],[3,1],[3,2]])
