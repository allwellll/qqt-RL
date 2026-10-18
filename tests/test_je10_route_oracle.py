import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts import je10_route_oracle as oracle


def geometry():
    d=jnp.broadcast_to((12-jnp.arange(13))[None,:,None],(1,13,15))
    return d,jnp.ones((1,5),bool),jnp.zeros((1,13,15)),jnp.zeros((1,13,15),bool),jnp.zeros((1,13,15))


@pytest.mark.parametrize('pos',[[1.8,9.64],[2.02,9.64],[7.2,5.1],[9.5,3.5]])
def test_oracle_disambiguates_continuous_southward_return(pos):
    target,valid=oracle.action_set(jnp.array([pos]),*geometry())
    assert bool(valid[0]);assert float(target[0,1])>0;assert float(target[0,0])==0


def test_equal_shortest_actions_are_set_valued_and_illegal_actions_rejected():
    y,x=jnp.indices((13,15));d=(abs(10-y)+abs(10-x))[None]
    _,legal,danger,blocked,fuse=geometry()
    target,valid=oracle.action_set(jnp.array([[5.5,5.5]]),d,legal,danger,blocked,fuse)
    np.testing.assert_allclose(target,[[0,.5,0,.5,0]])
    legal=legal.at[0,1].set(False)
    target,valid=oracle.action_set(jnp.array([[5.5,5.5]]),d,legal,danger,blocked,fuse)
    assert float(target[0,1])==0;assert float(target[0,3])==1


def test_wall_and_live_bomb_are_never_oracle_targets():
    d,legal,danger,blocked,fuse=geometry()
    blocked=blocked.at[0,2,9].set(True)
    target,valid=oracle.action_set(jnp.array([[1.8,9.5]]),d,legal,danger,blocked,fuse)
    assert float(target[0,1])==0
    target,valid=oracle.action_set(jnp.array([[1.8,9.5]]),d,legal,danger,jnp.zeros_like(blocked),fuse.at[0,2,9].set(5))
    assert float(target[0,1])==0


def test_threat_recovery_closes_empty_window_and_near_carrier_route():
    result=oracle.route_allowed(jnp.array([False,True,True,True]),jnp.array([True,True,True,False]),
        jnp.array([0,0,20,0]),jnp.array([10.,2.,2.,2.]))
    np.testing.assert_array_equal(result,[False,False,True,True])


def test_safe_long_return_has_no_120_tick_cutoff():
    assert bool(oracle.eligible(jnp.array([9]),jnp.array([True]),jnp.array([False]),
        jnp.array([0]),jnp.array([1]),jnp.array([False]),jnp.array([True]),jnp.array([900]))[0])


def test_set_loss_does_not_force_uniformity_inside_equivalent_oracle_set():
    legal=jnp.ones((1,5),bool);target=jnp.array([[0,.5,0,.5,0.]])
    high=jnp.array([[-20.,12.,-20.,-5.,-20.]])
    loss,_=oracle.supervised_loss(high,legal,target,jnp.array([True]),1.)
    assert float(loss)<1e-5
    zero=jax.grad(lambda v:oracle.supervised_loss(v,legal,target,jnp.array([False]),1.)[0])(high)
    np.testing.assert_array_equal(zero,jnp.zeros_like(zero))
