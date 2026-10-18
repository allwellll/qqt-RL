"""Set-valued preparation supervision must retain move/ability pairing."""
import jax
import jax.numpy as jnp
import numpy as np


def test_joint_mass_does_not_reward_illegal_cross_pairs():
    from scripts.prestock_joint_oracle import joint_set_loss
    # Only (up,place) and (down,wait) are acceptable. Independent marginal
    # labels would also reward (up,wait) and (down,place).
    target=jnp.zeros((1,5,3),bool).at[0,0,1].set(True).at[0,1,0].set(True)
    legal_m=jnp.ones((1,5),bool);legal_a=jnp.ones((1,3),bool);active=jnp.array([True])
    move=jnp.array([[8.,-8.,-8.,-8.,-8.]])
    good=joint_set_loss(move,jnp.array([[-8.,8.,-8.]]),legal_m,legal_a,target,active)[0]
    cross=joint_set_loss(move,jnp.array([[8.,-8.,-8.]]),legal_m,legal_a,target,active)[0]
    assert float(good)<.001 and float(cross)>10


def test_legal_mask_empty_set_and_inactive_have_finite_zero_gradient():
    from scripts.prestock_joint_oracle import joint_set_loss
    move=jnp.zeros((2,5));ability=jnp.zeros((2,3))
    legal_m=jnp.ones((2,5),bool);legal_a=jnp.ones((2,3),bool).at[:,1].set(False)
    target=jnp.zeros((2,5,3),bool).at[:,0,1].set(True)
    def loss(m,a):return joint_set_loss(m,a,legal_m,legal_a,target,jnp.array([True,False]))[0]
    value=loss(move,ability);grad=jax.grad(loss,argnums=(0,1))(move,ability)
    assert float(value)==0
    for g in grad:np.testing.assert_array_equal(g,0)


def test_set_mass_accepts_any_tied_joint_without_forcing_uniform_labels():
    from scripts.prestock_joint_oracle import joint_set_loss
    target=jnp.zeros((1,5,3),bool).at[0,0,1].set(True).at[0,1,1].set(True)
    lm=jnp.ones((1,5),bool);la=jnp.ones((1,3),bool);active=jnp.array([True])
    for chosen in [0,1]:
        move=jnp.full((1,5),-10.).at[0,chosen].set(10.)
        loss,_=joint_set_loss(move,jnp.array([[-10.,10.,-10.]]),lm,la,target,active)
        assert float(loss)<.001


def test_navigation_and_opening_have_one_supervision_owner(monkeypatch):
    from scripts.prestock_joint_oracle import preparation_targets
    from scripts.prestock_route_preparation import plan
    from scripts import frontier_native_witness as witness
    from test_frontier_native_witness import native_barrier
    state,goals,bricks=native_barrier()
    pos=state.pos[0:1];hard=state.wall[None];boxes=state.brick[None]
    home=goals[None];stock=jnp.zeros_like(home).at[0,6,2].set(True)
    prepared=plan(pos,hard,boxes,home,stock,jnp.array([True]))
    assert int(prepared.objective[0])==2
    single=jax.tree.map(lambda x:x[0],prepared)
    calls=[]
    from types import SimpleNamespace
    def fake(*args):
        calls.append(True)
        return SimpleNamespace(joint_set=jnp.zeros((5,3),bool).at[4,1].set(True))
    monkeypatch.setattr(witness,'placement_witness',fake)
    # No JIT here: instrument the authority decision without running a second
    # native trajectory. Native witness safety has its own actual-step tests.
    with jax.disable_jit():
        target,active=preparation_targets(state,jnp.int32(0),single,home[0],stock[0],
            jnp.array([0.,1.,0.,0.,0.]),jnp.array(True))
        np.testing.assert_array_equal(np.argwhere(target),[[1,0]])
        assert bool(active) and not calls
        target,active=preparation_targets(state,jnp.int32(0),single,home[0],stock[0],
            jnp.zeros(5),jnp.array(False))
        np.testing.assert_array_equal(np.argwhere(target),[[4,1]])
        assert bool(active) and len(calls)==1
