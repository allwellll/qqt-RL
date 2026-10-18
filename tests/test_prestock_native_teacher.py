"""Real native endpoints must certify the shared stock-preparation objective."""
import jax
import jax.numpy as jnp
import numpy as np


def transition(states,seat):
    from jax_bomb import je07_bun_env as env
    n=seat.size;r=jnp.arange(n)
    actions=jnp.zeros((n,5,2,2),jnp.int32).at[:,:,:,0].set(4)
    actions=actions.at[r[:,None],jnp.arange(5)[None],seat[:,None],0].set(jnp.arange(5))
    repeated=jax.tree.map(lambda x:jnp.repeat(x,5,0),states)
    after,_,_=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        repeated,actions.reshape(-1,2,2),jax.random.split(jax.random.PRNGKey(6106),n*5))
    legal,ability=jax.vmap(env.legal_mask)(states)
    return after,legal[r,seat],ability[r,seat]


def fixture(position=(5.5,10.5)):
    from test_prestock_training_state import native
    from scripts.prestock_training_state import state,update
    st,aux=native();st=st._replace(core=st.core._replace(pos=st.pos.at[0,0].set(jnp.array(position))))
    chosen=update(st,aux,state(1))
    chosen=jax.tree.map(lambda x:x[:,0],chosen)
    return st,chosen


def test_native_navigation_uses_shared_staging_and_physical_topology_without_wait_or_ability():
    from scripts.prestock_native_teacher import navigation,targets
    st,chosen=fixture();seat=jnp.array([0]);after,legal,ability=transition(st,seat)
    safe=jnp.ones((1,5,3),bool)
    move,valid=jax.jit(navigation)(st,after,seat,chosen,legal,safe)
    assert bool(valid[0]) and not bool(move[0,4])
    # Compare actual native endpoints with the exact shared PBRS stage field.
    from scripts.continuous_route_metric import continuous_distance
    blocked=st.wall|st.brick|st.core.pushable
    before=continuous_distance(st.pos[:,0],chosen['distance'],blocked)
    end=continuous_distance(after.pos[:,0],jnp.repeat(chosen['distance'],5,0),jnp.repeat(blocked,5,0))
    assert np.all(np.asarray((before-end)[np.asarray(move[0]>0)])>0)
    joint,valid=jax.jit(targets)(st,seat,chosen,move,valid,safe)
    assert bool(valid[0]) and bool(joint[0,:,0].any())
    assert not bool(joint[0,:,1:].any())
    # Temporal safety veto is joint-specific and has final authority.
    joint,valid=targets(st,seat,chosen,move,jnp.ones(1,bool),jnp.zeros_like(safe))
    assert not bool(valid.any()) and not bool(joint.any())


def test_native_navigation_rejects_premature_custody_and_changed_physical_topology():
    from scripts.prestock_native_teacher import navigation
    st,chosen=fixture();seat=jnp.array([0]);after,legal,_=transition(st,seat);safe=jnp.ones((1,5,3),bool)
    for changed in [after._replace(bun_carried=after.bun_carried.at[:,0].set(1)),
                    after._replace(core=after.core._replace(brick=after.brick.at[:,2,2].set(True))),
                    after._replace(trapped=after.trapped.at[:,0].set(10))]:
        move,valid=navigation(st,changed,seat,chosen,legal,safe)
        assert not bool(valid.any()) and not bool(move.any())


def test_native_opening_uses_shared_revision_and_joint_safety_no_navigation_fallback():
    from scripts.prestock_native_teacher import targets
    from test_frontier_native_witness import native_barrier
    from scripts.prestock_route_preparation import selection_state
    st,goals,bricks=native_barrier();st=jax.tree.map(lambda x:x[None],st)
    chosen=selection_state(1)
    chosen.update(active=jnp.ones(1,bool),ready=jnp.zeros(1,bool),objective=jnp.array([3],jnp.int8),
        staging=jnp.zeros((1,13,15),bool).at[0,6,6].set(True),first_bricks=bricks[None])
    # Supply native base destinations explicitly to retain this fixture's goal.
    safe=jnp.ones((1,5,3),bool)
    joint,valid=jax.jit(targets)(st,jnp.array([0]),chosen,jnp.zeros((1,5)),jnp.zeros(1,bool),safe,
        goals[None],goals[None])
    assert bool(valid[0]) and bool(joint[0,:,1].any())
    assert not bool(joint[0,:,0].any())
    veto=safe.at[:,:,1].set(False)
    joint,valid=targets(st,jnp.array([0]),chosen,jnp.zeros((1,5)),jnp.zeros(1,bool),veto,
        goals[None],goals[None])
    assert not bool(valid.any()) and not bool(joint.any())
