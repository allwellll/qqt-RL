"""Native trajectory witnesses for a restricted, proposal-only opening oracle."""
import jax
import jax.numpy as jnp
import numpy as np


def native_barrier():
    from scripts import je09_bun_scenes as scenes
    state=jax.tree.map(lambda x:x[0],scenes.init_batch(jax.random.PRNGKey(71),1,jax.nn.one_hot(8,10)))
    wall=jnp.zeros((13,15),bool).at[0].set(True).at[-1].set(True).at[:,0].set(True).at[:,-1].set(True)
    brick=jnp.zeros_like(wall).at[1:12,7].set(True)
    core=state.core._replace(pos=jnp.array([[6.5,6.5],[1.5,12.5]]),alive=jnp.array([True,False]),
        wall=wall,brick=brick,pushable=jnp.zeros_like(brick),fuse=jnp.zeros_like(state.fuse),
        owner=jnp.full_like(state.core.owner,-1),blast_linger=jnp.zeros_like(state.core.blast_linger),
        brick_linger=jnp.zeros_like(state.core.brick_linger),blast_cap=jnp.full_like(state.core.blast_cap,2))
    state=state._replace(core=core,trapped=jnp.zeros_like(state.trapped),
        bun_respawn=jnp.array([0,100],state.bun_respawn.dtype),bun_carried=jnp.full_like(state.bun_carried,-1))
    goals=jnp.zeros_like(brick).at[6,12].set(True)
    targets=jnp.zeros_like(brick).at[6,7].set(True)
    return state,goals,targets


def test_real_native_bomb_escape_opens_required_brick_and_improves_fixed_origin_topology():
    from scripts.frontier_native_witness import placement_witness
    state,goals,targets=native_barrier()
    result=jax.jit(placement_witness)(state,jnp.int32(0),goals,targets)
    assert np.asarray(result.joint_set).shape==(5,3)
    assert bool(result.joint_set[:,1].any())
    assert not bool(result.joint_set[:,0].any()) and not bool(result.joint_set[:,2].any())
    assert bool(result.accepted.any())
    assert np.asarray(result.safe)[np.asarray(result.accepted)].all()
    assert np.asarray(result.resolved)[np.asarray(result.accepted)].all()
    assert np.asarray(result.opened)[np.asarray(result.accepted)].all()
    assert np.asarray(result.topology_gain)[np.asarray(result.accepted)].all()


def test_no_placement_label_with_short_window_existing_hazard_or_wrong_target():
    from scripts.frontier_native_witness import placement_witness
    state,goals,targets=native_barrier();run=jax.jit(placement_witness)
    for modified,goal,target in [
        (state._replace(bun_carried=jnp.array([1,-1],state.bun_carried.dtype)),goals,targets),
        (state._replace(bun_respawn=jnp.array([0,8],state.bun_respawn.dtype)),goals,targets),
        (state._replace(core=state.core._replace(fuse=state.fuse.at[2,2].set(10),owner=state.core.owner.at[2,2].set(1))),goals,targets),
        (state,goals,jnp.zeros_like(targets)),
        (state,jnp.zeros_like(goals),targets)]:
        result=run(modified,jnp.int32(0),goal,target)
        assert not bool(result.joint_set.any())


def test_corridor_preparation_rejects_inventory_acquired_during_escape():
    from scripts.frontier_native_witness import placement_witness
    state,goals,targets=native_barrier()
    loose=jnp.zeros_like(state.bun_loose).at[...,1].set((~state.wall&~state.brick).astype(state.bun_loose.dtype))
    result=jax.jit(placement_witness)(state._replace(bun_loose=loose),jnp.int32(0),goals,targets)
    assert not bool(result.joint_set.any())
