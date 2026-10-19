"""Disconnected routes need reachable staging cells, never brick movement labels."""
import jax
import jax.numpy as jnp
import numpy as np


def geometry():
    hard=jnp.ones((1,5,7),bool).at[:,1:4,1:6].set(False)
    brick=jnp.zeros_like(hard).at[:,1:4,3].set(True)
    goals=jnp.zeros_like(hard).at[:,2,5].set(True)
    return jnp.array([[2.5,1.5]]),hard,brick,goals


def test_disconnected_home_proposes_reachable_frontier_before_first_brick():
    from scripts.route_frontier import plan
    pos,hard,brick,goals=geometry();result=jax.jit(plan)(pos,hard,brick,goals)
    assert bool(result.needed[0])
    np.testing.assert_array_equal(np.argwhere(result.staging[0]),[[2,2]])
    assert int(result.distance[0,2,1])==1
    assert not bool((result.staging&(hard|brick)).any())
    assert int(result.distance[0,2,3])==32767


def test_existing_open_detour_or_just_opened_brick_keeps_direct_home_objective():
    from scripts.route_frontier import plan
    pos,hard,brick,goals=geometry()
    for cleared in [brick.at[:,1,3].set(False),brick.at[:,2,3].set(False)]:
        result=plan(pos,hard,cleared,goals)
        assert not bool(result.needed[0]) and not bool(result.staging.any())
        assert int(result.distance[0,2,1])<32767


def test_hard_wall_disconnection_and_invalid_start_never_invent_frontier():
    from scripts.route_frontier import plan
    pos,hard,brick,goals=geometry()
    result=plan(pos,hard|brick,jnp.zeros_like(brick),goals)
    assert not bool(result.needed.any()) and not bool(result.staging.any())
    assert int(result.distance[0,2,1])==32767
    result=plan(jnp.array([[0.5,.5]]),hard,brick,goals)
    assert not bool(result.needed.any()) and not bool(result.staging.any())


def test_equal_brick_frontiers_are_sets_and_batched_spawn_choices_are_independent():
    from scripts.route_frontier import plan
    pos,hard,brick,goals=geometry()
    goals=goals.at[:,1:4,5].set(True)
    positions=jnp.array([[1.5,1.5],[3.5,1.5]])
    result=plan(positions,jnp.repeat(hard,2,0),jnp.repeat(brick,2,0),jnp.repeat(goals,2,0))
    np.testing.assert_array_equal(np.argwhere(result.staging[0]),[[1,2]])
    np.testing.assert_array_equal(np.argwhere(result.staging[1]),[[3,2]])
    # A centered start and symmetric goals admit both equally short ways.
    symmetric=goals.at[:,2,5].set(False)
    result=plan(pos,hard,brick,symmetric)
    np.testing.assert_array_equal(np.argwhere(result.staging[0]),[[1,2],[2,2],[3,2]])


def test_native_map_both_cages_all_legal_disconnected_starts_get_open_reachable_stage():
    from scripts.route_frontier import plan
    from scripts import je09_bun_scenes as scenes
    from scripts.je08_bun_potential import cpu_shortest,INF
    positions=[];goals=[]
    for team in range(2):
        direct=cpu_shortest(scenes.wall|scenes.brick,np.zeros_like(scenes.brick),scenes.goals[team])
        weighted=cpu_shortest(scenes.wall,scenes.brick,scenes.goals[team])
        cells=np.argwhere(~scenes.wall&~scenes.brick&(direct>=INF)&(weighted<INF))
        assert len(cells)>0
        positions.extend(cells+.5);goals.extend([scenes.goals[team]]*len(cells))
    n=len(positions)
    result=jax.jit(plan)(jnp.asarray(np.array(positions)),
        jnp.broadcast_to(scenes.wall,(n,13,15)),jnp.broadcast_to(scenes.brick,(n,13,15)),jnp.asarray(np.array(goals)))
    assert np.asarray(result.needed).all()
    assert np.asarray(result.staging).any((-2,-1)).all()
    cells=np.asarray(positions,dtype=int)
    assert (np.asarray(result.distance)[np.arange(n),cells[:,0],cells[:,1]]<INF).all()
    assert not np.asarray(result.staging[:,scenes.wall|scenes.brick]).any()


def test_frontier_names_actual_first_bricks_for_future_native_ability_witness():
    from scripts.route_frontier import plan
    pos,hard,brick,goals=geometry();result=plan(pos,hard,brick,goals)
    np.testing.assert_array_equal(np.argwhere(result.first_bricks[0]),[[2,3]])
    assert not bool((result.first_bricks&~brick).any())
    for opened in [brick.at[:,2,3].set(False),jnp.zeros_like(brick)]:
        direct=plan(pos,hard,opened,goals)
        assert not bool(direct.first_bricks.any())


def test_tied_first_bricks_keep_all_equal_routes_and_hard_walls_never_qualify():
    from scripts.route_frontier import plan
    pos,hard,brick,goals=geometry()
    goals=goals.at[:,1:4,5].set(True).at[:,2,5].set(False)
    result=plan(pos,hard,brick,goals)
    np.testing.assert_array_equal(np.argwhere(result.first_bricks[0]),[[1,3],[2,3],[3,3]])
    impossible=plan(pos,hard|brick,jnp.zeros_like(brick),goals)
    assert not bool(impossible.first_bricks.any())
