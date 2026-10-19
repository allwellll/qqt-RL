"""A carrier's home distance must not assume it can bomb through brick."""
import jax
import jax.numpy as jnp
import numpy as np


def fixture(h=7,w=9):
    wall=jnp.zeros((1,h,w),bool).at[:,0,:].set(True).at[:,-1,:].set(True)
    wall=wall.at[:,:,0].set(True).at[:,:,-1].set(True)
    brick=jnp.zeros_like(wall).at[:,1:-1,w//2].set(True)
    goals=jnp.zeros((1,4,h,w),bool)
    for channel,(r,c) in enumerate([(2,w-2),(h-3,1),(2,w-2),(h-3,1)]):
        goals=goals.at[0,channel,r,c].set(True)
    return wall,brick,goals


def test_drop_channels_keep_weighted_exact_while_both_home_channels_block_brick():
    from scripts.carrier_route_fields import fields
    from scripts.je08_bun_potential import shortest,INF
    wall,brick,goals=fixture();result=jax.jit(fields)(wall,brick,goals)
    legacy=shortest(wall[:,None],brick[:,None],goals)
    np.testing.assert_array_equal(result[:,:2],legacy[:,:2])
    assert int(legacy[0,2,2,1])<INF and int(result[0,2,2,1])==INF
    assert int(legacy[0,3,2,-2])<INF and int(result[0,3,2,-2])==INF
    assert int(result[0,2,2,-2])==0 and int(result[0,3,-3,1])==0


def test_opening_a_real_passage_restores_home_connectivity_for_both_seats():
    from scripts.carrier_route_fields import fields
    from scripts.je08_bun_potential import INF
    wall,brick,goals=fixture();opened=brick.at[0,3,4].set(False)
    result=fields(wall,opened,goals)
    assert int(result[0,2,2,1])<INF and int(result[0,3,2,-2])<INF


def test_alternate_corridor_takes_longer_walk_instead_of_shortcut_through_brick():
    from scripts.carrier_route_fields import fields
    from scripts.je08_bun_potential import cpu_shortest
    wall,brick,goals=fixture(9,11);brick=brick.at[0,7,5].set(False)
    result=np.asarray(fields(wall,brick,goals))
    for channel in [2,3]:
        expected=cpu_shortest(np.asarray(wall[0]|brick[0]),np.zeros((9,11),bool),np.asarray(goals[0,channel]))
        np.testing.assert_array_equal(result[0,channel],expected)


def test_open_map_is_exact_legacy_and_hard_wall_disconnection_stays_unreachable():
    from scripts.carrier_route_fields import fields
    from scripts.je08_bun_potential import shortest,INF
    wall,brick,goals=fixture();clear=jnp.zeros_like(brick)
    np.testing.assert_array_equal(fields(wall,clear,goals),shortest(wall[:,None],clear[:,None],goals))
    wall=wall.at[:,1:-1,4].set(True)
    result=fields(wall,clear,goals)
    assert int(result[0,2,2,1])==INF


def test_goal_inside_present_brick_is_not_a_free_carrier_destination():
    from scripts.carrier_route_fields import fields
    from scripts.je08_bun_potential import INF
    wall,brick,goals=fixture();brick=brick.at[0,2,7].set(True)
    result=fields(wall,brick,goals)
    assert bool((result[0,2]==INF).all())
