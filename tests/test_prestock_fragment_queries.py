"""One query per preparation revision and navigation/opening subphase."""
import jax.numpy as jnp
import numpy as np


def test_preparation_navigation_and_opening_get_separate_nonrepeating_queries():
    from scripts.prestock_fragment_queries import first_deviations
    mismatch=jnp.ones((8,1),bool);fragment=jnp.full((8,1),2);done=jnp.zeros_like(mismatch)
    revision=jnp.array([1,1,1,1,1,2,2,2])[:,None]
    opening=jnp.array([0,0,1,0,1,1,0,1],bool)[:,None]
    result=first_deviations(mismatch,fragment,done,jnp.ones_like(mismatch),revision,opening)
    np.testing.assert_array_equal(result[:,0],[1,0,1,0,0,1,1,0])


def test_ordinary_fragment_quota_is_exact_and_episode_reset_restores_authority():
    from scripts.prestock_fragment_queries import first_deviations
    from scripts.je09_stage_h6_ops import first_fragment_deviations
    mismatch=jnp.array([[1,0],[1,1],[1,1],[1,1],[1,1]],bool)
    fragment=jnp.array([[1,3],[2,3],[1,4],[2,4],[1,3]])
    done=jnp.array([[0,0],[0,0],[1,1],[0,0],[0,0]],bool)
    result=first_deviations(mismatch,fragment,done,jnp.zeros_like(mismatch),jnp.zeros_like(fragment),jnp.zeros_like(mismatch))
    np.testing.assert_array_equal(result,first_fragment_deviations(mismatch,fragment,done))


def test_mismatch_uses_actual_joint_pair_so_correct_move_wrong_ability_queries():
    from scripts.prestock_fragment_queries import joint_mismatch
    target=jnp.zeros((4,5,3),bool).at[:,2,1].set(True)
    actions=jnp.array([[2,0],[2,1],[3,1],[2,1]])
    mismatch=joint_mismatch(target,actions,jnp.array([1,1,1,0],bool))
    np.testing.assert_array_equal(mismatch,[1,0,1,0])
