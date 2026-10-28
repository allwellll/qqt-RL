"""H2 changes only initial item coverage; no custody or combat shortcuts."""
import jax
import jax.numpy as jnp
import numpy as np


def test_coverage_changes_only_real_drop_initial_items():
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    from scripts.je09_stage_h2_coverage import cover
    env.prepare()
    original = scenes.init_batch(jax.random.PRNGKey(2803000000), 32,
        jnp.array([.5, 0, 0, 0, 0, 0, 0, 0, 0, .5]), jnp.int32(0))
    covered = cover(original, jax.random.PRNGKey(2803000001), 1.)
    excluded = np.asarray(original.scene != 9)
    for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(covered)):
        np.testing.assert_array_equal(np.asarray(a)[excluded], np.asarray(b)[excluded])
    for name in ['core', 'bun_carried', 'trapped', 'bun_loose', 'bun_respawn', 'bun_spawn_pos']:
        for a, b in zip(jax.tree.leaves(getattr(original, name)), jax.tree.leaves(getattr(covered, name))):
            np.testing.assert_array_equal(a, b)
    rows = np.flatnonzero(np.asarray(original.scene == 9)); seats = rows % 2
    assert len(rows)
    np.testing.assert_array_equal(np.asarray(covered.bun_carried)[rows, seats], -1)
    np.testing.assert_array_equal(np.asarray(covered.item_slots)[rows, seats, :2], np.tile([2, 1], (len(rows), 1)))
    np.testing.assert_array_equal(np.asarray(covered.item_counts)[rows, seats, :2], 1)
    np.testing.assert_array_equal(np.asarray(covered.held_item)[rows, seats], 2)
    zero = cover(original, jax.random.PRNGKey(2803000001), 0.)
    for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(zero)):
        np.testing.assert_array_equal(a, b)


def test_fixed_budget_has_no_short_score_stopper():
    from scripts.je09_stage_h2_policy import TRAINING_GENERATIONS, POINTS, observation
    assert TRAINING_GENERATIONS == 32 and POINTS == [0, 4, 8, 12, 16, 24, 32]
    for scores in [[16, 14, 16], [16, 0, 0, 0, 0, 0, 0]]:
        assert observation(scores)['may_stop_training'] is False
