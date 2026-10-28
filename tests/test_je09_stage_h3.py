"""H3 changes lesson exposure only; H2 initialization and losses are retained."""
import importlib
import os
import jax
import jax.numpy as jnp
import numpy as np


def test_full_exposure_and_retention_mass():
    ops = importlib.import_module(os.environ.get('JE09_H3_TEST_OPERATOR', 'scripts.je09_stage_h3_ops'))
    weights = np.asarray(ops.curriculum_weights(jnp.zeros((10,65)), .60, .25))
    np.testing.assert_allclose(weights[[0,8,9]], [.60,.10,.25], atol=1e-7)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-7)
    from scripts import je09_bun_ops as old
    old_weights = np.asarray(old.curriculum_weights(jnp.zeros((10,65)), .60, .25))
    np.testing.assert_allclose(weights[1:8]/weights[1:8].sum(), old_weights[1:8]/old_weights[1:8].sum(), atol=1e-7)


def test_unchanged_production_losses_and_native_full_start():
    from scripts import je09_stage_h3_ops as new, je09_stage_h_ops as parent
    from scripts.je09_stage_h2_coverage import ORIGINAL_INIT, cover
    from jax_bomb import je07_bun_env as env
    env.prepare()
    assert new.collect is parent.collect and new.ppo is parent.ppo
    full = ORIGINAL_INIT(jax.random.PRNGKey(3003000000),32,jax.nn.one_hot(8,10),jnp.int32(0))
    covered = cover(full,jax.random.PRNGKey(3003000001),.75)
    for before,after in zip(jax.tree.leaves(full),jax.tree.leaves(covered)):
        np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(full.bun_carried,-1)
    np.testing.assert_array_equal(full.bun_loose,False)
    np.testing.assert_array_equal(full.alive,True)
    np.testing.assert_array_equal(full.trapped,0)


def test_mode_and_item_contracts_are_preserved():
    from scripts import je09_stage_h3_ops as new, je09_stage_h2_ops as parent
    assert new.reward_state is parent.reward_state
    assert new.combined_reward is parent.combined_reward
    assert new.return_expert is parent.return_expert
    assert new.chase_expert is parent.chase_expert
