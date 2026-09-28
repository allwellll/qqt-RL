import jax.numpy as jnp
import numpy as np

from jax_bomb.jax_train import compute_gae


def test_gae_stops_bootstrap_at_episode_boundary():
    reward = jnp.asarray([[1.0], [2.0], [3.0]])
    value = jnp.asarray([[0.5], [0.6], [0.7]])
    next_value = jnp.asarray([[0.6], [0.7], [9.0]])
    done = jnp.asarray([[False], [True], [False]])
    actual = np.asarray(compute_gae(reward, value, next_value, done, 0.9, 0.8))[:, 0]
    delta2 = 3.0 + 0.9 * 9.0 - 0.7
    delta1 = 2.0 - 0.6
    delta0 = 1.0 + 0.9 * 0.6 - 0.5
    expected = np.asarray([delta0 + 0.9 * 0.8 * delta1, delta1, delta2])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
