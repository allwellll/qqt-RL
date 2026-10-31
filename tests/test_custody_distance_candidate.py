import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import jax
import jax.numpy as j
import numpy as np
from scripts.custody_distance_candidate import shortest
from scripts.je08_bun_potential import shortest as reference


def test_hard_distance_exact_random_topology_multigoal_blocked_absent():
    rng = np.random.default_rng(2026208031)
    walls = rng.random((64, 1, 13, 15)) < .32
    goals = rng.random((64, 2, 13, 15)) < .02
    goals[0] = False
    walls[1] = True
    expected = jax.jit(reference)(j.asarray(walls), j.zeros_like(j.asarray(walls)), j.asarray(goals))
    actual = jax.jit(shortest)(j.asarray(walls), j.asarray(goals))
    np.testing.assert_array_equal(actual, expected)


def test_hard_distance_exact_long_winding_path():
    wall = np.ones((1, 1, 13, 15), bool)
    for row in range(0, 13, 2):
        wall[..., row, :] = False
        if row < 12:
            wall[..., row + 1, 14 if (row // 2) % 2 == 0 else 0] = False
    goals = np.zeros((1, 1, 13, 15), bool)
    goals[..., 0, 0] = True
    expected = reference(j.asarray(wall), j.zeros_like(j.asarray(wall)), j.asarray(goals))
    actual = jax.jit(shortest)(j.asarray(wall), j.asarray(goals))
    np.testing.assert_array_equal(actual, expected)
    assert int(actual[0, 0, 12, 14]) > 100


def test_fast2_sparse_fields_retain_clean_cache_and_padding():
    from scripts.bun_native_v2_custody_fast2 import sparse_fields
    walls = j.zeros((7, 13, 15), bool).at[1, :, 7].set(True)
    goals = j.zeros((7, 2, 13, 15), bool).at[:, 0, 1, 2].set(True).at[:, 1, 10, 12].set(True)
    expected = reference(walls[:, None], j.zeros_like(walls[:, None]), goals)
    dirty = j.arange(7) % 2 == 0
    previous = j.where(dirty[:, None, None, None], 32767, expected)
    np.testing.assert_array_equal(jax.jit(sparse_fields)(walls, goals, previous, dirty), expected)
