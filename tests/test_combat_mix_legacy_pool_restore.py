import numpy as np
import jax
import jax.numpy as j
from scripts.combat_mix_legacy import advance_pool

def test_pickled_numpy_league_pool_matches_device_pool_update():
    pool={'weights':np.arange(33,dtype=np.float32).reshape(11,3)}
    actor={'weights':np.array([100,101,102],np.float32)}
    # Original restore error occurs for NumPy payloads; device values are the reference.
    expected=advance_pool(jax.tree.map(j.asarray,pool),actor,16)
    actual=advance_pool(pool,actor,16)
    assert np.array_equal(actual['weights'],expected['weights'])
    assert np.array_equal(actual['weights'][0],actor['weights'])
    assert np.array_equal(actual['weights'][7],actor['weights'])
    assert np.array_equal(actual['weights'][1:7],pool['weights'][1:7])
