"""Bounded GPU distance microbenchmark; no optimizer or training writes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('JAXBOMB_DEVICE_OVERRIDE', '2'),
    JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import pickle
import statistics
import time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json, lock, sha

OUT = Path('runs/bun_native_v2_20261030/custody_performance_v1/distance_candidate')


def main():
    guard = lock(OUT / 'profile.lock')
    # Use real frozen topology. No model/environment state is manufactured.
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts import combat_item_engine as engine
    engine.configure()
    from scripts import bun_native_v2_custody_fast as fast
    from scripts.custody_distance_candidate import shortest as candidate
    from scripts.je08_bun_potential import shortest as reference
    source = json.loads(Path('runs/bun_native_v2_20261030/custody_performance_v1/ATOMIC_SWITCH_READY.json').read_text())['states']['c']['checkpoint']
    cp = Path(source['checkpoint'])
    assert sha(cp / 'state.pkl') == source['state_sha256']
    payload = pickle.load((cp / 'state.pkl').open('rb'))
    states = jax.tree.map(lambda x: j.asarray(x[0]), payload['states'])
    cache = jax.tree.map(lambda x: j.asarray(x[0]), payload['aux']['custody_v2'])
    _, ready, _ = jax.jit(fast.REFERENCE)(states, cache)
    blocked = ready['blocked'][:64, None]
    target = j.maximum(ready['target'][:64], 0)
    goals = j.zeros((64, 2, 13, 15), bool).at[j.arange(64)[:, None], j.arange(2)[None], target[..., 0], target[..., 1]].set(True)
    measurements = {}
    reference_value = None
    for name, fn in [('reference', jax.jit(lambda b, g: reference(b, j.zeros_like(b), g))),
                     ('eight_sweep', jax.jit(candidate))]:
        atomic_json(OUT / 'PROGRESS.json', dict(stage=name, status='COMPILE', unix=time.time()))
        began = time.monotonic()
        value = fn(blocked, goals)
        jax.block_until_ready(value)
        cold = time.monotonic() - began
        if reference_value is None:
            reference_value = np.asarray(value)
        else:
            np.testing.assert_array_equal(reference_value, np.asarray(value))
        for _ in range(3):
            jax.block_until_ready(fn(blocked, goals))
        times = []
        for _ in range(20):
            began = time.monotonic()
            jax.block_until_ready(fn(blocked, goals))
            times.append(time.monotonic() - began)
        measurements[name] = dict(cold_seconds=cold, warm_median_seconds=statistics.median(times), warm_seconds=times)
    atomic_json(OUT / 'COMPLETE.json', dict(status='PASS', source=source,
        source_sha256=sha(Path('scripts/custody_distance_candidate.py')),
        distance_bitwise_exact=True, actual_topologies=64, training_updates=0,
        results=measurements, speedup=measurements['reference']['warm_median_seconds']/measurements['eight_sweep']['warm_median_seconds'],
        limitations=['fixed saved topology/target microbenchmark; no full collector or training acceptance',
                     'not installed in any training worker'], unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        atomic_json(OUT / 'BLOCKED.json', dict(status='BLOCKED', reason='inspect profile.log; all training untouched', unix=time.time()))
        raise
