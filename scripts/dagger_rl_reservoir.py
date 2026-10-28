"""Candidate bounded DAgger memory; not imported by any active trainer.

Only selects aligned rows already supplied by the learner collector. It cannot
create environment states, labels or carried objects. The stream count belongs
in the checkpoint; seed belongs to the declared update's independent RNG.
"""
import numpy as np


def append(previous, new, *, seen, capacity, seed):
    """Algorithm R over a tuple of aligned arrays, without mutating inputs."""
    if capacity <= 0 or seen < 0 or not new:
        raise ValueError('positive capacity, nonnegative seen and row arrays required')
    n = len(new[0])
    if any(len(a) != n for a in new):
        raise ValueError('new row arrays must be aligned')
    if previous is None:
        if seen != 0:
            raise ValueError('missing reservoir for nonempty stream')
        previous = tuple(a[:0] for a in new)
    if len(previous) != len(new) or any(len(a) != min(seen, capacity) for a in previous):
        raise ValueError('reservoir length does not match saved stream count')
    if any(a.shape[1:] != b.shape[1:] or a.dtype != b.dtype for a, b in zip(previous, new)):
        raise ValueError('saved and new row schema differs')
    fill = min(n, capacity - len(previous[0]))
    pool = tuple(np.concatenate([old, fresh[:fill]]) for old, fresh in zip(previous, new))
    rng = np.random.default_rng(seed)
    replacements = {}
    for i in range(fill, n):
        slot = int(rng.integers(0, seen + i + 1))
        if slot < capacity:
            # Only the final replacement of a slot survives this update.
            replacements[slot] = i
    if replacements:
        slots = np.asarray(list(replacements), dtype=np.int64)
        indices = np.asarray(list(replacements.values()), dtype=np.int64)
        for dest, fresh in zip(pool, new):
            dest[slots] = fresh[indices]
    return pool, seen + n
