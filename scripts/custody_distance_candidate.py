"""Unweighted hard-obstacle distances; candidate only, never auto-installed.

Eight integer relaxations per convergence check reduce launch/synchronization
overhead. The finite grid bound and monotone fixed point preserve every cell.
"""
import jax
import jax.numpy as j
from scripts.je08_bun_potential import INF


def shortest(blocked, goals):
    h, w = goals.shape[-2:]
    initial = j.where(goals & ~blocked, 0, INF).astype(j.int32)

    def sweep(distance):
        entry = j.minimum(INF, distance + 1)
        padding = [(0, 0)] * (distance.ndim - 2) + [(1, 1), (1, 1)]
        p = j.pad(entry, padding, constant_values=INF)
        near = j.minimum(j.minimum(p[..., 2:, 1:-1], p[..., :-2, 1:-1]),
                         j.minimum(p[..., 1:-1, 2:], p[..., 1:-1, :-2]))
        return j.where(blocked, INF, j.minimum(distance, near))

    def body(carry):
        iteration, before, _ = carry
        distance = before
        for _ in range(8):
            distance = sweep(distance)
        return iteration + 8, distance, j.any(distance != before)

    return jax.lax.while_loop(lambda c: (c[0] < h * w) & c[2], body,
                             (j.int32(0), initial, j.array(True)))[1]
