#!/usr/bin/env python3
"""Audit danger_arena spawn buckets: valid pairs per bucket, sampled ratio on the
reset path, and the distance / direction mix on the jitted auto-reset path.

  JAXBOMB_RULE=bun python scripts/audit_spawn_buckets.py \
      --spawn-buckets native=0.3,near=0.1,... --envs 8192 --seed 20261001
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_env as E  # noqa: E402


def distances(opened):
    """All-pairs BFS steps on the native open-cell graph (-1 = unreachable)."""
    out = np.full((E.H, E.W, E.H, E.W), -1, np.int32)
    for start in map(tuple, np.argwhere(opened)):
        out[start][start] = 0
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (r + a, c + b)
                if 0 <= n[0] < E.H and 0 <= n[1] < E.W and opened[n] and out[start][n] < 0:
                    out[start][n] = out[start][r, c] + 1
                    queue.append(n)
    return out


def mix(cells, dist):
    p0, p1 = cells[:, 0], cells[:, 1]
    d = dist[p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]]
    dr, dc = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
    n = len(cells)
    return {
        "unreachable": int((d < 0).sum()),
        "overlap": int(((p0 == p1).all(-1)).sum()),
        "dist_2_4": float(((d >= 2) & (d <= 4)).sum() / n),
        "dist_5_8": float(((d >= 5) & (d <= 8)).sum() / n),
        "dist_9plus": float((d >= 9).sum() / n),
        "enemy_below": float(((dr >= 2) & (np.abs(dc) <= dr)).sum() / n),
        "enemy_upper_left": float(((dr <= -1) & (dc <= -1)).sum() / n),
        "enemy_upper_right": float(((dr <= -1) & (dc >= 1)).sum() / n),
        "mean_bfs_distance": float(d[d >= 0].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spawn-buckets", default=E.DEFAULT_SPAWN_BUCKETS)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    E.prepare()
    E.configure_training("danger_arena=1", 1, reward_profile="danger_arena")
    wall = np.asarray(E._BUN_LEVEL["wall"], np.bool_).reshape(E.H, E.W)
    brick = np.asarray(E._BUN_LEVEL["brick"], np.bool_).reshape(E.H, E.W)
    dist = distances(~wall & ~brick)

    key = jax.random.PRNGKey(args.seed)
    native = np.floor(np.asarray(E.init_batch(key, args.envs).core.pos)).astype(np.int32)
    audit = E.configure_spawn_buckets(args.spawn_buckets)
    states = jax.jit(E.init_batch, static_argnums=1)(key, args.envs)
    ids = np.asarray(E.spawn_bucket_ids(key, args.envs))
    observed = np.bincount(ids, minlength=len(E.SPAWN_BUCKET_NAMES)) / ids.size

    last = E.LESSON_MAX_STEPS[E.LESSON_DANGER_ARENA] - 1
    timed_out = states._replace(core=states.core._replace(t=jnp.full_like(states.core.t, last)))
    idle = jnp.tile(jnp.asarray([[[4, 0], [4, 0]]], jnp.int32), (args.envs, 1, 1))
    step = jax.jit(jax.vmap(lambda s, a, k: E.step(s, a, k)))
    reset, done = step(timed_out, idle, jax.random.split(jax.random.fold_in(key, 1), args.envs))

    report = {
        "spawn_buckets": args.spawn_buckets, "envs": args.envs, "seed": args.seed,
        "config": audit,
        "sampled_bucket_ratio": dict(zip(E.SPAWN_BUCKET_NAMES, map(float, observed))),
        "reset_mix": mix(np.floor(np.asarray(states.core.pos)).astype(np.int32), dist),
        "auto_reset_done": float(np.asarray(done).mean()),
        "auto_reset_mix": mix(np.floor(np.asarray(reset.core.pos)).astype(np.int32), dist),
        "native_mix": mix(native, dist),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    bad = (report["reset_mix"]["unreachable"] + report["reset_mix"]["overlap"]
           + report["auto_reset_mix"]["unreachable"] + report["auto_reset_mix"]["overlap"])
    sys.exit(1 if bad or report["auto_reset_done"] < 1.0 else 0)


if __name__ == "__main__":
    main()
