"""danger_arena spawn diversification: geometry, legality, reproducibility, JIT/auto-reset."""
import os

os.environ.setdefault("JAXBOMB_RULE", "bun")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bomb import bun_env
from jax_bomb import bun_safety
from jax_bomb import levels

ENVS = 512
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@pytest.fixture(scope="module", autouse=True)
def danger_arena():
    bun_env.prepare()
    bun_env.configure_training("danger_arena=1", 1, reward_profile="danger_arena")
    yield
    bun_env.configure_spawn_buckets(None)
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


@pytest.fixture(autouse=True)
def reset_buckets():
    # Spawn config is baked in at trace time; drop traces cached by other cases.
    jax.clear_caches()
    yield
    bun_env.configure_spawn_buckets(None)
    jax.clear_caches()


def _map():
    level = bun_env._BUN_LEVEL
    wall = np.asarray(level["wall"], np.bool_).reshape(bun_env.H, bun_env.W)
    brick = np.asarray(level["brick"], np.bool_).reshape(bun_env.H, bun_env.W)
    in_base = np.zeros_like(wall)
    for row, column in level["bun_bases"]:
        in_base[row:row + 3, column:column + 3] = True
    return wall, brick, in_base


def _bfs(opened, start):
    distance = {start: 0}
    queue = [start]
    for current in queue:
        for a, b in STEPS:
            nxt = (current[0] + a, current[1] + b)
            if (0 <= nxt[0] < bun_env.H and 0 <= nxt[1] < bun_env.W
                    and opened[nxt] and nxt not in distance):
                distance[nxt] = distance[current] + 1
                queue.append(nxt)
    return distance


def _cells(states):
    return np.floor(np.asarray(states.core.pos)).astype(np.int32)


def _in_bucket(name, d, dr, dc):
    return {
        "near": 2 <= d <= 4, "mid": 5 <= d <= 8, "far": d >= 9,
        "below": d >= 3 and dr >= 2 and abs(dc) <= dr,
        "upper_left": d >= 3 and dr <= -1 and dc <= -1,
        "upper_right": d >= 3 and dr <= -1 and dc >= 1,
    }[name]


@pytest.mark.parametrize("bucket", bun_env.SPAWN_BUCKET_NAMES[1:])
def test_each_bucket_geometry_and_legality(bucket):
    bun_env.configure_spawn_buckets(f"{bucket}=1")
    states = jax.jit(bun_env.init_batch, static_argnums=1)(
        jax.random.PRNGKey(20261001), ENVS)
    wall, brick, in_base = _map()
    opened = ~wall & ~brick
    cells = _cells(states)
    assert len({tuple(map(tuple, pair)) for pair in cells}) > 8
    for p0, p1 in cells:
        p0, p1 = tuple(p0), tuple(p1)
        assert opened[p0] and opened[p1] and not in_base[p0] and not in_base[p1]
        assert p0 != p1
        d = _bfs(opened, p0).get(p1)
        assert d is not None, "unreachable spawn"
        assert _in_bucket(bucket, d, p1[0] - p0[0], p1[1] - p0[1]), (p0, p1, d)
    np.testing.assert_array_equal(np.asarray(states.core.wall), np.broadcast_to(wall, (ENVS,) + wall.shape))
    np.testing.assert_array_equal(np.asarray(states.core.brick), np.broadcast_to(brick, (ENVS,) + brick.shape))
    np.testing.assert_array_equal(np.asarray(states.bun_spawn_pos), np.asarray(states.core.pos))
    assert (np.asarray(states.core.fuse) == 0).all()
    assert (np.asarray(states.lesson) == bun_env.LESSON_DANGER_ARENA).all()
    analysis = jax.jit(jax.vmap(bun_safety.analyze_actions))(states)
    assert not np.asarray(analysis.doomed).any()


def test_native_only_and_disabled_are_bit_identical():
    key = jax.random.PRNGKey(7)
    bun_env.configure_spawn_buckets(None)
    native = bun_env.init_batch(key, 256)
    bun_env.configure_spawn_buckets("native=1")
    forced = bun_env.init_batch(key, 256)
    for got, want in zip(jax.tree.leaves(forced), jax.tree.leaves(native)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_only_positions_change_vs_native():
    key = jax.random.PRNGKey(11)
    native = bun_env.init_batch(key, 256)
    bun_env.configure_spawn_buckets(bun_env.DEFAULT_SPAWN_BUCKETS)
    mixed = bun_env.init_batch(key, 256)
    skip = {"pos", "bun_spawn_pos"}
    for field in native._fields:
        if field == "core":
            for sub in native.core._fields:
                if sub not in skip:
                    np.testing.assert_array_equal(
                        np.asarray(getattr(mixed.core, sub)),
                        np.asarray(getattr(native.core, sub)), err_msg=sub)
        elif field not in skip:
            np.testing.assert_array_equal(
                np.asarray(getattr(mixed, field)), np.asarray(getattr(native, field)),
                err_msg=field)


def test_reproducible_and_ratio_matches_weights():
    audit = bun_env.configure_spawn_buckets(bun_env.DEFAULT_SPAWN_BUCKETS)
    assert audit["rejected_by_safety"] >= 0
    assert all(count > 0 for count in audit["valid_pairs"].values())
    key = jax.random.PRNGKey(20261001)
    first = _cells(bun_env.init_batch(key, 4096))
    np.testing.assert_array_equal(first, _cells(bun_env.init_batch(key, 4096)))
    assert not np.array_equal(first, _cells(bun_env.init_batch(jax.random.PRNGKey(1), 4096)))
    ids = np.asarray(bun_env.spawn_bucket_ids(key, 4096))
    expected = np.asarray([audit["weights"][n] for n in bun_env.SPAWN_BUCKET_NAMES])
    observed = np.bincount(ids, minlength=len(expected)) / ids.size
    np.testing.assert_allclose(observed, expected, atol=0.025)
    native_pairs = {tuple(map(tuple, p)) for p in np.asarray(bun_env._DANGER_ARENA_SPAWN_PAIRS)}
    native_pairs |= {(b, a) for a, b in native_pairs}
    for pair in first[ids == 0]:
        assert tuple(map(tuple, pair)) in native_pairs


def test_auto_reset_uses_diversified_spawns_under_jit():
    bun_env.configure_spawn_buckets("far=1")
    states = bun_env.init_batch(jax.random.PRNGKey(3), 128)
    last = bun_env.LESSON_MAX_STEPS[bun_env.LESSON_DANGER_ARENA] - 1
    states = states._replace(core=states.core._replace(
        t=jnp.full_like(states.core.t, last)))
    idle = jnp.tile(jnp.asarray([[[4, 0], [4, 0]]], jnp.int32), (128, 1, 1))
    step = jax.jit(jax.vmap(lambda s, a, k: bun_env.step(s, a, k)))
    after, done = step(states, idle, jax.random.split(jax.random.PRNGKey(4), 128))
    assert np.asarray(done).all()
    wall, brick, _ = _map()
    opened = ~wall & ~brick
    for p0, p1 in _cells(after):
        assert _bfs(opened, tuple(p0)).get(tuple(p1), 0) >= 9
    assert (np.asarray(after.core.t) == 0).all()


def test_safety_filter_rejects_trapping_spawns():
    bun_env.configure_spawn_buckets("near=1")
    pairs = np.asarray([[[0, 2], [1, 2]], [[1, 2], [0, 2]], [[4, 2], [4, 12]]], np.int32)
    np.testing.assert_array_equal(bun_env._spawn_pairs_safe(pairs), [False, False, True])


def test_bad_spec_rejected():
    with pytest.raises(ValueError):
        bun_env.configure_spawn_buckets("sideways=1")
    with pytest.raises(ValueError):
        bun_env.configure_spawn_buckets("near=0")
