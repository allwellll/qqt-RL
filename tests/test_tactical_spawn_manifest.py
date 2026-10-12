from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from jax_bomb.bun_tactical_eval import apply_spawn_manifest


class Core(NamedTuple):
    pos: object
    wall: object
    brick: object
    fuse: object


class State(NamedTuple):
    core: Core
    bun_spawn_pos: object
    seed_marker: object


def state():
    return State(Core(jnp.zeros((2, 2, 2)), jnp.zeros((2, 13, 15), dtype=bool),
                      jnp.zeros((2, 13, 15), dtype=bool), jnp.ones((2, 13, 15))),
                 jnp.zeros((2, 2, 2)), jnp.array([81, 92]))


def test_manifest_sets_initial_and_respawn_positions_only():
    original = state()
    cells = [[[7, 0], [4, 0]], [[6, 14], [3, 12]]]
    updated = apply_spawn_manifest(original, cells)
    np.testing.assert_array_equal(updated.core.pos, np.asarray(cells) + .5)
    np.testing.assert_array_equal(updated.bun_spawn_pos, updated.core.pos)
    np.testing.assert_array_equal(original.core.pos, 0)
    assert updated.core.wall is original.core.wall
    assert updated.core.brick is original.core.brick
    assert updated.core.fuse is original.core.fuse
    assert updated.seed_marker is original.seed_marker


@pytest.mark.parametrize('cells', [
    [[[7, 0], [4, 0]]],
    [[[7, 0], [7, 0]], [[6, 14], [3, 12]]],
    [[[7.5, 0], [4, 0]], [[6, 14], [3, 12]]],
    [[[float('nan'), 0], [4, 0]], [[6, 14], [3, 12]]],
    [[[13, 0], [4, 0]], [[6, 14], [3, 12]]],
    [[[7, -1], [4, 0]], [[6, 14], [3, 12]]],
])
def test_invalid_manifest_is_rejected(cells):
    with pytest.raises(ValueError):
        apply_spawn_manifest(state(), cells)


@pytest.mark.parametrize('field', ['wall', 'brick'])
def test_blocked_cell_is_rejected(field):
    original = state()
    blocked = getattr(original.core, field).at[0, 7, 0].set(True)
    original = original._replace(core=original.core._replace(**{field: blocked}))
    with pytest.raises(ValueError, match='blocked'):
        apply_spawn_manifest(original, [[[7, 0], [4, 0]], [[6, 14], [3, 12]]])
