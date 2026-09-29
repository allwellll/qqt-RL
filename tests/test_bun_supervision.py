"""RED-first fixed traces for deterministic danger obs + actor supervision.

Every quantity asserted here is derivable from the CURRENT public board
(fuses, walls, bricks, blast ranges, chain propagation, and the acting
player's own hypothetical bomb) via the existing safety oracle. Nothing here
consults an opponent policy or a future rollout, so the danger slices are
admissible as observation channels and the safe-action/escape quantities are
admissible as gradient targets. The traces pin the labels to hand-checked
scenarios before any of it is wired into obs, the network, or the loss.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_bomb import bun_env
from jax_bomb import bun_safety
from jax_bomb import bun_supervision
from jax_bomb import levels


@pytest.fixture(autouse=True)
def bun_level():
    bun_env.prepare()
    bun_env.configure_training("full=1", 1)
    yield
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


def _open_state(seed=0):
    state = bun_env._fresh(jax.random.PRNGKey(seed))
    core = state.core._replace(
        wall=jnp.zeros_like(state.core.wall),
        brick=jnp.zeros_like(state.core.brick),
        pushable=jnp.zeros_like(state.core.pushable),
        brick_linger=jnp.zeros_like(state.core.brick_linger),
        fuse=jnp.zeros_like(state.core.fuse),
        owner=jnp.full_like(state.core.owner, -1),
        bomb_blast=jnp.zeros_like(state.core.bomb_blast),
        blast_linger=jnp.zeros_like(state.core.blast_linger),
        pos=jnp.asarray([[6.5, 6.5], [10.5, 12.5]], jnp.float32),
        hp=jnp.ones_like(state.core.hp),
        alive=jnp.ones_like(state.core.alive),
        blast_cap=jnp.asarray([2.0, 2.0], jnp.float32),
        bombs_cap=jnp.asarray([2.0, 2.0], jnp.float32),
        spd_g=jnp.ones_like(state.core.spd_g),
    )
    return state._replace(
        core=core,
        blast_owner_linger=jnp.zeros_like(state.blast_owner_linger),
        move_status=jnp.zeros_like(state.move_status),
        status_ticks=jnp.zeros_like(state.status_ticks),
    )


def _bomb(state, row, column, fuse, owner, blast=2):
    return state._replace(core=state.core._replace(
        fuse=state.core.fuse.at[row, column].set(fuse),
        owner=state.core.owner.at[row, column].set(owner),
        bomb_blast=state.core.bomb_blast.at[row, column].set(blast)))


def test_danger_slices_are_temporally_monotone_and_respect_fuse():
    # bomb fuse=5 -> detonation at ~4 ticks after the -1 candidate step.
    state = _bomb(_open_state(1), 6, 6, 5, 0, blast=2)
    slices = bun_supervision.danger_slices(state, pid=0)

    assert slices.shape[0] == len(bun_supervision.DANGER_SLICE_TICKS)
    # slice ordering matches DANGER_SLICE_TICKS = (2,4,6,8,10)
    k2, k4 = slices[0], slices[1]
    assert float(k2[6, 6]) == 0.0        # not within 2 ticks
    assert float(k4[6, 6]) == 1.0        # within 4 ticks
    # monotone: an earlier-horizon danger cell stays dangerous at every later
    for earlier, later in zip(slices[:-1], slices[1:]):
        assert bool((later >= earlier).all())


def test_danger_slices_respect_walls_and_blast_range():
    state = _open_state(2)
    state = state._replace(core=state.core._replace(
        wall=state.core.wall.at[6, 7].set(True)))
    state = _bomb(state, 6, 6, 5, 0, blast=2)
    slices = bun_supervision.danger_slices(state, pid=0)
    within = slices[1]                    # k<=4 covers this bomb
    assert float(within[6, 5]) == 1.0     # left of bomb, open
    assert float(within[6, 8]) == 0.0     # shadowed by wall at (6,7)
    assert float(within[6, 9]) == 0.0     # beyond blast range anyway


def test_own_bomb_footprint_lights_where_my_new_bomb_would_burn():
    state = _open_state(3)                # empty board, no live bombs
    footprint = bun_supervision.own_bomb_footprint(state, pid=0)
    assert float(footprint[6, 6]) == 1.0  # my cell
    assert float(footprint[6, 8]) == 1.0  # blast_cap=2 reach
    assert float(footprint[6, 9]) == 0.0  # beyond reach
    assert float(footprint[10, 12]) == 0.0  # opponent cell, unaffected


def test_safe_action_label_matches_oracle_survivable_and_legal():
    state = _bomb(_open_state(4), 6, 6, 6, 0, blast=3)
    labels = bun_supervision.actor_supervision_labels(state)
    oracle = bun_safety.analyze_actions(state)
    expect = (oracle.survivable & oracle.legal).reshape(
        2, bun_env.N_MOVES * bun_env.N_BOMB).astype(jnp.float32)
    assert jnp.array_equal(labels["safe_action"], expect)
    assert labels["safe_action"].shape == (2, bun_env.N_MOVES * bun_env.N_BOMB)


def test_escape_label_is_zero_in_a_doomed_state():
    state = _open_state(5)
    walls = jnp.ones_like(state.core.wall).at[6, 6].set(False)
    state = state._replace(core=state.core._replace(wall=walls))
    state = _bomb(state, 6, 6, 1, 0, 1)   # boxed in, imminent detonation
    labels = bun_supervision.actor_supervision_labels(state)
    oracle = bun_safety.analyze_actions(state)
    assert bool(oracle.doomed[0])
    assert float(labels["escape"][0]) == 0.0
    assert float(labels["margin"][0]) == 0.0


def test_escape_margin_open_board_is_higher_than_boxed_board():
    open_state = _open_state(6)
    open_labels = bun_supervision.actor_supervision_labels(open_state)
    # a live enemy bomb two cells away removes some—but not all—safe moves
    boxed = _bomb(_open_state(6), 6, 8, 4, 1, blast=3)
    boxed_labels = bun_supervision.actor_supervision_labels(boxed)
    assert float(open_labels["margin"][0]) >= float(boxed_labels["margin"][0])
    assert 0.0 <= float(boxed_labels["margin"][0]) <= 1.0


def test_danger_obs_channels_stack_slices_then_own_footprint():
    state = _bomb(_open_state(7), 6, 6, 5, 0, blast=2)
    channels = bun_supervision.danger_obs_channels(state, pid=0)
    n_slice = len(bun_supervision.DANGER_SLICE_TICKS)
    assert channels.shape == (n_slice + 1, bun_env.H, bun_env.W)
    slices = bun_supervision.danger_slices(state, pid=0)
    footprint = bun_supervision.own_bomb_footprint(state, pid=0)
    assert jnp.array_equal(channels[:n_slice], slices)
    assert jnp.array_equal(channels[n_slice], footprint)


def test_labels_are_deterministic_under_fixed_seed():
    a = bun_supervision.actor_supervision_labels(_open_state(20260929))
    b = bun_supervision.actor_supervision_labels(_open_state(20260929))
    assert jnp.array_equal(a["safe_action"], b["safe_action"])
    assert jnp.array_equal(a["escape"], b["escape"])
    assert jnp.array_equal(a["margin"], b["margin"])
