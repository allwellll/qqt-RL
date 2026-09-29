"""Deterministic, leakage-free supervision features and labels for the Bun actor.

Everything here is a pure function of the CURRENT public game state — bomb
fuses, walls, bricks, blast ranges, deterministic chain propagation, and the
acting player's own hypothetical bomb. No opponent policy, no future rollout,
and no hidden planner state is consulted. That admissibility is what lets the
same machinery serve two roles:

  * ``danger_obs_channels`` — extra observation planes. A cell's near-future
    blast membership is something a perfect rules-reasoner reads straight off
    the board, so feeding it as input leaks nothing the agent could not compute.

  * ``actor_supervision_labels`` — auxiliary gradient targets. These predict the
    *consequences* the agent must reason about (which joint actions keep an
    escape branch alive, whether any escape exists, how boxed-in it is). PPO's
    lone "died -> -1" signal teaches this only through very long credit chains;
    a direct BCE target on the shared backbone teaches it in one step.

Design note vs. the reference sketch: the reference proposed a head that
*predicts the danger map*. Once the danger map is an observation input that is
redundant, so the heads here instead predict the escape/safe-action structure
that is NOT in the observation and genuinely requires simulating the acting
player's own bomb plus time-expanded reachability.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import bun_env as env
from . import bun_safety
from . import jax_env as base

# Escape movement is ~1/STEP ticks per cell (STEP=0.3 -> ~4 ticks). Bucketing
# detonation deadlines at these tick horizons yields crisp "will this cell burn
# within k ticks" planes spanning ~2.5 cells of escape range. The existing
# graded ch5 danger scalar already carries continuous timing; these add sharp
# temporal structure a patchify transformer can attend to directly.
DANGER_SLICE_TICKS: tuple[int, ...] = (2, 4, 6, 8, 10)


def _all_owner_deadlines(state: env.BunState) -> jnp.ndarray:
    """(H, W) int ticks until any live bomb's blast reaches each cell."""
    deadlines, _, _ = bun_safety._source_deadlines(
        state.core, state.blast_owner_linger, None)
    return deadlines.min(axis=0)


def danger_slices(state: env.BunState, pid: int,
                  ticks: tuple[int, ...] = DANGER_SLICE_TICKS) -> jnp.ndarray:
    """(len(ticks), H, W) float32 in {0,1}; plane k = deadline <= ticks[k].

    Wall/brick shadowing and chain propagation are inherited from
    ``_source_deadlines``. Planes are temporally monotone by construction
    (a cell dangerous within k ticks is dangerous within any larger horizon).
    ``pid`` is accepted for a symmetric call signature but the ambient danger
    field is owner-agnostic, so the slices are identical for either player.
    """
    del pid
    deadline = _all_owner_deadlines(state)
    thresholds = jnp.asarray(ticks, jnp.int32)[:, None, None]
    return (deadline[None] <= thresholds).astype(jnp.float32)


def own_bomb_footprint(state: env.BunState, pid: int) -> jnp.ndarray:
    """(H, W) float32 in {0,1}: cells that would burn if ``pid`` bombs now.

    Uses the acting player's own hypothetical bomb (``place_player=pid``) and
    keeps only that player's blast source, so the plane answers "the danger I
    am about to create" independent of fuse timing (a fresh FUSE-tick bomb sits
    far past any small horizon, so a footprint — not a thresholded slice — is
    the meaningful signal).
    """
    placed, _, _ = bun_safety._source_deadlines(
        state.core, state.blast_owner_linger, pid)
    return (placed[pid] < bun_safety._INF_TICK).astype(jnp.float32)


def danger_obs_channels(state: env.BunState, pid: int) -> jnp.ndarray:
    """(len(DANGER_SLICE_TICKS)+1, H, W): temporal slices then own footprint."""
    slices = danger_slices(state, pid)
    footprint = own_bomb_footprint(state, pid)[None]
    return jnp.concatenate([slices, footprint], axis=0)


N_DANGER_CHANNELS = len(DANGER_SLICE_TICKS) + 1
N_SAFE_ACTION = env.N_MOVES * env.N_BOMB


def actor_supervision_labels(state: env.BunState) -> dict[str, jnp.ndarray]:
    """Per-player deterministic gradient targets from the safety oracle.

    safe_action : (2, N_MOVES*N_BOMB) float in {0,1} — joint action is legal and
                  retains a survivable escape branch through the blast horizon.
    escape      : (2,) float in {0,1} — at least one legal action survives now.
    margin      : (2,) float in [0,1] — fraction of legal no-bomb first moves
                  that survive; a graded "how boxed in am I".
    """
    oracle = bun_safety.analyze_actions(state)
    safe = (oracle.survivable & oracle.legal).reshape(2, N_SAFE_ACTION)
    escape = (oracle.survivable & oracle.legal).any(axis=(1, 2))
    move_safe = (oracle.survivable[:, :, 0] & oracle.legal[:, :, 0]).sum(axis=1)
    margin = move_safe.astype(jnp.float32) / float(env.N_MOVES)
    return {
        "safe_action": safe.astype(jnp.float32),
        "escape": escape.astype(jnp.float32),
        "margin": margin,
    }
