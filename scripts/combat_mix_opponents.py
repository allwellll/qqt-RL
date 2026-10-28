"""Deterministic, auditable opponent and combat-scene allocation helpers."""
from __future__ import annotations

import numpy as np

OPPONENTS = ("rule3", "jax_hunter", "jax_hunter_hard", "selfplay_current",
             "selfplay_ring", "selfplay_fixed")
DEFAULT_WEIGHTS = (0.20, 0.25, 0.20, 0.15, 0.10, 0.10)
TIERS = {"rule3": None, "jax_hunter": "hunter", "jax_hunter_hard": "hunter_hard"}


def merge_opponent_actions(original, replacements, kinds):
    """Replace only P1 `(move, ability)` actions; preserve learner P0 exactly."""
    import jax.numpy as jnp
    actions = jnp.asarray(original)
    replacements = jnp.asarray(replacements)
    kinds = jnp.asarray(kinds, jnp.int32)
    if actions.ndim != 3 or actions.shape[1:] != (2, 2):
        raise ValueError("actions must have shape (N,2 players,2 action heads)")
    if replacements.shape != (actions.shape[0], len(OPPONENTS), 2):
        raise ValueError("replacements must contain every pool action for each environment")
    if kinds.shape != (actions.shape[0],):
        raise ValueError("one opponent id is required per environment")
    selected = replacements[jnp.arange(actions.shape[0]), kinds]
    return actions.at[:, 1, :].set(selected)


def validate_weights(weights=DEFAULT_WEIGHTS):
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(OPPONENTS),) or not np.isfinite(values).all():
        raise ValueError("opponent weights must be six finite values")
    if (values < 0).any() or not np.isclose(values.sum(), 1.0, atol=1e-7):
        raise ValueError("opponent weights must be non-negative and sum to one")
    return values


def fixed_allocation(count, seed, weights=DEFAULT_WEIGHTS):
    """Stable per-environment categorical pool; returned ids follow OPPONENTS."""
    if count <= 0:
        raise ValueError("environment count must be positive")
    rng = np.random.default_rng(int(seed))
    result = rng.choice(len(OPPONENTS), size=int(count), p=validate_weights(weights))
    return result.astype(np.int8)


def combat_scene_mask(count, seed, fraction=0.25):
    """Stable subset selector for clear-brick combat maps, independent of policy."""
    if count <= 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid scene allocation")
    n = int(round(count * fraction))
    mask = np.zeros(int(count), dtype=bool)
    if n:
        mask[np.random.default_rng(int(seed)).permutation(int(count))[:n]] = True
    return mask
