"""Compose route and item teachers without treating release as route movement.

Ability ``2`` is a release action.  A route label may include it only when the
item oracle has certified the complete placement, leave, contact and settlement
chain for the current public state.  The item oracle remains set-valued and has
priority when it is present.  This adapter is intentionally independent of the
legacy oracle so sealed historical namespaces keep their original semantics.
"""

import jax.numpy as j


def compose_teacher(route, item):
    """Return the public set-valued teacher for one learner-state batch.

    ``route`` and ``item`` have shape ``(batch, move, ability)``.  With no
    certified item plan, route movement and abilities 0/1 remain available,
    while ability 2 is removed.  A certified item plan replaces the route set
    exactly, preserving all safe equivalent item-chain actions.
    """
    if route.ndim != 3 or item.ndim != 3 or route.shape != item.shape:
        raise ValueError("route and item teachers must have equal rank-3 shape")
    certified = item.any(axis=(1, 2))
    route_without_release = route.at[:, :, 2].set(False)
    return j.where(certified[:, None, None], item, route_without_release)


__all__ = ["compose_teacher"]
