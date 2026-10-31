"""Experimental route-label qualification, separate from physical action masks."""
import jax.numpy as j
from scripts.dagger_rl_oracle import combine


def qualified_teacher(route, item, banana_inventory, *, enabled):
    """Qualify supervision only; item labels and deployment actions are unchanged.

    Item oracle overrides the entire route set. Without that certification,
    treating all legal abilities as equivalent provides no ability gradient.
    The experimental arm excludes banana release from those route sets. This
    is a falsifiable teacher preference, not proof that abstained plans are bad.
    Returns row counts: candidate, removed, route-set emptied.
    """
    no_item=~item.any((1,2))
    candidate=banana_inventory & no_item & route[:,:,2].any(1)
    remove=candidate & enabled
    restricted=route.at[:,:,2].set(route[:,:,2] & ~remove[:,None])
    emptied=remove & ~restricted.any((1,2))
    stats=j.stack([candidate.sum(),remove.sum(),emptied.sum()])
    return combine(restricted,item),stats
