"""H3 single-variable operator: raise full natural-scene exposure only."""
import jax.numpy as jnp
from scripts import je09_stage_h2_ops as _parent
from scripts.je09_stage_h2_ops import *

_base_curriculum_weights = _parent.curriculum_weights

def curriculum_weights(counts, combat_fraction=.60, bridge_fraction=.25):
    """Keep H2 combat/bridge mass; allocate a fixed 0.10 to full scene (8).

    No state is initialized with carry, loose, drop, or pickup here. All later
    events remain native, so this changes only the lesson mixture.
    """
    base = _base_curriculum_weights(counts, combat_fraction, bridge_fraction)
    target = jnp.float32(.10)
    rest = jnp.maximum(1. - combat_fraction - bridge_fraction, 0.)
    non_full = base.at[0].set(0.).at[8].set(0.).at[9].set(0.)
    non_full = non_full / jnp.maximum(non_full.sum(), 1e-6) * jnp.maximum(rest - target, 0.)
    return non_full.at[0].set(combat_fraction).at[8].set(target).at[9].set(bridge_fraction)
