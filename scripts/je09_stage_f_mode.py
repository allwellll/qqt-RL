"""Training-only native transport mode; no policy input or action override."""
import types
import jax
import jax.numpy as jnp
from scripts import je09_chase_expert as original_chase,je09_drop_expert as original_drop

def empty_transport_allowed(states):
    """Authoritative alive/trapped fields only; never movement debuffs/timers."""
    return jax.lax.stop_gradient((~states.alive[:,::-1]) | (states.trapped[:,::-1]>0))

def transition_allowed(before,after):
    # Close immediately at authoritative revival or sugar-bubble release.
    return empty_transport_allowed(before)&empty_transport_allowed(after)

def chase_gate(states,aux,seat):
    return original_chase.gate(states,aux,seat)&empty_transport_allowed(states)[jnp.arange(len(seat)),seat]

def chase_labels(states,aux,legal,seat,danger):
    target,active=original_chase.labels(states,aux,legal,seat,danger)
    return target,active&empty_transport_allowed(states)[jnp.arange(len(seat)),seat]

def drop_labels(states,aux,legal,seat,danger=None,hazard_mode='global'):
    target,active=original_drop.labels(states,aux,legal,seat,danger,hazard_mode)
    return target,active&empty_transport_allowed(states)[jnp.arange(len(seat)),seat]

chase=types.SimpleNamespace(**{k:getattr(original_chase,k) for k in ['state','update','supervised_loss']},gate=chase_gate,labels=chase_labels)
drop=types.SimpleNamespace(**{k:getattr(original_drop,k) for k in ['state','refresh','supervised_loss']},labels=drop_labels)
