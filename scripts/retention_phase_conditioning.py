"""R5 isolated phase-conditioning candidate; import never installs it."""
import jax
import jax.numpy as jnp
import numpy as np
from jax_bomb import jax_net as net

COMBAT_CONTROL, STOCK_WINDOW, DROP_RECOVERY, CARRY_RETURN, CAGE_DELIVERY = range(5)
_FORWARD = net.transformer_forward
_AUX_FORWARD = net.transformer_aux_forward
ENCODING = 'phase5-visible-v1'


def phase_id(obs, state):
    """Exclusive current-observation hint, never reward/teacher authority.

    Delivery > enemy cargo > visible enemy-origin drop > current stock window
    > combat. Dead/trapped noncarriers and own-origin cargo use combat fallback.
    Home proximity is splat mass >= .5 on the cage plus its four-neighbor halo.
    No pathfinder, scene, future, credit latch, or accepted oracle is accessed.
    """
    obs, state = jnp.asarray(obs), jnp.asarray(state)
    if obs.shape[-3:] != (24, 13, 15) or state.shape[-1] != 24:
        raise ValueError('phase5 requires native Bun24 obs and trap-timer state')
    home = obs[..., 14, :, :] > 0
    padded = jnp.pad(home, [(0, 0)] * (home.ndim-2) + [(1, 1), (1, 1)])
    halo = (home | padded[..., :-2, 1:-1] | padded[..., 2:, 1:-1]
            | padded[..., 1:-1, :-2] | padded[..., 1:-1, 2:])
    near = jnp.sum(obs[..., 0, :, :] * halo, axis=(-2, -1)) >= .5
    alive = state[..., 9] > 0
    free = state[..., 9] == 1
    cargo = alive & (state[..., 14] > 0)
    empty = state[..., 13] == 0
    drop = free & empty & jnp.any(obs[..., 17, :, :] > 0, axis=(-2, -1))
    window = (state[..., 10] <= 0) | (state[..., 10] > 1)
    stock = free & empty & window & (state[..., 20] > 0)
    return jnp.where(cargo, jnp.where(near, CAGE_DELIVERY, CARRY_RETURN),
                     jnp.where(drop, DROP_RECOVERY,
                               jnp.where(stock, STOCK_WINDOW, COMBAT_CONTROL)))


def features(obs, state):
    return jax.nn.one_hot(phase_id(obs, state), 5)


def migrate(params):
    """Fresh-input migration only; preserve old rows and all other leaves."""
    if isinstance(params, dict):
        result = {k: migrate(v) for k, v in params.items()}
        if 'state_w' in params:
            weight = params['state_w']
            if weight.shape[-2] != 24:
                raise ValueError('migration expects unconditioned raw parent')
            zeros = np.zeros(weight.shape[:-2] + (5, weight.shape[-1]), weight.dtype)
            concat = np.concatenate if isinstance(weight, np.ndarray) else jnp.concatenate
            result['state_w'] = concat([weight, zeros], axis=-2)
        return result
    if isinstance(params, (list, tuple)):
        return type(params)(migrate(v) for v in params)
    return params


def conditioned_state(params, obs, state, feature_override=None):
    if params['state_w'].shape[-2] == 24:
        return state
    if params['state_w'].shape[-2] != 29 or state is None:
        raise ValueError('conditioned network requires raw visible Bun24 inputs')
    encoded = features(obs, state) if feature_override is None else feature_override
    if encoded.shape != jnp.asarray(state).shape[:-1] + (5,):
        raise ValueError('phase override shape mismatch')
    return jnp.concatenate([jnp.asarray(state), encoded], -1)


def forward(params, obs, state=None, feature_override=None):
    return _FORWARD(params, jnp.asarray(obs),
                    conditioned_state(params, obs, state, feature_override))


def aux_forward(params, obs, state=None):
    return _AUX_FORWARD(params, obs, conditioned_state(params, obs, state))


def install():
    """Explicit per-process opt-in; old networks bypass feature extraction."""
    net.transformer_forward = forward
    net.transformer_aux_forward = aux_forward
