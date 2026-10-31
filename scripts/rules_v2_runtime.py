"""Explicit process-local rule selection; old campaign imports remain unchanged."""
import sys

RULE_VERSION = 'rules_v2_native_parity'


def install():
    """Select the frozen native ABI before importing collector/reward modules.

    Checkpoint classes retain their versioned module names, so loading an old
    midgame cannot accidentally masquerade as a new-rule state.
    """
    dependencies = ('scripts.carrier_mechanism_engine', 'scripts.je09_bun_scenes',
                    'scripts.je10_ops', 'jax_bomb.jax_train')
    if any(name in sys.modules for name in dependencies):
        raise RuntimeError('Rules must be selected before training modules load')
    import jax_bomb
    from jax_bomb import je05_core_env_v2 as core
    from jax_bomb import je07_bun_env_v2 as env
    for name, module in (('je05_core_env', core), ('je07_bun_env', env)):
        sys.modules['jax_bomb.' + name] = module
        setattr(jax_bomb, name, module)
    return env


def assert_native_state(state):
    """Fail explicitly on incomplete migration or fixed-capacity overflow."""
    import numpy as np
    import jax
    if type(state).__module__ != 'jax_bomb.je07_bun_env_v2':
        raise ValueError('Expected versioned native state; old midgame rejected')
    if any(v is None for v in jax.tree.leaves(state, is_leaf=lambda v: v is None)):
        raise ValueError('Incomplete native state')
    if np.any(np.asarray(state.native_items.overflow) != 0):
        raise ValueError('Native item queue overflow')
