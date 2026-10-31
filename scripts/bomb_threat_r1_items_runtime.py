"""Opt-in R1 native-item rules; existing workers never import this installer."""
import sys
RULE_VERSION='r1-native-items-stock-window-v2'


def install():
    if 'scripts.carrier_mechanism_engine' in sys.modules:
        raise RuntimeError('R1 item rules must be selected before engine import')
    import jax_bomb
    from jax_bomb import je05_core_env_r1_items as core,je07_bun_env_r1_items as env
    for name,module in [('je05_core_env',core),('je07_bun_env',env)]:
        sys.modules['jax_bomb.'+name]=module
        setattr(jax_bomb,name,module)
    return env
