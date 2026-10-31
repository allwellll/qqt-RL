"""Explicit process-local native-stock rules; never imported by R1 workers."""
import sys
from types import ModuleType

RULE_VERSION = 'bun-native-v2'


def install():
    from scripts.bomb_threat_r1_items_runtime import install as native_items
    if 'scripts.carrier_mechanism_engine' in sys.modules:
        raise RuntimeError('select native-v2 before importing the engine')
    env = native_items()
    original = env.step
    if not getattr(original, '_bun_native_v2', False):
        def step(*args, **kwargs):
            kwargs['require_stock_window'] = False
            return original(*args, **kwargs)
        step._bun_native_v2 = True
        env.step = step
    env.RULE_VERSION = RULE_VERSION
    # All legacy generated-worker install calls resolve to this explicit v2
    # installer in this process only. No stock-window wrapper is ever applied.
    proxy = ModuleType('scripts.bomb_threat_runtime')
    proxy.install = lambda: env
    proxy.RULE_VERSION = RULE_VERSION
    sys.modules['scripts.bomb_threat_runtime'] = proxy
    import scripts
    scripts.bomb_threat_runtime = proxy
    return env
