"""CC attack reward variant v2 (L2_ATK). Process-local patch of the LIVE attack credits in r3_items_reward.

v1 targeted bomb_threat_r1_progress_reward.settle, whose threat/selected/multi terms the live chain
(r3_items_collector) already replaces with raw=.04*topology+approach, so v1 had no effect. v2 rescales the
r3 one-shot, safety-settled, voidable attack slots (same mechanisms, only coefficients and caps):
  THREAT    .02 -> .05  explosion reduces enemy escape and covers a novel near cell (slot cap .45 unchanged)
  PRESSURE  .01 -> .02  repeated cover within 24 ticks from a new batch, <=10 per episode (slot cap .10 -> .20)
  CHAIN     .015*k -> .03*k  cross-batch chain, k<=4, void on bad/empty/home-stack/same batch (slot cap .30 -> .40)
  attack joint budget .45 -> .60 (includes inherited attack payment)
Item/road group (ATTR, ITEM_PATH, GLUE, BANANA, ROAD; joint .60) unchanged: genuine pickups are already paid
via r1_item_pay*1.5, brick topology via ROAD. No bomb-count, tick-stock, idle or fire-lane term.
usage: install_on_import() before the worker starts. The patch is applied when the worker itself imports
scripts.r3_items_reward (inside setup(), after XLA_FLAGS are set). Importing it earlier initialises the JAX
backend before the deterministic XLA flags exist (11:50 root cause of the L2_ATK restore digest mismatch).
"""
import inspect

VERSION = 'cc-attack-reward-v2'
THREAT, PRESSURE, CHAIN = .05, .02, .03
CAPS = (.36, .60, .36, .20, .20, .40, .20, .45)  # ATTR ITEM_PATH GLUE BANANA ROAD | CHAIN PRESSURE THREAT (order of range(8))
ATTACK_BUDGET = .60
SOURCES = {}  # patched source text (inspect.getsource cannot see exec'd code)


def _patch(fn, pairs, scope):
    src = inspect.getsource(fn)
    for old, new in pairs:
        assert src.count(old) == 1, (fn.__name__, old)
        src = src.replace(old, new)
    exec(compile(src, f'cc_attack_reward_v2.{fn.__name__}', 'exec'), scope)
    SOURCES[fn.__name__] = src
    return scope[fn.__name__]


def install(R=None):
    if R is None:
        from scripts import r3_items_reward as R
    if getattr(R, '_cc_attack_v2', False):
        return R
    old_caps = [round(float(x), 6) for x in R.CAPS]
    assert old_caps == [.36, .60, .36, .20, .20, .30, .10, .45], old_caps
    scope = R.__dict__
    R.chain_reward = _patch(R.chain_reward, [('.015*j.minimum(j.maximum(k,0),4)', f'{CHAIN}*j.minimum(j.maximum(k,0),4)')], scope)
    R.transition = _patch(R.transition, [
        ("candidate=candidate.at[:,PRESSURE].set(.01*pressure)", f"candidate=candidate.at[:,PRESSURE].set({PRESSURE}*pressure)"),
        ("candidate=candidate.at[:,THREAT].set(.02*(explosion", f"candidate=candidate.at[:,THREAT].set({THREAT}*(explosion")], scope)
    R.capped_payment = _patch(R.capped_payment, [("attack_budget=j.maximum(0.,.45-gross", f"attack_budget=j.maximum(0.,{ATTACK_BUDGET}-gross")], scope)
    import jax.numpy as j
    R.CAPS = j.array(CAPS)
    R._cc_attack_v2 = True
    return R


def install_on_import(name='scripts.r3_items_reward'):
    """Patch the module right after its first real import, without importing it (no early JAX backend init)."""
    import importlib.abc, sys
    if name in sys.modules:
        return install(sys.modules[name])
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            import importlib.util
            spec = importlib.util.find_spec(fullname)
            loader = spec.loader; original = loader.exec_module
            def exec_module(module):
                original(module); install(module)
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, Finder())
