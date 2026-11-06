"""CC fountain/chain attack reward variant v1 (L2_FTN). Process-local patch of the LIVE r3_items_reward attack slots,
built on cc_attack_reward_v2 (same mechanisms: one-shot, factual explosions, 30-tick quarantine, void on bad).
  CHAIN     .03*min(k,4) -> .04*min(k,6); x1.5 when the chained fire covers a cell within 3 of the live enemy
            or reduces the enemy's native safe exits. Void on bad/empty/same batch (unchanged). Slot cap .40 -> .60
  PRESSURE  (fountain) flat .02 x<=10 -> streak: the n-th effective own batch in a streak pays .01*min(n,5),
            <=20 effective batches per episode, slot cap .20 -> .40. Effective batch = factual own explosion from a new
            batch that covers a novel cell within 3 of the live enemy. Gap between consecutive effective batches >24 ticks
            restarts the streak at 1; bad (own death/trapped) resets it to 0 and voids the pending queue (settle).
  THREAT    .05 (unchanged)
  attack joint budget .60 -> .90 (includes inherited attack payment). Item/road group unchanged (.60).
  All attack credits additionally require the enemy alive before the explosion (no post-death credit).
State: no new aux key (restore keeps the pytree). pressure_count packs total + 64*streak; last_explosion(_batch) now
track the last EFFECTIVE batch (they are only read by this pressure term).
No bomb-count, tick-stock, idle or fire-lane term.
usage: install_on_import() before the worker starts (patch applied when the worker imports scripts.r3_items_reward
inside setup(), after XLA_FLAGS; importing earlier breaks deterministic restore, 11:50 root cause).
"""
import inspect

VERSION = 'cc-fountain-reward-v1'
THREAT, CHAIN, CHAIN_K, CHAIN_NEAR = .05, .04, 6, 1.5
FOUNTAIN, FOUNTAIN_N, FOUNTAIN_MAX, FOUNTAIN_GAP = .01, 5, 20, 24
CAPS = (.36, .60, .36, .20, .20, .60, .40, .45)  # ATTR ITEM_PATH GLUE BANANA ROAD | CHAIN PRESSURE THREAT
ATTACK_BUDGET = .90
PACK = 64  # pressure_count = total + PACK*streak (total <= FOUNTAIN_MAX < PACK)
SOURCES = {}


def fountain_step(count, last_t, last_batch, t, sourcebatch, effective, bad):
    """Pure fountain/pressure step. Returns (pay, new_count, new_last_t, new_last_batch, streak, hit)."""
    import jax.numpy as j
    total = count % PACK; streak = count // PACK
    hit = effective & (sourcebatch != last_batch) & (total < FOUNTAIN_MAX)
    n = j.where(t - last_t <= FOUNTAIN_GAP, streak + 1, 1)
    pay = j.where(hit, FOUNTAIN * j.minimum(n, FOUNTAIN_N), 0.)
    streak = j.where(bad, 0, j.where(hit, j.minimum(n, 31), streak))
    total = total + hit.astype(total.dtype)
    return (pay, total + PACK * streak, j.where(hit, t, last_t), j.where(hit, sourcebatch, last_batch), streak, hit)


def chain_reward(k, *, same_batch=False, bad=False, empty=False, home_stack=False, near=False):
    import jax.numpy as j
    base = CHAIN * j.minimum(j.maximum(k, 0), CHAIN_K) * j.where(near, CHAIN_NEAR, 1.)
    return j.where(bad | empty | home_stack | same_batch, 0., base)


TRANSITION = [
    # no credit for explosions once the enemy is already dead (killing blow keeps before.alive)
    ("explosion=source_valid.any((-2,-1))&useful&active",
     "explosion=source_valid.any((-2,-1))&useful&active&before.alive[rows,enemy]"),
    # CHAIN is computed after the exit/novel measurement (needs reduced/near)
    ("    candidate=candidate.at[:,CHAIN].set(chain_reward(k,bad=bad,empty=~explosion))\n", ""),
    ("    pressure=explosion&(cover&near).any((-2,-1))&(before.t-s['last_explosion']<=24)&(sourcebatch!=s['last_explosion_batch'])&(s['pressure_count']<10)\n"
     "    candidate=candidate.at[:,PRESSURE].set(.01*pressure)\n", ""),
    ("    candidate=candidate.at[:,THREAT].set(.02*(explosion&reduced&(novel&near).any((-2,-1))))\n",
     "    candidate=candidate.at[:,THREAT].set(THREAT_COEF*(explosion&reduced&(novel&near).any((-2,-1))))\n"
     "    chain_near=(cover&near).any((-2,-1))|reduced\n"
     "    candidate=candidate.at[:,CHAIN].set(chain_reward(k,bad=bad,empty=~explosion,near=chain_near))\n"
     "    effective=explosion&(novel&near).any((-2,-1))\n"
     "    fpay,fcount,flast,fbatch,fstreak,pressure=fountain_step(s['pressure_count'],s['last_explosion'],s['last_explosion_batch'],before.t,sourcebatch,effective,bad)\n"
     "    candidate=candidate.at[:,PRESSURE].set(fpay)\n"),
    ("s=dict(s,last_explosion=j.where(explosion,before.t,s['last_explosion']),last_explosion_batch=j.where(explosion,sourcebatch,s['last_explosion_batch']),pressure_count=s['pressure_count']+pressure.astype(j.int32))",
     "s=dict(s,last_explosion=flast,last_explosion_batch=fbatch,pressure_count=fcount)"),
    ("r3_pressure=pressure,", "r3_pressure=pressure,r3_fountain_streak=fstreak,r3_chain_near=chain_near&(k>0)&explosion,"),
]


def _patch(fn, pairs, scope):
    src = inspect.getsource(fn)
    for old, new in pairs:
        assert src.count(old) == 1, (fn.__name__, old)
        src = src.replace(old, new)
    exec(compile(src, f'cc_fountain_reward_v1.{fn.__name__}', 'exec'), scope)
    SOURCES[fn.__name__] = src
    return scope[fn.__name__]


def install(R=None):
    if R is None:
        from scripts import r3_items_reward as R
    if getattr(R, '_cc_fountain_v1', False):
        return R
    assert not getattr(R, '_cc_attack_v2', False), 'cc_attack_reward_v2 already installed'
    old_caps = [round(float(x), 6) for x in R.CAPS]
    assert old_caps == [.36, .60, .36, .20, .20, .30, .10, .45], old_caps
    scope = R.__dict__
    scope.update(chain_reward=chain_reward, fountain_step=fountain_step, THREAT_COEF=THREAT)
    R.transition = _patch(R.transition, TRANSITION, scope)
    R.capped_payment = _patch(R.capped_payment, [("attack_budget=j.maximum(0.,.45-gross", f"attack_budget=j.maximum(0.,{ATTACK_BUDGET}-gross")], scope)
    import jax.numpy as j
    R.CAPS = j.array(CAPS)
    R._cc_fountain_v1 = True
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
