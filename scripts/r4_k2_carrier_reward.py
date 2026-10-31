# 实验中 / EXPERIMENTAL: opt-in only; activation requires frozen gates.
"""R4 K2 opt-in conservative carrier interception credit.

Existing R2/R3 modules and running workers retain their frozen reward. This
module is installed only by an explicitly versioned K2 process. The +.1 forced
payment is an advance against a later custody delta, not an extra kill reward.
"""
from types import FunctionType
import jax.numpy as j

VERSION='r4-k2-carrier-conservative-v1'
FORCED_ADVANCE=.1
FORCED_WINDOW=20
_POTENTIAL=None


def graded_phi(home,enemy_carry,loose,progress_loose,progress_home,*,enabled=True):
    if not enabled:
        from scripts.bun_native_v2_custody_reward import graded_phi as legacy
        return legacy(home,enemy_carry,loose,progress_loose,progress_home)
    # The dropped/own-recovered package has a fixed -.1 potential. No progress
    # tick payments are added outside this single conservative state function.
    return j.where(home,0.,j.where(enemy_carry,-.40-.80*j.clip(progress_home,0.,1.),
        j.where(loose,-.10,-1.20)))


def delta(before,after,terminal=False):
    # Keep the existing first-terminal behavior: no failed-defense refund.
    return j.where(terminal,0.,j.asarray(after)-j.asarray(before))


def state(n):
    return dict(pending=j.zeros(n),age=j.full(n,-1,j.int32),
        source_batch=j.full(n,-1,j.int32),used_life=j.zeros(n,bool))


def advance(s,*,forced,source_batch,enemy_carrying_own,eligible,
            causal_kill,kill_batch,enemy_died,terminal):
    """Settle factual events supplied by the collector's provenance ledger.

    `forced` must already prove a newly installed own bomb/real chain removed
    every safe cell; `eligible` excludes spawn protection, cage farming and
    other invalid sources. A request/logit/geometric threat alone never pays.
    The source_batch identifies the physical causal batch through sugar death.
    A caller cannot manufacture these facts from a placement count.
    """
    start=forced&enemy_carrying_own&eligible&~s['used_life']&~terminal&(source_batch>=0)
    payment=j.where(start,FORCED_ADVANCE,0.)
    pending=j.where(start,FORCED_ADVANCE,s['pending'])
    age=j.where(start,0,j.where(s['age']>=0,s['age']+1,-1))
    source=j.where(start,source_batch,s['source_batch'])
    open_=pending>0
    matched=open_&causal_kill&eligible&(source==kill_batch)&(age<=FORCED_WINDOW)
    cancel=open_&~matched&(terminal|enemy_died|~enemy_carrying_own|(age>=FORCED_WINDOW))
    closed=matched|cancel
    # Matched credit is deducted from the same transition's custody gain.
    # Unmatched credit is clawed back. Both close the cash advance exactly.
    settlement=j.where(closed,pending,0.)
    payment=payment-settlement
    result=dict(pending=j.where(closed,0.,pending),age=j.where(closed,-1,age),
        source_batch=j.where(closed,-1,source),used_life=j.where(enemy_died,False,s['used_life']|start))
    return result,payment,dict(advance=j.where(start,FORCED_ADVANCE,0.),
        settled_advance=j.where(matched,pending,0.),clawback=j.where(cancel,pending,0.),
        pending=result['pending'],same_batch_kill=matched,forced_once=start)


def potential(states,cache):
    # Reuse the verified sparse distance fields exactly. Only the state->Phi
    # scalar map changes. Installation never reaches into another process.
    return _build_potential()(states,cache)


def _build_potential():
    global _POTENTIAL
    if _POTENTIAL is None:
        # Freeze the original function before installing into reference. The
        # legacy fast builder inspects reference.potential on first import.
        from scripts.bun_native_v2_custody_fast2 import potential as base
        _POTENTIAL=FunctionType(base.__code__,dict(base.__globals__,graded_phi=graded_phi))
    return _POTENTIAL


def install():
    from scripts import bun_native_v2_custody_reward as reference
    _build_potential()
    reference.potential=potential
