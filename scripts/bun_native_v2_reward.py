"""Versioned v2 route/antifarm wiring. Applied only in new R2 processes.

Kill discount is accrued at the factual kill tick, before delayed settlement;
legacy survival/clawback bookkeeping and terminal delivery amounts are intact.
"""
import inspect
from types import ModuleType
import jax.numpy as j

_INSTALLED=None


def victim_factors(states):
    from scripts.carrier_mechanism_engine import env
    # [batch, actor] factor for that actor's opposite victim.
    p=j.floor(states.pos[:,::-1]).astype(j.int32)
    base=j.asarray(env._BUN_BASES)[None]
    distance=j.maximum(base-p,0)+j.maximum(p-(base+2),0)
    farm=(states.bun_carried[:,::-1]>=0)|(distance.sum(-1)<=3)
    return j.where(farm,.3,1.)


def install(antifarm=False):
    global _INSTALLED
    if _INSTALLED is not None:
        if _INSTALLED!=bool(antifarm):raise RuntimeError('v2 arm cannot switch reward in-process')
        return
    from scripts import je09_route_pbrs as pbrs, je10_ops as ops
    from scripts import prestock_training_state as prep
    # Compile private phase/preparation functions with just the declared gate
    # change. Old module source and old process behavior remain unchanged.
    scope=dict(pbrs.__dict__)
    code=inspect.getsource(pbrs._phase_values_before_recovery_alignment)
    old='stock = empty & window & (stock_count > 0) & states.alive'
    assert code.count(old)==1
    code=code.replace(old,'stock = empty & states.alive & (states.trapped == 0)')
    # Unified empty-hand goal is the enemy stock cage, irrespective of enemy
    # survival or temporarily exhausted stock; carrying either origin goes home.
    code=code.replace('jnp.where(drop, PHASE_DROP, jnp.where(stock, PHASE_STOCK, PHASE_NONE))',
                      'jnp.where(stock, PHASE_STOCK, PHASE_NONE)')
    exec(code,scope)
    raw=scope['_phase_values_before_recovery_alignment']
    def phases(states,route,drop):
        from scripts.route_objective_consistency import align_recovery
        values=raw(states,route,drop)
        recovered=align_recovery(states,route,drop,values)
        # Recovery alignment may supply own-origin carry-home values, but may
        # not replace the declared empty-hand enemy-cage objective.
        carrying=states.bun_carried>=0
        return tuple(j.where(carrying,new,old) for new,old in zip(recovered,values))
    pbrs.phase_values=phases
    # configure() has already installed the former threat/window outer gate.
    # Future explicit install calls in this v2 worker retain the new target.
    ops.safe_phase_values=phases
    # The inherited .75 legal-stock-acquisition event also had its own native
    # death/trap eligibility gate. Remove it without falsifying event flags.
    from scripts import je09_stage_f_reward as transport_reward,je09_stage_h_reward as carry_reward
    scope=dict(transport_reward.__dict__)
    code=inspect.getsource(transport_reward.combined_reward)
    old="eligible=info['steal_window_trap']|info['steal_window_dead']"
    assert code.count(old)==1
    scope['transition_allowed']=lambda before,after:j.ones_like(before.alive)
    exec(code.replace(old,"eligible=info['steal']"),scope)
    carry_reward.parent_reward=scope['combined_reward']
    scope=dict(prep.__dict__)
    code=inspect.getsource(prep.update)
    code=code.replace('(states.scene!=0)[:,None]&~aux[\'terminal\'][:,None]&states.alive',"~aux['terminal'][:,None]&states.alive")
    code=code.replace('&~attributed_loose(states,aux)','')
    assert code.count('&(states.bun_carried<0)&window')==1
    exec(code.replace('&(states.bun_carried<0)&window','&(states.bun_carried<0)'),scope)
    prep.update=scope['update']
    # Native-custody item reward must recognize a real legal v2 acquisition.
    scope=dict(ops.__dict__)
    code=inspect.getsource(ops.update_custody)
    assert code.count('acquired=start&native_window&')==1
    exec(code.replace('acquired=start&native_window&','acquired=start&'),scope)
    from scripts import je09_stage_h5c_dagger as dagger
    dagger.update=scope['update_custody']
    _INSTALLED=bool(antifarm)
    if not antifarm:
        return
    from scripts import je04_bun_ops as legacy,je10_control_credit as credit
    fn=legacy.retention_reward
    scope=dict(fn.__globals__)
    code=inspect.getsource(fn)
    assert code.count("16.*info['surviving_kill']")==1
    exec(code.replace("16.*info['surviving_kill']","16.*info['surviving_kill']*info['v2_kill_factor']"),scope)
    legacy.retention_reward=scope['retention_reward']
    # Residual controlled->dead kill payment is also kill shaping. Control
    # ENTRY_REWARD itself is not a kill and is intentionally unchanged.
    scope=dict(credit.__dict__)
    code=inspect.getsource(credit.transition)
    assert code.count('extra=residual_hot.sum(1)*RESIDUAL_REWARD')==1
    exec(code.replace('extra=residual_hot.sum(1)*RESIDUAL_REWARD',"extra=(residual_hot*e['v2_victim_factor'][...,None]).sum(1)*RESIDUAL_REWARD"),scope)
    credit.transition=scope['transition']
    original=ops.combined_reward
    def combined(states,after,info,aux,settled,coefficient=1.,bridge_config=None):
        return original(states,after,dict(info,v2_kill_factor=victim_factors(states)),aux,settled,coefficient,bridge_config)
    ops.combined_reward=combined
    native=credit.native_events
    def events(before,after,info,terminal):
        return dict(native(before,after,info,terminal),v2_victim_factor=victim_factors(before)[:,::-1])
    credit.native_events=events
