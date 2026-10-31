"""Private custody-v2 collector; keeps all historical source seals intact."""
from pathlib import Path
from types import ModuleType
from scripts import bun_native_v2_custody_reward as rules

# R1's approach credit must not pay for chasing a carrier/home defender, or
# release approach credit accrued before the opponent entered the cage. All
# defensive interception shaping is represented by the custody change.
from scripts import bomb_threat_r1_progress_reward as _r1_reward
_reward_scope=dict(_r1_reward.__dict__)
import inspect
_reward_source=inspect.getsource(_r1_reward.transition)
_phase="phase=lambda st:st.alive[rows,seats]"
assert _reward_source.count(_phase)==1
_reward_source=_reward_source.replace(_phase,"phase=lambda st:(rules.victim_factors(st)[rows,seats]>0)&st.alive[rows,seats]")
_anchor="    e=dict(e,old_ids=old_ids,"
assert _reward_source.count(_anchor)==1
_reward_source=_reward_source.replace(_anchor,"    e=dict(e,active=e['active']&(rules.victim_factors(before)[rows,seats]>0),old_ids=old_ids,")
_reward_scope['rules']=rules
exec(compile(_reward_source,__file__,'exec'),_reward_scope)

source=Path('scripts/bomb_threat_r1_progress_collector.py').read_text()
source=source.replace('def configure(attack=False):','def configure(attack=False,antifarm=True):')
source=source.replace('    ATTACK=bool(attack)','    ATTACK=bool(attack)\n    rules.install()')
anchor="    r=r.at[rows,seats].add(extra);parts=parts.at[rows,seats,-1].set(original.select_player(r,seats))"
assert source.count(anchor)==1
source=source.replace(anchor,"""    # Custody is measured at the real learner transition, before any reset.
    metrics.update({('v2_'+name):value[rows,seats] for name,value in info.items() if name.startswith('custody_')})
    metrics.update(v2_enemy_stock_pickups=(info['steal'][rows,1-seats]&~base['terminal']).astype(j.int32),
        v2_kills_of_carrier=(info['surviving_kill'][rows,seats]&(st.bun_carried[rows,1-seats]>=0)&~base['terminal']).astype(j.int32))
"""+anchor)
anchor="_repl='''            diagnostics.update(policy_diag)"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+"\n            diagnostics.update({name:sm[name].sum(axis=0) for name in sm if name.startswith('v2_')})")
_module=ModuleType('custody_v2_r1_collector');_module.__file__=__file__;_module.rules=rules
exec(compile(source,__file__,'exec'),_module.__dict__)
_module.reward=ModuleType('custody_v2_attack_reward')
_module.reward.__dict__.update(_r1_reward.__dict__)
_module.reward.transition=_reward_scope['transition']
for name in ('E','configure','bank_selection','banks','fresh','metadata','transition','collector','seat_allocation',
             'inputs','choose','choose_with_diagnostics','route_teacher','combat_teacher','prediction'):
    globals()[name]=getattr(_module,name)


def banks(*args,**kwargs):
    states,aux=_module.banks(*args,**kwargs)
    return states,tuple(dict(a,custody_v2=rules.cache_state(len(s.t))) for s,a in zip(states,aux))
