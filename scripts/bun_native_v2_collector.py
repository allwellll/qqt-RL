"""R2 collector using R1 sampling/optimizer ABI, explicit native-v2 reward."""
from pathlib import Path
from types import ModuleType
from scripts import bun_native_v2_reward as rules

_module=ModuleType('native_v2_r1_collector')
_module.__file__=__file__
source=Path('scripts/bomb_threat_r1_progress_collector.py').read_text()
source=source.replace('ATTACK=False','ATTACK=False\nANTIFARM=False')
source=source.replace('def configure(attack=False):','def configure(attack=False,antifarm=False):')
source=source.replace('    global ATTACK\n    ATTACK=bool(attack)',
    '    global ATTACK,ANTIFARM\n    ATTACK=bool(attack)\n    ANTIFARM=bool(antifarm)\n    from scripts.bun_native_v2_reward import install\n    install(ANTIFARM)')
anchor="    r=r.at[rows,seats].add(extra);parts=parts.at[rows,seats,-1].set(original.select_player(r,seats))"
assert source.count(anchor)==1
source=source.replace(anchor,"""    steal=info['steal'][rows,1-seats]&~base['terminal']
    failure=j.where(ANTIFARM&steal,-.05,0.)
    r=r.at[rows,seats].add(extra+failure);parts=parts.at[rows,seats,-1].set(original.select_player(r,seats))
    metrics.update(v2_enemy_stock_pickups=steal.astype(j.int32),v2_defense_failure_pay=failure,
        v2_discounted_kills=(info['surviving_kill'][rows,seats]&(rules.victim_factors(st)[rows,seats]<1)&~base['terminal']).astype(j.int32))""")
# Private module needs the explicit helper in its own function global scope.
_module.__dict__['rules']=rules
anchor="_repl='''            diagnostics.update(policy_diag)"
# Extend the replacement payload, not the quoted source-anchor string.
assert source.count(anchor)==1
source=source.replace(anchor,anchor+"\n            diagnostics.update({name:sm[name].sum(axis=0) for name in sm if name.startswith('v2_')})")
exec(compile(source,__file__,'exec'),_module.__dict__)
for name in ('E','configure','bank_selection','banks','fresh','metadata','transition','collector','seat_allocation',
             'inputs','choose','choose_with_diagnostics','route_teacher','combat_teacher','prediction'):
    globals()[name]=getattr(_module,name)
