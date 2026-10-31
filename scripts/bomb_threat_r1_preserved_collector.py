"""New slot03 namespace; other active modules stay immutable."""
from pathlib import Path
from types import ModuleType
from scripts import bomb_threat_r1_preserved_reward as reward
_impl=ModuleType('r1_preserved_collector');_impl.__file__=__file__
s=Path('scripts/bomb_threat_r1_progress_collector.py').read_text().replace('bomb_threat_r1_progress_reward as reward','bomb_threat_r1_preserved_reward as reward')
anchor="                diagnostics.update({name:sm[name].sum(axis=0) for name in sm if name.startswith('r1_') or name=='approach'})"
add=anchor+"\n                for field in ('r1_legacy_pay','r1_legacy_threat_pay','r1_legacy_chain_pay','r1_legacy_topology_pay','r1_approach_pay','r1_item_pay','r1_additions_pay','r1_total_shaping_pay'):\n                    if field in sm:diagnostics[field+'_by_seat']=j.stack([j.where(seat==k,sm[field],0.).sum() for k in (0,1)])"
add+="\n                diagnostics['r1_total_reward_by_seat']=j.stack([j.where(seat==k,reward,0.).sum() for k in (0,1)])\n                diagnostics['r1_absolute_reward_by_seat']=j.stack([j.where(seat==k,j.abs(reward),0.).sum() for k in (0,1)])"
assert s.count(anchor)==1;s=s.replace(anchor,add)
exec(compile(s,__file__,'exec'),_impl.__dict__)
for name in ('E','configure','bank_selection','banks','fresh','metadata','transition','collector','seat_allocation','inputs','choose','choose_with_diagnostics','route_teacher','combat_teacher','prediction'):
    globals()[name]=getattr(_impl,name)

# Full-state migration preserves every in-flight credit until natural reset.
import jax
import jax.numpy as j
from scripts import bomb_threat_r1_progress_reward as previous_reward
from scripts import bomb_threat_r1_progress_collector as previous_collector
_new_transition=_impl.transition

def transition(st,ax,actions,keys,coefficient):
    after,na,r,done,info,parts,m=_new_transition(st,ax,actions,keys,coefficient)
    meta=ax['combat_mix'];credit=meta['bomb_threat'];mask=credit['pre_revision_episode'];seats=meta['learner_seat'];rows=j.arange(len(seats))
    base={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr','r1_attribute_highwater')}
    old,oldpay,oldmetrics,birth=previous_reward.transition(st,after,info,dict(base,r1_next_terminal=na['terminal']),dict(meta,bomb_threat=credit['pre_revision']),seats)
    newcredit=na['combat_mix']['bomb_threat'];newcredit=dict(newcredit,pre_revision=old,pre_revision_episode=mask)
    # Select complete previous credit on migrated episodes; new accounting is
    # inactive there and resets normally with the rest of episode metadata.
    newcredit=original_select(mask,dict(credit,pre_revision=old),newcredit)
    r=r.at[rows,seats].add(j.where(mask,oldpay-m['pay'],0.))
    parts=parts.at[rows,seats,-1].set(_impl.original.select_player(r,seats))
    na=dict(na,combat_mix=dict(na['combat_mix'],bomb_threat=newcredit))
    for k in m:
        v=oldmetrics.get(k,j.zeros_like(m[k]));shape=(len(mask),)+(1,)*(m[k].ndim-1)
        m[k]=j.where(mask.reshape(shape),v,m[k])
    m['r1_pre_revision_ticks']=mask.astype(j.int32)
    return after,na,r,done,info,parts,m

original_select=_impl.E.select_tree
_impl._impl.transition=transition

_base_configure=configure

def configure(attack=False):
    _base_configure(attack)
    # Migration evaluates the unchanged predecessor for in-flight episodes;
    # initialize BOTH safety caches before pmap, never inside a traced mesh.
    previous_collector.configure(attack)
