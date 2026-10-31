"""Slot03 only: frozen A3SR credit plus a separate capped R1 addition ledger."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import jax
import jax.numpy as j
from scripts import bomb_threat_reward as base
from scripts import bomb_threat_r1_item_progress as I
from scripts.bomb_threat_iter2_recipe import settlement
CAP=.10
legacy_settle=settlement(5.,.6)
legacy=ModuleType('r1_preserved_legacy_events')
exec(compile(Path('scripts/bomb_threat_seat5050_reward.py').read_text(),__file__,'exec'),legacy.__dict__)
legacy.settle=legacy_settle


def bounded_item_step(s,before,after,grew,*,budget,**flags):
    n,p,m=I.step(s,before,after,grew,**flags)
    invalid=flags['self_death']|flags['terminal']|flags['control_lost']
    admitted=j.where(invalid,0.,j.minimum(m['candidate'],j.maximum(0.,budget)))
    n=dict(n,pending=n['pending'].at[:,-1].set(admitted))
    m=dict(m,candidate=admitted,uncapped_candidate=m['candidate'],pending=n['pending'].sum(-1))
    # Current invalid candidate was never reserved; only previous queue voids.
    m['void']=j.where(invalid,s['pending'].sum(-1),0.)
    n['void']=s['void']+m['void']
    return n,p,m


def settle(e,s):
    rows=j.arange(len(s['gross']));valid=e['active']&~e['bad']&~e['terminal']
    ids=s['chain_ids'];count=s['chain_count'];candidates=e['old_ids']
    # The exact legacy chain already pays the selected earliest old source.
    candidate_valid=(candidates>=0)&(candidates!=e['old_id'][:,None])&~(candidates[:,:,None]==ids[:,None,:]).any(-1)
    selected=j.zeros(len(rows),j.int32)
    for _ in range(3):
        oldest=j.min(j.where(candidate_valid,candidates,1<<28),-1)
        ok=(oldest<(1<<28))&(count<3)&e['placed']&e['safe']&valid
        ids=ids.at[rows,j.minimum(count,2)].set(j.where(ok,oldest,ids[rows,j.minimum(count,2)]))
        count+=ok.astype(j.int32);selected+=ok.astype(j.int32)
        candidate_valid&=candidates!=oldest[:,None]
    multi=e['placed']&e['safe']&e['multibatch']&(e['batch_id']>s['multi_last'])&(s['multi_count']<6)&valid
    phi_after=j.where(e['danger_after'],e['phi_before'],e['phi_after'])
    approach=j.where(valid&e['phase_before']&e['phase_after'],.995*phi_after-e['phi_before'],0.)
    raw=j.stack([approach,.015*multi,.01*selected],-1)*s['r1_scale'][:,None]
    positive=j.maximum(raw,0.);room=j.maximum(0.,CAP-s['new_reserved'])
    factor=j.minimum(1.,room/j.maximum(positive.sum(-1),1e-12))
    admitted=(positive*factor[:,None]+j.minimum(raw,0.))*valid[:,None]
    components=s['new_pending_components']+admitted
    components=j.where(valid[:,None],components,0.)
    paid=j.where((e['settled']&valid)[:,None],components,0.)
    kept=j.where(e['settled'][:,None],0.,components)
    reserve=s['new_reserved']+j.maximum(admitted,0.).sum(-1)
    out=dict(s,new_reserved=reserve,new_pending_components=kept,pending=kept.sum(-1),gross=s['gross']+j.maximum(paid.sum(-1),0.),
        chain_ids=ids,chain_count=count,multi_last=j.where(multi,e['batch_id'],s['multi_last']),multi_count=s['multi_count']+multi.astype(j.int32),
        phi=j.where(e['phase_after'],phi_after,0.),phi_valid=e['phase_after']&valid)
    zero=j.zeros(len(rows),j.int32)
    m=dict(candidate=(admitted!=0).any(-1),settled_count=(paid.sum(-1)!=0).astype(j.int32),pay=paid.sum(-1),
        threat_events=zero,chain_events=selected,brick_events=zero,brick_paid_events=zero,threat_pending=kept.sum(-1),
        approach=approach,r1_raw=raw.sum(-1),r1_multi=multi.astype(j.int32),r1_pickups=zero,
        r1_void=j.where(~valid,s['pending'],0.),r1_pending=kept.sum(-1),r1_scale=s['r1_scale'],
        r1_approach_pay=paid[:,0],r1_multi_pay=paid[:,1],r1_extra_chain_pay=paid[:,2],r1_new_reserved=reserve)
    return out,paid.sum(-1),m


# Reuse frozen event qualification and physical accounting in an isolated module.
_impl=ModuleType('r1_preserved_new_events')
_src=Path('scripts/bomb_threat_r1_progress_reward.py').read_text()
a="        topo_age=j.full(n,1000,j.int32)"
b="        legacy=base.state(n),legacy_pending_components=j.zeros((n,3)),new_reserved=j.zeros(n),new_pending_components=j.zeros((n,3)),\n"+a
assert _src.count(a)==1;_src=_src.replace(a,b)
_src=_src.replace("items.step(s['items'],gb,ga,actual,active=", "items.step(s['items'],gb,ga,actual,budget=j.maximum(0.,CAP-n['new_reserved']),active=")
_src=_src.replace("    n['items']=ip", "    n['items']=ip; n['new_reserved']=n['new_reserved']+im['candidate']")
exec(compile(_src,__file__,'exec'),_impl.__dict__)
_impl.CAP=CAP;_impl.settle=settle
_impl.items=SimpleNamespace(INITIAL=I.INITIAL,fraction=I.fraction,state=I.state,step=bounded_item_step)
native_safety=_impl.native_safety
from scripts import bomb_threat_r1_progress_reward as previous_reward

def state(n):
    return dict(_impl.state(n),pre_revision=previous_reward.state(n),pre_revision_episode=j.zeros(n,bool))


def transition(before,after,info,aux,metadata,seats):
    s=metadata['bomb_threat']
    n,nr,m,birth=_impl.transition(before,after,info,aux,metadata,seats)
    old,lr,lm,oldbirth=legacy.transition(before,after,info,aux,dict(metadata,bomb_threat=s['legacy']),seats)
    # Allocate the actual capped legacy bundle by its pending component mix.
    components=s['legacy_pending_components']+j.stack([.10*lm['threat_events'],.05*lm['chain_events'],.20*lm['brick_events']],-1)
    weight=j.where(components.sum(-1,keepdims=True)>0,components/j.maximum(components.sum(-1,keepdims=True),1e-12),0.)
    legacy_paid=weight*lr[:,None]
    components=weight*old['pending'][:,None]
    n.update(legacy=old,legacy_pending_components=components)
    m.update(r1_legacy_pay=lr,r1_legacy_threat_pay=legacy_paid[:,0],r1_legacy_chain_pay=legacy_paid[:,1],r1_legacy_topology_pay=legacy_paid[:,2],
        r1_additions_pay=nr,r1_total_shaping_pay=lr+nr,r1_attack_pay=lr+m['r1_attack_pay'],
        pay=lr+nr,threat_events=lm['threat_events'],brick_events=lm['brick_events'],brick_paid_events=lm['brick_paid_events'],
        chain_events=lm['chain_events']+m['chain_events'])
    return n,lr+nr,m,birth
