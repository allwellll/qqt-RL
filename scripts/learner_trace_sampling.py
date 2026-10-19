"""Host-only diagnostics: transfer the selected learner state, never its batch."""
import hashlib
import numpy as np


def observation_state_hash(observations,global_states,index):
    return hashlib.sha256(np.asarray(observations[index]).tobytes()
        +np.asarray(global_states[index]).tobytes()).hexdigest()


def verifier_audit_counts(context):
    """Inherited verifier audit is the nine fields before final reward-GAE pair."""
    names=('eligible_active','geometrically_reachable','safety_accepted','first_deviation_query',
           'reachable_rejected','safety_accepted_drop','safety_accepted_stock',
           'safety_accepted_return','safety_accepted_delivery')
    values=np.asarray(context)[...,-11:-2]
    assert values.shape[-1]==len(names)
    return dict(zip(names,values.reshape(-1,len(names)).sum(0).tolist()))


def executed_opponent_exposure(context):
    """Successor provenance92:95 = final source, sampled pool slot, valid."""
    data=np.asarray(context)[...,92:95]
    source=data[...,0].astype(np.int32);slot=data[...,1].astype(np.int32);valid=data[...,2]>0
    names=['responsive_rule','fixed_g0','fixed_stage1','league','weak_courier','weak_guard','carry_assist']
    counts={name:int(((source==i)&valid).sum()) for i,name in enumerate(names)}
    assert sum(counts.values())==int(valid.sum())
    league=valid&(source==3)
    categories=dict(current=int((league&(slot==0)).sum()),ring=int((league&(slot>=7)).sum()),
        fixed_league=int((league&(slot>=1)&(slot<7)).sum()))
    assert sum(categories.values())==counts['league']
    return dict(counts=counts,**categories,**{k:counts[k] for k in names if k!='league'},
        valid_decision_ticks=int(valid.sum()),fraction_of_valid={k:v/max(1,int(valid.sum())) for k,v in categories.items()},
        scope='Final action-producing source after weak-course and carry-assist overrides; valid means learner alive, matching training funnel.')
