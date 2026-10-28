"""Pure, versioned selection/exploration rules; never consumes holdout for tuning."""
import copy
import numpy as np

def score(hunter):
    return sum(w*hunter[k]['safe_win'] for k,w in [('hard',3),('normal',2),('easy',1)])-2*sum(hunter[k]['death_episode'] for k in ('hard','normal','easy'))

def eligible(candidate,parent):
    return parent>0 and candidate>=.9*parent

def explore(recipe,seed):
    r=copy.deepcopy(recipe);rng=np.random.default_rng(seed)
    r['lr']*=float(rng.choice([.8,1.25]));r['entropy_coefficient']*=float(rng.choice([.8,1.25]))
    r['combat_shaping']=float(rng.choice([0,.1,.2,.3]));r['curriculum_fraction']=float(rng.choice([.25,.4,.6]))
    w=np.array(r['opponent_weights'],float)
    # Transfer mass, never create a negative weight or alter sum. Two moves
    # explicitly bias hard/current by +/- .1 with available-mass clipping.
    for target in [2,3]:
        donor=0 if target==2 else 1
        delta=float(rng.choice([-.1,.1]));delta=min(delta,w[donor]) if delta>=0 else max(delta,-w[target])
        w[target]+=delta;w[donor]-=delta
    r['opponent_weights']=w.tolist();return r

def replacements(cohort):
    """Exactly bottom/top quarters of all8 observed slots; no missing-as-zero."""
    if len(cohort)!=8:return []
    ordered=sorted(cohort,key=lambda r:(r['score'],r['slot']))
    donors=[r for r in reversed(ordered[-2:]) if r['eligible']]
    if not donors:return []
    result=[]
    for i,loser in enumerate(ordered[:2]):
        donor=donors[i%len(donors)]
        if donor['slot']!=loser['slot'] and donor['score']-loser['score']>=6:
            result.append(dict(loser=loser['slot'],donor=donor['slot'],checkpoint=donor['checkpoint'],score_gap=donor['score']-loser['score']))
    return result
