"""Pure asynchronous queue policy; counters never compare inherited PPO updates."""
def order(r):
    return r.get('chunk',-1), r.get('update',-1)

def pending_latest(rows):
    latest={}
    keep=set()
    for r in rows:
        if r.get('selection_eligible') is False:continue
        if r.get('split')=='holdout' or r['name'].startswith('parent_'):
            keep.add(r['name'])
        elif 'slot' in r:
            s=r['slot']
            if s not in latest or order(r)>order(latest[s]):latest[s]=r
    return keep | {r['name'] for r in latest.values()}

def priority(r):
    return (0 if r.get('priority')=='best_candidate' else 1 if r['name'].startswith('parent_') else 2 if r.get('split')=='holdout' else 3, -r.get('chunk',-1),r['name'])

def fresh_cohort(generation,current_generations):
    return bool(current_generations) and all(0<=g-generation<=2 for g in current_generations.values())

def lag_report(rows,scored_names):
    result={}
    for slot in sorted({r['slot'] for r in rows if 'slot' in r and r.get('split')=='dev' and r.get('selection_eligible') is not False}):
        saves=[r for r in rows if r.get('slot')==slot and r.get('split')=='dev' and r.get('selection_eligible') is not False]
        saved=max(saves,key=order);scores=[r for r in saves if r['name'] in scored_names]
        scored=max(scores,key=order) if scores else None
        result[str(slot)]=dict(latest_saved=saved['name'],latest_scored=scored['name'] if scored else None,saved_chunk=saved['chunk'],scored_chunk=scored['chunk'] if scored else None,lag_chunks=saved['chunk']-scored['chunk'] if scored else None)
    return result
