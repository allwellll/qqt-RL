"""Recompute the carry-stage decision funnel from complete raw JS episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/r4_combat_20261031/interceptor')


def read(p):
    return json.loads(p.read_text())


def stats(xs):
    if not xs:
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(n=len(xs), mean=statistics.mean(xs), median=statistics.median(xs), minimum=min(xs), maximum=max(xs))


def episodes(folder, count=64):
    es=[]
    for i in range(0,count,8):
        f=folder/f'shard_{i:03d}.json.outcome.json';receipt=read(Path(str(f).replace('.outcome.json','.ACCEPTED.json')))
        assert all(sha(Path(p))==h for p,h in receipt['files'].items())
        es+=read(f)['episodes']
    assert len(es)==count and len({(e['seed'],e['seat']) for e in es})==count
    return es


def pressure(es):
    def group(games):
        possessions=[p for e in games for p in e['r4_pressure']['possessions'][e['seat']] if p['origin']==1-e['seat']]
        delivered=sum(p['delivered'] for p in possessions)
        return dict(games=len(games),acquisitions=len(possessions),deliveries=delivered,
            success=delivered/len(possessions) if possessions else None,
            model_deaths=sum(e['deaths'][e['seat']] for e in games),
            bot_self_deaths=sum(e['self_deaths'][1-e['seat']] for e in games),
            model_self_deaths=sum(e['self_deaths'][e['seat']] for e in games),
            bot_bombs=sum(e['placed'][1-e['seat']] for e in games),
            bot_bombs_per_game=sum(e['placed'][1-e['seat']] for e in games)/len(games))
    return dict(all=group(es),**{f'P{p}':group([e for e in es if e['seat']==p]) for p in (0,1)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default=str(BASE/'diagnostic_v1'));args=parser.parse_args()
    folder=Path(args.folder);es=episodes(folder);rows=[r for e in es for r in e['interceptor_diagnostic']['rows']]
    active=[r for r in rows if r['bot_alive'] and not r['bot_trapped']]
    planned=[r for r in rows if r.get('planner',{}).get('common_cells') is not None]
    proposed=[r for r in rows if r.get('bomb',{}).get('proposed')]
    no_best=[r for r in planned if not r['planner'].get('best')]
    places=[p for e in es for p in e['interceptor_diagnostic']['placements']]
    by_reason={}
    for reason in sorted({p['reason'] for p in places}):
        ps=[p for p in places if p['reason']==reason]
        by_reason[reason]=dict(placements=len(ps),target_defined=sum(p['impact'] is not None for p in ps),
            actual_target_arrivals=sum(p['actual_arrival_tick'] is not None for p in ps),
            arrival_error_ticks=stats([p['arrival_error_ticks'] for p in ps if 'arrival_error_ticks' in p]),
            arrival_censor=dict(Counter(p['arrival_censor'] for p in ps)),
            observed_explosions=sum(p['explosion_tick'] is not None for p in ps),
            carrier_distance_at_explosion=stats([p['carrier_distance_at_explosion'] for p in ps if 'carrier_distance_at_explosion' in p]),
            same_possession_explosions=sum(p.get('carrier_still_same_possession',False) for p in ps),
            same_possession_distance_at_explosion=stats([p['carrier_distance_at_explosion'] for p in ps if p.get('carrier_still_same_possession')]),
            explosion_ray_hits_carrier_cell=sum(p.get('ray_hits_carrier_cell',False) for p in ps),
            note='ray cell coincidence is not a causal kill; post-move actual position and same-possession flag retained in raw')
    results=dict(status='COMPLETE',games=len(es),passive_action_parity=True,
        exposure=dict(all_enemy_carry_ticks=len(rows),bot_actionable=len(active),
            bot_dead=sum(not r['bot_alive'] for r in rows),bot_controlled=sum(r['bot_alive'] and r['bot_trapped']>0 for r in rows),
            bot_simultaneously_carrying=sum(r['bot_carrying']>=0 for r in rows)),
        modes=dict(Counter(r['mode'] for r in rows)),actionable_modes=dict(Counter(r['mode'] for r in active)),
        goal_sources=dict(Counter(r.get('goal_source','not_actionable') for r in rows)),
        plan_absence=dict(Counter(r['planner']['plan_absent'] for r in rows if 'plan_absent' in r.get('planner',{}))),
        planner=dict(eligible=len(planned),with_common_cell=sum(r['planner']['common_cells']>0 for r in planned),
            without_common_cell=sum(not r['planner']['common_cells'] for r in planned),best_found=len(planned)-len(no_best),no_best=len(no_best),
            reject_counts={k:sum(r['planner'].get(k,0) for r in no_best) for k in ('reject_no_hit','reject_unreachable','reject_time','reject_passed','valid')},
            note='candidate counts nested within ticks, not independent opportunities; no-common and time failures can overlap'),
        bombs=dict(route_proposed_ticks=len(proposed),escape_veto=sum(r['bomb']['own_escape']==0 for r in proposed),
            robust_intersection_veto=sum(r['bomb']['own_escape'] is not None and r['bomb']['own_escape']!=0 and bool(r['bomb']['robust']) and not(r['bomb']['own_escape'] & r['bomb']['robust'][0]) for r in proposed),
            all_first_own_escape_calls=sum(r.get('bomb',{}).get('own_escape') is not None for r in rows),
            all_first_own_escape_zero=sum(r.get('bomb',{}).get('own_escape')==0 for r in rows),
            all_first_robust_intersection_zero=sum(r.get('bomb',{}).get('own_escape') is not None and r['bomb']['own_escape']!=0 and bool(r['bomb']['robust']) and not(r['bomb']['own_escape'] & r['bomb']['robust'][0]) for r in rows),
            all_carry_placements=len(places),by_reason=by_reason,
            inventory_absent_ticks=sum(r.get('bomb',{}).get('inventory') is False for r in rows),
            deliberate_target_wait_ticks=sum(bool(r.get('bomb',{}).get('wait')) for r in rows)),
        holds=dict(total=sum(r['reason']=='hold' for r in rows),
            by_goal_source=dict(Counter(r.get('goal_source') for r in rows if r['reason']=='hold'))),
        replay_pressure=pressure(es),hard_same64=pressure(episodes(BASE/'evaluation/u3040/hunter_hard')),
        limitations=['all tick shares descriptive, correlated within games','unknown and censored arrivals retained, no imputation',
            'self_deaths is native ownBombDefeat causal death counter, not every trap/contact','no held-out tuning or JAX training admission'],
        raw_receipts={str(p):sha(p) for p in sorted(folder.glob('*.ACCEPTED.json'))},unix=time.time())
    output=folder/('SUMMARY_RECOMPUTED.json' if (folder/'SUMMARY.json').exists() else 'SUMMARY.json')
    atomic_json(output,results)
    print(json.dumps({k:v for k,v in results.items() if k not in ('raw_receipts',)},indent=2))


if __name__=='__main__':main()
