"""Read only completed fixed-checkpoint evidence; never controls GPU training."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def ratio(n,d):
    return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def normalized_route_delta(current,previous):
    """Compare pooled opportunity rates, keeping missing rates unknown.

    Legacy evaluator fields named oracle_agreement/progress_per_target_tick
    contain sums of episode ratios. Only route_rates' count-derived ratios
    are comparable here; immutable raw evaluator artifacts stay untouched.
    """
    result={}
    for name,value in current.items():
        prior=previous.get(name)
        if not isinstance(value,dict) or not isinstance(prior,dict):continue
        if not all(k in value and k in prior for k in ['numerator','denominator','rate']):continue
        result[name]=dict(numerator=value['numerator']-prior['numerator'],
            denominator=value['denominator']-prior['denominator'],
            rate=value['rate']-prior['rate'] if value['rate'] is not None and prior['rate'] is not None else None)
    return result


def interpret_verifier_audit(audit,native_progress=False):
    if not audit:return None
    # H6 overwrites label8 with selected before appending the audit. The
    # successor host initially inherited misleading geometric field names.
    names={'geometrically_reachable':'selected_for_verifier',
           'safety_accepted':'safety_and_native_progress_accepted' if native_progress else 'safety_and_geometry_accepted',
           'reachable_rejected':'selected_rejected'}
    return dict(counts={names.get(k,k):v for k,v in audit.items()},
                geometric_candidate_count=None,
                scope=('Bounded selection precedes native safety and native endpoint progress set; independent geometric coverage was not logged.' if native_progress else
                    'Bounded selection precedes native safety and original geometry gate; independent geometric coverage was not logged.'))


def phase_coverage(roles,audit):
    """All-route active->accepted->query counts on matching phase populations.

    Bounded selection is inside accepted/active; this is observed label coverage,
    never a safety pass rate conditional on all geometrically valid candidates.
    """
    fields={'drop':('candidate_a','path_a','safety_accepted_drop'),
        'stock':('candidate_b','path_b','safety_accepted_stock'),
        'return':('candidate_carry_return','carry_return','safety_accepted_return'),
        'delivery':('candidate_cage_delivery','cage_delivery','safety_accepted_delivery')}
    if not roles or not all('candidate_b' in r for r in roles.values()):return None
    result={}
    for name,(active_key,query_key,accepted_key) in fields.items():
        active=sum(r.get(active_key,0) for r in roles.values())
        query=sum(r.get(query_key,0) for r in roles.values())
        accepted=audit.get(accepted_key)
        result[name]=dict(active=active,selected=None,geometric=None,accepted=accepted,query=query,
            accepted_per_active=ratio(accepted,active) if accepted is not None else None,
            query_per_active=ratio(query,active),
            scope='All native route scenes; bounded selection included, per-phase selected/geometric counts unlogged; query may repeat an episode across rollout chunks')
    assert sum(r['active'] for r in result.values())==sum(r.get('eligible',0) for r in roles.values())
    assert sum(r['query'] for r in result.values())==sum(r.get('query',0) for r in roles.values())
    if audit:
        assert sum(r['active'] for r in result.values())==audit['eligible_active']
        assert sum(r['accepted'] for r in result.values())==audit['safety_accepted']
        assert sum(r['query'] for r in result.values())==audit['first_deviation_query']
    return result


def route_rates(r):
    episodes=r.get('episodes',[])
    window_episodes=sum(e.get('window_events',0)>0 for e in episodes)
    window_target_episodes=sum(e.get('window_events',0)>0 and e.get('target_events',0)>0
        for e in episodes)
    return dict(window_to_target=ratio(window_target_episodes,window_episodes),
        window_to_target_scope='Episode co-occurrence, not event-level causal conversion; targets of initialized carriers without a native window are excluded.',
        first_oracle=ratio(r.get('first_oracle_match_episodes',0),r.get('oracle_opportunity_episodes',0)),
        sustained_oracle=ratio(r.get('oracle_matches',0),r.get('oracle_opportunities',0)),
        progress_per_target_tick=ratio(r.get('progress_ticks',0),r.get('target_ticks',0)),
        pickup_per_game=ratio(r.get('pickup_episodes',0),r['games']),
        delivery_per_game=ratio(r.get('delivery_episodes',0),r['games']),
        switch_on_recovery=ratio(r.get('recovery_switches',0),r.get('recovery_switch_opportunities',0)),
        control_attribution=ratio(r.get('attributed_first_control',0),r.get('first_control',0)),
        # final_pop also contains initialized fixture control, so its denominator
        # cannot be pooled with newly attributed controls into a conversion rate.
        control_raw={k:r.get(k,0) for k in ['first_control','attributed_first_control','final_pop','active_contact_pop','causal_kill','surviving_kill','native_causal_kill']},
        safety={k:r.get(k,0) for k in ['death','self_death']})


def route_strata(r):
    groups={}
    for e in r.get('episodes',[]):
        key=f"team{e['team']}:{e['geometry']}"
        group=groups.setdefault(key,Counter())
        group.update(games=1,pickup_episodes=int(e.get('pickup',0)>0),delivery_episodes=int(e.get('delivery',0)>0),
            death=e.get('death',0),self_death=e.get('self_death',0))
        for phase,counts in e.get('stage_funnel',{}).items():
            group.update({phase+':'+k:v for k,v in counts.items()})
    return {k:dict(v) for k,v in groups.items()}


def fixed_review(out,task):
    path=out/f'fixed-{task}-2026101800.trace.json'
    if not path.exists():return None
    trace=json.loads(path.read_text());first=trace.get('first_deviation')
    ticks=trace['diagnostic_ticks'];phases=[t.get('phase') for t in ticks]
    phase_counts=dict(Counter(phases))
    phase_transitions=dict(Counter(f'{a}->{b}' for a,b in zip(phases,phases[1:]) if a!=b))
    if not first:return {'seed':2026101800,'first_deviation':None,
                         'phase_ticks':phase_counts,'phase_transitions':phase_transitions}
    idx=next((i for i,t in enumerate(ticks) if t['tick']==first['tick']+1),None)
    row=ticks[idx] if idx is not None else {}
    previous=ticks[idx-1] if idx else {}
    advantage=row.get('approximate_reward_gae');prior=previous.get('approximate_reward_gae')
    return dict(seed=2026101800,first_deviation=first,phase_ticks=phase_counts,phase_transitions=phase_transitions,
        action_tick=first['tick'],recorded_transition_tick=row.get('tick'),
        reward_components=row.get('approximate_reward_components'),
        approximate_reward_gae=advantage,previous_approximate_reward_gae=prior,
        approximate_advantage_change=advantage-prior if advantage is not None and prior is not None else None,
        scope='Fixed game0; exact production Review replay generated and verified by unique evaluator. JS reward reconstruction and zero-value reward-GAE are approximate, not checkpointed training reward or Q.')


def analyze(base):
    base=Path(base)
    if base.is_absolute():base=base.relative_to(ROOT)
    summaries=[];training=[]
    plan=json.loads((base/'plan.json').read_text())
    events=base/'formal/bun392_bridge_pfsp8/events.jsonl'
    if events.exists():training=[json.loads(line) for line in events.read_text().splitlines()]
    for out in sorted((base/'authority_je10').glob('g*')):
        marker=out/'COMPLETE.json'
        if not marker.exists():continue
        proof=json.loads(marker.read_text());raw=(out/'SUMMARY.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==proof['shards']['SUMMARY.json']
        s=json.loads(raw);g=s['generation'];rows=[r for r in training if r['generation']<=g]
        phases={};audit=Counter();rewards=Counter();transitions=Counter();scenes=Counter();league=Counter()
        for row in rows:
            rewards.update(dict(zip(row['component_names'],row['reward_components'])))
            from scripts.je10_diagnostics import native_full_event_counts
            transitions.update(native_full_event_counts(row))
            scenes.update(dict(zip(row['scene_names'],row['actual_scene_valid_ticks'])))
            audit.update(row['dagger'].get('verifier_audit',{}))
            for role,counts in row['dagger']['opponent_course'].items():
                p=phases.setdefault(role,Counter())
                p.update({k:v for k,v in counts.items() if isinstance(v,(int,float))})
            league.update({k:row.get('actual_league_exposure',{}).get(k,0) for k in ['current','ring','fixed']})
        league_total=sum(league.values())
        summary=dict(generation=g,retention={'hunter_js':s['hunter_js'],'tactical_python':s['tactical_python']},
            delta_parent=s['delta_parent'],delta_previous=s['delta_previous'],
            routes={task:route_rates(r) for task,r in s['route'].items()},
            route_strata={task:route_strata(r) for task,r in s['route'].items()},
            fixed_first_deviations={task:fixed_review(out,task) for task in s['route']},
            training=dict(generations=len(rows),native_full_event_counts=dict(transitions),scene_valid_ticks=dict(scenes),
                scene_valid_tick_fraction={k:v/max(1,sum(scenes.values())) for k,v in scenes.items()},
                native_full_events_per_million_full_ticks={k:v*1e6/max(1,scenes['full_bun']) for k,v in transitions.items()},reward_components=dict(rewards),
                control_gross_positive=rewards['control_entry_reward']+rewards['control_residual_reward'],
                control_net=rewards['control_entry_reward']+rewards['control_residual_reward']+rewards['control_clawback'],
                route_positive=rewards['route_positive'],route_negative=rewards['route_negative'],route_net=rewards['route_shaped_reward'],
                capture_terminal_total=rewards['transport_capture'],
                dagger_role_and_phase={k:dict(v) for k,v in phases.items()},
                dagger_phase_opportunities=phase_coverage(phases,audit),
                dagger_verifier_audit=interpret_verifier_audit(audit,bool(plan.get('oracle_certificate'))),
                actual_opponents=s['actual_training_exposure'],
                sampled_league_counts=dict(league),sampled_league_conditional_fraction={k:v/league_total if league_total else None for k,v in league.items()}))
        for label,reference in [('parent',summaries[0] if summaries else summary),
                                ('previous',summaries[-1] if summaries else summary)]:
            summary['normalized_route_delta_'+label]={task:normalized_route_delta(r,reference['routes'][task])
                for task,r in summary['routes'].items()}
        summaries.append(summary)
    result=dict(schema='je10.completed-curve-analysis/v1',run=str(base),controls_training=False,
        completed_generations=[s['generation'] for s in summaries],checkpoints=summaries,
        limits=['No incomplete checkpoint or partial evaluator artifact is consumed.',
                'Raw evaluator oracle_agreement/progress_per_target_tick are sums of episode ratios; use these pooled count-derived route rates and normalized_route_delta fields. Zero denominators remain null.',
                'Initialized control fixtures are distinct from newly caused control events; no pooled pop/control conversion rate.',
                'JAX training counts, Python Tactical and JS Hunter/native tasks are separate opportunity populations.',
                'Zero denominator yields null, never a false zero success rate.',
                'Current/ring/fixed are underlying sampled league exposure; weak-role overrides remain separately disclosed.',
                'Old r01 logs lack selected/accepted verifier audit; query/candidate alone cannot identify rejection cause.',
                'Role candidate/query fields include all route scenes; role visited counts cover natural full only. Do not divide across these scopes.',
                'Scene mixture weights apply at reset; differing horizons and termination produce different actual tick occupancy.',
                'Only native Bun806 and open/wall variants are observed; cross-map generalization remains unproven.'])
    (base/'CURVE_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    result=analyze(args.run)
    for s in result['checkpoints']:
        print(json.dumps(dict(generation=s['generation'],hunter=[s['retention']['hunter_js'][k]['safe_win'] for k in ['hard','normal','easy']],
            tactical=s['retention']['tactical_python']['safe_win'],route_delivery={k:v['delivery_per_game']['numerator'] for k,v in s['routes'].items()},
            native_full=s['training']['native_full_event_counts'],sampled_league_fraction=s['training']['sampled_league_conditional_fraction'])))
