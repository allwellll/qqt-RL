"""Evidence-only checkpoint diagnostics. Never launches or stops GPU work."""
import json
from collections import Counter
from pathlib import Path
from scripts.population_run_state import atomic_json


def sampling_diagnostic_targets():
    # Retired eventless-custody scenes have intentionally zero exposure.
    return ('vulnerable_steal','defend_and_steal','route_break_capture')


def native_full_event_counts(row):
    names=('own_window_enter','own_kill','enemy_base_entry','loose_pickup',
           'inventory_or_native_steal','attributed_custody_start','capture','native_transfer')
    counts=Counter()
    for role in row['dagger']['opponent_course'].values():
        values=role['native_transitions']
        if len(values)!=len(names):raise ValueError('native-full transition schema mismatch')
        counts.update(dict(zip(names,values)))
    return dict(counts)


def diagnose(base, evaluation):
    base=Path(base)
    if base.is_absolute():base=base.relative_to(Path(__file__).resolve().parents[1])
    generation=evaluation['generation']
    rows=[json.loads(s) for s in (base/'formal/bun392_bridge_pfsp8/events.jsonl').read_text().splitlines()] if generation else []
    rows=[r for r in rows if r['generation']<=generation]
    rewards=Counter();transitions=Counter();scenes=Counter();queries=candidates=0.
    for row in rows:
        rewards.update(dict(zip(row['component_names'],row['reward_components'])))
        transitions.update(native_full_event_counts(row))
        scenes.update(dict(zip(row['scene_names'],row['actual_scene_valid_ticks'])))
        queries+=row.get('signal_decomposition',{}).get('oracle_queries',sum(r['query'] for r in row['dagger']['opponent_course'].values()))
        candidates+=row.get('signal_decomposition',{}).get('oracle_candidates',sum(r['eligible'] for r in row['dagger']['opponent_course'].values()))
    failures=[]
    for path in sorted((base/'diagnostics').glob('FIRST_DEVIATIONS_*.json')):
        if int(path.stem.rsplit('_',1)[1])<=generation:
            data=json.loads(path.read_text());failures.extend(dict(generation=data['generation'],**r) for r in data['failures'])
    routes={}
    for task,r in evaluation['route'].items():
        routes[task]={k:r.get(k,0) for k in ['games','window_ticks','target_ticks','oracle_opportunities','oracle_matches','oracle_opportunity_episodes','first_oracle_match_episodes','progress_ticks','pickup_episodes','return_started','delivery_episodes','recovery_switch_opportunities','recovery_switches','death','self_death']}
        routes[task].update(acquisition_scope=r.get('acquisition_scope','legacy_fixture_scope_unverified'),
            synthetic_carrier_diagnostic_only=task in ['carry_near','carry_mid','carry_far'],
            proves_post_pickup_unreachability=False,oracle_agreement=r.get('oracle_matches',0)/max(1,r.get('oracle_opportunities',0)),progress_per_target_tick=r.get('progress_ticks',0)/max(1,r.get('target_ticks',0)))
    denom=max(1,abs(rewards['combined_total']))
    hypotheses=[]
    zero_routes=[k for k in sampling_diagnostic_targets() if scenes.get(k,0)==0]
    if rows and zero_routes:
        hypotheses.append(dict(rank=1,hypothesis='Legal empty-handed route lessons have zero observed sampling mass; learner-state route exposure can remain concentrated on bridge rather than the weak full-game stage.',evidence=dict(zero_sampled_lessons=zero_routes,natural_full_transitions=dict(transitions)),minimal_next_change='Change only scene mixture, preserving credit/PBRS/DAgger/network: combat .35, bridge .20, full .35, remaining .10 legal empty-handed scenes1/5/7 only; never restore retired scenes2/3/4/6. Predeclare new identity, focused sampler TDD and technical smoke; healthy current fixed run continues.'))
    if rows:
        hypotheses.append(dict(rank=len(hypotheses)+1,hypothesis='Bounded strict-verifier queries may cover a small fraction of learner route candidates; candidate count alone does not prove useful supervision coverage.',evidence=dict(query_ticks=queries,candidate_ticks=candidates,query_per_candidate=queries/max(1,candidates)),minimal_next_change='Inspect phase-specific first deviations and verifier rejection causes before changing query budget; do not change scene mix and query budget in the same causal follow-up.'))
        hypotheses.append(dict(rank=len(hypotheses)+1,hypothesis='Route shaping may be small or negative relative to task/control rewards; evaluate gross positive and negative terms, not just net.',evidence={k:rewards[k] for k in ['control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward','route_positive','route_negative','settled_kill','transport_capture','combined_total']},minimal_next_change='Use phase/target first-deviation traces and reward-GAE approximation to determine signal resolution before any coefficient change.'))
    result=dict(retired_scene_ids=[2,3,4,6],retired_scene_zero_exposure_expected=True,post_pickup_unreachability_inference_withdrawn=True,generation=generation,diagnostic_only=True,controls_training=False,opportunities_scope='Training native transitions are natural-full counters; JS route fixtures and native full are separate tasks; no cross-simulator pooled rate.',training_generations=len(rows),reward_components=dict(rewards),reward_fraction_of_absolute_combined_total={k:rewards[k]/denom for k in ['control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward','route_positive','route_negative']},actual_scene_ticks=dict(scenes),native_full_transitions=dict(transitions),oracle=dict(candidate_ticks=candidates,query_ticks=queries,query_per_candidate=queries/max(1,candidates)),first_deviation_sample=dict(selection='first16 encountered per generation, capped diagnostic sample; not population failure rate',count=len(failures),phases=dict(Counter(str(r['phase']) for r in failures)),first=failures[0] if failures else None),route_funnel=routes,ranked_root_cause_hypotheses=hypotheses,retention=dict(hunter_js=evaluation['hunter_js'],tactical_python=evaluation['tactical_python']),actual_opponent_exposure=evaluation['actual_training_exposure'],delta_parent=evaluation['delta_parent'],delta_previous=evaluation['delta_previous'],limits=['Only Bun806 native/open geometries available; cross-map generalization unproven.','JS observable reconstruction and zero-value reward GAE are approximate, not checkpointed training reward or counterfactual Q.','Intermediate metrics diagnose and prepare successor proposals; never capability/admission/promotion gates.'])
    atomic_json(base/'authority_je10'/f'g{generation:06d}'/'DIAGNOSIS.json',result)
    if generation==128:
        atomic_json(base/'FINAL_ROOT_CAUSE_PROPOSAL.json',dict(generation=generation,evidence=result,decision='Prepare the highest-ranked minimum verifiable change; predeclared successor already starts before final evaluation. Retain combat/safety checks as diagnostics, not admission gates.'))
    report=Path('reports/je10_training_20261018.md')
    heading=f'<!-- JE10-AUTO:{base.name}:g{generation}:BEGIN -->';ending=f'<!-- JE10-AUTO:{base.name}:g{generation}:END -->'
    lines=[heading,f'\n运行 `{base.name}` g{generation} 异步固定评估完成。该点只作诊断，GPU固定预算继续；各simulator严格分列。\n', '| 协议 | 安全击杀 / 局数 | 自伤局 |', '|---|---:|---:|']
    for name,r in [('Python Tactical v2',evaluation['tactical_python'])]+[(f'JS Hunter {k}',v) for k,v in evaluation['hunter_js'].items()]:lines.append(f"| {name} | {r.get('safe_win',0)}/{r.get('games',64)} | {r.get('self_episodes',0)} |")
    lines+=['\n| JS 固定路线 | oracle机会局 | 首方向一致局 | 拾包局 | 交付局 / 局数 |','|---|---:|---:|---:|---:|']
    for task,r in routes.items():lines.append(f"| {task} | {r['oracle_opportunity_episodes']} | {r['first_oracle_match_episodes']} | {r['pickup_episodes']} | {r['delivery_episodes']}/{r['games']} |")
    lines.append(f"\n截至该点训练累计：native full事件 `{dict(transitions)}`；DAgger query/candidate={queries:g}/{candidates:g}；route净/总绝对合计={rewards['route_shaped_reward']:g}/{denom:g}。原始机会分母、相对父点/前一点变化、首次偏航样本、保留指标及实际对手暴露见 `{base}/authority_je10/g{generation:06d}/DIAGNOSIS.json`。")
    for h in hypotheses:lines.append(f"\n根因假设{h['rank']}（尚非单变量因果结论）：{h['hypothesis']} 下一轮最小候选改动：{h['minimal_next_change']}")
    lines+=['',ending,''];block='\n'.join(lines);content=report.read_text() if report.exists() else ''
    if heading in content:
        a=content.index(heading);b=content.index(ending,a)+len(ending);content=content[:a]+block+content[b:]
    else:content+='\n'+block
    report.write_text(content)
    return result
