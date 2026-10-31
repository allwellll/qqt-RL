"""Recompute delayed carrier attribution from the frozen full-game records.

The instantaneous Sim causalKill flag does not carry attribution through a
sugar timeout. Use the evaluator's existing persistent ControlFunnel certificate
and disclose both sources. No new games or source-label fabrication.
"""
import json
from pathlib import Path
from scripts.population_run_state import atomic_json,sha

B=Path('runs/r4_combat_20261031/carrier_diagnostic_v2')


def recompute(episode,seat):
    raw=episode['r4_carrier'];deaths=[e for e in raw['events'] if e['type']=='carrier_death' and e['learner']==seat]
    certified=[k for k in episode['kills'] if k['attacker']==seat and k['victim_carried']==seat]
    matching=[k for k in certified if any(e['tick']==k['tick'] for e in deaths)]
    # Every certified kill used here must have the independently recorded
    # factual carrier death and the same origin, seat and physical tick.
    assert len(matching)==len(certified)
    events=[]
    events += [dict(tick=e['tick'],kind='pickup') for e in raw['events'] if e['type']=='own_stock_taken' and e['learner']==seat]
    events += [dict(tick=k['tick'],kind='kill') for k in matching]
    for e in episode['antifarm']['events']:
        if e['type']=='delivery' and e['origin']==seat:
            events.append(dict(tick=e['tick'],kind='return' if e['player']==seat else 'loss'))
    priority={'kill':0,'return':1,'loss':1,'pickup':2}
    events.sort(key=lambda e:(e['tick'],priority[e['kind']]))
    current=None;cycles=[]
    for e in events:
        if e['kind']=='pickup':current=dict(pickup_tick=e['tick'],kill_tick=None)
        elif e['kind']=='kill' and current and current['pickup_tick']<e['tick']:current['kill_tick']=e['tick']
        elif e['kind']=='return':
            if current and current['kill_tick'] is not None:cycles.append(dict(current,return_tick=e['tick']))
            current=None
        elif e['kind']=='loss':current=None
    return dict(carrier_causal_kills_funnel=len(matching),carrier_causal_kills_sim_flag=raw['carrier_causal_kills'][seat],
        completed_antifarm_cycles_funnel=len(cycles),cycles=cycles,certified_carrier_kills=matching)


def main():
    collection=json.loads((B/'COMPLETE.json').read_text());assert collection['status']=='COMPLETE'
    es=[]
    for start in range(0,256,8):
        f=B/f'shard_{start:03d}.json.outcome.json';assert sha(f)==collection['files'][str(f)]
        es+=json.loads(f.read_text())['episodes']
    records=[dict(seed=e['seed'],seat=e['seat'],**recompute(e,e['seat'])) for e in es]
    results={}
    for p in (0,1):
        v=dict(collection['results'][f'P{p}']);group=[r for r in records if r['seat']==p]
        for k in ('carrier_causal_kills_funnel','carrier_causal_kills_sim_flag','completed_antifarm_cycles_funnel'):
            v[k]=sum(r[k] for r in group)
        v['carrier_killed_rate_funnel']=v['carrier_causal_kills_funnel']/v['exposed_carrier_lives'] if v['exposed_carrier_lives'] else None
        v['completed_antifarm_cycles_funnel_per_game']=v['completed_antifarm_cycles_funnel']/v['games']
        # Disambiguate the initial raw counters; never silently call them the
        # complete delayed causal chain in downstream summaries.
        for k in ('carrier_causal_kills','carrier_killed_rate','completed_antifarm_cycles','completed_antifarm_cycles_per_game'):v.pop(k)
        results[f'P{p}']=v
    d=dict(status='COMPLETE',measurement_version=3,games=256,new_games=0,results=results,records=records,
        collection_sha256=sha(B/'COMPLETE.json'),source_sha256={str(Path(__file__).relative_to(Path.cwd())):sha(Path(__file__))},
        attribution='production evaluator persistent ControlFunnel certificate matched to recorded carrier death; instantaneous Sim causalKill flag reported separately',
        limitation='observational carrier path proximity cannot prove useful interception, reward sufficiency or a causal training bottleneck')
    atomic_json(B/'FINAL_METRICS.json',d)
    f=Path('reports/r4_combat_20261031.md');text=f.read_text();marker='\n## u4002 持包者拦截基线（22:25修订）'
    start=text.find(marker)
    if start>=0:
        end=text.find('\n## ',start+len(marker));end=len(text) if end<0 else end
        text=text[:start]+text[end:]
    lines=[marker,'','Hunter hard/greedy、老师关闭，256完整局（每席128），动作SHA逐局与旧诊断一致。未新增重跑；从已记录ControlFunnel延迟归因和同tick持包死亡重算。','',
        '|席位|可行动敌持我包tick|路径3格内tick/比例|暴露持包生命|延迟因果杀持包者/比例|敌送达/局|己送达/局|完整取杀夺回循环/局|',
        '|---|---|---|---|---|---|---|---|']
    for seat,v in results.items():
        lines.append(f"|{seat}|{v['own_actionable_ticks']}|{v['interception_ticks']}/{v['interception_rate']:.4%}|{v['exposed_carrier_lives']}|{v['carrier_causal_kills_funnel']}/{v['carrier_killed_rate_funnel']:.2%}|{v['enemy_deliveries_per_game']}|{v['own_deliveries_per_game']}|{v['completed_antifarm_cycles_funnel_per_game']}|")
    lines+=['','瞬时Sim `causalKill` 在糖浆延迟死亡中两席均漏记为0；原始计数保留于采集收据，不能据此声称杀持包者为0。新表采用已有生产评估的持久ControlFunnel证书，逐条匹配死亡tick/席位/包来源。几何接近比例高不能等同有效封路；本样本不足以证明“不会拦截”或“奖励不够”是唯一根因。生命去重，无原生JAX训练奖励账本。',
        '', '原始逐局与SHA：`runs/r4_combat_20261031/carrier_diagnostic_v2/COMPLETE.json`；重算公式、逐局证书与最终指标：`runs/r4_combat_20261031/carrier_diagnostic_v2/FINAL_METRICS.json`。','']
    f.write_text(text+'\n'.join(lines))
    print(json.dumps(dict(status='COMPLETE',results=results),ensure_ascii=False))


if __name__=='__main__':main()
