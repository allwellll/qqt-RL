"""Paired successor results by original map, brick stage and pickup threat."""
import json
import time
from collections import defaultdict
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.combat_regions_report_v2 import paired as original_paired,measures as original_measures
import numpy as np

B=Path('runs/corner_counterplay_20261104');E=B/'stage_evaluation'


def measures(row):
    result=original_measures(row);r3=row['r3'];seat=row['seat']
    custody=row['primary_custody']['groups']['all']
    result.update(item_pickups=sum(r3['pickups'][seat]),item_releases=sum(r3['releases'][seat]),
        custody_timeouts=custody['timeouts'],custody_unknown_drops=custody['unknown_drops'],
        custody_other_terminal=custody['other_terminal'])
    return result


def paired(parent,candidate):
    result=original_paired(parent,candidate)
    left={(r['seed'],r['seat'],r['mode']):r for r in parent}
    right={(r['seed'],r['seat'],r['mode']):r for r in candidate}
    clusters=defaultdict(list)
    for key in sorted(left.keys()&right.keys()):clusters[key[0]].append((measures(left[key]),measures(right[key])))
    if not clusters:return result
    blocks=list(clusters.values());ids=np.random.default_rng(2026543110).integers(0,len(blocks),(2000,len(blocks)))
    sizes=np.asarray([len(block) for block in blocks],float)
    for name in ('item_pickups','item_releases','custody_timeouts','custody_unknown_drops','custody_other_terminal'):
        deltas=np.asarray([sum(b[name]-a[name] for a,b in block) for block in blocks],float)
        boot=deltas[ids].sum(-1)/sizes[ids].sum(-1)
        result['intervals'][name]=dict(mean=float(deltas.sum()/sizes.sum()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)))
    return result


def main():
    models=defaultdict(lambda:defaultdict(list));p=json.loads((E/'PROTOCOL.json').read_text())
    for file in E.glob('*/**/ACCEPTED.json'):
        proof=json.loads(file.read_text());assert all(sha(f)==h for f,h in proof['files'].items())
        job=proof['job'];models[job['candidate']][job['target']].extend(json.loads((file.parent/'COMPLETE.json').read_text())['results'])
    output=dict(status='PARTIAL',capability_improvement='NOT_ESTABLISHED',models={},comparisons={},unix=time.time())
    if not models:
        status_path=E/'STATUS.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status')=='WAITING_SOURCE_CHECKPOINT':
            output.update(status='WAITING_SOURCE_CHECKPOINT',waiting_reason=status.get('reason'),compilation_started=False,
                expected_source_update=19382,observed_upstream_update=status.get('observed_upstream_update'))
    for label,targets in sorted(models.items()):
        output['models'][label]={}
        for target,rs in targets.items():
            groups=defaultdict(list)
            for r in rs:groups[(r['start_region'],r.get('brick_stage','native_complete'))].append(r)
            output['models'][label][target]=dict(episodes=len(rs),expected=len(p['heavy_games'] if target=='heavy_v6' else p['pair_games']),strata=[])
            for (region,stage),data in sorted(groups.items()):
                vals=[measures(r) for r in data];tot={k:sum(v[k] for v in vals if v[k] is not None) for k in vals[0] if k!='eligible_gap_median'}
                output['models'][label][target]['strata'].append(dict(region=region,brick_stage=stage,**tot,
                    corner_controls=sum(r.get('corner',{}).get('controls',0) for r in data),corner_kills=sum(r.get('corner',{}).get('kills',0) for r in data)))
    baseline=models.get('parent19382',{})
    for label,targets in models.items():
        if label=='parent19382':continue
        output['comparisons'][label]={}
        for target,rs in targets.items():
            parent=baseline.get(target,[]);groups={}
            for region,stage in sorted({(r['start_region'],r.get('brick_stage','native_complete')) for r in parent}):
                filt=lambda r:r['start_region']==region and r.get('brick_stage','native_complete')==stage
                groups[region+'/'+stage]=paired([r for r in parent if filt(r)],[r for r in rs if filt(r)])
            output['comparisons'][label][target]=groups
        # Heavy-v6 remains the exact old natural item/transport sentinel.
        natural=output['comparisons'][label].get('heavy_v6',{}).get('natural/native_complete',{})
        iv=natural.get('intervals',{})
        guards=all(iv.get(k,{}).get(side,default)>=threshold if side=='lower95' else iv.get(k,{}).get(side,default)<=threshold
            for k,side,threshold,default in [('wins','lower95',-.10,-999),('self_deaths','upper95',.10,999),('deliveries','lower95',-.10,-999),('carry_failures','upper95',.10,999)])
        natural_rows=lambda source:[r for r in source.get('heavy_v6',[]) if r['start_region']=='natural']
        natural_exposure=min(sum(measures(r)['acquisitions'] for r in natural_rows(source)) for source in (baseline,targets))>=8
        item_guard=all(iv.get(k,{}).get('lower95',-999)>=-.25 for k in ('item_pickups','item_releases'))
        complete=all(len(targets.get(t,[]))==len(p['heavy_games'] if t=='heavy_v6' else p['pair_games']) and len(baseline.get(t,[]))==len(p['heavy_games'] if t=='heavy_v6' else p['pair_games']) for t in ('heavy_v6','corner17999','parent18102'))
        brick_checks={}
        for stage in ('complete','partial','asymmetric','open'):
            comp=output['comparisons'][label].get('parent18102',{}).get('natural/'+stage,{})
            ci=comp.get('intervals',{})
            exposure=min(sum(measures(r)['acquisitions'] for r in source.get('parent18102',[]) if r['start_region']=='natural' and r.get('brick_stage')==stage) for source in (baseline,targets))>=8
            brick_checks[stage]=exposure and ci.get('deliveries',{}).get('lower95',-999)>=-.10 and ci.get('carry_failures',{}).get('upper95',999)<=.10
        corner_proven=True
        for target in ('corner17999','parent18102'):
            parent=[r for r in baseline.get(target,[]) if r['start_region']=='bottom_right']
            candidate=[r for r in targets.get(target,[]) if r['start_region']=='bottom_right']
            # Keep actual contact denominators; replace only the counted kill
            # column with strictly corner-origin uninterrupted control kills.
            def corner_rows(rs):
                result=[]
                for row in rs:
                    copy=dict(row);ca=dict(row['continuous_attack']);ca['groups']={}
                    for i,(name,g) in enumerate(row['continuous_attack']['groups'].items()):
                        ca['groups'][name]=dict(g,causal_kills=row.get('corner',{}).get('kills',0) if i==0 else 0)
                    copy['continuous_attack']=ca;result.append(copy)
                return result
            c=paired(corner_rows(parent),corner_rows(candidate))
            output['comparisons'][label][target]['strict_corner_origin_kills']=c
            corner_proven&=c.get('intervals',{}).get('causal_kills_per_contact_game',{}).get('lower95',-999)>0
        all_regions=lambda source:[r for r in source.get('parent18102',[]) if r['start_region']!='natural']
        attack=paired(all_regions(baseline),all_regions(targets))
        attack_proven=attack.get('intervals',{}).get('repeated_attack_kills',{}).get('lower95',-999)>0
        output['comparisons'][label]['continuous_attack_all_regions']=attack
        checks=dict(complete_fixed_pairs=complete,heavy_natural_transport_safety=guards and natural_exposure,
            heavy_natural_item_guard=item_guard,all_brick_density_transport_guards=all(brick_checks.values()),
            strict_corner_kill_improvement=corner_proven,continuous_attack_kill_conversion=attack_proven)
        output['comparisons'][label]['acceptance']=dict(checks=checks,brick_checks=brick_checks,status='PASS_WITHIN_SMALL_FIXED_PROTOCOL' if all(checks.values()) else 'NOT_ESTABLISHED')
        if all(checks.values()):output['capability_improvement']='ESTABLISHED_WITHIN_SMALL_FIXED_PROTOCOL_ONLY'
    threats=defaultdict(lambda:dict(acquisitions=0,deliveries=0,failures=0))
    for label,targets in models.items():
        for target,rs in targets.items():
            for r in rs:
                for event in r.get('pickups',[]):
                    if event['player']!=r['seat']:continue
                    d=threats[(label,target,r.get('brick_stage','native_complete'),event['band'])];d['acquisitions']+=1
                    outcome=(event.get('custody') or {}).get('outcome');d['deliveries']+=int(outcome=='same_custody_delivery');d['failures']+=int(outcome in ('death_drop','controlled_drop'))
    output['pickup_threat_strata']=[dict(model=k[0],target=k[1],brick_stage=k[2],threat=k[3],**v) for k,v in sorted(threats.items())]
    if (E/'COMPLETE.json').exists():output['status']='COMPLETE'
    atomic_json(E/'SUMMARY.json',output)
    report=Path('reports/corner_counterplay_stage_20261104.md')
    lines=['# 下一完整阶段：多砖布局与角落反制（20261104）','',
        '阶段源u19382，320次真实更新目标u19702；原1280更新保持独立完整。训练、CPU恢复、实际PPO曝光与固定配对评估分别验收。任何未完整对照不宣称能力改善。','',
        '冻结原始自然heavy-v6道具/运包对照、历史角落偏好模型、父点、四种砖密度与七类区域。按取包时敌死/受控/活远/活近可截返程/活近其他分层，同链交付及失败分别计数。布局是806衍生冷启动训练地图，不冒称新原版地图。','',
        '|模型|对手|区域/砖阶段|局/胜|因果杀/自死|取包/交付/失败|',
        '|---|---|---|---:|---:|---:|']
    for label,targets in output['models'].items():
        for target,block in targets.items():
            for r in block['strata']:lines.append(f"|{label}|{target}|{r['region']}/{r['brick_stage']}|{r['games']}/{r['wins']}|{r['causal_kills']}/{r['self_deaths']}|{r['acquisitions']}/{r['deliveries']}/{r['carry_failures']}|")
    lines+=['','能力结论：`'+output['capability_improvement']+'`。按环境seed聚类的配对bootstrap区间、原始样本量和各项验收见 `runs/corner_counterplay_20261104/stage_evaluation/SUMMARY.json`。低样本或缺少真实取包不能当保护证明。','']
    report.write_text('\n'.join(lines))


if __name__=='__main__':main()
