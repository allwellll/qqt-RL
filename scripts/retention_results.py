"""Paired training-seed screen statistics; never drives a healthy trainer."""
import itertools
import json
from pathlib import Path
import numpy as np
from scripts.retention_screen import declaration,ROOT,read,ARM
from scripts.population_run_state import atomic_json


def metrics(summary):
    route=summary['route'];natural=route['natural_full2400']
    result=dict(natural=natural['delivery_episodes']/natural['games'],
        pickup=natural['pickup_episodes']/natural['games'],
        tactical=summary['tactical_python']['safe_win']/64,
        tactical_self=summary['tactical_python']['self_episodes']/64,
        tactical_bombs=summary['tactical_python']['bombs_per_game'],
        natural_self=natural['self_death']/natural['games'],
        natural_kills=natural.get('causal_kill',0)/max(1,natural.get('attributed_first_control',0)))
    for tier,h in summary['hunter_js'].items():
        result['hunter_'+tier]=h['safe_win']/h['games']
        result['hunter_self_'+tier]=h['self_episodes']/h['games']
        result['hunter_bombs_'+tier]=h['bombs_per_game']
    return result


def paired_interval(values):
    # Enumerate the seed bootstrap rather than inventing independent games.
    samples=[float(np.mean(sample)) for sample in itertools.product(values,repeat=len(values))]
    return np.quantile(samples,[.025,.975]).tolist()


def summarize(base):
    d=declaration();runs=[];complete={};lines=[]
    for spec in d['runs']:
        run=ROOT/spec['namespace'];summaries=[read(run/'authority_je10'/f'g{g:06d}'/'SUMMARY.json') for g in d['points']]
        points=[metrics(s) for s in summaries];tail={k:float(np.mean([p[k] for p in points[-3:]])) for k in points[0]}
        values=[p['natural'] for p in points];auc=float(sum((b-a)*(values[i]+values[i+1])/2 for i,(a,b) in enumerate(zip(d['points'][:-1],d['points'][1:])))/32)
        holdout=metrics(read(run/'authority_retention_test'/'g000032'/'SUMMARY.json'))
        events=[json.loads(line) for line in (run/'formal'/ARM/'events.jsonl').read_text().splitlines()]
        assert len(events)==32 and all(r['finite'] and r['nonzero_update'] for r in events)
        record=dict(**spec,tail=tail,natural_worst=min(values[-3:]),natural_auc=auc,curve=values,holdout=holdout,
            training_ticks=events[-1]['fresh_ticks'],core_final=events[-1]['dagger'],all_fixed_summaries=summaries)
        runs.append(record);complete[spec['arm'],spec['seed']]=record
    diffs=[];directions=0;noninfer=True;worst=[];aucs=[];holdout=[];avoidance=False
    for seed in d['train_seeds']:
        a,b=complete['A0',seed],complete['A1',seed]
        delta={k:b['tail'][k]-a['tail'][k] for k in a['tail']};diffs.append(delta['natural']);directions+=delta['natural']>0
        worst.append(b['natural_worst']-a['natural_worst']);aucs.append(b['natural_auc']-a['natural_auc'])
        holdout.append(b['holdout']['natural']-a['holdout']['natural'])
        for k in ['tactical','hunter_hard','hunter_normal','hunter_easy']:noninfer &= delta[k]>=-.05
        for k in ['tactical_self','natural_self','hunter_self_hard','hunter_self_normal','hunter_self_easy']:noninfer &= delta[k]<=.02
        for k in ['tactical_bombs','hunter_bombs_hard','hunter_bombs_normal','hunter_bombs_easy']:
            avoidance |= b['tail'][k]<.8*a['tail'][k]
    candidate='A1_pending_confirmation' if (np.mean(diffs)>0 and directions>=2 and np.mean(worst)>=0 and np.mean(aucs)>=0 and np.mean(holdout)>=0 and noninfer and not avoidance) else 'A0_or_no_improving_candidate'
    result=dict(passed=True,candidate=candidate,training_ticks=sum(r['training_ticks'] for r in runs),runs=runs,
        paired_natural_tail_delta=diffs,paired_seed_bootstrap95=paired_interval(diffs),positive_seeds=int(directions),
        paired_worst_delta=worst,paired_auc_delta=aucs,paired_holdout_delta=holdout,
        noninferiority_point_estimates=bool(noninfer),avoidance_flag=bool(avoidance),
        limitation='3 independent training seeds: exploratory screening only; discrete wide seed bootstrap is not proof of noninferiority or stable improvement. Holdout consumed once after all dev.')
    lines=['\n## 已完成筛选结果\n',f'实际训练量：{result["training_ticks"]:,} fresh ticks，6个独立run各32代；候选决定：**{candidate}**。3个训练seed仅用于筛选，不能声称稳定显著提升。\n',
        '|臂|训练seed|自然交付曲线0/8/16/24/32|尾均值|尾最差|AUC|test自然|尾Tactical|尾Hunter难/中/易|尾Tactical自伤|',
        '|---|---|---|---:|---:|---:|---:|---:|---|---:|']
    for r in runs:
        t=r['tail'];lines.append(f'|{r["arm"]}|{r["seed"]}|'+ '/'.join(f'{v:.3f}' for v in r['curve'])+f'|{t["natural"]:.3f}|{r["natural_worst"]:.3f}|{r["natural_auc"]:.3f}|{r["holdout"]["natural"]:.3f}|{t["tactical"]:.3f}|'+ '/'.join(f'{t["hunter_"+tier]:.3f}' for tier in ['hard','normal','easy'])+f'|{t["tactical_self"]:.3f}|')
    lines += ['\nA1−A0自然尾均值配对差：'+str(diffs)+'；训练seed bootstrap95%区间：'+str(result['paired_seed_bootstrap95'])+'。',
        f'非劣点估计满足：{noninfer}；规避交战代理标记：{avoidance}。bombs/game不能独自证明交战保持，完整control/kill机会分母及动态对手实际暴露在各点SUMMARY中独立保存。',
        '\n八路线保持原分母：人工near/mid/far仅诊断；库存/恢复为受控fixture；natural为原生完整局。下面保留全部开发点，不择峰。\n',
        '|臂/seed/g|D/S/Drop/Near/Mid/Far/Recovery/Full交付|自然拾/交|首oracle匹配/机会|推进/目标ticks|自然死亡/自伤|',
        '|---|---|---|---|---|---|']
    for r in runs:
        for s in r['all_fixed_summaries']:
            rt=s['route'];n=rt['natural_full2400']
            lines.append(f'|{r["arm"]}/{r["seed"]}/{s["generation"]}|'+ '/'.join(str(v['delivery_episodes']) for v in rt.values())+f'|{n["pickup_episodes"]}/{n["delivery_episodes"]}|{n["first_oracle_matches"]}/{n["first_oracle_actions"]}|{n["progress_ticks"]}/{n["target_ticks"]}|{n["death"]}/{n["self_death"]}|')
    report=ROOT/d.get('experiment_report','reports/retention_experiment_20261019.md')
    body=report.read_text().split('\n## 已完成筛选结果')[0];report.write_text(body+'\n'+'\n'.join(lines)+'\n')
    ledger=ROOT/'docs/training_experiment_ledger.md'
    with ledger.open('a') as f:f.write(f'\nRETENTION-20261019-R1终态：6run×32代，{result["training_ticks"]:,} fresh ticks；候选 `{candidate}`；配对自然尾均值差{diffs}，seed-bootstrap95%{result["paired_seed_bootstrap95"]}。完整逐seed/点报告 `reports/retention_experiment_20261019.md`。未恢复r19、未启动r20，无能力门禁。\n')
    atomic_json(base/'RESULTS.json',result)
    return result
