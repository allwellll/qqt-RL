"""Recompute the fixed Bank endpoints' phase report, without new rollouts."""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from scripts.population_run_state import atomic_json, sha
from scripts.passive_evaluation_metrics import summarize_cohort
from scripts.campaign_remaining_cpu import linked_funnel

BASE = Path('runs/authorized_campaign_20261026/bank_ratio_ab_v1')
PROTOCOL = Path('runs/authorized_campaign_20261026/protocols/bank_ratio_eval_v1.json')
REPORT = Path('reports/bank_ratio_ab_funnel_20261027.md')


def rate(a, b):
    return dict(numerator=int(a), denominator=int(b), value=float(a/b) if b else None)


def median(values):
    return float(np.median(values)) if values else None


def summarize(games):
    pooled = summarize_cohort(games)
    linked = linked_funnel(games)
    windows = {}
    for kind in ('death', 'native_syrup'):
        all_windows = [w for g in games for w in g['window_funnel']['windows'] if w['kind']==kind]
        closed = [w for w in all_windows if not w['censored'] and w['start_tick'] is not None]
        first = [min(w['stock_pickup_ticks'])-w['start_tick'] for w in closed if w['stock_pickup_ticks']]
        windows[kind] = dict(count=len(all_windows),closed=len(closed),censored=len(all_windows)-len(closed),
            attribution=dict(Counter(w.get('cause_attribution','unknown') for w in all_windows)),
            pickup_per_closed_window=rate(sum(bool(w['stock_pickup_ticks']) for w in closed),len(closed)),
            first_pickup_latency_ticks_median=median(first), latency_observations=len(first),
            same_custody_linked=linked[kind])
    custody = [e for g in games for e in g['carry_episodes']['episodes']]
    closed = [e for e in custody if not e['outcome'].startswith('censored_')]
    delivered = [e for e in custody if e['outcome']=='delivered']
    outcomes = Counter(e['outcome'] for e in custody)
    timeout = sum(g['termination'] in ('timeout','horizon','time_limit','timeout_horizon') for g in games)
    # Native runner names finite-horizon terminal explicitly; preserve the
    # complete histogram so an unfamiliar label is never silently a timeout.
    histogram = Counter(g['termination'] for g in games)
    return dict(games=len(games), combat=dict(windows=windows,
        own_deaths_per_game=pooled['own_deaths_per_match'],timeout_games=timeout,
        termination_counts=dict(histogram)),
        pickup=dict(free_alive_stock=pooled['pickup_event_counts'].get('enemy_stock_while_enemy_free',0),
                    sources=pooled['pickup_event_counts']),
        return_chain=dict(acquisitions=len(custody),outcomes=dict(outcomes),
            same_chain_delivery_per_closed=rate(len(delivered),len(closed)),
            delivery_per_all_acquisitions=rate(len(delivered),len(custody)),
            delivered_return_ticks_median=median([e['duration_ticks'] for e in delivered]),
            censor_count=sum(v for k,v in outcomes.items() if k.startswith('censored_'))),
        full_game_delivery=pooled['at_least_one_delivery_rate'],
        phases=pooled['phases'],
        item_by_type={kind:{phase:dict(requests=None,actual_placements=None,self_contact=None,enemy_contact=None,
            reason='Frozen telemetry has no held item type or consumed-event IDs; pooled hits and enemy effect proxies cannot identify all typed chains')
            for phase in ('combat_free_enemy','controlled_window','carrying')} for kind in ('banana','slow_glue')},
        critic_return_EV=None,critic_return_bias=None,
        critic_missing_reason='No per-tick V or terminal MC targets in this frozen endpoint evaluation',
        source_protocol='funnel_complete_v1 accounting; no new games; old native JAX rule3 only')


def primitives(game):
    result = {}
    for kind in ('death','native_syrup'):
        w=[x for x in game['window_funnel']['windows'] if x['kind']==kind and not x['censored'] and x['start_tick'] is not None]
        result[kind+'_window_pickup']=(sum(bool(x['stock_pickup_ticks']) for x in w),len(w))
    custody=[e for e in game['carry_episodes']['episodes'] if not e['outcome'].startswith('censored_')]
    result['same_chain_delivery']=(sum(e['outcome']=='delivered' for e in custody),len(custody))
    result['natural_delivery']=(game['full_game']['at_least_one_delivery']['numerator'],1)
    result['deaths_per_game']=(game['full_game']['own_deaths']['count'],1)
    return result


def paired_comparison(parent, candidate, repetitions=2000):
    if set(parent)!=set(candidate):
        raise ValueError('Matched source seed set differs')
    seeds=sorted(parent)
    p=[primitives(parent[s]) for s in seeds];c=[primitives(candidate[s]) for s in seeds]
    rng=np.random.default_rng(20261027)
    indices=rng.integers(0,len(seeds),(repetitions,len(seeds)))
    result={}
    for key in p[0]:
        x=np.asarray([r[key] for r in p],float);y=np.asarray([r[key] for r in c],float)
        def ratio(z):
            total=z.sum(0)
            return total[0]/total[1] if total[1] else None
        px,cx=ratio(x),ratio(y)
        xr=x[indices].sum(1);yr=y[indices].sum(1)
        valid=(xr[:,1]>0)&(yr[:,1]>0)
        delta=yr[valid,0]/yr[valid,1]-xr[valid,0]/xr[valid,1]
        result[key]=dict(parent=px,candidate=cx,paired_difference=cx-px if px is not None and cx is not None else None,
            source_game_paired_bootstrap_CI=np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
            valid_bootstrap_replicates=len(delta),source_games=len(seeds))
    return result


def main():
    protocol=json.loads(PROTOCOL.read_text());sources=[];cohorts={};rows=[]
    for name,spec in protocol['conditions'].items():
        folder=Path(protocol['namespace'])/name
        done=json.loads((folder/'COMPLETE.json').read_text());source=folder/'SUMMARY.json'
        assert done['passed'] and sha(source)==done['summary_sha256']
        sources.append(dict(path=str(source),sha256=sha(source)))
        value=json.loads(source.read_text())
        for row in value['rows']:
            if len(row['episodes'])!=spec['games_per_mode']:raise ValueError('Episode budget mismatch')
            episodes=row['episodes']
            if any(e['artificial_truncation'] for e in episodes):raise ValueError('Artificially truncated game')
            if len({e['seed'] for e in episodes})!=len(episodes):raise ValueError('Duplicate source seed')
            games=[e['passive_metrics'] for e in episodes]
            mode=row['action_mode'];cohorts[name,mode]={e['seed']:e['passive_metrics'] for e in episodes}
            arm='parent' if name.startswith('parent_') else name.split('_s')[0]
            seed=spec['run_seed'] if arm=='parent' else 2026102751+(spec['run_seed']-2026103151)
            rows.append(dict(condition=name,arm=arm,paired_parent_seed=seed,split=spec['split'],mode=mode,
                             checkpoint=spec['checkpoint'],metrics=summarize(games)))
    comparisons=[]
    for split in ('dev','holdout'):
        for mode in protocol['action_modes']:
            for i in range(3):
                parent=f'parent_s{2026102751+i}_u064_{split}'
                control=f'control_s{2026103151+i}_u320_{split}'
                reduced=f'reduced_s{2026103151+i}_u320_{split}'
                for a,b in ((parent,control),(parent,reduced),(control,reduced)):
                    comparisons.append(dict(split=split,mode=mode,parent=a,candidate=b,
                        paired_seed=i,metrics=paired_comparison(cohorts[a,mode],cohorts[b,mode])))
    pooled=[]
    for split in ('dev','holdout'):
        for mode in protocol['action_modes']:
            for arm in ('parent','control','reduced'):
                chosen=[r for r in rows if r['split']==split and r['mode']==mode and r['arm']==arm]
                games=[g for r in chosen for g in cohorts[r['condition'],mode].values()]
                pooled.append(dict(split=split,mode=mode,arm=arm,training_seeds=3,metrics=summarize(games)))
    missing=dict(JS_Hunter=None,JS_Hunter_reason='Not included in frozen BANK evaluation protocol; no extra games requested',
        Python_Tactical=None,Python_Tactical_reason='Not included in frozen BANK endpoint protocol',
        typed_items=None,typed_items_reason='Frozen old telemetry lacks type/actual native release event identity',
        complete_banana_self_slide_chain=None,complete_banana_self_slide_chain_reason='Cannot reconstruct self-placement→self-contact→full slide→same-chain delivery from this trace ABI',
        complete_glue_net_benefit=None,complete_glue_net_benefit_reason='No matched counterfactual or full pursuit delay chain',
        preterminal_vs_settlement_item_metrics=None,preterminal_vs_settlement_item_metrics_reason='Per-phase summary includes ledger tail; first-terminal event not exported as typed item-stage boundary',
        critic_EV_bias=None,critic_EV_bias_reason='No per-tick critic predictions/terminal MC targets',
        own_alive_actionable=None,own_alive_actionable_reason='This old passive ABI lacks own_alive_before; no action inferred from no death')
    result=dict(status='COMPLETE_WITH_MISSING_DATA',sources=sources,protocol_sha256=sha(PROTOCOL),
        models=9,evaluation_conditions=18,cohorts=len(rows),total_games=sum(r['metrics']['games'] for r in rows),
        rows=rows,pooled=pooled,paired_comparisons=comparisons,missing=missing,
        item_warning='actual_placements in inherited pooled phases is request+pre-legality inference, not actual consumption evidence; report as placement proxy',
        inference='CI resamples paired source games within each training seed. Three seed point differences retained; no pooled source-only CI claimed as training-seed uncertainty.',
        banana_primary='self acceleration while carrying, delivery ticks; no enemy-kill objective',unix=time.time())
    target=BASE/'FUNNEL_RESULT_20261027.json';atomic_json(target,result)
    text=['# BANK-RATIO 分环节固定终点评估（20261027）','',
        f'六个u320与三个各自父点；18条件、36个split/动作模式 cohort，共{result["total_games"]}自然完整局。JAX旧规则/rule3，老师关闭；不混JS、Python Tactical或新规则。dev64、holdout304局/模型/模式；所有固定终点，无择峰。',
        '', '分子分母先汇总；死亡与原生糖浆窗分开；同链结局由原passive ID关联。窗口率只用已闭合窗，censor另报。香蕉主语义是持包自踩加速，不以敌滑倒或杀敌替代。', '',
        '| split | 模式 | 组（3seed） | 交付局/局 | 自死/局 | 死亡窗取包/闭合窗 | 糖浆窗取包/闭合窗 | 同链交付/闭合链 | censor | 交付返程中位tick |',
        '|---|---|---|---|---|---|---|---|---|---|']
    def fraction(v):return f'{v["numerator"]}/{v["denominator"]}'
    for r in pooled:
        m=r['metrics'];w=m['combat']['windows'];c=m['return_chain']
        text.append('| '+' | '.join(map(str,[r['split'],r['mode'],r['arm'],fraction(m['full_game_delivery']),fraction(m['combat']['own_deaths_per_game']),fraction(w['death']['pickup_per_closed_window']),fraction(w['native_syrup']['pickup_per_closed_window']),fraction(c['same_chain_delivery_per_closed']),c['censor_count'],c['delivered_return_ticks_median']]))+' |')
    text += ['', '逐seed比较：reduced−control（概率.125−.25），区间为配对来源局bootstrap95%；训练seed不确定性须看三seed差值，不能用来源局CI替代。', '',
        '| split | 模式 | seed尾号 | 自然交付差 | 来源局CI | 同链交付差 | 自死/局差 |',
        '|---|---|---|---|---|---|---|']
    for r in comparisons:
        if not r['parent'].startswith('control_'):continue
        a=r['metrics']['natural_delivery'];c=r['metrics']['same_chain_delivery'];d=r['metrics']['deaths_per_game']
        text.append(f'| {r["split"]} | {r["mode"]} | {r["paired_seed"]+1} | {a["paired_difference"]} | {a["source_game_paired_bootstrap_CI"]} | {c["paired_difference"]} | {d["paired_difference"]} |')
    text+=['','每模型逐环节、归因窗口数、自由存活取包、窗首取包延时、全部持包结局、各阶段道具请求/放置代理/接触计数及全部父点差值见可重算JSON。组汇总不用于掩盖seed尾部。', '',
        '道具限制：既有`actual_placements`由请求+前态合法性推断，本报告明确降为放置代理；未保存真实消费ID与held类型，香蕉/胶各阶段实际放置、自吃完整链分别为null，不能把0或命中proxy当真值。敌滑倒proxy只作次要参考。Critic EV/偏差、可行动时间与首次terminal前/账本尾段分阶段指标亦为null并附原因。没有为补齐报告新增GPU或对局。', '',
        f'原始输入与SHA、全部计数和缺失原因：`{target}`；报告脚本`{__file__}`。冻结BANK协议不含JS Hunter/Python Tactical，本报告不伪造这些结果。','']
    REPORT.write_text('\n'.join(text))
    atomic_json(BASE/'FUNNEL_REPORT_COMPLETE.json',dict(passed=True,report=str(REPORT),report_sha256=sha(REPORT),
        raw=str(target),raw_sha256=sha(target),source_games=result['total_games'],new_rollouts=0,new_GPU_tasks=0,unix=time.time()))
    atomic_json(BASE/'notifications'/f'{time.time_ns()}_FUNNEL_REPORT_COMPLETE.json',dict(status='COMPLETE_WITH_MISSING_DATA',report=str(REPORT),games=result['total_games'],new_rollouts=0))
    print(json.dumps(dict(report=str(REPORT),games=result['total_games'],cohorts=len(rows))))


if __name__=='__main__':main()
