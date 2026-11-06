"""Regenerate auditable region/attack results and conservative paired gates."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json

B = Path('runs/combat_regions_independent_20261104')
OUT = B / 'evaluation'
REPORT = Path('reports/combat_regions_independent_20261104.md')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def measures(row):
    groups = list(row['continuous_attack']['groups'].values())
    total = lambda key: sum(g[key] for g in groups)
    custody = row['primary_custody']['groups']['all']
    gaps = [gap for g in groups for gap in g['eligible_gaps']]
    return dict(games=1, contact_games=int(row['continuous_attack']['contact']),
        wins=int(row['winner'] == row['seat']), deaths=total('deaths'), self_deaths=total('self_deaths'),
        causal_controls=total('causal_controls'), causal_kills=total('causal_kills'),
        contact_repeat30=total('contact_repeat30'), placed=total('placed'),
        contact_placed=total('contact_placed'), route_cut_proxy=total('route_cut_proxy'),
        repeated_attack_controls=total('repeated_attack_controls'), repeated_attack_kills=total('repeated_attack_kills'),
        legal_capacity_ticks=total('legal_capacity_ticks'), eligible_gap_median=float(np.median(gaps)) if gaps else None,
        acquisitions=custody['acquisitions'], deliveries=custody['same_chain_deliveries'],
        carry_failures=custody['carrying_deaths'] + custody['controlled_drops'])


def key(row):
    return row['seed'], row['seat'], row['mode']


def paired(parent, candidate):
    left = {key(r): r for r in parent}; right = {key(r): r for r in candidate}
    common = sorted(left.keys() & right.keys())
    if not common:
        return dict(status='NO_PAIRED_EPISODES', pairs=0)
    clusters = defaultdict(list)
    for k in common:
        clusters[k[0]].append((measures(left[k]), measures(right[k])))
    metrics = ['wins', 'self_deaths', 'causal_kills', 'contact_repeat30', 'repeated_attack_kills', 'deliveries', 'carry_failures']
    result = dict(status='PAIRED_DESCRIPTIVE', pairs=len(common), independent_seeds=len(clusters), intervals={})
    rng = np.random.default_rng(2026543110)
    ids = rng.integers(0, len(clusters), (2000, len(clusters)))
    values = list(clusters.values())
    for name in metrics:
        # Cluster by environment seed so two inference modes and both physical
        # seats are not silently treated as independent environments.
        delta = np.asarray([sum(b[name] - a[name] for a, b in block) for block in values], float)
        size = np.asarray([len(block) for block in values], float)
        bootstrap = delta[ids].sum(-1) / size[ids].sum(-1)
        result['intervals'][name] = dict(mean=float(delta.sum()/size.sum()),
            lower95=float(np.quantile(bootstrap, .025)), upper95=float(np.quantile(bootstrap, .975)))
    # Causal kills per game with actual contact, with denominators re-sampled.
    blocks = np.asarray([[sum(a['causal_kills'] for a,b in block),sum(a['contact_games'] for a,b in block),
                          sum(b['causal_kills'] for a,b in block),sum(b['contact_games'] for a,b in block)] for block in values], float)
    sampled = blocks[ids].sum(1)
    valid = (sampled[:,1] > 0) & (sampled[:,3] > 0)
    sums = blocks.sum(0)
    if valid.any() and sums[1] and sums[3]:
        delta = sampled[valid,2]/sampled[valid,3] - sampled[valid,0]/sampled[valid,1]
        result['intervals']['causal_kills_per_contact_game'] = dict(mean=float(sums[2]/sums[3]-sums[0]/sums[1]),
            lower95=float(np.quantile(delta,.025)),upper95=float(np.quantile(delta,.975)))
    result['contact_games'] = [int(sums[1]),int(sums[3])]
    return result


def main():
    models = defaultdict(lambda: defaultdict(list))
    for file in OUT.glob('*/**/ACCEPTED.json'):
        proof = read(file); job = proof['job']; complete = read(file.parent/'COMPLETE.json')
        models[job['label']][job['condition']].extend(complete['results'])
    protocol = read(OUT/'PROTOCOL.json', {})
    output = dict(status='PARTIAL', capability_improvement='NOT_ESTABLISHED', models={}, comparisons={})
    lines = ['# 独立多位置对拼与连续进攻优化（20261104）', '',
        '消费回执：`.hermes/TASK_CONSUMED.json`；实验：`runs/combat_regions_independent_20261104`。原工作区仅只读，未 push、未改生产 `web/sim.js`，风险训练 GPU3/GPU5 保留。', '',
        '课程：70% 原自然配方 + 30% 合法区域起点；全体期望 52.5% 干净自然、17.5% 真实 Bank、30% 区域起点。区域起点排除笼区/推箱格，使用原始地形，保留本侧原生复活点；不旋转地图，不增加泡量或跑圈奖励。G 的 move5/ability3 与 mode token 保留；50% 轻量安全拦截继续留在独立风险线。', '',
        '原 336 起点中 24 行涉及笼区，新 train/dev/holdout 各 336 行均排除此问题，双席位/距离/相对方向平衡。中心没有合法 pair，明确缺覆盖。dev 与 holdout 环境 seed 分离；有限地图几何可能重合，不声称地图泛化。', '',
        'CPU 初态与 8 局 ×32 tick 真实奖励转移通过，所有值 finite，无初态死亡、虚构取包或额外历史奖。旧 R1 折扣势函数本身有每 tick 最高约 0.00006 的静止项，已单列；额外未解释收益为零，R3 历史支付为零。该项未被冒称课程新收益。', '',
        '连续进攻观察器通过 200 tick 完整物理/RNG 轨迹一致性验证。合法机会排除携包/失控/满容量，并计入本 tick 引信到期释放的容量；死亡断链。封路为新增泡格减少可达区域的图论代理，不能冒称强制控制。续泡后的控制/击杀为攻击链时序关联；严格 unique-other 归因与逐泡因果归因分开。', '',
        '固定评估：冻结 heavy-v6 对手、双席位、greedy/categorical；dev 区域每区每模式12局、自然32个seed双席双模式，holdout 每区每模式48局。80/320/640/960/1280新增更新评估，640及末点完整holdout；训练不等待短评。低样本、无接敌、未完成单列，不以训练loss判能力。', '']
    for name in ('CPU_VALIDATION.json','CPU_FRESH_RESTORE.json','PREPARED.json','FIRST_FORMAL_UPDATE.json'):
        value = read(B/name)
        if value:
            lines.append('收据 `'+name+'`：`'+str(value.get('status'))+'`。')
    lines += ['', '|模型/条件|起始区域|模式/席位|局/接敌|因果控/杀|死/自死|接敌续泡≤30|封路代理|取包/同链交付|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for label, conditions in sorted(models.items()):
        output['models'][label] = {}
        for condition, rows in sorted(conditions.items()):
            buckets = defaultdict(list)
            for row in rows:
                buckets[(row['start_region'], row['mode'], row['seat'])].append(row)
            summaries=[]
            for (region,mode,seat), bucket in sorted(buckets.items()):
                vals=[measures(r) for r in bucket]
                totals={k:sum(v[k] for v in vals) for k in vals[0] if k!='eligible_gap_median'}
                summaries.append(dict(region=region,mode=mode,seat=seat,**totals))
                t=totals
                lines.append(f"|{label}/{condition}|{region}|{mode}/P{seat}|{t['games']}/{t['contact_games']}|{t['causal_controls']}/{t['causal_kills']}|{t['deaths']}/{t['self_deaths']}|{t['contact_repeat30']}|{t['route_cut_proxy']}|{t['acquisitions']}/{t['deliveries']}|")
            output['models'][label][condition] = dict(episodes=len(rows), expected=len(protocol.get('conditions',{}).get(condition,[])), summaries=summaries,
                no_contact=[{k:r[k] for k in ('seed','seat','mode','start_region')} for r in rows if not r['continuous_attack']['contact']])
    parent = models.get('parent',{})
    # Weakest regions are chosen on the parent dev set, before holdout results.
    parent_dev = parent.get('dev_region',[])
    rates={}
    for region in set(r['start_region'] for r in parent_dev):
        vals=[measures(r) for r in parent_dev if r['start_region']==region]
        contacts=sum(v['contact_games'] for v in vals)
        if contacts>=8:
            rates[region]=sum(v['causal_kills'] for v in vals)/contacts
    weakest=[r for r,v in rates.items() if v==min(rates.values())] if rates else []
    parent_dev_complete=len(parent_dev)==len(protocol.get('conditions',{}).get('dev_region',[])) and bool(parent_dev)
    output['parent_dev_weakest_regions']=weakest
    output['weakest_region_selection_frozen']=parent_dev_complete
    for label, conditions in models.items():
        if label=='parent':continue
        output['comparisons'][label]={}
        for condition, rows in conditions.items():
            baseline=parent.get(condition,[])
            comparisons={region:paired([r for r in baseline if r['start_region']==region],
                [r for r in rows if r['start_region']==region]) for region in set(r['start_region'] for r in baseline)}
            output['comparisons'][label][condition]=comparisons
        checks={}
        required=('dev_region','dev_natural','holdout_region')
        checks['complete_frozen_pairs']=all(len(conditions.get(c,[]))==len(protocol.get('conditions',{}).get(c,[])) and
            len(parent.get(c,[]))==len(protocol.get('conditions',{}).get(c,[])) and bool(conditions.get(c,[])) for c in required)
        comparisons=output['comparisons'][label]
        holdout=comparisons.get('holdout_region',{})
        def interval_ok(comp,metric,side,threshold,strict=False):
            interval=comp.get('intervals',{}).get(metric)
            if not interval:return False
            value=interval[side]
            return (value>threshold if strict else value>=threshold) if side=='lower95' else value<=threshold
        checks['all_weakest_regions_improve']=parent_dev_complete and bool(weakest) and all(
            min(holdout.get(r,{}).get('contact_games',[0,0]))>=8 and
            interval_ok(holdout.get(r,{}),'causal_kills_per_contact_game','lower95',0,True) for r in weakest)
        br=holdout.get('bottom_right',{})
        checks['bottom_right_no_material_regression']=interval_ok(br,'wins','lower95',-.10) and interval_ok(br,'self_deaths','upper95',.10)
        natural=comparisons.get('dev_natural',{}).get('natural',{})
        checks['natural_no_material_regression']=interval_ok(natural,'wins','lower95',-.10) and interval_ok(natural,'self_deaths','upper95',.10)
        region_all=paired(parent.get('holdout_region',[]),conditions.get('holdout_region',[]))
        checks['real_continuous_attack_conversion']=interval_ok(region_all,'repeated_attack_kills','lower95',0,True)
        pv=[measures(r) for r in parent.get('dev_natural',[])];cv=[measures(r) for r in conditions.get('dev_natural',[])]
        pa=sum(v['acquisitions'] for v in pv);ca=sum(v['acquisitions'] for v in cv)
        # Fixed-seed deliveries and carry failures remain whole-game outcomes.
        # A candidate with no real pickup exposure cannot pass transport safety.
        checks['transport_has_exposure']=min(pa,ca)>=8
        checks['transport_no_material_regression']=checks['transport_has_exposure'] and interval_ok(natural,'deliveries','lower95',-.10) and interval_ok(natural,'carry_failures','upper95',.10)
        verdict='PASS_FROZEN_CAPABILITY_GATE' if all(checks.values()) else 'NOT_ESTABLISHED'
        output['comparisons'][label]['acceptance']=dict(status=verdict,checks=checks,
            thresholds='predeclared protocol; cluster bootstrap by environment seed; temporal chain association is not per-bubble causality')
        if verdict=='PASS_FROZEN_CAPABILITY_GATE':output['capability_improvement']='ESTABLISHED_WITHIN_FROZEN_PROTOCOL'
    if (OUT/'COMPLETE.json').exists():output['status']='COMPLETE'
    conclusion='当前能力结论：**尚未证明最弱区域改善或自然/运包无退步。** 需完整冻结配对结果后判定；泡数、续泡间隔或驻留均匀均不能单独通过验收。' if output['capability_improvement']=='NOT_ESTABLISHED' else '当前能力结论：冻结配对协议验收通过；具体检查项、置信区间与样本量见 `evaluation/SUMMARY.json`，不扩展为逐泡因果或新地图泛化结论。'
    lines += ['',conclusion,'',
        '原始逐局 RESULT、SOURCE_EVENTS、完整 review、模型/地图/对手与状态哈希保留在 `evaluation/`。技术样本固定选择左上/右下前缀，左上2400tick无放泡，右下951tick获胜并完成一次真实取包同链交付；这只是父点案例，不能当训练改善。', '']
    for label,value in output['comparisons'].items():
        lines.append('模型 `'+label+'` 验收：`'+json.dumps(value['acceptance'],ensure_ascii=False)+'`。')
    lines += ['', '完整轨迹简图录像：`reports/combat_regions_replays_20261104/top_left_parent.gif`、`reports/combat_regions_replays_20261104/bottom_right_parent.gif`。未发布网页。', '']
    atomic_json(OUT/'SUMMARY.json',output)
    REPORT.write_text('\n'.join(lines))


if __name__=='__main__':main()
