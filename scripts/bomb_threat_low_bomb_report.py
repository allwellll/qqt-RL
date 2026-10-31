"""Aggregate matched, per-tick native JS evidence; no parameter selection."""
import json,time,statistics,collections
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/low_bomb_diag_u800');R=Path('runs/bomb_threat_v1/replay_best_u800')
def read(p):return json.loads(p.read_text())
def aggregate(rows):
    sumf=lambda key:sum(r[key] for r in rows)
    n=len(rows);opp=sumf('opportunity_ticks');leg=sumf('bomb_legal_ticks');active=sumf('active_both_ticks');alive=sumf('alive_ticks');ticks=sumf('ticks')
    return dict(games=n,total_ticks=ticks,active_both_ticks=active,alive_ticks=alive,bombs_per_game=sumf('bombs_placed')/n,
        median_bombs=statistics.median(r['bombs_placed'] for r in rows),own_half_alive_fraction=sumf('own_half_ticks')/alive,own_cage_near_alive_fraction=sumf('own_cage_near_ticks')/alive,
        enemy_carry_ticks=sumf('carry_enemy_ticks'),return_progress_ticks=sumf('return_progress_ticks'),
        near5_ticks=sumf('near5_active_ticks'),near5_active_fraction=sumf('near5_active_ticks')/active if active else None,
        brick_only_active_fraction=sumf('brick_only_ticks')/active if active else None,connected_active_fraction=sumf('connected_active_ticks')/active if active else None,
        opportunity_ticks=opp,mean_bomb_probability_threat=sumf('opportunity_probability_sum')/opp if opp else None,
        mean_bomb_probability_legal=sumf('legal_bomb_probability_sum')/leg if leg else None,bomb_legal_ticks=leg,
        opportunity_rank_counts=[sum(r['opportunity_rank_counts'][i] for r in rows) for i in range(3)],rank2_mid_opportunity_ticks=sumf('rank2_mid_opportunity_ticks'),
        move_counts=[sum(r['move_counts'][i] for r in rows) for i in range(5)],ability_counts=[sum(r['ability_counts'][i] for r in rows) for i in range(3)],
        distance_histogram=dict(sum((collections.Counter(r['distance_histogram']) for r in rows),collections.Counter())),
        bomb_requests=sumf('bomb_requests'),rejected=sumf('rejected_requests'),rejection_reasons=dict(sum((collections.Counter(r['rejection_reason_counts']) for r in rows),collections.Counter())),
        dead_ticks=sumf('dead_ticks'),trapped_ticks=sumf('trapped_ticks'),stationary_position_ticks=sumf('stationary_position_ticks'),
        first_death_by400=sum(r['first_death_tick'] is not None and r['first_death_tick']<=400 for r in rows))

def main():
    cfg=read(B/'MANIFEST.json');rows=[];files={}
    for e in cfg['games']:
        d=B/f"game_{e['index']:03d}";cm=read(d/'COMPLETE.json');m=read(d/'METRICS.json');match=read(d/'MATCH.json')
        assert cm['status']==match['status']=='MATCH'
        for f,k in [('METRICS.json','metrics_sha256'),('TRACE.json','trace_sha256'),('review.json','review_sha256'),('MATCH.json','match_sha256')]:assert sha(d/f)==cm[k]
        rows.append(m);files[str(d/'COMPLETE.json')]=sha(d/'COMPLETE.json')
    groups={f'P{s}_{g}':aggregate([r for r in rows if r['seat']==s and r['group']==g]) for s in [0,1] for g in ['low','high']}
    categories=['碰不到敌人','能碰到但躲着不打','去打了但argmax选不到放泡','被对手压制','其他']
    low=[r for r in rows if r['group']=='low'];counts={c:sum(r['category']==c for r in low) for c in categories}
    representatives=[min((r for r in low if r['category']==c),key=lambda r:r['index']) for c in categories if counts[c]]
    result=dict(status='COMPLETE',groups=groups,category_counts=counts,category_counts_by_seat={f'P{s}':dict(collections.Counter(r['category'] for r in low if r['seat']==s)) for s in [0,1]},representatives=[dict(index=r['index'],seed=r['seed'],seat=r['seat'],category=r['category']) for r in representatives],games=rows,files=files,protocol_sha256=sha(B/'PROTOCOL.json'),unix=time.time())
    atomic_json(B/'RESULTS.json',result)
    manifest=read(R/'MANIFEST.json');prior=list(manifest['games']);added=[]
    for rep in representatives:
        src=next(e for e in cfg['games'] if e['index']==rep['index']);e=dict(src,rule='少泡诊断·'+rep['category']);manifest['games'].append(e);added.append(e)
        # Reuse standard Review files already verified in the diagnostic; no new physics/inference.
        import shutil
        dest=R/f"game_{rep['index']:03d}";dest.mkdir(exist_ok=True)
        for f in ['review.json','MATCH.json']:
            if not (dest/f).exists():shutil.copyfile(B/f"game_{rep['index']:03d}"/f,dest/f)
            else:
                old=read(dest/'MATCH.json');assert old['actual']==read(B/f"game_{rep['index']:03d}/MATCH.json")['actual']
    if not (R/'MANIFEST_before_low_bomb.json').exists():atomic_json(R/'MANIFEST_before_low_bomb.json',dict(read(R/'MANIFEST.json')))
    # Idempotent rebuild of this amendment only, preserving all prior17 entries.
    base=read(R/'MANIFEST_before_low_bomb.json');base['games']+=added;base['low_bomb_protocol']=str(B/'PROTOCOL.json');atomic_json(R/'MANIFEST.json',base)
    atomic_json(B/'REPRESENTATIVES.json',dict(status='SELECTED',selection='first original index per nonempty low-group category; priority/thresholds frozen before collection',games=added,previous_entries=len(prior),unix=time.time()))
    text=['# u800少泡局诊断（20261029）','', '执行`.hermes/TASK_low_bomb_diag_20261029_2120.md`。仅CPU被动观测，同一候选policy、生产JS stock-window-v1/hard Hunter、greedy、老师关闭；奖励/训练/解码不改。','',
        '## 冻结样本与口径','', '各seat按原index升序，低泡≤5且排除双方零泡，P0仅15符合条件全部纳入，P1前16；高泡≥16各前16，共63独立完整自然局。全部winner/placed/kills及额外终局字段与原fresh256匹配，只有MATCH样本入统计。固定前缀样本不是随机配对因果实验，不能将组差直接归因某个机制。','',
        '路径每tick当前格BFS，wall/brick/pushable/crate作阻挡；另算忽略brick的BFS及含未爆泡动态阻挡的BFS。-1只是本帧该投影未连通，非游戏规则无解，未规划推箱/等待/炸砖/连续碰撞；忽略砖连通只标可能需清砖，不证明能安全炸通。初始及每200tick含终局快照单列。距离只在双活未包裹的可达tick分桶，未连通及对手失控另列。','',
        '概率为实际JS生产模型同一次前向结果的masked softmax，不额外forward/RNG，不拿logit当Q。ability为none/bomb/item，与移动头独立；rank只对合法ability排序，同分按较小index。每tick logits/mask/概率/rank和未mask概率保存。主威胁机会=双方活且未包裹、本方空手且mask允许放泡、敌无无敌、当前位置新泡直射线在blastCap内能到当前敌中心格。只是几何机会，不是安全放置、追击预测或因果击杀；无敌射线另计。己方半场=距己笼中心不大于距敌笼，笼附近曼哈顿≤3；持包时间按真实敌原属包；返程推进tick为同包持续持有时到己笼足迹的BFS距离下降。','',
        '## 同seat低泡与高泡对照','', '|组|局数|泡/局|泡中位|近敌≤5格tick/双活tick|砖隔开/双活|己半场/己活|己笼附近/己活|持敌包tick|返程推进tick|威胁合法tick|威胁P(bomb)|rank1/2/3|rank2且P .2–.5|','|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for key,g in groups.items():
        f=lambda x:'未覆盖' if x is None else f'{x:.4f}'
        text.append(f"|{key}|{g['games']}|{g['bombs_per_game']:.2f}|{g['median_bombs']}|{g['near5_ticks']}/{g['active_both_ticks']}|{f(g['brick_only_active_fraction'])}|{f(g['own_half_alive_fraction'])}|{f(g['own_cage_near_alive_fraction'])}|{g['enemy_carry_ticks']}|{g['return_progress_ticks']}|{g['opportunity_ticks']}|{f(g['mean_bomb_probability_threat'])}|{g['opportunity_rank_counts']}|{g['rank2_mid_opportunity_ticks']}|")
    text+=['','|组|上/下/左/右/停意图|none/bomb/item|实际位置不动tick|请求/拒绝|拒绝原因（可重叠）|距离0–5/6–10/11–20/21+/未连通/非双活|','|---|---|---|---:|---|---|---|']
    for key,g in groups.items():text.append(f"|{key}|{g['move_counts']}|{g['ability_counts']}|{g['stationary_position_ticks']}|{g['bomb_requests']}/{g['rejected']}|{g['rejection_reasons']}|{g['distance_histogram']}|")
    text+=['','## 每局指标与描述性归类','', '归类阈值和互斥优先级在采集前写入`PROTOCOL.json`。被压制优先：首死≤400或死亡+包裹≥25%全tick；砖隔开：双活中≥50%只有去砖才连通；argmax瓶颈：合法射线机会中至少10tick且≥50%为rank2、P∈[.2,.5]；回避：双活≥50%可连通，但近敌≤5格不足10%，己半场≥60%全tick；其余为其他。名称沿用用户任务，但属于可观测模式，不能单凭阈值证明意图/因果。类别总数仅低泡31局，高泡32局是对照。','', '|类别|P0局数|P1局数|合计|首index代表|','|---|---:|---:|---:|---|']
    for c in categories:
        r=next((r for r in representatives if r['category']==c),None);text.append(f"|{c}|{sum(x['category']==c and x['seat']==0 for x in low)}|{sum(x['category']==c and x['seat']==1 for x in low)}|{counts[c]}|{r['index'] if r else '无样本，不造代表'}|")
    text+=['','|组/P/index|tick|泡|可连通双活tick|砖隔开tick|近敌5tick|己半场占全tick|笼近占全tick|持敌包/推进|移动分布|请求/拒绝|合法P(bomb)|威胁P(bomb)/rank1,2,3|首死/死亡tick/包裹tick|分类|','|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---|']
    for r in rows:
        f=lambda v:'无机会' if v is None else f'{v:.4f}'
        text.append(f"|{r['group']}/P{r['seat']}/{r['index']}|{r['ticks']}|{r['bombs_placed']}|{r['connected_active_ticks']}|{r['brick_only_ticks']}|{r['near5_active_ticks']}|{r['fractions']['own_half']:.3f}|{r['fractions']['own_cage_near']:.3f}|{r['carry_enemy_ticks']}/{r['return_progress_ticks']}|{r['move_counts']}|{r['bomb_requests']}/{r['rejected_requests']}|{f(r['mean_bomb_probability_legal'])}|{f(r['mean_bomb_probability_threat'])}/{r['opportunity_rank_counts']}|{r['first_death_tick']}/{r['dead_ticks']}/{r['trapped_ticks']}|{r['category']}|")
    text+=['','## 证据边界、录像与复现','',
        '机会为0的局不能证明argmax瓶颈；高放泡也不能等同打得好，必须保留胜负/击杀/交付与危险后果。被砖隔开、回避、rank2和被压制可共存，本表按冻结优先级只给一个主描述类，原始各指标均保留，未先归因权重不足、不调奖励。完整滑行、长期安全效用和反事实强制放泡不在本任务内，未冒充已完成。','',
        '录像代表来自每个非空低泡类别最早index，添加到同一OSS对象原入口后；重复局明确披露，未采新seed替换。URL： <REPLAY_URL> 。上传/浏览器以`RECEIPT.json`最终验收为准。','',
        '原始每tick `game_XXX/TRACE.json`（columns+rows）、每200tick快照及逐局指标`METRICS.json`、原outcome核对`MATCH.json`、标准Review和SHA `COMPLETE.json`；聚合`RESULTS.json`。复现：`CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.bomb_threat_low_bomb_owner`，已完成局不重跑；`CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.bomb_threat_low_bomb_report`重新验SHA聚合。4个CPU worker，避开训练物理核，未启动GPU/训练owner、未向训练worker发信号，未push/发布模型/BEST/改网页。']
    Path('reports/bomb_threat_v1_low_bomb_diag_20261029.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(groups=groups,category_counts=counts,representatives=result['representatives']),ensure_ascii=False))
if __name__=='__main__':main()
