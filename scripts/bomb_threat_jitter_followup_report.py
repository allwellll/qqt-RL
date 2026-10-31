"""Passive follow-up: real original paths, finite full-game hold diagnostics."""
import json,math,time,statistics
from pathlib import Path
from collections import Counter
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/jitter_followup_u800')
def read(p):return json.loads(p.read_text())
def rate(rows,move='move_ticks',rev='lr_reversals',ints='reversal_intervals'):
    steps=sum(r[move] for r in rows);events=sum(r[rev] for r in rows);intervals=[v for r in rows for v in r[ints]]
    return dict(move_ticks=steps,reversals=events,per100_move_ticks=100*events/steps if steps else None,median_event_interval=statistics.median(intervals) if intervals else None,interval_histogram=dict(Counter(intervals)))
def summary(rows):
    w=sum(r['winner']==r['seat'] for r in rows);l=sum(r['winner'] not in [None,-1,r['seat']] for r in rows)
    return dict(games=len(rows),win=w,draw=len(rows)-w-l,loss=l,win_minus_loss=w-l,causal_kills=sum(r['causal_kills'][r['seat']] for r in rows),bombs_per_game=sum(r['placed'][r['seat']] for r in rows)/len(rows),unilateral_zero_bomb_games=sum(r['placed'][r['seat']]==0 and r['placed'][1-r['seat']]>0 for r in rows),all_zero_own_games=sum(r['placed'][r['seat']]==0 for r in rows),self_deaths=sum(r['self_deaths'][r['seat']] for r in rows),deaths=sum(r['deaths'][r['seat']] for r in rows),carry_stall_ticks=sum(r['carry_stall_ticks'] for r in rows),carry_ticks=sum(r['carry_ticks'] for r in rows),oscillation_ticks=sum(r['oscillation_ticks'] for r in rows),reversals=rate(rows))
def main():
    cfg=read(B/'PROTOCOL.json');rv=read(B/'REVERSIBILITY.json');danger=read(B/'DANGER_AND_HUNTER.json');native=read(B/'native/COMPLETE.json');arms={};games={};files={}
    for arm in cfg['inference_arms']:
        rr=[];stats=Counter()
        for start in range(0,64,8):
            p=B/arm/f'shard_{start:03d}.json.diagnostic.json';x=read(p);assert x['model_sha256']==cfg['model']['model_sha256'] and len(x['episodes'])==8
            assert {r['seed'] for r in x['episodes']}==set(range(cfg['seed_start']+start,cfg['seed_start']+start+8));assert all(r['done'] and 0<r['ticks']<=2400 for r in x['episodes'])
            rr+=x['episodes'];stats.update({k:v for k,v in x['hold'].items() if k!='period'});files[str(p)]=sha(p)
            for suffix in ['', '.outcome.json','.trace.json']:
                q=B/arm/f'shard_{start:03d}.json{suffix}';files[str(q)]=sha(q)
        rr.sort(key=lambda r:r['seed']);assert len(rr)==64 and len({r['seed'] for r in rr})==64 and all(r['seat']==(r['seed']-cfg['seed_start'])%2 for r in rr)
        games[arm]=rr;arms[arm]=dict(overall=summary(rr),by_seat={f'P{p}':summary([r for r in rr if r['seat']==p]) for p in (0,1)},hold=dict(stats))
    paired={}
    for arm in ['H2','H4']:
        b,a=arms['H0']['overall'],arms[arm]['overall'];diff={k:a[k]-b[k] for k in ['win_minus_loss','causal_kills','unilateral_zero_bomb_games','carry_stall_ticks']};ok=diff['win_minus_loss']>=5 and diff['causal_kills']>=0 and diff['unilateral_zero_bomb_games']<=0
        pairs=[]
        for x,y in zip(games['H0'],games[arm]):
            assert (x['seed'],x['seat'])==(y['seed'],y['seat']);score=lambda r:1 if r['winner']==r['seat'] else 0 if r['winner'] in (None,-1) else -1
            pairs.append(dict(seed=x['seed'],seat=x['seat'],H0_score=score(x),candidate_score=score(y),delta=score(y)-score(x),H0_kills=x['causal_kills'][x['seat']],candidate_kills=y['causal_kills'][y['seat']]))
        paired[arm]=dict(differences=diff,task_observation_threshold_met=ok,pairs=pairs,not_promotion=True)
    jr=read(Path(cfg['jitter_source'])/'RESULTS.json');js_p0=[g for g in games['H0'] if g['seat']==0]
    comparisons=dict(jax_native_categorical_P0=rate(native['episodes']),JS_dev_argmax_P0=rate(js_p0),JS_dev_Hunter_vs_P0=rate(js_p0,'enemy_move_ticks','enemy_lr_reversals','enemy_reversal_intervals'))
    low=[r for r in jr['games'] if r['group']=='low'];high=[r for r in jr['games'] if r['group']=='high']
    for group,rows in [('original_P0_low',low),('original_P0_high',high)]:
        # Recompute exact denominator from stored original before-state audit, not overall game tick.
        movement=0;intervals=[]
        for r in rows:
            d=read(Path(cfg['jitter_source'])/f"game_{r['index']:03d}.json");movement+=sum(v['controlled'] and v['move']<4 for v in d['rows']);intervals+=d['reversals']['intervals']
        ev=sum(r['direct_lr_reversals'] for r in rows);comparisons[group]=dict(games=len(rows),move_ticks=movement,reversals=ev,per100_move_ticks=100*ev/movement,median_event_interval=statistics.median(intervals))
    dg={}
    for group in ['low','high']:
        rr=[x for x in danger['danger'] if x['group']==group];dg[group]=dict(oscillation_ticks=sum(r['oscillation_ticks'] for r in rr),type_ticks=dict(sum((Counter(r['type_ticks']) for r in rr),Counter())),enemy_relative_sign_ticks=dict(sum((Counter(r['enemy_relative_sign_ticks']) for r in rr),Counter())))
    results=dict(status='COMPLETE',arms=arms,paired=paired,reversal_comparisons=comparisons,danger=dg,files=files,unix=time.time());atomic_json(B/'RESULTS.json',results)
    text=['# 持包可逆性、振荡定位与动作保持对照（20261029）','',
        '执行`.hermes/TASK_jitter_stall_followup_20261029_2300.md`。CPU诊断，七路训练及GPU2评估继续，不改奖励、网页解码、BEST或门禁。协议`runs/bomb_threat_v1/jitter_followup_u800/PROTOCOL.json`在采集前冻结。', '',
        f"冻结候选u800 policy SHA `{cfg['model']['policy_sha256']}`；JS JSON SHA `{cfg['model']['model_sha256']}`。H0/H2/H4每组完整64局，同dev seed2026122900–2026122963，P0/P1各32，生产JS stock-window-v1，带砖2400tick或自然提前终局，hard Hunter，argmax，老师关闭。只诊断，不择峰或晋升。", '',
        '持包可逆性：真实到达路径通常可退回，不代表该段起点就是己笼。这六局取包前300tick均不从己笼出发，出生时到己笼的静态投影已经不连通；六局取包时都还能连回窗口起点，未发现推箱移动封路。不能把上一报告BFS−1解释成角色走过的路永久封死。', '',
        '|index|出生连续位置/己笼BFS|取包tick/前窗口起点|取包后到原窗口起点BFS|路径新阻挡及责任|持包终段去向|','|---|---|---|---|---|---|']
    for r in sorted(rv['games'],key=lambda r:r['index']):
        for e in r['pickups']:
            z=e['snapshots'][-1];blocks=e['blocked_arrival_cells_at_pickup'];description='无' if not blocks else '; '.join(f"格{x['cell']}={x['kind']}；变化{x['changes']}" for x in blocks)
            text.append(f"|{r['index']}|{r['birth_pos']}/{r['birth_home_static']}|{e['pickup_tick']}/{e['path_start']}|{e['reverse_arrival_graph_distance']}|{description}|{z['pos']}，原窗口起点距{z['reverse_arrival_graph_distance']}，己笼静态{z['home_static']}|")
    text+=['','6局完整原动作从出生重放，记录所有格与连续坐标、每次wall/brick/box/bomb类型变化、取包前300tick路径（出生不足300则全取）以及取包后每200tick+终局快照。无推箱位移，因此无需臆测推箱责任；index42格56在982tick出现敌方owner1未爆泡，上次经过为966tick，取包1002tick时原访问格有泡，但替代路线仍连回窗口起点（12格）。其余5局取包时原访问格无当前障碍。持包后部分时刻到窗口起点暂时不通要按泡变化看，不能统称砖永久封路。`reversibility_INDEX.json`与`REVERSIBILITY.json`含封路时刻、owner、所有变化和轨迹；`PATH_OBSTRUCTIONS.json`逐局列原路径上所有新出现阻挡及本方/对手责任，初始砖墙归初始地图，不归后来角色造成。', '',
        '结论措辞：六局取包时均为“原到达路径起点仍连通，但己笼并非这条到达路径的起点”。不把后续全部时间都写成原路可返；部分未爆泡会暂时阻断，后续爆掉后又可恢复。六局出生至取包均未访问己笼、静态投影从未连到己笼；取包前300tick路径无大于1.2格的瞬移步，取包后无新增wall/brick/push_box、无推箱位移。不能无条件写“原路回城可达”，也不支持本方推箱封住原路。BFS本身是格点连接，不保证连续转角可行、时序火焰安全或敌方响应；报告不宣称物理无解。', '',
        '合法脱困动作：Actor只有移动5类×ability none/bomb/item3类，无主动丢包动作；`_bunDrop`仅死亡路径调用。原生糖浆被接触终结时可转移给敌方，属于丧失控制，不是可选安全丢包。持包禁止放泡；若有香蕉/胶库存，item仍可能合法，香蕉自踩滑行或推箱/等待敌清砖可能改变局面，但六局审计没有证明这些能安全回城。不通过直接调用内部_bunDrop伪造合法动作。', '',
        'Actor输入：无上一步动作、无last_move_dir、无朝向、无速度向量；global[5]/[8]含己/敌spdG归一化速度能力，global[23]仅“有非NONE移动状态”布尔，未含滑行方向或剩余ticks。obs ch0/ch2是当前连续坐标双线性splat，未堆帧。更改lastMoveDir/slideDir不会改变输入，而速度成长改变global[5]，单测验证。JAX make_obs/global_vec与生产JS编码逐字段来源见INPUT_OBSERVABILITY.json；不能把物理内部字段存在当作Actor可观察。', '',
        '|来源|移动tick|LR直接反转|每100移动tick反转|反转事件间隔中位tick|','|---|---:|---:|---:|---:|']
    for k,v in comparisons.items():text.append(f"|{k}|{v['move_ticks']}|{v['reversals']}|{v['per100_move_ticks']:.4f}|{v['median_event_interval']}|")
    text+=['','JAX为同u800训练原生规则/Actor的8局CPU自然categorical探针、dev前8seed、P0、native Hunter tier3，完整终局；没有optimizer、Bank或课程，也不是已录历史混合训练collector。因此不能单因素证明argmax原因：JS/JAX对手实现、初态及随机流不同。JS H0 dev P0与其Hunter是相同对局直接比较；原31局为按放泡数选择的另一fresh集合，分开列。非零采样反转只能否定“仅argmax才可能反转”，不能直接证明同等振荡。','',
        '|原31局组|振荡tick|敌泡范围边缘|敌泡范围内部|范围外紧邻|其他泡/当前余焰|无观察到爆泡危险|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in dg.items():
        t=v['type_ticks'];text.append(f"|{k}|{v['oscillation_ticks']}|{t.get('enemy_blast_edge',0)}|{t.get('enemy_blast_interior',0)}|{t.get('adjacent_outside_enemy_blast',0)}|{t.get('other_or_active_flame_risk',0)}|{t.get('no_observed_blast_danger',0)}|")
    text+=['','危险为当前敌泡blastCap射线遇wall/brick/bomb停止，并计算泡连锁闭包；边缘=当前格在预测覆盖内、至少一个邻格不在；外紧邻单列；当前余焰来源不明归“其他”。未模拟未来放泡/移动/清砖或敌意图。边缘关联可称可能躲避，无观察危险可称可能犹豫，不能把意图分类当事实。敌相对位置按(dy,dx)符号九分桶，在DANGER_AND_HUNTER逐局与RESULTS按组保存，tick分母闭合。', '',
        '动作保持固定按episode tick分块：H2每2tick、H4每4tick选择一次移动，块内复用请求；Actor forward与ability照常每tick。保持动作后mask变非法时继续记录原请求，由原生物理阻挡，不偷加安全重选。H0逐位原样。每组独立同种子对手RNG流，但干预改变状态会改变后续对手决策和随机消耗，属于真实响应对照，不是相同后续状态。', '',
        '|组/席位|局数|胜/平/负|胜−负|因果击杀|泡/局|单方零泡/己方全零泡|自伤/死亡|LR反转/100移动tick|持包停滞tick|','|---|---:|---|---:|---:|---:|---|---|---:|---:|']
    for a,v in arms.items():
        for label,s in [('整体',v['overall']),*v['by_seat'].items()]:text.append(f"|{a}/{label}|{s['games']}|{s['win']}/{s['draw']}/{s['loss']}|{s['win_minus_loss']}|{s['causal_kills']}|{s['bombs_per_game']:.4f}|{s['unilateral_zero_bomb_games']}/{s['all_zero_own_games']}|{s['self_deaths']}/{s['deaths']}|{s['reversals']['per100_move_ticks']:.4f}|{s['carry_stall_ticks']}|")
    text+=['','持包停滞沿上一冻结50tick、真实carry、中心不推进且静态BFS不降规则；crate可走过，不当障碍；尾窗口不补成50。零泡同时报告单方零泡（敌有泡）与己方全零，避免口径混淆。击杀=因果归因，不等于死亡数；自伤采用生产ownBombDefeat事件。', '',
        f"H2相对H0：{paired['H2']['differences']}，任务观察阈值满足={paired['H2']['task_observation_threshold_met']}；H4：{paired['H4']['differences']}，满足={paired['H4']['task_observation_threshold_met']}。阈值为胜−负至少+5、因果击杀不降、单方零泡不增，只供编排方决策，不进入BEST/闸门。逐seed成对胜负表保留RESULTS，不以总表替代配对数据，不声称训练泛化或独立holdout提升。", '',
        '复现：`node --test tests/bomb_threat_jitter_followup_test.js tests/bomb_threat_p0_jitter_test.js tests/bomb_threat_iter2_actions_test.js`；`CUDA_VISIBLE_DEVICES= .venv/bin/python -m scripts.bomb_threat_jitter_followup_owner`；`CUDA_VISIBLE_DEVICES= taskset -c 116-119 .venv/bin/python -m scripts.bomb_threat_jitter_native_probe`；`node scripts/bomb_threat_jitter_followup_audit.js`；`.venv/bin/python -m scripts.bomb_threat_jitter_followup_report`。所有路径仓库相对；逐局/分片/trace SHA、排除/重试与最终训练现场见RECEIPT。','']
    Path('reports/bomb_threat_v1_jitter_followup_20261029.md').write_text('\n'.join(text));print(json.dumps({a:v['overall'] for a,v in arms.items()},ensure_ascii=False))
if __name__=='__main__':main()
