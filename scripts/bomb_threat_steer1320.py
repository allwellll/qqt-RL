"""13:20 authorized boundary change and read-only kill-to-win diagnosis."""
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/bomb_threat_v1'


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {} if default is None else default


def a3_bomb_kill_observation(arms,records):
    points=[]
    for arm in sorted(arms,key=lambda x:x['update']):
        if arm['update']<680 or arm['update']%40:continue
        controls={}
        for slot in ('0','3','6'):
            eligible=[r for r in records if r['slot']==slot and not r.get('arm') and r.get('train_unix')
                and arm.get('train_unix') and 0<=arm['train_unix']-r['train_unix']<=600]
            if eligible:controls[slot]=max(eligible,key=lambda x:x['train_unix'])
        m=arm['metrics'];complete=len(controls)==3
        c={k:statistics.mean(r['metrics'][k] for r in controls.values()) if complete else None
            for k in ('bombs_per_game','causal_kills')}
        points.append(dict(update=arm['update'],bombs_per_game=m['bombs_per_game'],causal_kills=m['causal_kills'],
            draw_kill_difference=m.get('draw_kill_difference'),control=c,
            control_markers={s:r.get('marker') for s,r in controls.items()},
            qualifies=m['bombs_per_game']>c['bombs_per_game']+5 and m['causal_kills']<=c['causal_kills'] if complete else None,
            bombs29_kills_below_control=m['bombs_per_game']>=29 and m['causal_kills']<c['causal_kills'] if complete else None,
            negative_draw_kill_difference=m.get('draw_kill_difference') is not None and m['draw_kill_difference']<0))
    trigger=[]
    for i in range(2,len(points)):
        tail=points[i-2:i+1]
        if tail[1]['update']-tail[0]['update']==40 and tail[2]['update']-tail[1]['update']==40 and all(x['qualifies'] is True for x in tail):
            trigger=tail;break
    return dict(status='RESTORE_ORIGINAL_RECIPE' if trigger else 'PENDING' if len(points)<3 or any(x['qualifies'] is None for x in points) else 'CONTINUE',
        points=points,trigger_points=trigger,control_slots=['0','3','6'],matching='equal-slot latest preceding <=600s; missing never zero filled',
        rule='3 consecutive cadence40 dev points bombs>control+5 AND kills<=control; next completed chunk only',
        observation_only=['bombs>=29 and kills<control','draw kill difference<0'])


def summarize_d4(records):
    draws=[r for r in records if r['done'] and r['winner'] is None]
    kd=[r for r in draws if r['causal_kills'][r['seat']]>0]
    hist=Counter();enemy_stock=Counter();bins={k:0 for k in ('1..600','601..1200','1201..1800','1801..2400')}
    events=[];details=[];missing=0
    for r in kd:
        seat=r['seat'];stored=r['stored'];nat=r.get('natural',{})
        kills=[k for k in r['kills'] if k['attacker']==seat]
        missing+=r['causal_kills'][seat]-len(kills)
        for k in kills:
            t=k['tick'];events.append(t)
            bins['1..600' if t<=600 else '601..1200' if t<=1200 else '1201..1800' if t<=1800 else '1801..2400']+=1
        own=stored[seat];opp=stored[1-seat]
        hist[str(sum(own)-sum(opp))]+=1
        enemy_stock[str(own[1-seat])]+=1
        details.append(dict(seed=r['seed'],seat=seat,ticks=r['ticks'],stored=stored,
            own_stock_total=sum(own),opponent_stock_total=sum(opp),stock_total_difference=sum(own)-sum(opp),
            enemy_origin_stored=own[1-seat],opponent_enemy_origin_stored=opp[seat],
            own_causal_kills=r['causal_kills'][seat],kill_ticks=[k['tick'] for k in kills],
            first_kill_remaining_ticks=r['ticks']-min(k['tick'] for k in kills) if kills else None,
            delivery=nat.get('delivery'),pickup=nat.get('pickup'),window_stock_pickup=nat.get('window_stock_pickup'),
            window_stock_delivery=nat.get('window_stock_delivery'),
            post_kill_delivery='none: full-game delivery=0' if nat.get('delivery')==0 else 'unknown timing: delivery event timestamps not recorded'))
    return dict(games=len(records),draw_games=len(draws),kill_draw_games=len(kd),kill_draw_kills=sum(r['causal_kills'][r['seat']] for r in kd),
        kill_draw_missing_event_count=missing,kill_draw_stock_total_difference_histogram=dict(sorted(hist.items())),
        kill_draw_enemy_origin_stored_histogram=dict(sorted(enemy_stock.items())),
        kill_draw_no_delivery_games=sum(r.get('natural',{}).get('delivery')==0 for r in kd),
        kill_draw_no_pickup_games=sum(r.get('natural',{}).get('pickup')==0 for r in kd),
        kill_draw_delivery_timing_unavailable_games=sum(r.get('natural',{}).get('delivery',0)>0 for r in kd),
        kill_draw_tick_bins=bins,kill_draw_tick_median=statistics.median(events) if events else None,
        kill_draw_first_kill_remaining_ge600_games=sum(d['first_kill_remaining_ticks'] is not None and d['first_kill_remaining_ticks']>=600 for d in details),
        kill_draw_all_kills_last300_games=sum(bool(d['kill_ticks']) and min(d['kill_ticks'])>2100 for d in details),
        details=details,limitations=['no forced-action counterfactual; remaining ticks alone do not prove achievable delivery',
            'nonzero delivery lacks event timing; no invented after-kill ordering',
            'stock difference is terminal stock, not cumulative deliveries; kills do not directly decide native Bun winner'])


def d4():
    authorization=read(BASE/'iter2/STEER1320_AUTHORIZATION.json');out=BASE/'iter2/D4';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
    summaries=[];all_records=[];inputs={}
    for point in authorization['D4']['points']:
        marker=ROOT/point['marker'];assert sha(marker)==point['marker_sha256']
        doc=read(marker);assert doc['policy_sha256']==point['policy_sha256'] and doc['split']=='dev'
        episodes=[]
        for file in sorted(marker.parent.glob('natural_greedy_*.json.outcome.json')):
            assert sha(file)==doc['files'][file.name];inputs[str(file.relative_to(ROOT))]=sha(file)
            raw=read(file);assert raw['engine']=='production_JS' and raw['teacher'] is False and raw['mode']=='greedy'
            naturalfile=Path(str(file).removesuffix('.outcome.json'));assert sha(naturalfile)==doc['files'][naturalfile.name]
            inputs[str(naturalfile.relative_to(ROOT))]=sha(naturalfile)
            ngames=read(naturalfile)['results']['natural_full2400']['episodes']
            index={(n['seed'],n['team']):n for n in ngames}
            for r in raw['episodes']:
                n=index[(r['seed'],r['seat'])];assert n['ticks']==r['ticks'] and r['done']
                episodes.append(dict(r,natural={k:n.get(k) for k in ('delivery','pickup','window_stock_pickup','window_stock_delivery','death')}))
        assert len(episodes)==64 and len({(r['seed'],r['seat']) for r in episodes})==64
        assert {r['seed'] for r in episodes}==set(range(2026122900,2026122964))
        assert sum(r['causal_kills'][r['seat']] for r in episodes)==doc['natural_js_full2400']['greedy']['causal_kills']
        summary=summarize_d4(episodes);assert summary['kill_draw_missing_event_count']==0
        record=dict(point=point,summary=summary,records=episodes)
        atomic_json(out/f'u{point["update"]:06d}.json',record);summaries.append(dict(update=point['update'],**summary));all_records.extend(episodes)
    value=dict(status='COMPLETE',frozen_points=authorization['D4']['points'],points=summaries,aggregate=summarize_d4(all_records),
        evaluation_games=len(all_records),unique_seed_seat_pairs=len({(r['seed'],r['seat']) for r in all_records}),
        independent_game_inputs=64,paired_checkpoints=True,input_sha256=inputs,
        source='existing production JS full terminal trajectories, teacher off; no new reruns',unix=time.time())
    atomic_json(out/'COMPLETE.json',value);return value


def refresh():
    d=d4();a=read(BASE/'iter2/DEV_BUCKETS_AND_ROLLBACK.json');slots=read(BASE/'SLOTS.json');live=[]
    from scripts.bomb_threat_owner import workers
    active=workers()
    for s,e in slots.items():
        out=BASE/'formal'/e['run'];p=read(out/'PROGRESS.json');last=p.get('last',{});proto=read(BASE/'protocols'/f"{e['run']}.json")
        live.append(dict(slot=s,run=e['run'],update=p.get('iterations'),active=e['run'] in active,
            arm=e.get('iter2_arm'),fixed_entropy=bool(e.get('entropy_restored_1320')),finite=last.get('finite'),
            entropy_coefficient=last.get('entropy_coefficient',last.get('configuration',{}).get('entropy_coefficient')),
            update_l2=last.get('update_l2'),env_per_second=262144/last['cycle_seconds'] if last else None,
            boundary_update=proto['chunk_start_update']+proto['chunk_updates'],smoke=read(out/'SMOKE.json').get('passed'),restore=read(out/'RESTORE.json').get('passed')))
    a2=next(r for r in live if r['slot']=='7');watch=a.get('a3_bomb_kill_observation',{})
    path=ROOT/'reports/bomb_threat_v1_iter2_round5_20261029.md';ag=d['aggregate']
    lines=['# ITER2 round5：13:20决定执行与D4击杀转胜诊断（20261029）','',
        '授权 `.hermes/STEER_20261029_1320.md`；口径先登记 `runs/bomb_threat_v1/iter2/amendment06_steer1320_boundary_and_D4.json`。训练不停，不改终局奖励、闸门、网页默认/解码、models.json或BEST；不push/发布。',
        '', '## A2自然边界恢复固定熵','',
        '编排方决定停止熵退火：近期argmax平局均值56、零泡17.5、击杀约减半，没有缩小argmax与采样差距。此为本臂操作决定和观察，未隔离证明熵为全部变化的唯一根因。slot07当前c15在u960自然完整chunk终态保留最新完整Actor/Critic/Adam/env/aux/RNG/ring和累计步数；新namespace只将熵改为固定0.005，其余保持A1配方，不新Critic、不重做20步预热。按smoke→独立fresh restore→正式续训；不向健康worker发信号。',
        '当前状态：`'+json.dumps(a2,ensure_ascii=False)+'`。边界回执 `STEER1320_ENTROPY_HANDOFF.json`，切换后的真实非零更新验收由owner持久刷新。',
        '本次授权消费时冻结的原始双模式复算（20点，比编排19点快照多1点；不能混成同一截止）：`'+json.dumps({k:v for k,v in read(BASE/'iter2/STEER1320_PRE_SWITCH_OBSERVATION.json').items() if k not in ('rows','limitations')},ensure_ascii=False)+'`。',
        '', '## A3放泡—击杀观察','',
        'u640击杀/放泡不差，仅胜减负较差，按本决定继续。新增u680起连续3个cadence40 dev点：泡/局>同期对照+5且击杀≤对照，才在下一完整chunk按已有回退流程执行。同期沿11:20口径：slot00/03/06各取训练时间最近前置≤600秒的已评dev点，槽等权；缺项PENDING。泡≥29且击杀<对照、平局击杀差<0单列观察，单独不触发回退。',
        '`'+json.dumps(watch,ensure_ascii=False)+'`。',
        '', '## D4：有击杀的平局为何未变胜局','',
        '固定本次冻结时slot04最新三个完整dev greedy点u680/u720/u760；每点64局，共192个checkpoint×局观测，共用64个seed/席位，不能当192独立seed。已逐文件校验marker/模型/原始分片SHA及64seed集合、完整终局；未重跑旧局、未用holdout挑点。',
        '', '|update|局数|平局|有己方击杀平局|其中击杀|全局无交付|全局无取包|末库存差分布|击杀tick中位数|', '|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in d['points']:
        lines.append(f"|{r['update']}|{r['games']}|{r['draw_games']}|{r['kill_draw_games']}|{r['kill_draw_kills']}|{r['kill_draw_no_delivery_games']}|{r['kill_draw_no_pickup_games']}|{r['kill_draw_stock_total_difference_histogram']}|{r['kill_draw_tick_median']}|")
    lines+=['', '合并：`'+json.dumps({k:v for k,v in ag.items() if k!='details'},ensure_ascii=False)+'`。',
        '', f"观察事实：{ag['kill_draw_games']}个有击杀平局中，{ag['kill_draw_no_delivery_games']}个整局交付为0，{ag['kill_draw_no_pickup_games']}个整局取包为0；因此这些局的击杀没有形成完整交付链。{ag['kill_draw_first_kill_remaining_ge600_games']}局首杀距终局仍至少600tick，{ag['kill_draw_all_kills_last300_games']}局全部击杀集中最后300tick。不能把时间余量直接称为可交付能力，不能给未记录时序的交付强加“击杀后”标签。",
        '库存规则代码：`web/sim-stock-window-v1.js` `_bunUpdate`带包回己笼增加库存，`_bunHasCapturedAll`触发即时胜方；超时则比较两队笼内库存总数，只有相同才平局，不按击杀数破平。28个有击杀平局末矩阵均为[[1,0],[0,1]]，双方总数1，敌方来源入己笼0。每局末双方按来源库存矩阵、库存总差、击杀tick、全局取包/交付与缺失字段在D4逐点JSON。末库存相同本身不等同累计交付为0，本批交付0由独立逐局指标验证。',
        '结论边界：主要可观测断点是击杀没有转成取包/交付与终局库存条件；部分晚杀可受剩余时间限制，但没有反事实证据证明其唯一原因。已有分片未记录非零交付的精确tick，无法全部分清“交付发生在击杀前/后”；明确缺失，不编造。下一配方改动仍由编排方决定。',
        '', '## 当前训练','', '|slot|run|u|finite|entropy系数|env/s|', '|---|---|---:|---|---:|---:|']
    for r in sorted(live,key=lambda x:int(x['slot'])):
        lines.append(f"|{r['slot']}|{r['run']}|{r['update']}|{r['finite']}|{r['entropy_coefficient']}|{r['env_per_second']}|")
    round1=read(BASE/'iter2/DECISION_ROUND1.json',{})
    if round1.get('status')=='AUTHORIZED':
        gate=read(BASE/'iter2/ROUND1_A1_U960_DECISION.json',{})
        current_slot07=next((r for r in live if r['slot']=='7'),{})
        lines+=['','## 14:00决定接续（覆盖此前A2继续训练的计划）','',
            'A2只允许运行到slot07当前c16的u1280原子边界；该边界后禁止再启动A2，转为slot01最新原子Actor派生的A3第二seed（2026112907），新初始化Critic/Adam/env/RNG并预热20次Critic更新。A3b仅提案，不启动。',
            'A1门槛读取固定dev检查点slot04 c15 u960：'+json.dumps(gate,ensure_ascii=False)+'。',
            'slot07当前持久状态：'+json.dumps(current_slot07,ensure_ascii=False)+'。',
            'A3 u680/u720 holdout优先请求及双模式64局状态：`runs/bomb_threat_v1/eval_requests/bt_slot01_c000015_u000680_holdout.json`、`runs/bomb_threat_v1/eval_requests/bt_slot01_c000015_u000720_holdout.json`；完整结果未落盘前标记pending，不据此晋升。']
    lines+=['','复现：`.venv/bin/python -m scripts.bomb_threat_steer1320`。原始输入SHA、逐局结果：`runs/bomb_threat_v1/iter2/D4/COMPLETE.json`；持久执行状态：`STEER1320_EXECUTION.json`。']
    path.write_text('\n'.join(lines)+'\n')
    done=a2['fixed_entropy'] and a2['smoke'] and a2['restore'] and a2['finite'] and min(a2['update_l2'].values())>0 and a2['update']>960+2
    round1=read(BASE/'iter2/DECISION_ROUND1.json',{})
    round1_status=None
    if round1.get('status')=='AUTHORIZED':
        slot07=next((r for r in live if r['slot']=='7'),{})
        if slot07.get('arm')=='A3':
            last=slot07.get('last',{});round1_status='ROUND1_A3_SECOND_SEED_FORMAL' if last.get('finite') and last.get('update_l2',{}).get('actor',0)>0 and last.get('update_l2',{}).get('critic',0)>0 and slot07.get('formal_update',0)>2 else 'ROUND1_A3_SECOND_SEED_INITIALIZING'
        else:round1_status='WAITING_A2_U1280_ATOMIC_BOUNDARY'
    receipt=dict(status=round1_status or ('COMPLETED_TRAINING_CONTINUES' if done else 'WAITING_NATURAL_A2_CHUNK_BOUNDARY'),
        A2=a2,A3=watch,round1_decision=round1.get('decision'),round1_status=round1_status,
        round1_a1_u960_gate=read(BASE/'iter2/ROUND1_A1_U960_DECISION.json',{}),
        D4_complete='runs/bomb_threat_v1/iter2/D4/COMPLETE.json',report=str(path.relative_to(ROOT)),report_sha256=sha(path),active=live,training_worker_signals=0,unix=time.time())
    atomic_json(BASE/'iter2/STEER1320_EXECUTION.json',receipt)
    return receipt


if __name__=='__main__':print(json.dumps(refresh(),ensure_ascii=False))
