"""Passive steering receipt, current throughput and production-JS latency."""
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.bomb_threat_train_monitor import read_updates, active_run_names
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def js_latency(base=BASE):
    completed=[];pending=[];incomplete_markers=[];now=time.time()
    for req in (base/'eval_requests').glob('*.json'):
        q=read(req)
        if not q.get('checkpoint'):continue
        cp=ROOT/q['checkpoint'];cm=read(cp/'COMMITTED.json')
        if not cm or (base/'formal'/cp.parent.name/'EXCLUDED_INVALID.json').exists():continue
        stamp=cm.get('committed_unix') or cm.get('unix') or (cp/'COMMITTED.json').stat().st_mtime
        marker=base/'eval'/req.stem/'NATURAL_COMPLETE.json';d=read(marker)
        if d:
            if d.get('policy_sha256')!=cm.get('policy_sha256'):continue
            modes=d.get('natural_js_full2400',{})
            if not all(modes.get(mode,{}).get('games')==64 for mode in ('greedy','categorical')):
                incomplete_markers.append(req.stem)
                continue
            finish=d.get('unix',marker.stat().st_mtime)
            proto=read(base/'protocols'/(cp.parent.name+'.json'))
            completed.append(dict(request=req.stem,checkpoint=q['checkpoint'],split=q.get('split'),
                slot=proto.get('slot'),iter2_arm=q.get('iter2_arm'),
                seconds=max(0,finish-stamp),commit_unix=stamp,JS_complete_unix=finish,
                marker=str(marker.relative_to(ROOT))))
        else:pending.append(dict(request=req.stem,age_seconds=max(0,now-stamp)))
    values=[r['seconds'] for r in completed]
    iter2=[r['seconds'] for r in completed if r['iter2_arm']]
    by_slot={}
    for slot in (0,1,3,4,5,6,7):
        rr=[r for r in completed if r['slot']==slot]
        by_slot[str(slot)]=dict(completed=len(rr),p50_seconds=statistics.median(r['seconds'] for r in rr) if rr else None,
            max_seconds=max((r['seconds'] for r in rr),default=None))
    return dict(measurement='COMMITTED receipt mtime to both JS64-mode NATURAL_COMPLETE unix; no JAX/quick-screen mixing',
        iter2=dict(completed=len(iter2),p50_seconds=statistics.median(iter2) if iter2 else None,max_seconds=max(iter2) if iter2 else None),
        by_slot=by_slot,
        completed=len(values),p50_seconds=statistics.median(values) if values else None,
        max_seconds=max(values) if values else None,pending=len(pending),
        incomplete_dual64_markers=incomplete_markers,
        max_pending_age_seconds=max((r['age_seconds'] for r in pending),default=0),records=completed)


def refresh():
    from scripts.bomb_threat_iter2_analysis import refresh as analysis_refresh
    a=analysis_refresh();status=read(BASE/'TRAIN_MONITOR_STATUS.json');activation=read(BASE/'iter2/DIAGNOSTICS_ACTIVATION.json')
    try:alerts=[json.loads(line) for line in (BASE/'TRAIN_ALERTS.jsonl').read_text().splitlines()]
    except (OSError,ValueError):alerts=[]
    low_counts=dict(Counter(r['scope'] for r in alerts if r['code']=='LOW_ENV_PER_SECOND'))
    current_low_windows=[r for r in alerts if r['code']=='LOW_ENV_PER_SECOND' and r['scope']=='CURRENT']
    current=[];technical=[]
    slots=read(BASE/'SLOTS.json');live=active_run_names()
    for slot,entry in slots.items():
        run=entry['run'];rr=read_updates(BASE/'formal'/run/'updates.jsonl')
        if not rr:
            launch=read(BASE/'formal'/run/'LAUNCH.json')
            technical.append(dict(slot=slot,run=run,active=run in live,
                stage=launch.get('stage','NO_RECORDED_UPDATE'),
                stage_age_seconds=max(0,time.time()-launch.get('unix',time.time())),
                interpretation='尚无实际更新，不能用进程存在或冷编译冒充有效训练'))
            continue
        last=rr[-1];warm=rr[-20:];speed=sum(r.get('actual_environment_steps_this_update',0) for r in warm)/sum(r['cycle_seconds'] for r in warm)
        lastspeed=last['actual_environment_steps_this_update']/last['cycle_seconds'];slow=lastspeed<15000
        proto=read(BASE/'protocols'/(run+'.json'))
        cold_boundary=last['new_update']<=proto.get('chunk_start_update',-100)+2
        reason='chunk首两步smoke/恢复冷编译周期；尚待warm周期核验' if slow and cold_boundary else ('实际当前吞吐偏低，尚无独立原因证据' if slow else '当前最新更新≥15000 env/s')
        current.append(dict(slot=slot,run=run,active=run in live,update=last['new_update'],finite=last.get('finite'),
            env_per_second=lastspeed,last20_env_per_second=speed,latest_cycle_seconds=last['cycle_seconds'],
            reason=reason,update_age_seconds=time.time()-last.get('unix',0),update_l2=last.get('update_l2'),
            loss=last.get('loss'),diagnostics=last.get('policy_diagnostics_version'),
            entropy=last.get('loss',[None]*4)[3] if len(last.get('loss',[])) in (7,8) else None))
    deployment=[]
    for file in (BASE/'protocols').glob('*.json'):
        proto=read(file)
        if proto.get('passive_diagnostics_version')!='masked-live-collector-v1':continue
        rr=read_updates(BASE/'formal'/file.stem/'updates.jsonl')
        actual=next((r for r in rr if r.get('policy_diagnostics_version')),None)
        source=proto.get('continuation_source') or {}
        previous=read(ROOT/source['checkpoint']/'COMMITTED.json').get('metrics',{}) if source.get('checkpoint') else {}
        out=BASE/'formal'/file.stem
        record=dict(run=file.stem,arm=proto.get('iter2_arm'),module=proto.get('worker_module'),status='RECORDED' if actual else 'WAITING_CHUNK_TECHNICAL_CHECKS',
            smoke_passed=read(out/'SMOKE.json').get('passed',False),
            restore_passed=read(out/'RESTORE.json').get('passed',False),
            previous_atomic=dict(update=previous.get('new_update'),loss=previous.get('loss'),update_l2=previous.get('update_l2')),
            first_diagnostic_update=None,interpretation='adjacent random live updates are observational; exact controlled equality is in UPDATE_PARITY')
        if actual:
            e=actual['exposure'];n=e['policy_diagnostic_ticks']
            record['first_diagnostic_update']=dict(update=actual['new_update'],finite=actual['finite'],loss=actual['loss'],update_l2=actual['update_l2'],
                collector_entropy=e['policy_entropy_sum']/n,bomb_probability=e['bomb_probability_sum']/n,bomb_argmax_fraction=e['bomb_argmax_count']/n,
                denominator=n,env_per_second=actual['actual_environment_steps_this_update']/actual['cycle_seconds'])
        formal=next((r for r in rr if r.get('policy_diagnostics_version') and
            r['new_update']>proto['chunk_start_update']+2),None)
        record['first_post_restore_update']=None if formal is None else dict(
            update=formal['new_update'],finite=formal['finite'],update_l2=formal['update_l2'],
            env_per_second=formal['actual_environment_steps_this_update']/formal['cycle_seconds'])
        deployment.append(record)
    atomic_json(BASE/'iter2/DIAGNOSTICS_DEPLOYMENT.json',dict(status='TRACKING_NATURAL_CHUNK_BOUNDARIES',runs=deployment,unix=time.time()))
    latency=js_latency();atomic_json(BASE/'iter2/JS_EVAL_LATENCY.json',latency)
    early=a['a3_early_observation'];p=ROOT/'reports/bomb_threat_v1_iter2_round4_20261029.md'
    lines=['# ITER2 round4：11:20纠偏执行（20261029）','',
        '训练不停；不改奖励、配方、闸门、网页默认或解码。旧collector/worker源码与冻结run协议保留；新只读诊断通过受控parity后，在下一自然320-update chunk边界按新worker版本接入。已启动chunk不会原位换ABI。',
        '', '## 诊断接线与数值边界','',
        '源码证明loss[3]为优化minibatch的move+ability总熵，历史7/8项向量已回填并标来源。7项：[loss,ppo,value,entropy,approx_kl,clip_fraction,gradient_norm]；8项最后另为ability_head_gradient_norm（不是Critic范数）。collector新诊断是masked、live tick的move+ability总熵、bomb=1概率、ability argmax=1比例，共用原前向与原RNG，不接梯度/GAE。历史bomb概率和argmax不可倒推。',
        '', '激活状态：`'+json.dumps(activation,ensure_ascii=False)+'`。',
        '同起点单个真实on-policy update的before/after loss、Actor/Critic L2、Adam、完整env/aux/RNG逐位对照：`runs/bomb_threat_v1/iter2/STEER1120_UPDATE_PARITY.json`。此受控同输入比较与相邻随机训练update不同，不能宣称相邻update数值相等。',
        '实际chunk部署及边界前后loss/L2逐项收据：`runs/bomb_threat_v1/iter2/DIAGNOSTICS_DEPLOYMENT.json`。'+json.dumps(deployment,ensure_ascii=False),
        '', '## 告警分级与当前吞吐','',
        'UNAVAILABLE归DATA_GAP，单独计数，不标红；ANOMALY才标红。历史和当前分开；只对真实活跃worker的最后update判断15分钟停更。此前把活跃run早期窗口当停更的告警已撤回。settled_kill是奖励而非事件计数，历史击杀/100tick改为未记录；本次三个新诊断不增加击杀事件字段，继续明确缺失。',
        '监控快照：`'+json.dumps({k:status.get(k) for k in ('windows','alerts','alert_counts_by_severity','alert_counts_by_scope','alert_counts_by_scope_and_severity')},ensure_ascii=False)+'`。',
        '低吞吐告警按40-update窗口聚合，历史/当前数量：`'+json.dumps(low_counts,ensure_ascii=False)+'`；当前低速窗口：`'+json.dumps(current_low_windows,ensure_ascii=False)+'`。窗口可能仍含首更冷编译，不能将窗口均值当作最新热态速度；下表另列最后一更与近20更。',
        '', '|slot/run|最新u|finite|最新env/s|近20更新env/s|最新秒|当前低吞吐原因|', '|---|---:|---|---:|---:|---:|---|']
    for r in sorted(current,key=lambda r:int(r['slot'])):lines.append(f"|{r['slot']}/{r['run']}|{r['update']}|{r['finite']}|{r['env_per_second']:.0f}|{r['last20_env_per_second']:.0f}|{r['latest_cycle_seconds']:.2f}|{r['reason']}|")
    lines+=['','尚无更新的技术阶段：`'+json.dumps(technical,ensure_ascii=False)+'`。',
        '', '## A3加严观察','',
        'u640原比较继续；新增u200/240/280/320连续3点负≥6且胜减负<同期对照均值−5，才在下一完整chunk原子终态按既定fallback恢复原配方。同期口径：每个A3训练时刻，取slot00/03/06各自最近前置且≤600秒的已评dev点，槽等权；缺任一槽或任一点则PENDING，不零填。该确定性匹配口径已先登记amendment。',
        '状态：`'+json.dumps(early,ensure_ascii=False)+'`。','',
        '|A3 update|负局|直接自伤死亡|平局击杀差|三项塑形实付/训练净回报|','|---:|---:|---:|---:|---:|']
    for r in a['a3_points']:lines.append(f"|{r['update']}|{r['loss']}|{r['self_deaths']}|{r['draw_kill_difference']}|{r['threat_paid_fraction'] if r['threat_paid_fraction'] is not None else '未记录'}|")
    lines+=['','实付是目标chunk前缀，不是JS整局回算；自伤只覆盖直接ownBombDefeat，不泛化为所有自伤。u160当前原始NATURAL_COMPLETE的self_deaths=0、deaths=26、delivery=10、draw_own_causal_kills=10，未复现指令快照中“自伤10”；保留分歧，不把其他10计为自伤。每评估点原始marker见DEV_BUCKETS_AND_ROLLBACK。',
        '', '## JS评估延迟','',
        f"匹配完成请求 {latency['completed']}；checkpoint原子COMMITTED写盘至双模式JS出成绩 P50={latency['p50_seconds']}秒，最大={latency['max_seconds']}秒。未完成{latency['pending']}，最大已等{latency['max_pending_age_seconds']:.1f}秒（右删失，不能当已完成延迟）。逐请求记录见 `runs/bomb_threat_v1/iter2/JS_EVAL_LATENCY.json`。CPU/JS与GPU2辅助异步，不阻塞健康训练。",
        'ITER2子集：`'+json.dumps(latency['iter2'],ensure_ascii=False)+'`；各slot分列：`'+json.dumps(latency['by_slot'],ensure_ascii=False)+'`。',
        '', '复现：`.venv/bin/python -m scripts.bomb_threat_train_monitor`；`.venv/bin/python -m scripts.bomb_threat_steer1120_report`。未push/发布、未改models.json/BEST。']
    p.write_text('\n'.join(lines)+'\n')
    deployed=any(r['restore_passed'] and r['smoke_passed'] and r['first_post_restore_update'] for r in deployment)
    receipt=dict(status='COMPLETED_TRAINING_CONTINUES' if deployed else 'IMPLEMENTED_TRAINING_CONTINUES',report=str(p.relative_to(ROOT)),report_sha256=sha(p),
        diagnostics_activation=activation.get('status','NOT_YET_READY'),current=current,technical_stages=technical,
        deployment=deployment,
        low_throughput_window_counts=low_counts,current_low_throughput_windows=current_low_windows,
        a3_early_status=early['status'],JS_latency={k:v for k,v in latency.items() if k!='records'},
        worker_signals=0,reward_recipe_gate_changes=False,unix=time.time())
    atomic_json(BASE/'iter2/STEER1120_EXECUTION.json',receipt);return receipt


if __name__=='__main__':print(json.dumps(refresh(),ensure_ascii=False))
