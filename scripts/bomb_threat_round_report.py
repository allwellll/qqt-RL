"""Two-hour auditable round reports, independent from GPU launches."""
import json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
from scripts.combat_mix_owner import read
from scripts.bomb_threat_owner import invalid_ancestry
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def lineage_limits(run_name):
    """Allow only evaluations in the active causal branch, capped at each fork."""
    from scripts.bomb_threat_owner import causal_lineage_limits
    return causal_lineage_limits(run_name,BASE)


def snapshot():
    slots=read(BASE/'SLOTS.json',{});data=[]
    for gpu,slot in sorted(slots.items(),key=lambda kv:int(kv[0])):
        run=BASE/'formal'/slot['run'];progress=read(run/'PROGRESS.json',{});latest=progress.get('iterations',slot['source']['update'])
        candidates=[];lineage=lineage_limits(slot['run']);depth={n:i for i,n in enumerate(lineage)}
        for p in (BASE/'eval').glob('*/GATE_METRICS.json'):
            r=read(p,{})
            if not r.get('checkpoint') or (p.parent/'EXCLUDED_INVALID.json').exists():continue
            cp=Path(r['checkpoint']);u=int(cp.name[1:])
            if cp.parent.name not in lineage or u>lineage[cp.parent.name]:continue
            if u%40 or invalid_ancestry(cp.parent.name) or (BASE/'formal'/cp.parent.name/'EXCLUDED_EVALUATION.json').exists():continue
            marker=read(p.parent/'NATURAL_COMPLETE.json',{})
            if marker.get('split')!='dev':continue
            candidates.append((u,p,r))
        candidates.sort(key=lambda x:(-depth[Path(x[2]['checkpoint']).parent.name],x[0],x[1].parent.name));last=candidates[-1] if candidates else None
        commits=sorted(run.glob('u*/COMMITTED.json'))
        current_cp=read(commits[-1]) if commits else slot['source']
        ref=read(BASE/'eval'/('baseline_parent_dev' if int(gpu) in (0,1,3,4,7) else 'baseline_u1120')/'NATURAL_COMPLETE.json',{})
        data.append(dict(gpu=int(gpu),role=slot['role'],initial_checkpoint=slot['old_boundary'],
            current_chunk_source=slot['source']['checkpoint'],active_lineage=list(lineage),
            current_run=slot['run'],current_update=latest,latest_atomic=current_cp['checkpoint'],
            stage=read(run/'LAUNCH.json',{}).get('stage'),last_metric=progress.get('last'),
            latest_evaluated_update=last[0] if last else None,latest_gate=str(last[1].relative_to(ROOT)) if last else None,
            evaluated_in_current_run=bool(last and Path(last[2]['checkpoint']).parent.name==slot['run']),
            gate=last[2] if last else None,lag_updates=latest-last[0] if last else None))
    return data


def write_round(force=False):
    guard=lock(BASE/'ROUND_REPORT.lock');receipt=read(BASE/'ROUND_REPORT_STATUS.json',{})
    now=time.time()
    if not force and receipt and now<receipt['next_due_unix']:return None
    number=int(receipt.get('last_round',0))+1;date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    path=ROOT/'reports'/f'bomb_threat_v1_round{number}_{date}.md';rows=snapshot()
    text=f'# BOMB-THREAT-V1 第{number}轮（{date}）\n\n'
    text+='本轮为现场快照，非完整长训终态。按07:00批准决定，slot03/04从slot00 u360复制并关闭新增威胁/拓扑/连锁塑形；其他训练配方不变，slot05/06不复制。当前dev seed 2026122900，argmax和categorical各64完整带砖局；生产JS normal Hunter、老师关闭。dev父点同协议13胜45平6负、因果击杀34、死亡18、32.59375泡/局、平局击杀差14。14:20零泡口径：父点单方零泡3、双方零泡4、合计7；单方仅我方实际0且对手实际>0，按真实seat取数，任何局仍计胜负分母。网页使用argmax，以argmax为主、采样单列；dev不能替代holdout晋升。\n\n'
    text+='|slot/起点|当前run/update/阶段|最新已评u/滞后|相对PBT/绝对门槛|胜/平/负|胜-负|因果击杀|死亡|平局击杀差|泡/局|单方零泡|双方零泡|\n|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        g=(r['gate'] or {}).get('natural_js_full2400',{}).get('greedy',{})
        gate=r['gate'];relative=gate.get('pbt_gate_status',('PASS' if gate['pbt_selection_eligible'] else 'FAIL')) if gate else 'INCOMPLETE'
        absolute=gate.get('absolute_gate_status',('PASS' if gate.get('absolute_gate_pass',gate['best_eligible']) else 'FAIL')) if gate else 'INCOMPLETE'
        if gate and not r['evaluated_in_current_run']:
            relative='祖先'+relative;absolute='祖先'+absolute
        text+=f"|{r['gpu']:02d}/{r['role']}|{r['current_run']} / {r['current_update']} / {r['stage']}|{r['latest_evaluated_update']} / {r['lag_updates']}|{relative}/{absolute}|{g.get('win','—')}/{g.get('draw','—')}/{g.get('loss','—')}|"+'|'.join(str(g.get(k,'—')) for k in ('win_minus_loss','causal_kills','deaths','draw_kill_difference','bombs_per_game','zero_bomb_games','structural_zero_bomb_games'))+'|\n'
    text+='\n采样同行指标（与argmax分开）：\n\n|slot/已评点|胜/平/负|胜-负|因果击杀|死亡|平局击杀差|泡/局|单方零泡|双方零泡|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        g=(r['gate'] or {}).get('natural_js_full2400',{}).get('categorical',{})
        text+=f"|{r['gpu']:02d}/u{r['latest_evaluated_update']}|{g.get('win','—')}/{g.get('draw','—')}/{g.get('loss','—')}|"+'|'.join(str(g.get(k,'—')) for k in ('win_minus_loss','causal_kills','deaths','draw_kill_difference','bombs_per_game','zero_bomb_games','structural_zero_bomb_games'))+'|\n'
    text+='\n各slot起点与完整证据、闸门失败原因：\n\n'
    for r in rows:
        text+=f"- slot{r['gpu']:02d} 当前谱系记录起点`{r['initial_checkpoint']}`；当前chunk实际起点`{r['current_chunk_source']}`；当前原子点`{r['latest_atomic']}`；最新已评`{r['latest_gate']}`。"
        if r['gate'] and not r['evaluated_in_current_run']:text+='此评分属于已验祖先，不能称新分支已通过。'
        if r['gate']:
            text+='相对失败：'+', '.join(r['gate']['pbt_gate_failures'])+'；绝对失败：'+', '.join(r['gate']['best_gate_failures'])+'。'
            text+='相对缺数据：'+', '.join(r['gate'].get('pbt_gate_missing',[]))+'；绝对缺数据：'+', '.join(r['gate'].get('best_gate_missing',[]))+'。'
            text+='新塑形/原生净总奖励='+str(r['gate']['hack_monitor'].get('threat_reward_fraction_total'))+'（仅该chunk前缀，含部分对局）。'
        text+='\n'
    text+='''
🔴 **红色预警：少放泡退化继续，不能因胜局小涨而忽略。** 用户验收历史点：slot04 u1120为11.5泡/局、零泡16；slot07 u1080为13.6泡/局、零泡16；slot00从u360的28.5泡/局降至u1120的20.6。slot03旧分支u920→960→1000放泡19.2→26.0→29.4，击杀23→23→23、胜10→10→7；slot04旧分支u960→1000→1040放泡9.7→11.4→13.8、击杀14→9→9、胜6→6→5，均已按规则在原子边界暂停旧塑形谱系。

🔴 **红色预警：slot05控制线显著退化。** u1520–1800约4泡/局、零泡21–35/64；u2080为2胜47平15负、死亡44。只保留既定对照，不复制、不追加专项资源。🔴 **笼内击杀超过parent+10pp（约13%）的历史点单列且不能晋升：** slot04 u1000 22%、slot07 u600 19%、slot03 u720及slot04 u640约17%。直接自伤持续为0只覆盖ownBombDefeat，不能推成无间接自伤。

argmax与采样分化单列：已验旧分支同点采样约多10–20泡/局、零泡少5–10局；这是特定检查点观察，不推广到所有新点。网页以argmax为主，不能用采样改善掩盖确定性退化；上表保留双模式完整分母。

调度纠偏：CPU自然局12个worker、新worker3分片，36个独立物理核，避开训练的SMT同核；旧在途分片保留不重跑；总配额约88核，保留GPU训练与原生辅助。最新各slot40倍数点先评，再按80更新间隔回补，u+1/u+2仅技术记录；无效谱系移到excluded_requests并保留证据。先写自然局/GATE，再异步保留清砖快筛，报表/上传独立进程不阻塞领取。新chunk320更新=83,886,080环境步，保存/评估仍每40更新。独立prewarm命名空间在当前GPU训练重叠做2次真实技术更新与fresh restore，严禁把它们计正式步数或能力样本；新chunk从真实最终原子点完整恢复。旧80协议和在途worker不改。GPU2继续新规则JAX rule3辅助，与JS Hunter分列，弱对手饱和不能当晋升证据。

07:00批复与本轮执行：

1. 仅批准信号漏斗审计，不调权重。首次威胁0.02、拓扑0.04、时间连接0.01及cap0.20在带塑形线保持；系数翻倍暂缓。当前约0.5%的实付占比不能证明唯一根因。下面原生聚合审计严格标出历史缺失字段，不能冒称事件级转化链已齐。
2. 熵下限/探索损失提案驳回，没有加入任何此类新损失或mask改动。可被动观察分阶段熵和概率；少放泡本身不证明熵塌缩。
3. 批准的两次复制父点固定为`runs/bomb_threat_v1/formal/bt_slot00_c000003/u000360`，dev 21/41/2、击杀27、28.5泡/局、零泡5；holdout 17/46/1、击杀31、31.3泡/局、零泡11，均通过相对门槛。两个子slot从该点完整接管Actor/Critic/Adam/env/aux/ring，子seed为2026112903/2026112904；原样source NEXT_STEP先验收，再对RNG重新派生、仅清零新塑形余额，最后按目标slot已冻结对手配方重映射combat_mix.kind，完整前后摘要写PBT_HANDOFF/LAUNCH。此重映射不能叫完整aux逐位不变。旧失败预热及原子暂停点全部保留、不续训。
4. 后续复制授权：仅在安全原子边界、相对闸门通过且同checkpoint已完成holdout的候选可复制parent线排名末位，每轮最多2条；slot05/06永不复制。当前这轮的2次额度由slot03/04占满，不因新轮报告提前重复复制同一父点。风险仍包括选择噪声和种群趋同，不把相对合格等同绝对改善。

两条无塑形分支属于批准的PBT观察对照。与持续带塑形线还存在复制时点/训练年龄/策略谱系及目标slot原配方差异，不能把后续分数差全部归因于关闭塑形；没有新增宣称严格成对单变量实验。

晋升：本轮明确不晋升BEST、不push、不改网页默认或models.json。后续任何晋升仍必须同checkpoint在holdout seed 2026132900双模式各64完整局并通过绝对闸门。旧O5/R2仍暂停。09:00修订2覆盖旧阈值：PBT相对不要求holdout；BEST要求双模式holdout64、绝对条件及fresh每档256复核，当前不晋升。

限制：直接自伤只计ownBombDefeat，不覆盖所有间接死因；reward占比为目标chunk的原生前缀，非JS推算且含部分对局；尚未评的当前点不能写成通过或改善。逐局分片、策略hash、spawn/Sim hash均在各评估目录；原有NATURAL_COMPLETE不可变，派生GATE_METRICS可重算。
'''
    acceptance=read(BASE/'PBT_COPY_ACCEPTANCE_20261029_0700.json',{})
    if acceptance:
        text+='\n复制正式首更新验收：`runs/bomb_threat_v1/PBT_COPY_ACCEPTANCE_20261029_0700.json`，'+acceptance['status']+'。\n\n'
        text+='|slot/run|首更新/新增环境步|Actor/Critic ΔL2|state SHA256|policy SHA256|\n|---|---|---|---|---|\n'
        for r in acceptance.get('slots',[]):
            first=r['formal_first_update'];m=first['metrics'];delta=m['update_l2']
            text+=f"|{r['slot']:02d}/{r['run']}|{first['update']}/{m['actual_environment_steps_this_update']}|{delta['actor']:.6f}/{delta['critic']:.6f}|{first['state_sha256']}|{first['policy_sha256']}|\n"
        text+='\n首周期约68秒含技术准备，不是稳态吞吐。用户30分钟恢复目标未达到：donor校验前误改combat_mix.kind、worker未消费已批准暂停覆盖及新命名空间冷编译造成延迟；失败证据保留。当前首更新已验，健康训练继续，评估异步。\n'
    first_evals=[]
    for name in ('bt_slot03_c000012_u000400','bt_slot04_c000011_u000400',
                 'bt_slot03_c000012_u000400_holdout','bt_slot04_c000011_u000400_holdout'):
        metric=read(BASE/'eval'/name/'GATE_METRICS.json',{})
        if metric:first_evals.append((name,metric))
    if first_evals:
        text+='\n新分支固定首评u400（不是择峰；最新后续点见开头快照）：\n\n|checkpoint/集合|模式/局数|胜/平/负|击杀/死亡|泡/局/零泡局|相对/绝对闸门|\n|---|---|---|---|---|---|\n'
        for name,metric in first_evals:
            for mode,g in metric['natural_js_full2400'].items():
                text+=f"|{name}|{mode}/{g['games']}|{g['win']}/{g['draw']}/{g['loss']}|{g['causal_kills']}/{g['deaths']}|{g['bombs_per_game']:.3f}/{g['zero_bomb_games']}|{metric['pbt_selection_eligible']}/{metric.get('absolute_gate_pass',metric['best_eligible'])}|\n"
        text+='\n闸门为checkpoint整体判定、以argmax为主，并非表内每种动作模式独立判定。holdout仅完成者列数值，不能把dev相对通过替代holdout；本轮不晋升BEST，后续回落也保留。\n'
    throughput=read(BASE/'PBT_WARM_THROUGHPUT_20261029.json',{})
    if throughput:
        text+='\n暖态实测（每路20个真实正式更新，原始行与SHA见`PBT_WARM_THROUGHPUT_20261029.json`）：'
        text+='；'.join(f"slot{r['slot']:02d} u{r['updates'][0]}–{r['updates'][1]}，中位周期{r['median_cycle_seconds']:.3f}s、{r['median_environment_steps_per_second']:.0f}环境步/s" for r in throughput['rows'])
        text+='。这些是演进策略的观察性稳态周期，与冻结参考20255.36步/s并非同轨迹因果性能对照；未用显存占用冒称计算。首验连续GPU3/4利用率81%–87%、功率约283–292W。\n'
    audit=read(BASE/'signal_audit_round3/RESULT.json',{})
    if audit:
        text+='\n信号审计：`runs/bomb_threat_v1/signal_audit_round3/RESULT.json`。7个冻结原子窗口、每个40更新/10,485,760环境tick；完整state/policy及所选原始行SHA验收。仅原生训练计数，无JS奖励回算，未改训练。\n\n'
        text+='|slot/u窗口|请求→实际落泡/拒绝|首次安全威胁/实际落泡|拓扑事件/实际落泡|信号并集|正实付env-tick|新塑形/净总奖励|\n|---|---|---|---|---:|---:|---:|\n'
        for a in audit['windows']:
            s=a['raw_counters']
            text+=f"|{a['slot']:02d}/{a['start_update']}–{a['end_update']}|{s['learner_bomb_requests']:.0f}→{s['learner_bombs_placed']:.0f}/{s['learner_bomb_requests_rejected']:.0f}|{a['first_safe_threat_events_per_actual_place']:.3%}|{a['topology_events_per_actual_place']:.3%}|{s['chain_candidates']:.0f}|{a['positive_payment_transitions']:.0f}|{a['payment_over_net_reward']:.3%}|\n"
        text+='\n已实现/可复算：请求=实际落泡+拒绝逐窗严格成立；接受的首次安全威胁约占实际落泡14%–21%、拓扑改善约3.5%–4.3%，正奖励确有结算，新塑形占净总奖励约0.49%–0.57%。因此不是“从未产生任何信号/未接入奖励”；也不能由此断言权重弱是少放泡唯一原因。**缺失：**历史日志未保存合法/安全机会总分母、chain-only事件、pending创建/按原因作废/窗口首尾余额、事件ID到终局的链接，以及原生分阶段动作概率/熵。`chain_candidates`是三类信号并集，`chain_settled`是正付款env-tick，不是逐信号转化率或完成局数；拓扑付款计数可含跨窗余额，不能据接近1的比值证明无作废。完整事件级漏斗仍未完成，后续只补被动审计，不改当前冻结训练。\n'
    trace_manifest=read(BASE/'signal_trace_round3/MANIFEST.json',{})
    if trace_manifest:
        trace_rows=[read(p) for p in sorted((BASE/'signal_trace_round3').glob('*/RESULT.json'))]
        trace_complete=read(BASE/'signal_trace_round3/COMPLETE.json',{})
        text+=f"\n原生被动轨迹补审计：{len(trace_rows)}/{len(trace_manifest['checkpoints'])}来源已完成，整体状态`{trace_complete.get('status','RUNNING')}`。固定每来源8个真实保存环境、256tick上限，无reset/优化/老师接管；源对手kind与source RNG保留。各来源首步在CPU同状态/同动作/同RNG下，新增统计前后实际transition与原有metrics摘要逐位一致；不宣称所有后续轨迹逐位对齐或与GPU轨迹一致，也不是从出生开始的完整自然局。\n\n"
        text+='|来源|有效转移/结束局/删失局|实落泡/安全落泡|威胁/拓扑/连锁事件|期初pending+准入|实付/坏事件作废/终态作废/期末pending|cap裁剪/守恒残差|\n|---|---|---|---|---|---|---|\n'
        for r in trace_rows:
            c=r['counters']
            text+=f"|{r['name']}|{r['actual_observed_transitions']}/{r['completed_source_episodes']}/{r['censored_source_episodes']}|{c['actual_place']:.0f}/{c['safe_placed']:.0f}|{c['first_safe_threat']:.0f}/{c['topology_events']:.0f}/{c['chain_events']:.0f}|{r['opening_pending']:.4f}+{c['admitted']:.4f}|{c['paid']:.4f}/{c['cancel_bad']:.4f}/{c['cancel_terminal']:.4f}/{r['closing_pending']:.4f}|{c['clipped']:.4f}/{r['mass_balance_residual']:.2g}|\n"
        text+='\n分阶段合法放泡观察（能力头masked categorical分布，与实际评估argmax分开）：\n\n|来源/阶段|可观察tick|合法放泡机会|条件平均P(放泡)|条件平均ability熵|请求/实际落泡|\n|---|---:|---:|---:|---:|---|\n'
        for r in trace_rows:
            for label,p in r['phases'].items():
                prob=p['mean_bomb_probability_given_legal'];entropy=p['mean_ability_entropy_given_legal']
                text+=f"|{r['name']}/{label}|{p['observed_ticks']}|{p['legal_bomb_opportunities']}|{prob if prob is not None else '未覆盖'}|{entropy if entropy is not None else '未覆盖'}|{p['bomb_requests']}/{p['actual_bombs']}|\n"
        text+='\n余额满足期初pending+准入=实付+按原因作废+期末pending。期末余额是删失，不当作失败；offered与cap裁剪单列，不把cap裁剪混同危险作废。每源只有8个保存状态，来源/对手/阶段暴露不同，观察到低放泡概率或低熵不证明训练期熵塌缩，不支持加探索损失。缺失覆盖及逐信号到终局因果链仍按上节限制报告，不用短窗口代替能力或完整自然局奖励对齐。所有trace JSON及SHA见`signal_trace_round3/*/RESULT.json`。\n'
    trace_verified=read(BASE/'signal_trace_round3/VERIFICATION.json',{})
    if trace_verified:
        text+=f"\n被动轨迹独立复算验收：`signal_trace_round3/VERIFICATION.json`，{trace_verified['status']}，{len(trace_verified['sources'])}来源/{trace_verified['actual_transitions']}有效转移；校验源checkpoint、源码与逐段trace SHA，逐tick余额/取消原因守恒、完成/删失分母及分阶段概率和熵。复算命令：`PYTHONPATH=. .venv/bin/python runs/bomb_threat_v1/signal_trace_round3/verify_trace.py`；只读CPU统计，不创建训练更新。\n"
    tests=read(BASE/'PBT_COPY_TEST_ACCEPTANCE_20261029.json',{})
    if tests:
        text+=f"\n本次回归验收：`runs/bomb_threat_v1/PBT_COPY_TEST_ACCEPTANCE_20261029.json`，{tests['status']}。Python相关{tests['python']['passed_progress_marks']}项无失败，JS {tests['javascript']['files_passed']}个测试文件通过（含14个席位×库存状态案例）；原始日志SHA在验收收据。全套启动后追加的队列/谱系报告小改另由focused回归覆盖，不将早期suite冒称覆盖后续修改。\n"
    tmp=path.with_suffix('.md.tmp');tmp.write_text(text);tmp.replace(path)
    evidence=BASE/'round_reports'/f'round{number}.json';atomic_json(evidence,dict(round=number,report=str(path.relative_to(ROOT)),report_sha256=sha(path),slots=rows,unix=now))
    atomic_json(BASE/'ROUND_REPORT_STATUS.json',dict(last_round=number,report=str(path.relative_to(ROOT)),last_unix=now,next_due_unix=now+7200))
    from scripts.iter_pbt_owner import notify
    notify(f'BOMB_THREAT_ROUND_{number}',f'BOMB-THREAT round{number}已落盘：批准的无塑形复制/逐slot闸门/原生信号审计与缺项；健康冻结训练继续，CPU补评独立。',report=str(path.relative_to(ROOT)))
    return str(path.relative_to(ROOT))

if __name__=='__main__':print(write_round(force='--force' in __import__('sys').argv))
