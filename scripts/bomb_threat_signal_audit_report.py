"""Recount retained traces and distinguish native aggregates from missing data."""
import json,time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/bomb_threat_v1'

def fullgame_rows(output):
    complete=json.loads((output/'COMPLETE.json').read_text());manifest=json.loads((output/'MANIFEST.json').read_text())
    assert complete['manifest_sha256']==sha(output/'MANIFEST.json')
    rows=[]
    for r in complete['results']:
        chunks=[]
        for file,h in r['trace_files'].items():
            assert sha(ROOT/file)==h;chunks.append(json.loads((ROOT/file).read_text()))
        j={k:np.concatenate([np.asarray(c[k]) for c in chunks]) for k in chunks[0]}
        assert len(r['episodes'])==manifest['games_per_mode']
        live=j['live'];assert np.max(np.abs(j['shadow_balance'][live]),initial=0)<1e-6
        assert np.max(np.abs(j['balance_residual'][live]),initial=0)<1e-6
        for e in r['episodes']:
            i=e['index'];v=live[:,i]
            for key,metric in [('first_safe_threat','threat_events'),('topology_events','topology_events'),('chain_events','chain_events'),
                ('bomb_request','requests'),('actual_place','actual_bombs'),('rejected','rejected'),('paid','paid'),('cancelled','cancelled')]:
                np.testing.assert_allclose(j[key][v,i].sum(),e[metric],atol=1e-6)
            assert int(j['completed'][v,i].sum())==int(e['complete'])
            assert e['requests']==e['actual_bombs']+e['rejected']
            assert sum(e['reject_reasons'].values())==e['rejected']
        n=len(r['episodes']);sumkey=lambda k:sum(e[k] for e in r['episodes'])
        row=dict(name=r['name'],mode=r['mode'],games=n,completed=r['complete_games'],
            **{k:sumkey(k) for k in ('threat_events','topology_events','chain_events','requests','actual_bombs','rejected','paid','admitted',
                'cancelled','cancel_bad','cancel_terminal','cancel_inactive','closing_pending','deaths','deliveries','net_reward','preterminal_ticks','tail_ticks')},
            policy_sha256=r['policy_sha256'],trace_files=len(chunks),first_transition_passive_exact=r['first_transition_passive_exact'])
        row['paid_by_type_proportional_shadow']=np.sum([e['paid_by_type_proportional_shadow'] for e in r['episodes']],axis=0).tolist()
        np.testing.assert_allclose(sum(row['paid_by_type_proportional_shadow']),row['paid'],atol=1e-6)
        row['cancelled_over_admitted']=row['cancelled']/row['admitted'] if row['admitted'] else None
        row['reject_reasons']={str(i):sum(e['reject_reasons'][str(i)] for e in r['episodes']) for i in range(6)}
        rows.append(row)
    atomic_json(output/'VERIFICATION.json',dict(status='PASS',complete_sha256=sha(output/'COMPLETE.json'),
        verifier_sha256=sha(__file__),rows=rows,actual_games=sum(r['games'] for r in rows),new_training_steps=0,
        historical_state_reverified=False,unix=time.time()))
    return complete,rows


def write_report():
    aggregate_path=BASE/'signal_audit_round3/RESULT.json';aggregate=json.loads(aggregate_path.read_text())
    old=BASE/'signal_trace_round3';historical=json.loads((old/'COMPLETE.json').read_text())
    for r in historical['results']:
        for file,h in r['trace_files'].items():assert sha(ROOT/file)==h
    output=BASE/'signal_audit_fullgames_20261029_attempt02'
    full=None;rows=[]
    if (output/'COMPLETE.json').exists():full,rows=fullgame_rows(output)
    text=['# BOMB-THREAT 信号漏斗审计（20261029）','',
        '09:00修订2与07:00批准的被动审计，不改奖励/权重/闸门之外的训练语义。三项系数仍为威胁0.02、拓扑0.04、连锁0.01，每局gross cap0.20；slot03/04批准的无塑形支线仍全0。老师关闭；没有新增探索损失。',
        '', '已实现、可复算：原生训练7个冻结40更新窗口（各10,485,760环境tick）、已有3组8个保存状态的256tick有界trace、以及下面独立自然完整局补测。账本支付、撤销、pending/cap与真实放泡事件均读训练引擎，不用JS近似回算。历史逐行/trace SHA可复核；按删除授权，历史state SHA只来自COMMITTED，不能重验其已删除文件或重建当时env/critic。',
        '', '历史训练聚合（分母是重置/采样窗口，不是完整自然局）：',
        '', '|slot/窗口|请求/实际/拒绝|威胁/实际泡|拓扑/实际泡|三项实付/净奖励|', '|---|---|---:|---:|---:|']
    for w in aggregate['windows']:
        c=w['raw_counters'];text.append(f"|{w['slot']}/{w['start_update']}–{w['end_update']}|{c['learner_bomb_requests']}/{c['learner_bombs_placed']}/{c['learner_bomb_requests_rejected']}|{w['first_safe_threat_events_per_actual_place']:.3%}|{w['topology_events_per_actual_place']:.3%}|{w['payment_over_net_reward']:.3%}|")
    text+=['', '历史三项实付仅有一个合并channel，`chain_candidates`是事件并集，不是chain-only；`chain_settled`是正支付env-tick，不是成功局数。约0.5%的实付与少放泡同时出现，只是线索，不能证明唯一根因或调高系数必然改善。',
        '', '已有有界状态trace（不能回答整局每局问题）：', '', '|来源|有效转移|完结/删失|威胁/拓扑/连锁|付/撤销/尾余额|', '|---|---:|---|---|---|']
    for r in historical['results']:
        c=r['counters'];text.append(f"|{r['name']}|{r['actual_observed_transitions']}|{r['completed_source_episodes']}/{r['censored_source_episodes']}|{c['first_safe_threat']}/{c['topology_events']}/{c['chain_events']}|{c['paid']:.6f}/{c['cancelled']:.6f}/{r['closing_pending']:.6f}|")
    text+=['', '这24个状态并非从出生采集；即使其中9个走到终局，也不能把缺少的前缀当成0。撤销分母须含opening pending，尾余额仍删失。',
        '', '独立自然完整局：共同seed2026142900起8个固定episode index，layout0、seat0、JAX rule3（非生产JS Hunter、非训练current/ring/fixed混合）；donor u360和slot00 u960，每点greedy/categorical各8局，总32。出生空手、真实库存取包，原生2400tick+最多160结算尾，teacher/optimizer关闭。不能用此小样本调参或替代训练/JS能力评估。',
        '', '首尝试在CPU初始化/编译阶段超过6分钟仍0轨迹；保留无观测停机收据，未删除已完成局。attempt02仅将params/mode动态传入并复用JIT及显式阶段进度，科学条件不变。']
    if full:
        text+=['', f"补测状态：{full['status']}；墙钟{full['elapsed_seconds']:.1f}秒，独立逐trace复算PASS。", '',
        '|来源/模式|完整/请求局|每局威胁/拓扑/连锁|总实付/每局实付|撤销/准入|拒绝/请求|死亡/交付|', '|---|---|---|---|---|---|---|']
        for r in rows:
            n=r['games'];ratio=r['cancelled_over_admitted'];text.append(f"|{r['name']}/{r['mode']}|{r['completed']}/{n}|{r['threat_events']/n:.3f}/{r['topology_events']/n:.3f}/{r['chain_events']/n:.3f}|{r['paid']:.6f}/{r['paid']/n:.6f}|{r['cancelled']:.6f}/{r['admitted']:.6f} ({ratio if ratio is not None else 'undefined'})|{r['rejected']}/{r['requests']}|{r['deaths']}/{r['deliveries']}|")
        text+=['', '每类支付的边界：原引擎只有合并pending，不保留原生按类归属；下面采用“每tick按实际准入offer比例分摊cap、撤销与合并支付”的被动影子账本。三类之和逐步等于真实合并支付，不能称引擎原生逐类因果贡献；真正原生分项历史不可获得。',
            '', '|来源/模式|威胁/拓扑/连锁分摊实付|bad撤销/terminal撤销/inactive撤销|拒绝原因0/1/2/3/4/5|首terminal前/结算尾tick|', '|---|---|---|---|---|']
        for r in rows:
            text.append('|'+r['name']+'/'+r['mode']+'|'+'/'.join(f'{x:.6f}' for x in r['paid_by_type_proportional_shadow'])+'|'+ '/'.join(f'{r[k]:.6f}' for k in ('cancel_bad','cancel_terminal','cancel_inactive'))+'|'+'/'.join(str(r['reject_reasons'][str(i)]) for i in range(6))+f"|{r['preterminal_ticks']}/{r['tail_ticks']}|")
    else:text+=['','补测尚在CPU运行，未写COMPLETE，不把launch/进程存在或编译计成完成；以该目录PROGRESS.json和后续VERIFICATION.json为准。']
    text+=['', '拒绝原因分层：0=取动作前mask合法但最终未落泡，具体移动/执行顺序原因未知；1=terminal账本尾；2=死亡；3=原生包裹；4=持包；5=其他取动作前mask不合法。只对实际请求且未落泡归因，优先级terminal→死亡→包裹→持包→其他mask→未知，互斥计数与请求=落泡+拒绝均可复算。mask合法不是执行一定成功，也不是动作有益。',
        '', '缺历史数据：训练整局来源ID/首tick到终局链接、原生每类支付channel、完整请求拒绝原因、反制与直接自伤的分别撤销金额，不能从聚合恢复。补测bad撤销仍是死亡/被控等联合条件，不能擅称自伤；bad事件标签不等于已证明被敌人反制。未知放置顺序原因保留未知。没有撤销不证明不存在风险，低覆盖不能推全局。',
        '', '不可获得：已删除历史state的重新rollout/完整Critic/Adam/env/RNG核验；未记录的旧事件ID归因。保留的policy、COMMITTED、逐局JSON和trace仍可核验，不把历史state SHA记录当作当前文件存在证明。',
        '', '结论与最小后续：当前不调整系数或加入熵/探索约束。先区分动作选择（合法放泡概率/模式差异）、真实落泡、触发、cap/撤销、结算、终局链；若提出改系数，必须基于原生整局实付量级给出具体新值/预计占比/终局优先与cap、防重复事件，提交编排批准后另版本实施。当前小样本/历史聚合不足以给可靠预期倍数，故不擅报系数方案已生效。',
        '', '复现：`.venv/bin/python -m scripts.bomb_threat_full_signal_audit runs/bomb_threat_v1/signal_audit_fullgames_20261029_attempt02/MANIFEST.json`（COMPLETE存在不重跑）；`.venv/bin/python -m scripts.bomb_threat_signal_audit_report`只重算保存trace。完整逐局/逐tick/检查点policy SHA与实际墙钟均在上述目录。']
    target=ROOT/'reports/bomb_threat_v1_signal_audit_20261029.md';target.write_text('\n'.join(text)+'\n')
    atomic_json(BASE/'SIGNAL_AUDIT_REPORT_STATUS.json',dict(status='COMPLETE_WITH_DISCLOSED_LIMITATIONS' if full else 'PENDING_FULLGAME_AUDIT',
        report=str(target.relative_to(ROOT)),report_sha256=sha(target),fullgame_status=full['status'] if full else None,unix=time.time()))
    return target

if __name__=='__main__':print(write_report().relative_to(ROOT))
