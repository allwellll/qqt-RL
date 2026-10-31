"""Finalize only after complete fixed native and production-engine evidence."""
import json,time,subprocess
import numpy as np
from scripts.carrier_mechanism_run import B,R,read,notify
from scripts.carrier_mechanism_stats import cluster_interval
from scripts.population_run_state import atomic_json,sha


def f(x):return 'NA' if x is None else f'{x:.4f}'


def natural_observation(result):
    """Use episode counts, never the legacy merger's sum of per-episode ratios."""
    episodes=result['results']['natural_full2400']['episodes'];n=len(episodes)
    sums={key:sum(e.get(key,0) for e in episodes) for key in [
        'window_events','target_events','first_oracle_actions','first_oracle_matches',
        'oracle_opportunities','oracle_matches','target_ticks','progress_ticks',
        'window_stock_pickup','stock_pickup_dead','stock_pickup_native_sugar',
        'stock_pickup_free_alive','window_stock_delivery','free_stock_delivery',
        'nonstock_pickup','nonstock_delivery','recovery_switch_opportunities','recovery_switches']}
    binary={key:np.array([int(e.get(key,0)>0) for e in episodes]) for key in ['pickup','delivery','death','self_death']}
    stages={}
    for e in episodes:
        for phase,metrics in e.get('stage_funnel',{}).items():
            dest=stages.setdefault(phase,{})
            for key,value in metrics.items():dest[key]=dest.get(key,0)+value
    return dict(games=n,episode_ids=[(e['game'],e['seed'],e['team']) for e in episodes],
        episode_counts={k:int(v.sum()) for k,v in binary.items()},event_counts=sums,stage_funnel=stages,
        tick_oracle_agreement=sums['oracle_matches']/max(1,sums['oracle_opportunities']),
        first_phase_oracle_agreement=sums['first_oracle_matches']/max(1,sums['first_oracle_actions']),
        progress_per_target_tick=sums['progress_ticks']/max(1,sums['target_ticks']),
        denominator_warning='Event counts, ticks and episode counts are separate; no cross-unit chain conversion; legacy summed ratios and INF-sensitive net_progress excluded'),binary


def finalize():
    if (B/'DIAGNOSTIC_COMPLETE.json').exists():return
    assert read(B/'C_NATIVE_BANK_COMPLETE.json')['passed'] and read(B/'CPU_EVALUATION_COMPLETE.json')['passed']
    from scripts.carrier_mechanism_acceptance import verify
    if not (B/'C_ATOMIC_ACCEPTANCE.json').exists():verify()
    assert read(B/'C_ATOMIC_ACCEPTANCE.json')['passed']
    while not (B/'GPU_PIPELINE_COMPLETE.json').exists():
        if (B/'USER_STOP.json').exists():return
        time.sleep(2)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    active=[]
    from pathlib import Path
    for procdir in Path('/proc').glob('[0-9]*'):
        try:
            args=(procdir/'cmdline').read_bytes().split(b'\0')
            if ((b'scripts/carrier_mechanism_run.py' in args and args[-2:-1] in [[b'C'],[b'AB'],[b'collect'],[b'parity']])
                or b'scripts.carrier_mechanism_analysis' in args
                or b'scripts.carrier_mechanism_train' in args):active.append(True)
        except (OSError,PermissionError):pass
    assert not active,'scoped GPU worker still active'
    atomic_json(B/'GPU_RELEASE_RECEIPT.json',dict(scoped_workers=0,worker_exit_verified=True,gpus=[list(map(int,line.split(','))) for line in gpu.strip().splitlines()],old_continuous_resume=False,automatic_successor=False,unix=time.time()))
    assert read(B/'PARITY_COMPLETE.json')['passed'];AB=read(B/'AB_COMPLETE.json');C=read(B/'C_NATIVE_BANK_COMPLETE.json');P=read(B/'PROTOCOL.json')
    base=B/'C/no_update/u000';control=read(base/'retention/COMPLETE.json');rows=[];retention={}
    for arm in P['C']['arms']:
        observations=[]
        for seed in P['C']['paired_seeds']:
            cp=B/'C'/f'{arm}_s{seed}'/'u016';r=read(cp/'retention/COMPLETE.json');observations.append(r)
            native=read(cp/'native_holdout.json')
            observations[-1]['native']=native
        delta={}
        for tier in ['hard','normal','easy']:
            metrics={}
            for key,value in control['hunter'][tier].items():
                if not isinstance(value,(int,float)) or key=='games':continue
                d=np.array([x['hunter'][tier][key]-value for x in observations]);metrics[key]=dict(delta=d.tolist(),mean=float(d.mean()),paired_training_seed_CI=cluster_interval(d,P['C']['paired_seeds']))
            delta[tier]=metrics
        baseline,base_binary=natural_observation(control['natural']);natural=[];binary=[]
        for observation in observations:
            row,values=natural_observation(observation['natural']);assert row['episode_ids']==baseline['episode_ids']
            natural.append(row);binary.append(values)
        natural_delta={}
        for key in base_binary:
            paired=np.array([x[key]-base_binary[key] for x in binary]);seed_delta=paired.mean(-1)
            rng=np.random.default_rng(20261023);si=rng.integers(0,5,(5000,5));ei=rng.integers(0,baseline['games'],(5000,baseline['games']))
            boot=paired[si[:,:,None],ei[:,None,:]].mean((1,2))
            natural_delta[key]=dict(delta_rate=seed_delta.tolist(),mean_delta_rate=float(seed_delta.mean()),
                paired_training_seed_CI=cluster_interval(seed_delta,P['C']['paired_seeds']),
                paired_seed_and_episode_CI=np.quantile(boot,[.025,.975]).tolist())
        update_signals=[]
        for seed in P['C']['paired_seeds']:
            updates=[json.loads(line) for line in (B/'C'/f'{arm}_s{seed}'/'updates.jsonl').read_text().splitlines()]
            assert [u['update'] for u in updates]==list(range(1,P['C']['updates']+1))
            exposure={phase:{key:sum(u['route_exposure'][phase][key] for u in updates) for key in ['custody_rows','teacher_rows','oracle_consistent_actions']} for phase in ['near','mid','far']}
            counts={key:sum(u['signal'][key] for u in updates) for key in ['eligible_rows','set_disagreement_rows','disagreement_positive_raw_GAE','disagreement_positive_normalized_GAE']}
            update_signals.append(dict(seed=seed,phase_exposure=exposure,counts=counts,
                first_update_l2=updates[0]['update_l2'],last_update_l2=updates[-1]['update_l2'],
                environment_steps=updates[-1]['environment_steps'],optimizer_steps=updates[-1]['optimizer_steps']))
        retention[arm]=dict(Hunter_delta=delta,tactical=[x['tactical'] for x in observations],native_holdout=[x['native']['delivery'] for x in observations],
            native_holdout_calibration=[x['native']['calibration_terminal_complete_states'] for x in observations],
            native_holdout_initial_teacher_fit=[x['native']['initial_safe_teacher_fit'] for x in observations],
            actual_update_signals=update_signals,natural=natural,natural_control=baseline,natural_paired_delta=natural_delta)
    atomic_json(B/'FINAL_COMPARISON.json',dict(passed=True,AB=AB['models'],C_native=C['summary'],C_retention=retention,selection='bounded mechanism identification; no publication/promotion or automatic successor',old_continuous_resume=False,holdout_tuning=False,unix=time.time()))
    lines=['\n## 固定诊断终态\n',f"Bank {read(B/'BANK_FROZEN.json')['states']}独立状态；完整入站/取包账本/新进程next-step parity全部通过。所有结果由固定样本产生，无择峰。\n",'| Checkpoint | terminal MC EV | RMSE | bias | target variance | Bank交付 | 死亡 | 掉包 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,result in AB['models'].items():
        v=result['V'];c=v['calibration_terminal_complete_states'] or {}
        lines.append('| '+model+' | '+' | '.join(f(c.get(k)) for k in ['explained_variance','rmse','bias','target_variance'])+' | '+' | '.join(f(v[k]) for k in ['delivery','death','drop'])+' |')
    lines+=['\n| Checkpoint | subset Q contrast | ΔQ | 来源局cluster95% CI | 重建实际GAE差 |','|---|---|---:|---|---:|']
    for model,result in AB['models'].items():
        for name,row in result['Q'].items():lines.append(f"| {model} | {name} | {f(row['mean'])} | {row['source_episode_cluster_CI']} | {f(row['GAE_first_delta_mean'])} |")
    lines+=['\nC固定最终holdout（三臂各5配对seed；共享无更新control）：','\n| 臂 | 5 seed交付 | 相对control平均 | 配对训练seed95% CI |','|---|---|---:|---|']
    for arm,row in C['summary']['holdout']['arms'].items():lines.append(f"| {arm} | {row['delivery']} | {f(np.mean(row['delta_to_control']))} | {row['paired_training_seed_CI']} |")
    lines+=['\n自然生产JS完整64局/训练seed（control为共享父点，5个seed不等于5份独立control）：',
        '\n| 臂 | 5 seed交付局数/64 | 相对control交付率均差 | 训练seed×对局配对95% CI |','|---|---|---:|---|']
    for arm,row in retention.items():
        d=row['natural_paired_delta']['delivery']
        lines.append(f"| {arm} | {[x['episode_counts']['delivery'] for x in row['natural']]} | {f(d['mean_delta_rate'])} | {d['paired_seed_and_episode_CI']} |")
    lines+=['\n自然完整对局同时报告拾取/死亡/自伤的配对率变化，以及原始事件/阶段/机会分母；不拿条件Bank成功代替自然链。旧合并器的逐局比例求和和含INF跃迁的net_progress不作能力依据。',
        '\n原生首偏航Bank-P保留全部死亡/丢包，事件按偏航transition及之后统计。首次老师集合外动作不等于失败，更不自动证明负优势；重建原始GAE与Q子集对照单列。生产JS Review沿用固定首seed唯一链路，其奖励/优势若为离线近似或未记录，则明确标注；不冒充本诊断JAX实际重建GAE。']
    if (B/'Q_GAE_SIGNAL_ALIGNMENT.json').exists():
        lines.append('\nQ/GAE来源状态均值排序一致性见 `Q_GAE_SIGNAL_ALIGNMENT.json`；有限8尾MC不确定性保留，不由负EV或部分符号不一致直接断言PPO无效。')
    lines+=['\n生产JS Hunter难/中/易各64、native JS自然完整64、Python Tactical v2各64逐checkpoint结果与保留差值在 `FINAL_COMPARISON.json`；三类引擎独立报告，不互换。完整dev保存点曲线均保留，holdout仅最终固定点。5训练seed CI与来源局重复rollout CI分列。','\n解释约束：低EV必须结合回报方差、偏差、截断延长敏感性解释，不能直接归因为PPO噪声。B只覆盖前进/倒退/等待，不是全合法联合动作优势。C三臂会产生不同learner-state分布和老师有效曝光；即便单臂胜出也不能单凭这一结果证明干扰。网络为独立Actor/Critic，没有共享trunk因果结论。Bank是条件于真实取得库存的返程估计，自然完整链和战斗保持独立。','\n复现（须保留本诊断scoped授权且旧暂停hash不变）：','```sh','.venv/bin/python scripts/carrier_mechanism_run.py owner','.venv/bin/python scripts/carrier_mechanism_cpu.py','```','owner按原子marker跳过已完成阶段，绝不恢复旧continuous。所有状态、环境/更新步、排除、输入hash、恢复路径与源身份均保存在独立namespace。','\n本诊断GPU预算结束，不启动旧O5、储备队列或后继，不发布。']
    from scripts.carrier_mechanism_plot import plot
    plot()
    lines.append('\n![固定实测图](carrier_mechanism_diagnostic_20261023.png)')
    report=R/'reports/carrier_mechanism_diagnostic_20261023.md';report.write_text(report.read_text()+'\n'.join(lines)+'\n')
    ledger=R/'docs/training_experiment_ledger.md';ledger.write_text(ledger.read_text()+'\n### CAUSAL-CARRIER-MECHANISM-20261023-v1 完整终态\n\nA/B实测、C三臂五seed固定更新、全部保存点dev/最终holdout与CPU/JS战斗保持已消费。固定结果和限制见 `reports/carrier_mechanism_diagnostic_20261023.md` 与独立namespace FINAL_COMPARISON；无模型发布、无自动后继，旧O5/储备暂停保留。\n')
    atomic_json(B/'DIAGNOSTIC_COMPLETE.json',dict(passed=True,fixed_results_complete=True,report='reports/carrier_mechanism_diagnostic_20261023.md',comparison_sha256=sha(B/'FINAL_COMPARISON.json'),old_continuous_resume=False,automatic_successor=False,unix=time.time()))
    notify('真实持包诊断完整终态：A/B parity与V/Q、C三臂五seed和固定CPU/JS保留集全部完成；报告已冻结，GPU停止，旧O5/储备继续暂停，不发布。')

if __name__=='__main__':finalize()
