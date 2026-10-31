"""Recomputable scale summary from frozen runs, evaluations and GPU samples."""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

B=Path('runs/authorized_campaign_20261026')


def read(p):return json.loads(Path(p).read_text())


def main():
    campaign=read(B/'CAMPAIGN.json'); evidence=[]; runs=[]; evals=[]; stages=[]
    for group in ('scale_env_v1','scale_minibatch_v1','scale_lr_v1','multigpu_v1'):
        path=B/group/'SUMMARY.json';evidence.append(dict(path=str(path),sha256=sha(path)))
        for r in read(path)['rows']:
            root=B/group/f'{r["arm"]}_s{r["seed"]}'
            updates=root/'updates.jsonl';u=[json.loads(x) for x in updates.read_text().splitlines() if x.strip()]
            assert sha(updates)==r['raw_updates_sha256']
            cp=Path(r['endpoint']['checkpoint'])
            assert sha(cp/'state.pkl')==r['endpoint']['state_sha256']
            assert sha(cp/'policy.pt')==r['endpoint']['policy_sha256']
            job=next(j for j in campaign['jobs'] if j.get('complete')==str(root/'COMPLETE.json'))
            samples_path=B/'gpu_samples'/f'{job["id"]}.jsonl'
            sample_rows=[json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
            # Begin at completion of u2 so u3 collector/update and all warm boundaries are included.
            warm=[x for x in sample_rows if u[1]['unix']<=x['unix']<=u[-1]['unix']]
            devices=r['endpoint']['metrics']['configuration']['devices']
            samples=[g for row in warm for g in row['gpus'] if g['index']<devices]
            power=dict(samples=len(warm),device_samples=len(samples),
                       mean_active_card_watts=S.mean(g['power_watts'] for g in samples) if samples else None,
                       mean_active_card_util_percent=S.mean(g['utilization_percent'] for g in samples) if samples else None,
                       sampled_all_eight_card_watts=S.mean(sum(g['power_watts'] for g in row['gpus']) for row in warm) if warm else None,
                       window_start_unix=u[1]['unix'],window_end_unix=u[-1]['unix'],
                       excludes_cold_compile=True,not_continuous_energy_measurement=True)
            run=dict(group=group,arm=r['arm'],seed=r['seed'],
                     env_per_second=r['cycle_warm_env_per_second'],median_cycle_seconds=r['warm_cycle_median_seconds'],
                     total_env_steps=r['actual_environment_steps'],optimizer_steps=r['actual_optimizer_steps'],
                     configuration=r['endpoint']['metrics']['configuration'],power=power,
                     endpoint=r['endpoint']['checkpoint'],endpoint_policy_sha256=r['endpoint']['policy_sha256'],
                     finite_all=r['finite_all'],wall_seconds=job.get('terminal_unix',0)-job.get('started_unix',0),
                     checkpoint_state_sha256=r['endpoint']['state_sha256'])
            runs.append(run)
            evidence.extend([dict(path=str(updates),sha256=sha(updates)),dict(path=str(samples_path),sha256=sha(samples_path))])
            prefix=f'{group}_{r["arm"]}_s{r["seed"]}_'
            for summary_path in sorted((B/'scale_async_eval_v1').glob(prefix+'*/SUMMARY.json')):
                summary=read(summary_path)
                for row in summary['rows']:
                    cohort=row['passive_cohort']
                    # Model/policy equality from per-episode provenance, not filename identity alone.
                    if summary_path.parent.name.endswith(('_u000320_holdout','_u000160_holdout')):
                        assert all(ep['provenance']['policy_sha256']==run['endpoint_policy_sha256'] for ep in row['episodes'])
                    result=dict(group=group,arm=r['arm'],seed=r['seed'],condition=summary_path.parent.name,
                                split='holdout' if summary_path.parent.name.endswith('_holdout') else 'dev',
                                update=int(summary_path.parent.name.split('_u')[-1].split('_')[0]),mode=row['action_mode'],
                                games=row['games'],delivery=cohort['at_least_one_delivery_rate'],
                                deaths=cohort['own_deaths_per_match'],any_death=cohort['any_own_death_rate'],
                                net_wins=cohort['net_wins'],carry=cohort['carry_episodes'],windows=cohort['windows'],
                                source=str(summary_path),sha256=sha(summary_path))
                    evals.append(result)
                evidence.append(dict(path=str(summary_path),sha256=sha(summary_path)))
    calibration=read(B/'EVAL_GPU_CALIBRATION.json')
    atomic_json(B/'SCALE_RECOMPUTABLE_SUMMARY_20261027.json',dict(
        runs=runs,evaluations=evals,raw_sources=evidence,async_calibration=calibration,
        source_protocols={str(p):sha(p) for p in (B/'protocols').glob('scale*json')},
        multigpu_capability_evaluations_available=False,
        evaluation_scope='independent native JAX layout0/P0/rule3; not production JS Hunter or Python Tactical',
        created_by='scripts.campaign_scale_summary'))
    lines=['# 规模实验汇总（20261027）','',
        '结论：env32 是当前最有依据的吞吐候选，暖周期平均吞吐约提升六成，三seed自然采样交付也改善；但死亡增加、greedy 第二seed回落，且缺生产JS/Hunter和Tactical独立评估，**尚不能称“吞吐提升且整体能力/安全不降”**。保守兼容配置仍为 env16、batch16384、minibatch512、lr7.5e-5。lr5e-5 的能力结果值得单独保留，但它没有吞吐收益，不能与env32未测组合后推荐。','',
        '所有训练/checkpoint与GPU采样均为已有固定预算结果，本次没有重训或新增GPU任务；不择峰，报告固定终点与全部dev保存点。','',
        '## 配置与可比性','',
        '- SCALE共同父为各自R1 s2751/52/53 u064完整Adam/env/aux/RNG；三配对seed为2026103051/52/53；每run新增5,242,880真实环境步。固定legacy奖励、网络、rule3、Bank比例.25、DAgger/KL=0、rollout128、epoch1。',
        '- env32仅并行环境16→32/卡，global batch16384→32768，终点u160；其余u320。实际优化步以原始记录为准，env32与control均10,240；mb256为20,480。',
        '- minibatch实验仅512→256；LR实验仅7.5e-5→5e-5；不能合并两变量解释。各组control策略hash须分别核验，不能把重复相同轨迹当额外独立seed。',
        '- MULTIGPU固定global128环境、batch16384、minibatch512，仅重分片，1/2/4/8卡各131,072步、8更新、单seed。不是强扩展长训能力对照，shuffle/浮点归约未承诺逐位一致。','',
        '## 吞吐与设备观测','',
        'env/s使用排除前两更新的有效样本数/周期耗时；周期不包含全部冷启动、保存恢复墙钟。功率/util是u2完成至终点之间的离散采样均值（包含周期内host与保存空档），不是峰值kernel利用率或能耗积分。','',
        '| 组/臂 | seed尾号 | env/s | 暖周期中位s | active卡平均W/util% | 全8卡平均W | 采样数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        p=r['power'];lines.append(f'| {r["group"]}/{r["arm"]} | {str(r["seed"])[-2:]} | {r["env_per_second"]:.1f} | {r["median_cycle_seconds"]:.3f} | {p["mean_active_card_watts"]:.1f}/{p["mean_active_card_util_percent"]:.1f} | {p["sampled_all_eight_card_watts"]:.1f} | {p["samples"]} |')
    lines+=['','| 组 | control均值 env/s | 实验臂均值 env/s | 配对均值变化 |','|---|---:|---:|---:|']
    for group in ('scale_env_v1','scale_minibatch_v1','scale_lr_v1'):
        rows=[r for r in runs if r['group']==group];control=[r for r in rows if r['arm']=='control'];variant=[r for r in rows if r['arm']!='control']
        delta=[v['env_per_second']/next(c for c in control if c['seed']==v['seed'])['env_per_second']-1 for v in variant]
        lines.append(f'| {group} | {S.mean(r["env_per_second"] for r in control):.1f} | {S.mean(r["env_per_second"] for r in variant):.1f} | {S.mean(delta)*100:+.1f}% |')
    lines+=['','## 固定终点的独立能力评估','',
        '每个seed/臂dev64局/动作模式，holdout304局/动作模式；u1/中点dev另列在后表。所有为自然完整native JAX JE07、layout0、P0、固定rule3、老师关闭，评估CPU独立于训练；这些结果不能替代JS Hunter或Python Tactical。D/父点的其他引擎评估不是SCALE候选评估。表内“死亡”是事件数/局数，可大于1；交付是一局至少一次。','',
        '| 组/臂 | seed尾号 | dev categorical 交付/死亡 | dev greedy 交付/死亡 | holdout categorical 交付/死亡 | holdout greedy 交付/死亡 |',
        '|---|---:|---|---|---|---|']
    def cell(r):return f'{r["delivery"]["numerator"]}/{r["games"]}；{r["deaths"]["numerator"]}/{r["games"]}'
    for r in runs:
        if r['group']=='multigpu_v1':continue
        final=int(Path(r['endpoint']).name[1:]);rs=[e for e in evals if (e['group'],e['arm'],e['seed'],e['update'])==(r['group'],r['arm'],r['seed'],final)]
        cells=[cell(next(e for e in rs if e['split']==split and e['mode']==mode)) for split in ('dev','holdout') for mode in ('categorical','greedy')]
        lines.append(f'| {r["group"]}/{r["arm"]} | {str(r["seed"])[-2:]} | '+ ' | '.join(cells)+' |')
    lines+=['','三seed逐seed差值比把912局看作912个训练seed更重要；相同游戏seed用于配对，只有3个独立训练seed，不能凭均值或窄逐局CI证明泛化/安全非劣。','',
        '| 组 | holdout模式 | 实验臂−control交付局数（seed51/52/53，各304局） | 死亡事件差（同序） |','|---|---|---|---|']
    for group in ('scale_env_v1','scale_minibatch_v1','scale_lr_v1'):
        arm=next(r['arm'] for r in runs if r['group']==group and r['arm']!='control')
        for mode in ('categorical','greedy'):
            deltas=[];death=[]
            for seed in (2026103051,2026103052,2026103053):
                a=next(e for e in evals if (e['group'],e['arm'],e['seed'],e['split'],e['mode'])==(group,arm,seed,'holdout',mode))
                c=next(e for e in evals if (e['group'],e['arm'],e['seed'],e['split'],e['mode'])==(group,'control',seed,'holdout',mode))
                deltas.append(a['delivery']['numerator']-c['delivery']['numerator']);death.append(a['deaths']['numerator']-c['deaths']['numerator'])
            lines.append(f'| {group} | {mode} | {deltas} | {death} |')
    lines+=['','## Async eval与多卡建议','',
        f'- 同run异步评估扩容校准：低并发{calibration["low_median_env_per_second"]:.1f} env/s，扩容{calibration["high_median_env_per_second"]:.1f} env/s，变化{100*calibration["throughput_change_fraction"]:+.2f}%。这是调度短窗校准，非能力A/B；之后滚动回落触发了自动缩容，故推荐保留5%吞吐保护与资源分区，而非无限加CPU worker。',
        '- 1卡吞吐接近8卡；当前工作负载主要host/collector受限，不能声称8卡有线性加速。1/2/4卡可作为独立实验分组候选，但本次没有并发组训练验证，没有multiGPU checkpoint dev/holdout独立能力评估，不推荐直接以短probe替换生产训练。',
        '- mb256吞吐无稳定收益且交付明显下降；不推荐。lr5e-5自然采样尾部改善、greedy不同seed有得失，平均吞吐略低；只能作为能力候选，非效率推荐。',
        '- env32可以作为后续优先性能配置候选，配置为8卡×32env、rollout128、batch32768、minibatch512、epoch1、lr7.5e-5；当前不为它新增训练。死亡增量及生产JS缺口使“效果不降”的证据不足。','',
        '## 终点holdout窗口/返程与胜负（各seed分列）','',
        '窗口取包率分母为完整闭合的死亡/原生糖浆窗口；携包交付分母为已闭合custody，censored另列于机器可读记录；net win由完整局胜负计算，不能等同包子交付。','',
        '| 组/臂/seed尾号/动作模式 | 死亡窗口取包 | 原生糖浆窗口取包 | 同custody交付 | 净胜/局 |',
        '|---|---|---|---|---|']
    def frac(x):return f'{x["numerator"]}/{x["denominator"]}'
    for e in evals:
        if e['split']!='holdout':continue
        w=e['windows'];lines.append(f'| {e["group"]}/{e["arm"]}/{str(e["seed"])[-2:]}/{e["mode"]} | '+
            ' | '.join((frac(w['death']['pickup_per_closed_window']),frac(w['native_syrup']['pickup_per_closed_window']),
                        frac(e['carry']['same_episode_delivery_rate']),frac(e['net_wins'])))+' |')
    lines+=['',
        '## 全部dev检查点（禁止择峰）','',
        '| 条件 | categorical交付/死亡 | greedy交付/死亡 |','|---|---|---|']
    bycondition=defaultdict(list)
    for e in evals:
        if e['split']=='dev':bycondition[e['condition']].append(e)
    for condition,rs in sorted(bycondition.items()):
        lines.append('| '+condition+' | '+' | '.join(cell(next(e for e in rs if e['mode']==m)) for m in ('categorical','greedy'))+' |')
    lines+=['','## 复算与执行状态','',
        '复算：`.venv/bin/python -m scripts.campaign_scale_summary`。机器可读数据及全部原始SHA：`runs/authorized_campaign_20261026/SCALE_RECOMPUTABLE_SUMMARY_20261027.json`；每个终点state/policy SHA与逐更新日志在生成时实际核验。',
        'GPU空闲原因：已登记GPU预算全部完成，无已登记GPU后继；剩余三项原入口存在输出自等待，修复后只启动CPU生产者。BANK-RATIO正文与CPU入口/GPU限制冲突单列，不伪造训练完成；不自动恢复旧队列。']
    Path('reports/scale_summary_20261027.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(runs=len(runs),evaluation_conditions=len(bycondition),evaluation_rows=len(evals),report='reports/scale_summary_20261027.md')))


if __name__=='__main__':main()
