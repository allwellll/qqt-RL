"""Read immutable Bun Stage1/2 evidence and emit portable, hash-audited results.

No training, process control, checkpoint writes, or model selection for deployment.
All candidates are exploratory because checkpoint selection uses the screening set.
"""
import argparse
import json
from pathlib import Path

from scripts.population_run_state import atomic_json, sha


def read(path):
    return json.loads(Path(path).read_text())


def checked_complete(directory, policy_hash):
    main_path = directory / 'COMPLETE.json'
    full_path = directory / 'COMPLETE_WITH_BUN_TASKS.json'
    main, full = read(main_path), read(full_path)
    assert main['passed'] and full['passed']
    assert main['policy_sha256'] == full['policy_sha256'] == policy_hash
    assert full['authority_sha256'] == sha(main_path)
    for name, digest in {**main['shards'], **full['additional_shards']}.items():
        assert sha(directory / name) == digest, directory / name
    tasks = {k: {field: value for field, value in row.items() if field != 'episodes'}
             for k, row in full['tasks'].items()}
    return dict(hunter=main['hunter'], tactical=main['tactical'], tasks=tasks,
                policy_sha256=policy_hash, main_sha256=sha(main_path),
                full_sha256=sha(full_path))


def stage_summary(run, authority_name):
    run = Path(run)
    protocol = read(run / 'plan.json')
    out = run / authority_name
    terminal = read(out / 'EVALUATION_COMPLETE.json')
    assert terminal['passed']
    if (out / 'OWNER_FINISHED.json').exists():
        owner = read(out / 'OWNER_FINISHED.json')
        assert owner['passed'] and owner['evaluation_sha256'] == sha(out / 'EVALUATION_COMPLETE.json')
    arms = protocol['arms']
    if isinstance(arms, dict):
        arms = list(arms)
    # Retention's plan declares recipes rather than an arms list.
    rows = {}
    for g in protocol['checkpoints']:
        rows[str(g)] = {}
        for arm in arms:
            cp = run / 'formal' / arm / f'gen{g:06d}'
            committed = read(cp / 'COMMITTED.json')
            assert committed['passed'] and committed['plan_sha256'] == sha(run / 'plan.json')
            assert committed['policy_sha256'] == sha(cp / 'policy.pt')
            assert committed['state_sha256'] == sha(cp / 'state.pkl')
            rows[str(g)][arm] = checked_complete(out / f'{arm}_g{g:06d}', committed['policy_sha256'])
        if 'pair_hashes' in terminal:
            assert sha(out / f'PAIR_g{g:06d}.json') == terminal['pair_hashes'][str(g)]
    training = {}
    for arm in arms:
        target = run / f'TRAINING_COMPLETE_{arm}.json'
        t = read(target)
        assert t['passed'] and t['generation'] == protocol['generations']
        if 'training_hashes' in terminal:
            assert sha(target) == terminal['training_hashes'][arm]
        training[arm] = {k: t[k] for k in ['generation', 'fresh_ticks', 'compute_seconds', 'wall_seconds']}
    return dict(experiment_id=protocol['experiment_id'], run=run.name,
        authority=authority_name, plan_sha256=sha(run / 'plan.json'),
        protocol_sha256=sha(out / 'PROTOCOL.json'),
        evaluation_complete_sha256=sha(out / 'EVALUATION_COMPLETE.json'),
        training=training, curve=rows)


def metric_cell(m):
    return '/'.join(str(m[k]) for k in ['safe_win', 'death_episode', 'self_episodes', 'trade_episodes'])


def markdown(summary):
    lines = ['# Bun392 Stage1/Stage2 完整验收（20261012）', '',
        '所有数字来自已提交检查点及权威评估 shard；脚本重新校验 plan、policy、state、'
        'COMPLETE、附加任务、pair 与终态 hash。每档 Hunter/Tactical 每臂64局，'
        '任务每类16局。单训练seed、多检查点同一快筛集；不作显著性、独立确认或部署晋级结论。', '',
        'Stage1 教师保持组合明显优于无教师对照的战斗安全退化轨迹；但 KL 和45%战斗重采样'
        '共同改变，不能分别归因。Stage1 teacher g2048 是实际冻结续训起点，'
        'teacher g4096（安全击杀22/192、死亡0、Tactical24/64）应保留为另一个确认候选；'
        '不把 late CPU 结果回填成当时已知选择依据。', '',
        'Stage2 动态12%联赛尚未形成稳定三档共同改善。动态 g3072 是本轮最值得扩大样本'
        '确认的探索候选：Hunter 0/4/23，死亡1/0/3，自炸0/0/0，Tactical24；'
        'g4096回落为0/3/16，死亡1/4/2、自炸0/3/1、Tactical19。'
        '保留g3072，不能默认终点最好，也不能称它已通过安全非劣门槛：g3072 full任务有2局自炸。', '',
        '共同Stage2 g0 Hunter2/4/10、Tactical21；动态终点相对固定终点安全击杀差'
        'hard/normal/easy=-3/+1/+2，总数相同19/192，死亡+2、自炸+3、换命-1，'
        'Tactical+1。动态g3072相对固定g3072安全击杀差0/+1/+11、Tactical+2，'
        '但前后点不稳定。多数增加来自easy，hard能力仍弱。', '',
        'Stage2所有点carry_home capture14/16、near_steal steal16/16，'
        'route_break和full capture/steal/return均0；因此任务局部保持，完整Bun任务仍未补齐。'
        '任务为原生JAX冻结idle诊断，不冒称网页Hunter/完整产品验收。', '',
        'retention_v2 名义奖励会被后续 lesson 分支覆盖；本轮保留真实执行语义与分项遥测，'
        '不能把未生效的安全奖励候选说成已验证机制。', '']
    for stage_name, s in summary.items():
        lines += [f'## {stage_name}', '',
            '表内每格为 **安全击杀(胜)/死亡/自炸/换命**；Hunter按hard、normal、easy分别列。', '',
            '| generation | arm | hard | normal | easy | Tactical | carry capture | near steal | full death/self |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for g, arms in s['curve'].items():
            for arm, m in arms.items():
                b = m['tasks']; full = b['full']
                lines.append(f"| {g} | {arm.split('_')[-1]} | " + ' | '.join(
                    metric_cell(m['hunter'][d]) for d in ['hard', 'normal', 'easy']) +
                    f" | {metric_cell(m['tactical'])} | {b['carry_home']['capture_episodes']} | "
                    f"{b['near_steal']['steal_episodes']} | {full['death_episodes']}/{full['self_episodes']} |")
        lines += ['', 'Stage1任务旧产物没有trade列，记为未采集，禁止补造0；Stage2新增观察列全曲线task trade为0。', '']
        for arm, t in s['training'].items():
            lines.append(f"- {arm}: {t['generation']}代，{t['fresh_ticks']:,} fresh ticks，"
                f"有效训练{t['compute_seconds']/3600:.3f}小时，墙钟{t['wall_seconds']/3600:.3f}小时，"
                f"平均有效吞吐{t['fresh_ticks']/t['compute_seconds']:.0f} ticks/s。")
        lines += ['', f"plan SHA `{s['plan_sha256']}`；终态评估 SHA `{s['evaluation_complete_sha256']}`。", '']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--stage2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = dict(Stage1=stage_summary(args.stage1, 'authority_bun831_v1'),
                   Stage2=stage_summary(args.stage2, 'authority_stage2_recovery_v2'))
    atomic_json(args.output.with_suffix('.json'), summary)
    args.output.with_suffix('.md').write_text(markdown(summary))


if __name__ == '__main__':
    main()
