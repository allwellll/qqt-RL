"""Evidence summary for saved points; observation only, never a training gate."""
import json
from pathlib import Path

from scripts.population_run_state import atomic_json, sha
from scripts.ppo_scale_longrun_run import B, R, read

FIELDS = ['games', 'delivery_episodes', 'pickup_episodes', 'death', 'self_death',
          'attributed_first_control', 'final_pop', 'surviving_kill', 'window_events',
          'window_stock_pickup', 'stock_pickup_dead', 'stock_pickup_native_sugar',
          'stock_pickup_free_alive', 'window_stock_delivery', 'free_stock_delivery',
          'nonstock_pickup', 'nonstock_delivery', 'delivery_without_tracked_acquisition',
          'first_oracle_actions', 'first_oracle_matches', 'progress_ticks', 'target_ticks']


def summarize():
    rows = []
    for marker in sorted((B / 'formal').glob('*/u*/retention/COMPLETE.json')):
        cp = marker.parent.parent
        atom = read(cp / 'COMMITTED.json')
        assert sha(cp / 'policy.pt') == atom['policy_sha256']
        d = read(marker)
        n = d['natural']['results']['natural_full2400']
        row = dict(checkpoint=str(cp.relative_to(R)), run=cp.parent.name,
                   update=atom['update'], environment_steps=atom.get('environment_steps', 0),
                   policy_sha256=atom['policy_sha256'],
                   natural={k: n.get(k) for k in FIELDS},
                   stage_funnel=n['stage_funnel'], tactical=d['tactical'], hunter=d['hunter'])
        for version in [4, 5]:
            audit = cp / f'natural_action_audit_v{version}/COMPLETE.json'
            if audit.exists():
                a = read(audit)
                row[f'natural_action_audit_v{version}'] = a['summary']
        holdout = marker.parent / 'natural_holdout64.json'
        if holdout.exists():
            h = read(holdout)['results']['natural_full2400']
            row['holdout'] = {k: h.get(k) for k in FIELDS}
        rows.append(row)
    telemetry = {}
    for log in sorted((B / 'formal').glob('*_s*/updates.jsonl')):
        updates = [json.loads(x) for x in log.read_text().splitlines()]
        assert len({x['update'] for x in updates}) == len(updates)
        telemetry[log.parent.name] = dict(updates=len(updates),
            last_environment_steps=updates[-1]['environment_steps'],
            all_finite=all(x['finite'] for x in updates),
            actor_critic_all_nonzero=all(all(v>0 for v in x['update_l2'].values()) for x in updates),
            ability_nonzero_updates=sum(x['actor_ability_update_l2']>0 for x in updates),
            last_loss=updates[-1]['loss'],
            loss_fields=['total', 'ppo', 'value', 'entropy', 'approximate_kl',
                         'clip_fraction', 'gradient_norm', 'ability_gradient_norm'],
            critic_ev='not_collected; value loss and GAE variance are not EV',
            opponent_exposure=updates[-1]['actual_opponent_exposure'])
    final_rows = {r['run']: r for r in rows
                  if (R / r['checkpoint']).parent.joinpath('COMPLETE.json').exists()
                  and r['update'] == read(B/'PROTOCOL.json')['updates_by_arm'].get(
                      r['run'].split('_s')[0])}
    comparison = []
    for seed in read(B/'PROTOCOL.json')['paired_seeds']:
        base = final_rows.get(f'ppo_base_batch_s{seed}')
        scaled = final_rows.get(f'ppo_scaled_batch_s{seed}')
        if base and scaled:
            pair = dict(seed=seed, matched_environment_steps=base['environment_steps'])
            for split in ['natural', 'holdout']:
                if split in base and split in scaled:
                    pair[split] = {k: scaled[split][k]-base[split][k]
                                   for k in ['delivery_episodes', 'death', 'self_death']}
            comparison.append(pair)
    tails = {}
    protocol = read(B/'PROTOCOL.json')
    for run in telemetry:
        arm=run.split('_s')[0]
        available=[r for r in rows if r['run']==run]
        aligned=[r for r in available if r['environment_steps']>=65536]
        tail=[r for r in aligned if r['environment_steps']>=524288]
        if aligned:
            tails[run]=dict(available_matched_points=[r['environment_steps'] for r in aligned],
                worst_fixed_dev_delivery_episodes=min(r['natural']['delivery_episodes'] for r in aligned),
                worst_fixed_dev_death_events=max(r['natural']['death'] for r in aligned),
                observed_tail_points=[r['environment_steps'] for r in tail],
                observed_tail_mean_delivery_episodes=(sum(r['natural']['delivery_episodes'] for r in tail)/len(tail)) if tail else None,
                fixed_tail_complete=len(tail)==3,
                no_peak_selection=True)
    result = dict(protocol_sha256=sha(B/'PROTOCOL.json'), points=rows,
        training=telemetry, final_paired_differences_scaled_minus_base=comparison,
        fixed_tail_and_worst=tails,
        training_complete=(B/'TRAINING_COMPLETE.json').exists(),
        cpu_complete=(B/'CPU_EVALUATION_COMPLETE.json').exists(),
        capability_gate=False, promotion=False, old_queues_resume=False,
        caveats=['Three seeds are a screen, not a safety equivalence proof.',
                 'u1 is not environment-step matched between arms.',
                 'v3/v4 chain proxies have disclosed attribution limitations; raw release/contact remain separate.',
                 'v5 excludes pre-existing learner movement effects; slow-field delay remains observational, not counterfactual Q.',
                 'Bank is old single-map P0 all-far; it is not natural capability.'])
    atomic_json(B/'LIVE_COMPARISON.json', result)
    lines = ['# PPO长训固定点原始比较（20261025）', '',
        '自动消费已完成点，不控制GPU，不择峰。完整合同和测量更正见同日长训报告。', '',
        '| run/点 | 环境步 | 自然交付局/64 | 死亡事件 | 合法窗口库存拾取 | 同链交付 | Tactical安全胜/64 | Hunter难/中/易安全胜/64 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        n=row['natural']; h=row['hunter']
        lines.append(f"| {row['run']}/u{row['update']} | {row['environment_steps']} | "
                     f"{n['delivery_episodes']} | {n['death']} | {n['window_stock_pickup']} | "
                     f"{n['window_stock_delivery']} | {row['tactical']['safe_win']} | "
                     + '/'.join(str(h[t]['safe_win']) for t in ['hard','normal','easy'])+' |')
    lines += ['', '自然完整、Python Tactical和JS Hunter分别报告；死亡事件与死亡对局不混用。',
              '完整阶段漏斗、自由存活自主取包、掉包/接触转移及道具原始统计保存在LIVE_COMPARISON。',
              'v4正净值不证明独立道具因果收益；v5排除已有本人移动效果污染，胶延迟仍是观察proxy。',
              '训练EV未采集；早期startup flag不能捕获所有编译。', '',
              f"GPU预算完成：{result['training_complete']}；CPU完整完成：{result['cpu_complete']}。"]
    target=R/'reports/ppo_scale_longrun_comparison_20261025.md'
    temp=target.with_suffix('.md.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(target)
    if result['training_complete'] and result['cpu_complete']:
        atomic_json(B/'FINAL_COMPARISON.json',result)
        atomic_json(B/'FINAL_ANALYSIS_RECEIPT.json',dict(
            passed=True,comparison_sha256=sha(B/'FINAL_COMPARISON.json'),
            report_sha256=sha(target),paired_final_seeds=len(comparison),
            next_direction='Use all paired tails and safety versus common parent; no default DAgger or peak promotion',
            old_queues_resume=False))
    return result


if __name__ == '__main__':
    summarize()
