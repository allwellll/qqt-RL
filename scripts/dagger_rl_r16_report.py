"""Inactive fixed three-arm schedule comparison; no peak or training dispatch."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
import pickle
import numpy as np
import jax
from scripts.dagger_rl_r16_run import B, R, read, notify
from scripts.population_run_state import atomic_json, sha, lock
from scripts.terminal_checkpoint_restore import digest
from scripts.combat_item_report import natural, paired_uncertainty


def run():
    unique = lock(B/'analysis.lock')
    for name in ['TRAINING_COMPLETE.json', 'CPU_EVALUATION_COMPLETE.json', 'NATIVE_BANK_COMPLETE.json']:
        assert read(B/name)['passed']
    p = read(B/'PROTOCOL.json')
    source = pickle.load((B/'inputs/parent.pkl').open('rb'))
    parent = B/'formal/no_update/u000'
    parent_dev, _ = natural(parent)
    parent_holdout, parent_binary = natural(parent, True)
    parent_retention = read(parent/'retention/COMPLETE.json')
    results, binaries, atomic_rows, initial = {}, {}, [], {}
    for arm in p['arms']:
        results[arm], binaries[arm] = [], []
        for seed in p['paired_seeds']:
            base = B/'formal'/f'{arm}_s{seed}'
            rows = [__import__('json').loads(line) for line in (base/'updates.jsonl').read_text().splitlines()]
            assert [x['update'] for x in rows] == list(range(1, 33))
            assert all(x['finite'] and x['update_l2']['actor'] > 0 for x in rows)
            assert all((x['update_l2']['critic'] == 0) == (x['mode'] == 'dagger') for x in rows)
            assert all((x['dagger_coefficient'] == 0) == (x['mode'] == 'ppo') for x in rows)
            for u in p['checkpoints']:
                cp = base/f'u{u:03d}'
                m = read(cp/'COMMITTED.json')
                assert sha(cp/'state.pkl') == m['state_sha256'] and sha(cp/'policy.pt') == m['policy_sha256']
                x = pickle.load((cp/'state.pkl').open('rb'))
                assert digest(pickle.load((cp/'policy.pt').open('rb'))) == digest(x['params']['actor'])
                assert int(x['adam'][1][0].count) == int(source['adam'][1][0].count)+u*32
                assert all(np.isfinite(v).all() for v in jax.tree.leaves((x['params'], x['adam'])))
                if u == 0:
                    assert digest(x['params']) == digest(source['params']) and digest(x['adam']) == digest(source['adam'])
                    identity = digest((x['states'], x['aux'], x['keys'], x['auxiliary']))
                    if seed in initial:
                        assert initial[seed] == identity
                    else:
                        initial[seed] = identity
                if arm == 'sequential_anchor' and u <= 16:
                    assert digest(x['params']['critic']) == digest(source['params']['critic'])
                    for name in ['mu', 'nu']:
                        assert digest(getattr(x['adam'][1][0], name)['critic']) == digest(getattr(source['adam'][1][0], name)['critic'])
                atomic_rows.append(dict(checkpoint=str(cp.relative_to(B)), state_sha256=m['state_sha256'], policy_sha256=m['policy_sha256']))
            cp = base/'u032'
            dev, _ = natural(cp)
            holdout, binary = natural(cp, True)
            mid, _ = natural(base/'u016')
            retention = read(cp/'retention/COMPLETE.json')
            stages = {}
            for mode in ['dagger', 'ppo', 'anchor']:
                selected = [x for x in rows if x['mode'] == mode]
                if not selected:
                    continue
                stages[mode] = dict(updates=len(selected),
                    environment_steps=16384*len(selected), optimizer_steps=32*len(selected),
                    fresh_PPO_environment_steps_adopted=sum(x['adopted_fresh_PPO_environment_steps_this_update'] for x in selected),
                    supervised_optimizer_steps=sum(x['supervised_loss_optimizer_steps_this_update'] for x in selected),
                    critic_parameter_update_steps=sum(x['critic_parameter_updates_this_update'] for x in selected),
                    item_labels=sum(x['aggregation']['new_item_rows'] for x in selected),
                    raw_ability_gradient_mean=float(np.mean([x['ability_head_gradient_norm'] for x in selected])))
            results[arm].append(dict(seed=seed, dev=dev, holdout=holdout, midpoint=mid,
                tail_dev_delivery=(dev['episode_counts']['delivery']+mid['episode_counts']['delivery'])/2,
                Hunter=retention['hunter'], Tactical=retention['tactical'], stages=stages,
                Bank=read(cp/'native_holdout.json'),
                BankP={k: read(cp/f'bankP_{k}_holdout.json') for k in ['backward', 'wait']},
                curves={str(u): read(base/f'u{u:03d}/native_dev.json') for u in [1, 8, 16, 17, 24, 32]},
                switch_neighbor_metrics={str(u): read(base/f'u{u:03d}/COMMITTED.json')['metrics'] for u in [16, 17]},
                actual_opponents=dict(current=0, ring=0, fixed=524288, rule3=524288,
                    identity='JAX rule3, not Python Tactical or production JS Hunter')))
            binaries[arm].append(binary)
    assert len(atomic_rows) == 63
    contrasts = {arm: {metric: paired_uncertainty(np.stack([
        np.asarray(a[metric], float)-np.asarray(c[metric], float)
        for a, c in zip(binaries[arm], binaries['ppo_anchor'])])) for metric in parent_binary}
        for arm in ['sequential_anchor', 'joint_anchor']}
    parent_contrasts = {arm: {metric: paired_uncertainty(np.stack([
        np.asarray(a[metric], float)-np.asarray(parent_binary[metric], float)
        for a in binaries[arm]])) for metric in parent_binary} for arm in p['arms']}
    atomic_json(B/'ATOMIC_ACCEPTANCE.json', dict(passed=True, declared_checkpoints=63, rows=atomic_rows,
        actor_exports_Adam_counts_and_paired_full_u0_exact=True, sequential_critic_and_moments_frozen=True))
    result = dict(passed=True, id=p['id'], arms=results,
        parent=dict(dev=parent_dev, holdout=parent_holdout, Hunter=parent_retention['hunter'], Tactical=parent_retention['tactical']),
        paired_vs_PPO_combat_anchor_control=contrasts, vs_shared_parent=parent_contrasts,
        no_peak_selection=True, no_promotion=True, bank_limit=p['bank_limits'])
    atomic_json(B/'FINAL_COMPARISON.json', result)
    lines = ['\n## R16固定三臂终态比较\n',
             '| 臂 | 自然holdout交付/64（三seed） | 死亡对局/64 | dev尾部 |', '|---|---|---|---|']
    for arm, rs in results.items():
        lines.append(f"| {arm} | {[r['holdout']['episode_counts']['delivery'] for r in rs]} | {[r['holdout']['episode_counts']['death'] for r in rs]} | {[r['tail_dev_delivery'] for r in rs]} |")
    lines.append(f"\n共同未训练父holdout交付{parent_holdout['episode_counts']['delivery']}/64；所有CI与保留见FINAL_COMPARISON，不能以退化control称整体改善。预算、阶段损失实际采用步、16/17相邻点、所有固定端点保留。Bank仍为已使用单图P0全far；没有新奖励、道具目标、对手、课程或网络改动。旧队列保持暂停，不发布。")
    with (R/'reports/dagger_rl_iteration_20261025.md').open('a') as f:
        f.write('\n'.join(lines)+'\n')
    atomic_json(B/'ANALYSIS_COMPLETE.json', dict(passed=True, comparison_sha256=sha(B/'FINAL_COMPARISON.json')))
    notify('R16九run固定比较完成：顺序/联合/PPO+共同战斗保持、父点、阶段步数和尾部安全均已保存；完整证据继续按协议消费，不自动晋升。')


if __name__ == '__main__':
    run()
