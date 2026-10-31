"""Consume every predeclared non-adopted update diagnostic, without tuning."""
import json
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha

R=Path(__file__).resolve().parents[1]
B=R/'runs/dagger_rl_20261024/r7_update_mechanism'
read=lambda p:json.loads(p.read_text())


def run():
    assert read(B/'COMPLETE.json')['passed']
    protocol=read(B/'PROTOCOL.json')
    budget=read(B/'COMPLETE.json')
    rows=[]
    branches=['joint_anchor','ppo_anchor','dagger_anchor','no_update','module_norm_matched_random']
    for seed in protocol['source_seeds']:
        folder=B/f'seed_{seed}'
        assert read(folder/'JOINT_EXACT.json')['passed'] and read(folder/'FRESH_RESTORE.json')['passed']
        d=read(folder/'COMPLETE.json');grad=read(folder/'COMPONENT_GRADIENTS.json')
        assert d['executed_optimizer_steps']==96 and d['logical_environment_steps']==8192
        rows.append(dict(seed=seed,branches=d['branches'],gradients=grad,
            exact_reference_sha256=sha(folder/'JOINT_EXACT.json'),fresh_restore_sha256=sha(folder/'FRESH_RESTORE.json')))
    summary={}
    for branch in branches:
        summary[branch]=dict(parameter_L2={m:[r['branches'][branch]['parameter_L2'][m] for r in rows] for m in ['actor','critic']},phases={})
        for phase in ['all','combat','window','carry','near','mid','far','natural','real_pickup_bank','item_release','item_contact']:
            groups=[r['branches'][branch]['probe']['groups'][phase] for r in rows]
            metrics=['move_parent_KL','ability_parent_KL','move_greedy_changed','ability_greedy_changed',
                'release_probability','bomb_probability','teacher_joint_mass','teacher_greedy_matches']
            summary[branch]['phases'][phase]=dict(states=[g['states'] for g in groups],teacher_rows=[g['teacher_rows'] for g in groups],
                **{k:[g.get(k) for g in groups] for k in metrics})
    evidence=dict(passed=True,protocol_sha256=sha(B/'PROTOCOL.json'),all_declared_inputs_consumed=True,
        original_joint_params_Adam_env_aux_RNG_memory_exact=True,fresh_restore_exact=True,
        rows=rows,summary=summary,logical_environment_steps=24576,executed_optimizer_steps=288,
        technical_regeneration_environment_steps=budget.get('technical_regeneration_environment_steps',0),
        total_executed_environment_steps=budget.get('total_executed_environment_steps',24576),
        new_independent_environment_samples=0,adopted_updates=0,extra_evaluation_rollouts=0,
        limits=['three regenerated existing TRAIN batches, not three new learned models',
            'static64 per-seed probe, phase/label coverage may be zero',
            'raw gradient cosine or static KL is not harmful long-run interference',
            'no measured Q contrast or independent capability claim',
            'module-norm random displacement is not an Adam-matched training arm'],
        old_continuous_resume=False)
    atomic_json(B/'FINAL_MECHANISM_EVIDENCE.json',evidence)
    lines=['\n### R7固定匹配更新机制诊断（20261024）\n',
        '原R6 control三seed的真实u1参数、Adam、完整env/aux/RNG及source-memory/replay精确重现；每seed8192行只重放一次。',
        '以下为相同TRAIN输入上的静态变化，不是能力、Q或长期干扰因果结论。DAgger-only冻结Critic及moments；共享Adam时钟推进。',
        '\n| 诊断分支 | Actor L2（三批） | Critic L2 | combat ability KL（三组TRAIN状态） |',
        '|---|---|---|---|']
    for branch in branches:
        r=summary[branch]
        fmt=lambda a:str([round(x,6) if x is not None else None for x in a])
        lines.append(f"| {branch} | {fmt(r['parameter_L2']['actor'])} | {fmt(r['parameter_L2']['critic'])} | {fmt(r['phases']['combat']['ability_parent_KL'])} |")
    lines += [f"\n完整分量梯度/原始cosine、phase/距离/来源/标签分母、greedy/集合质量、固定随机位移与fresh恢复收据见 `{str(B.relative_to(R))}/FINAL_MECHANISM_EVIDENCE.json`（SHA `{sha(B/'FINAL_MECHANISM_EVIDENCE.json')}`）。",
        f"\n声明24576重复逻辑环境步，另有{budget.get('technical_regeneration_environment_steps',0)}技术恢复重生步，实际执行环境步{budget.get('total_executed_environment_steps',24576)}；288诊断优化步、0新独立样本、0采用更新、0额外评估rollout。首批已完成joint按SHA复用，没有重复优化。同GPU fresh输出精确，CPU/GPU输出差异单列为后端测量，不以放宽容差掩盖。未以holdout选量级；不从低EV推断PPO噪声、不从静态漂移证明长期有害干扰。旧O5/continuous/储备暂停不变。"]
    if not (B/'REPORT_WRITTEN.json').exists():
        for name in ['reports/dagger_rl_iteration_20261024.md','docs/training_experiment_ledger.md']:
            with (R/name).open('a') as f:f.write('\n'.join(lines)+'\n')
        atomic_json(B/'REPORT_WRITTEN.json',dict(passed=True,evidence_sha256=sha(B/'FINAL_MECHANISM_EVIDENCE.json')))
    atomic_json(B/'COMPLETE_FULL_TERMINAL_NOTICE.json',dict(passed=True,evidence_sha256=sha(B/'FINAL_MECHANISM_EVIDENCE.json'),
        adopted_updates=0,old_continuous_resume=False,successor='requires explicit evidence-ranked bounded scientific selection'))


if __name__=='__main__':run()
