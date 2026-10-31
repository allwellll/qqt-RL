"""Materialize only a full-evidence selected, bounded retention-schedule screen."""
import copy
import shutil
import time
from scripts.dagger_rl_r16_run import B, R, read
from scripts.population_run_state import atomic_json, sha


def prepare():
    prior = R/'runs/dagger_rl_20261024/r15_carry_mask'
    choice = read(prior/'NEXT_SELECTED_EXPERIMENT.json')
    assert choice['execute_retention_schedule'] is True
    for name in ['COMPLETE_EVIDENCE_CONSUMED.json', 'SUPPLEMENTAL_EVIDENCE_CONSUMED.json',
                 'ZERO_STEP_REMAINING_AUDIT.json', 'GPU_COMPLETE.json']:
        assert read(prior/name)['passed']
    assert choice['R15_complete_evidence_sha256'] == sha(prior/'COMPLETE_EVIDENCE_SYNTHESIS.json')
    assert choice['R15_supplement_sha256'] == sha(prior/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json')
    assert not (B/'PROTOCOL.json').exists(), 'do not overwrite or rerun a namespace'
    old = read(prior/'PROTOCOL.json')
    arms = ['ppo_anchor', 'sequential_anchor', 'joint_anchor']
    p = dict(id='DAGGER-RL-20261025-R16-RETENTION-SCHEDULE', mode=old['mode'],
        parent=old['parent'], parent_input_sha256=old['parent_input_sha256'], optimizer=old['optimizer'],
        arms=arms, paired_seeds=[2026102661, 2026102662, 2026102663], updates=32, switch_update=16,
        single_variable='route/item supervised and PPO loss schedule under identical common combat-parent KL',
        schedule=dict(ppo_anchor='PPO/value/entropy+combat-parent KL throughout; no route/item DAgger',
            sequential_anchor='u1..16 DAgger+combat-parent KL with independent Critic and moments frozen; '
                              'u17..32 fresh on-policy PPO/value/entropy+combat-parent KL, DAgger exactly0',
            joint_anchor='PPO/value/entropy+DAgger.2+combat-parent KL throughout'),
        common='legacy ch7 and legacy collector movement mask in all arms; no R14/R15 input treatment; '
               'same reward, teacher, opponents, reset mixture, network, saved Adam, lr and reservoir',
        hypothesis='Does removing or separating supervised pressure change natural retention and sustained '
                   'item/route transfer under the same combat anchor? Not assume shared Actor/Critic trunk or harmful interference.',
        scale_selection='none; reuse .2 supervised/.05 combat KL/7.5e-5 lr and fixed16/17 switch; no holdout tuning',
        train_mixture=old['train_mixture'], opponents=dict(current=0, ring=0, fixed=1., identity='JAX rule3'),
        bank_source=old['bank_source'], bank_sha256=old['bank_sha256'], bank_limits=old['bank_limits'],
        oracle='same public-state joint safe route/banana teacher, unchanged labels and legacy mask binding; '
               'stock only native death/sugar, glue not native sugar; real cargo continues return after revival',
        aggregation='same categorical PPO and auxiliary student streams, uniform reservoir8192 and same replay/query budgets; '
                    'PPO always fresh stream only, auxiliary advantages discarded. Pure PPO still queries/collects '
                    'matched supervision opportunities for audit but adopts none; realized learner states may diverge.',
        envs=64, steps=128, environment_steps_per_run=524288, total_environment_steps=4718592,
        optimizer_steps_per_run=1024, total_optimizer_steps=9216,
        checkpoints=[0, 1, 8, 16, 17, 24, 32], lr=7.5e-5, gamma=.995, gae_lambda=.95,
        clip=.1, value=.5, entropy=.005, gradient_clip=.5, epochs=1, minibatch=256,
        anchor_coefficient=.05, dagger_coefficient_by_arm={a: 0. if a=='ppo_anchor' else .2 for a in arms},
        learning_rate_by_arm={a: 7.5e-5 for a in arms}, auxiliary_greedy={a: False for a in arms},
        release_qualification={a: False for a in arms}, memory=copy.deepcopy(old['memory']),
        technical_smoke=dict(total_executed_environment_steps=3072, total_executed_optimizer_steps=18,
            adopted_formal_steps=0, checks='three branches finite/real Actor-ability gradients and updates; '
            'PPO Critic nonzero; DAgger Critic parameters and mu/nu frozen; full env/aux/RNG/Adam/history '
            'fresh-process all-branch next update hash exact, no capability threshold'),
        evaluation=copy.deepcopy(old['evaluation']),
        fixed_fit='own u1/u16 memory first<=64 arrivals per natural/real-Bank and route/release/contact; '
                  'parent/u1/u16/u17/u32 same-input joint/ability fit, not independent capability or proof of forgetting',
        reporting='all fixed finals/tails/worst seed versus matched PPO+combat KL control and untrained parent; '
                  'real JS Hunter H/N/E64 each, Python-host native JAX Tactical64, natural dev/holdout64, '
                  'used single-map P0 allfar Bank/P separately; actual loss-adopted stage budgets and item opportunity denominators',
        successor='credible predeclared sequential whole-chain/retention improvement permits separately frozen five-new-seed '
                  'confirmation against matched controls; otherwise zero-step earliest drift/coverage/switch-fit audit to choose '
                  'one bounded failure-chain question. If all matched arms degrade, report that explicitly and do not '
                  'extend budgets or fill GPUs with unrelated runs. No automatic multipliers/rewards/old queues.',
        stopping='nine fixed32-update runs plus declared Bank/P; only technical invalidity or user stop interrupts; '
                 'no midpoint capability gates, no peak selection or automatic promotion',
        R15_evidence_sha256=choice['R15_complete_evidence_sha256'],
        R15_supplement_sha256=choice['R15_supplement_sha256'],
        R15_zero_step_audit_sha256=sha(prior/'ZERO_STEP_REMAINING_AUDIT.json'), unix=time.time())
    p['memory']['teacher_loss_coefficient_by_arm'] = p['dagger_coefficient_by_arm']
    p['memory']['sequential_after_u16_coefficient'] = 0.
    p['evaluation']['natural_holdout_seed'] = 2026111800
    p['evaluation']['checkpoints'] = 'immutable0/1/8/16/17/24/32; dev64 at16/32; parent+9finals natural holdout64, Hunter each64, Tactical64, fixed Bank/P32x8'
    p['evaluation']['selection'] = 'all three matched seeds/fixed final/tail/worst; no holdout scale or switch tuning'
    B.mkdir(parents=True, exist_ok=True)
    for folder in ['inputs', 'cpu_eval_inputs', 'inputs/sealed_sources', 'cache']:
        (B/folder).mkdir(parents=True, exist_ok=True)
    atomic_json(B/'PROTOCOL.json', p)
    shutil.copyfile(prior/'inputs/parent.pkl', B/'inputs/parent.pkl')
    assert sha(B/'inputs/parent.pkl') == p['parent_input_sha256']
    for name in ['policy.pt', 'state.pkl']:
        assert sha(R/p['parent']['checkpoint']/name) == p['parent']['policy_sha256' if name=='policy.pt' else 'state_sha256']
    for name in ['spawns.json', 'regions.json']:
        shutil.copyfile(prior/'cpu_eval_inputs'/name, B/'cpu_eval_inputs'/name)
    ev = dict(read(prior/'CPU_EVAL_PROTOCOL.json'), holdout_seed_start=2026111800)
    atomic_json(B/'CPU_EVAL_PROTOCOL.json', ev)
    sources = dict(read(prior/'SOURCE_SEAL.json')['sources'])
    for name, h in sources.items():
        assert sha(R/name) == h, name
    names = [f'scripts/dagger_rl_r16_{name}.py' for name in ['prepare', 'run', 'train', 'cpu', 'bank', 'report', 'diagnostics']]
    names += ['scripts/dagger_rl_retention_schedule.py', 'scripts/dagger_rl_retention_loop.py',
              'scripts/dagger_rl_item_opportunity_telemetry.py', 'tests/test_dagger_rl_retention_schedule.py',
              'tests/test_dagger_rl_item_opportunity_telemetry.py', 'tests/test_dagger_rl_r16_authorization.py']
    for name in names:
        sources[name] = sha(R/name)
    for name in sources:
        dest = B/'inputs/sealed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R/name, dest)
    atomic_json(B/'SOURCE_SEAL.json', dict(sources=sources, protocol_sha256=sha(B/'PROTOCOL.json')))
    cache_hashes = []
    for src in (prior/'cache').glob('*'):
        if src.is_file():
            dest = B/'cache'/src.name
            shutil.copyfile(src, dest)
            assert sha(dest) == sha(src)
            cache_hashes.append(dict(name=src.name, sha256=sha(src)))
    atomic_json(B/'COMPILED_CACHE_REUSE.json', dict(compiler_files_only=True, no_params_Adam_env_RNG=True, files=cache_hashes))
    pause = sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    assert pause == '358dc9c05479ea351ac8434a79ffc53333c896d0f04f84f7671d5c8333ffdbea'
    atomic_json(B/'SCOPED_GPU_AUTHORIZATION.json', dict(authorized_by='explicit_user_instruction',
                protocol_sha256=sha(B/'PROTOCOL.json'), old_global_pause_sha256=pause, old_continuous_resume=False))
    atomic_json(B/'SCIENTIFIC_SELECTION.json', dict(choice, protocol_sha256=sha(B/'PROTOCOL.json')))
    prep = R/'runs/dagger_rl_20261025/r16_schedule_preparation'
    atomic_json(B/'FOCUSED_CORRECTNESS.json', dict(passed=True, tests=19,
        GREEN_sha256=sha(prep/'FOCUSED_GREEN.log'), authorization_RED_sha256=sha(prep/'AUTHORIZATION_RED.log'),
        real_TRAIN_loss_path_sha256=sha(prep/'REAL_TRAIN_LOSS_PATH.json'), capability_gate=False))
    atomic_json(B/'MATERIALIZED.json', dict(passed=True, formal_not_started=True,
                protocol_sha256=sha(B/'PROTOCOL.json'), old_continuous_resume=False))
    print(p['id'], sha(B/'PROTOCOL.json'), flush=True)


if __name__ == '__main__':
    prepare()
