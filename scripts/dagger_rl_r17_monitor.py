"""Independent real-update/terminal acceptance; no dispatch or GPU use."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import sys
import time
import json
import pickle
from pathlib import Path
import numpy as np
import jax
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from scripts.dagger_rl_r17_run import B, read, notify
from scripts.population_run_state import atomic_json, sha, lock
from scripts.terminal_checkpoint_restore import digest


def checked(cp):
    m = read(cp/'COMMITTED.json')
    assert sha(cp/'state.pkl') == m['state_sha256'] and sha(cp/'policy.pt') == m['policy_sha256']
    x = pickle.load((cp/'state.pkl').open('rb'))
    assert digest(pickle.load((cp/'policy.pt').open('rb'))) == digest(x['params']['actor'])
    assert all(np.isfinite(a).all() for a in jax.tree.leaves((x['params'], x['adam'])))
    return m, x


def run():
    unique = lock(B/'independent_terminal.lock')
    protocol = read(B/'PROTOCOL.json')
    while not (B/'USER_STOP.json').exists():
        completed = []
        for seed in protocol['paired_seeds']:
            for arm in protocol['arms']:
                base = B/'formal'/f'{arm}_s{seed}'
                if (base/'u001/COMMITTED.json').exists() and not (base/'FIRST_UPDATE_INDEPENDENT.json').exists():
                    m, x = checked(base/'u001')
                    _, initial = checked(base/'u000')
                    parent = pickle.load((B/'inputs/parent.pkl').open('rb'))
                    assert digest(initial['params']) == digest(parent['params'])
                    assert digest(initial['adam']) == digest(parent['adam'])
                    row = m['metrics']
                    assert row['finite'] and row['update_l2']['actor'] > 0 and row['ability_head_update_l2'] > 0
                    assert (row['update_l2']['critic'] == 0) == (arm == 'sequential_anchor')
                    assert x['history'] is not None and set(x['auxiliary']) == {'states', 'aux', 'keys'}
                    atomic_json(base/'FIRST_UPDATE_INDEPENDENT.json', dict(passed=True, finite=True,
                        update=1, state_sha256=m['state_sha256'], policy_sha256=m['policy_sha256'],
                        actual_update_L2=row['update_l2'], ability_update_L2=row['ability_head_update_l2'],
                        parent_params_Adam_exact=True, smoke_not_inherited=True,
                        environment_steps=16384, optimizer_steps=32))
                if not (base/'COMPLETE.json').exists():
                    continue
                target = base/'TERMINAL_INDEPENDENT.json'
                if not target.exists():
                    m, x = checked(base/'u032')
                    rows = [json.loads(line) for line in (base/'updates.jsonl').read_text().splitlines()]
                    assert [r['update'] for r in rows] == list(range(1, 33))
                    assert all(r['finite'] and r['update_l2']['actor'] > 0 for r in rows)
                    assert all((r['update_l2']['critic'] == 0) == (r['mode'] == 'dagger') for r in rows)
                    assert all((r['dagger_coefficient'] == 0) == (r['mode'] == 'ppo') for r in rows)
                    receipt = dict(passed=True, arm=arm, seed=seed, update=32,
                        state_sha256=m['state_sha256'], policy_sha256=m['policy_sha256'],
                        environment_steps=524288, optimizer_steps=1024,
                        adopted_PPO_environment_steps=sum(r['adopted_fresh_PPO_environment_steps_this_update'] for r in rows),
                        supervised_optimizer_steps=sum(r['supervised_loss_optimizer_steps_this_update'] for r in rows),
                        critic_parameter_update_steps=sum(r['critic_parameter_updates_this_update'] for r in rows))
                    atomic_json(target, receipt)
                    text = (f'\nR17固定run终态 `{base.name}/u032`：原子state SHA `{m["state_sha256"]}`，'
                            f'524288环境步/1024优化步，实际采用PPO环境步{receipt["adopted_PPO_environment_steps"]}、'
                            f'监督优化步{receipt["supervised_optimizer_steps"]}、Critic更新步{receipt["critic_parameter_update_steps"]}。'
                            '32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。\n')
                    for doc in ['reports/dagger_rl_iteration_20261025.md', 'docs/training_experiment_ledger.md']:
                        with (R/doc).open('a') as f:
                            f.write(text)
                    notify(f'R17独立终态 {base.name}/u32：524288环境步/1024优化步，原子SHA {m["state_sha256"][:16]}；原owner串行接续固定队列。')
                completed.append(base.name)
        atomic_json(B/'INDEPENDENT_TERMINAL_PROGRESS.json', dict(completed=completed, expected=3, controls_training=False))
        if len(completed) == 3:
            atomic_json(B/'FORMAL_TERMINAL_ACCEPTANCE.json', dict(passed=True, runs=3,
                formal_environment_steps=1572864, formal_optimizer_steps=3072))
            return
        if (B/'TECHNICAL_FAILURE.json').exists():
            return
        time.sleep(5)


if __name__ == '__main__':
    run()
