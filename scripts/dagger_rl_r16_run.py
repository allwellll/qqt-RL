"""Inactive bounded retention-schedule owner; requires frozen full R15 choice."""
import os
import sys
import time
import subprocess
from pathlib import Path
from scripts.population_run_state import atomic_json, sha, lock

R = Path(__file__).resolve().parents[1]
B = R/'runs/dagger_rl_20261025/r16_retention_schedule'
read = lambda p: __import__('json').loads(Path(p).read_text())


class UserStopped(Exception):
    pass


def authorize():
    if (B/'USER_STOP.json').exists():
        raise UserStopped('safe atomic user stop')
    p = read(B/'PROTOCOL.json')
    a = read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert a['authorized_by'] == 'explicit_user_instruction' and not a['old_continuous_resume']
    assert a['protocol_sha256'] == sha(B/'PROTOCOL.json')
    assert a['old_global_pause_sha256'] == sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    prior = R/'runs/dagger_rl_20261024/r15_carry_mask'
    for name in ['COMPLETE_EVIDENCE_CONSUMED.json', 'SUPPLEMENTAL_EVIDENCE_CONSUMED.json',
                 'ZERO_STEP_REMAINING_AUDIT.json', 'GPU_COMPLETE.json']:
        assert read(prior/name)['passed']
    choice = read(B/'SCIENTIFIC_SELECTION.json')
    assert choice['execute_retention_schedule'] is True
    assert choice['R15_complete_evidence_sha256'] == sha(prior/'COMPLETE_EVIDENCE_SYNTHESIS.json')
    assert choice['R15_supplement_sha256'] == sha(prior/'SUPPLEMENTAL_EVIDENCE_CONSUMED.json')
    assert choice['protocol_sha256'] == sha(B/'PROTOCOL.json')
    assert len(p['arms']) == 3 and set(p['arms']) == {'ppo_anchor', 'sequential_anchor', 'joint_anchor'}
    assert len(p['paired_seeds']) == len(set(p['paired_seeds'])) == 3
    assert p['updates'] == 32 and p['switch_update'] == 16
    assert p['total_environment_steps'] == 4718592 and p['total_optimizer_steps'] == 9216
    assert sha(B/'inputs/parent.pkl') == p['parent_input_sha256'], 'common clean parent/Adam SHA mismatch'


def notify(message):
    print(message, flush=True)
    atomic_json(B/'notifications'/f'{time.time_ns()}.json', dict(message=message, unix=time.time()))
    try:
        subprocess.run(['jdme-notify', 'send', message], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def owner():
    unique = lock(B/'owner.lock')
    authorize()
    p = read(B/'PROTOCOL.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', JAX_COMPILATION_CACHE_DIR=str(B/'cache'))

    def child(args, stage):
        authorize()
        for name, expected in read(B/'SOURCE_SEAL.json')['sources'].items():
            assert sha(R/name) == expected, 'sealed source changed: '+name
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True).strip():
            authorize()
            atomic_json(B/'RESOURCE_WAIT.json', dict(stage=stage, no_preempt=True))
            time.sleep(5)
        atomic_json(B/'OWNER_STAGE.json', dict(stage=stage, unix=time.time()))
        with (B/f'{stage.split("/")[0]}.log').open('a') as f:
            worker = subprocess.Popen([sys.executable, '-m']+args, cwd=R, env=env,
                                      stdout=f, stderr=subprocess.STDOUT)
            while worker.poll() is None:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True)
                with (B/'GPU_ACTIVITY.jsonl').open('a') as g:
                    import json
                    g.write(json.dumps(dict(unix=time.time(), stage=stage,
                        devices=[[int(v.strip()) for v in row.split(',')]
                                 for row in raw.strip().splitlines()]))+'\n')
                time.sleep(5)
            if (B/'USER_STOP.json').exists():
                raise UserStopped('worker stopped after last atomic checkpoint')
            assert worker.returncode == 0, f'{stage} exit={worker.returncode}'
    try:
        if not (B/'TECHNICAL_SMOKE.json').exists():
            child(['scripts.dagger_rl_r16_train', 'smoke'], 'smoke')
        assert read(B/'TECHNICAL_SMOKE.json')['passed']
        records = []
        for seed in p['paired_seeds']:
            for arm in p['arms']:
                cp = B/'formal'/f'{arm}_s{seed}'
                if not (cp/'COMPLETE.json').exists():
                    child(['scripts.dagger_rl_r16_train', 'formal', arm, str(seed)],
                          f'formal/{arm}_s{seed}')
                x = read(cp/'COMPLETE.json')
                assert x['update'] == 32 and x['state_sha256'] == sha(cp/'u032/state.pkl')
                records.append(x)
        atomic_json(B/'TRAINING_COMPLETE.json', dict(passed=True, runs=records,
                    environment_steps=p['total_environment_steps'],
                    optimizer_steps=p['total_optimizer_steps'], old_continuous_resume=False))
        if not (B/'NATIVE_BANK_COMPLETE.json').exists():
            child(['scripts.dagger_rl_r16_bank'], 'bank')
        assert read(B/'NATIVE_BANK_COMPLETE.json')['passed']
        atomic_json(B/'GPU_COMPLETE.json', dict(passed=True, old_continuous_resume=False))
        notify('R16九个固定run及声明Bank GPU阶段完成；CPU独立，后继仅按冻结证据规则选择，旧队列保持暂停。')
    except UserStopped as e:
        atomic_json(B/'PAUSED_OWNER.json', dict(safe_atomic=True, reason=str(e), old_continuous_resume=False))
        notify('R16用户安全原子暂停：禁止领取下一更新/run/smoke，保存点与恢复入口保留。')
    except Exception as e:
        atomic_json(B/'TECHNICAL_FAILURE.json', dict(reason=str(e).replace(str(R), '<WORKTREE>'), unix=time.time()))
        notify('R16技术故障，完整原子证据保留；禁止静默续错或恢复旧队列。')
        raise
    finally:
        atomic_json(B/'OWNER_EXIT.json', dict(unix=time.time(), old_continuous_resume=False))


def complete():
    unique = lock(B/'completion.lock')
    required = ['GPU_COMPLETE.json', 'CPU_EVALUATION_COMPLETE.json', 'FIXED_FIT_COMPLETE.json']
    while not all((B/name).exists() for name in required):
        if any((B/name).exists() for name in ['USER_STOP.json', 'TECHNICAL_FAILURE.json', 'CPU_TECHNICAL_FAILURE.json']):
            return
        time.sleep(5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (B/'analysis.log').open('a') as f:
        subprocess.run([sys.executable, '-m', 'scripts.dagger_rl_r16_report'], cwd=R, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    atomic_json(B/'COMPLETE_EVIDENCE_CONSUMED.json', dict(passed=True, formal_runs=9,
        formal_environment_steps=4718592, formal_optimizer_steps=9216,
        comparison_sha256=sha(B/'FINAL_COMPARISON.json'), old_continuous_resume=False))
    notify('R16九run完整固定结果及切换拟合已消费；按预声明科学问题决定有界后继，旧队列不恢复。')


if __name__ == '__main__':
    complete() if len(sys.argv) > 1 and sys.argv[1] == 'complete' else owner()
