"""Bounded source-replay experiment owner; no legacy queue entrypoints."""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
B = R / 'runs/dagger_rl_20261024/r5_source_candidate'
from scripts.population_run_state import atomic_json, sha, lock


def read(path):
    return json.loads(Path(path).read_text())


def authorize():
    assert (B/'SCIENTIFIC_SELECTION.json').exists(), 'R5 scientific selection pending'
    d = read(B/'SCIENTIFIC_SELECTION.json')
    assert d['execute_source_test'] is True, 'R5 scientific selection declined'
    assert d['protocol_sha256'] == sha(B/'PROTOCOL.json')
    prior = R/'runs/dagger_rl_20261024/r4_reservoir_candidate'
    assert d['prior_comparison_sha256'] == sha(prior/'FINAL_COMPARISON.json')
    assert d['prior_evidence_sha256'] == sha(prior/'COMPLETE_EVIDENCE_SYNTHESIS.json')
    auth = read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert auth['authorized_by'] == 'explicit_user_instruction'
    assert not auth['old_continuous_resume']
    assert auth['bounded_protocol_sha256'] == sha(B/'PROTOCOL.json')
    assert sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json') == auth['old_global_pause_sha256']
    assert not (B/'USER_STOP.json').exists(), 'safe atomic boundary user stop'


def notify(message):
    message = message.replace('R4', 'R5')
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
    assert (B/'READY_TECHNICAL.json').exists(), 'focused technical regressions incomplete'
    prior = R/'runs/dagger_rl_20261024/r4_reservoir_candidate'
    assert (prior/'GPU_COMPLETE.json').exists() and (prior/'OWNER_EXIT.json').exists()
    atomic_json(B/'OWNER.json', dict(unique=True, bounded=True, old_continuous_resume=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               JAX_PLATFORMS='cuda', JAX_PLATFORM_NAME='cuda', JAXBOMB_RULE='bun',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
    try:
        for stage, marker in [('smoke','TECHNICAL_SMOKE.json'),
                              ('formal','TRAINING_COMPLETE.json'),
                              ('bank','NATIVE_BANK_COMPLETE.json')]:
            if (B/marker).exists():
                continue
            authorize()
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                          '--format=csv,noheader'], text=True).strip():
                authorize()
                atomic_json(B/'RESOURCE_WAIT.json', dict(stage=stage, no_preempt=True))
                time.sleep(5)
            atomic_json(B/'OWNER_STAGE.json', dict(stage=stage, unix=time.time()))
            module = 'scripts.dagger_rl_r5_bank' if stage == 'bank' else 'scripts.dagger_rl_r5_train'
            command = [sys.executable, '-m', module] + ([] if stage == 'bank' else [stage])
            code = subprocess.run(command, cwd=R, env=env,
                                  stdout=(B/f'{stage}.log').open('a'),
                                  stderr=subprocess.STDOUT).returncode
            if code or not (B/marker).exists():
                raise RuntimeError(f'{stage} exit={code}; required atomic marker absent={not (B/marker).exists()}')
            notify(f'R5 {stage}固定阶段完成，证据{marker}已保存。')
        atomic_json(B/'GPU_COMPLETE.json', dict(passed=True, old_continuous_resume=False))
        notify('R5三seed六run及固定Bank GPU预算结束；独立CPU/JS继续，无旧队列拉起。')
    except Exception as e:
        atomic_json(B/'TECHNICAL_FAILURE.json', dict(reason=str(e), unix=time.time()))
        notify(f'R5技术故障：{e}；最近原子点保留，需要修复，不恢复旧队列。')
    finally:
        atomic_json(B/'OWNER_EXIT.json', dict(unix=time.time(), old_continuous_resume=False))


def complete():
    unique = lock(B/'completion_owner.lock')
    while not ((B/'GPU_COMPLETE.json').exists() and (B/'CPU_EVALUATION_COMPLETE.json').exists()):
        if (B/'USER_STOP.json').exists():
            return
        if (B/'TECHNICAL_FAILURE.json').exists() or (B/'CPU_TECHNICAL_FAILURE.json').exists():
            notify('R5持久验收检测到技术故障，原子模型与待评估队列保留。')
            return
        time.sleep(5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    code = subprocess.run([sys.executable,'-m','scripts.dagger_rl_r5_report'], cwd=R, env=env,
                          stdout=(B/'analysis.log').open('a'), stderr=subprocess.STDOUT).returncode
    if code:
        atomic_json(B/'EVIDENCE_CONSUMPTION_FAILURE.json', dict(code=code))
        notify('R5报告消费故障；原始评估完整保留，不声称已全部验收。')
        return
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True).strip()
    atomic_json(B/'COMPLETE_TERMINAL_NOTICE.json', dict(
        passed=True, formal_runs=6, last_update_each=32,
        formal_environment_steps=1572864, formal_optimizer_steps=6144,
        comparison_sha256=sha(B/'FINAL_COMPARISON.json'),
        all_GPU_free=not apps, CPU_JS_complete=True, old_continuous_resume=False,
        successor='requires fixed complete-chain and parent/safety evidence; no equivalent automatic extension'))
    notify('R5完整三seed比较和原子验收已完成；旧队列仍暂停，下一轮须据真实来源/全链证据选择。')


if __name__ == '__main__':
    complete() if len(sys.argv)>1 and sys.argv[1]=='complete' else owner()
