"""Unique bounded R10 owner; no route to legacy continuous/reserve queues."""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
B=R/'runs/dagger_rl_20261024/r10_release_qualification'
from scripts.population_run_state import atomic_json,sha,lock
read=lambda p:json.loads(Path(p).read_text())


def authorize():
    d=read(B/'SCIENTIFIC_SELECTION.json')
    assert d['execute_release_qualification'] is True and d['protocol_sha256']==sha(B/'PROTOCOL.json')
    p=R/'runs/dagger_rl_20261024/r9_step_scale'
    assert d['prior_evidence_sha256']==sha(p/'COMPLETE_EVIDENCE_SYNTHESIS.json')
    assert d['prior_comparison_sha256']==sha(p/'FINAL_COMPARISON.json')
    assert read(p/'COMPLETE_FULL_TERMINAL_NOTICE.json')['passed']
    assert d['composition_sha256']==sha(p/'oracle_composition/COMPLETE.json')
    a=read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert a['authorized_by']=='explicit_user_instruction' and not a['old_continuous_resume']
    assert a['bounded_protocol_sha256']==sha(B/'PROTOCOL.json')
    assert a['old_global_pause_sha256']==sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    assert not (B/'USER_STOP.json').exists(),'safe atomic user stop'


def notify(message):
    print(message,flush=True)
    atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=message,unix=time.time()))
    try:subprocess.run(['jdme-notify','send',message],timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.TimeoutExpired):pass


def owner():
    unique=lock(B/'owner.lock');authorize();assert (B/'READY_TECHNICAL.json').exists()
    prior=R/'runs/dagger_rl_20261024/r9_step_scale'
    assert (prior/'OWNER_EXIT.json').exists() and read(prior/'COMPLETE_FULL_TERMINAL_NOTICE.json')['passed']
    atomic_json(B/'OWNER.json',dict(unique=True,bounded=True,old_continuous_resume=False))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
    try:
        for stage,marker in [('smoke','TECHNICAL_SMOKE.json'),('formal','TRAINING_COMPLETE.json'),('bank','NATIVE_BANK_COMPLETE.json')]:
            if (B/marker).exists():continue
            authorize()
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip():
                authorize();atomic_json(B/'RESOURCE_WAIT.json',dict(stage=stage,no_preempt=True));time.sleep(5)
            atomic_json(B/'OWNER_STAGE.json',dict(stage=stage,unix=time.time()))
            module='scripts.dagger_rl_r10_bank' if stage=='bank' else 'scripts.dagger_rl_r10_train'
            code=subprocess.run([sys.executable,'-m',module]+([] if stage=='bank' else [stage]),cwd=R,env=env,
                stdout=(B/f'{stage}.log').open('a'),stderr=subprocess.STDOUT).returncode
            if code or not (B/marker).exists():raise RuntimeError(f'{stage} exit={code}; atomic marker absent={not (B/marker).exists()}')
            notify(f'R10 {stage}固定阶段完成，{marker}已保存。')
        atomic_json(B/'GPU_COMPLETE.json',dict(passed=True,old_continuous_resume=False))
        notify('R10六固定run及Bank GPU预算终态；CPU/JS独立消费，旧队列保持暂停。')
    except Exception as e:
        reason=str(e).replace(str(R),'<WORKTREE>');atomic_json(B/'TECHNICAL_FAILURE.json',dict(reason=reason,unix=time.time()))
        notify(f'R10技术故障：{reason}；保存原子点，需修复后继续，不恢复旧队列。')
    finally:atomic_json(B/'OWNER_EXIT.json',dict(unix=time.time(),old_continuous_resume=False))


def complete():
    unique=lock(B/'completion.lock')
    while not all((B/n).exists() for n in ['GPU_COMPLETE.json','CPU_EVALUATION_COMPLETE.json']):
        if (B/'USER_STOP.json').exists():return
        if any((B/n).exists() for n in ['TECHNICAL_FAILURE.json','CPU_TECHNICAL_FAILURE.json']):return
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code=subprocess.run([sys.executable,'-m','scripts.dagger_rl_r10_report'],cwd=R,env=env,stdout=(B/'analysis.log').open('a'),stderr=subprocess.STDOUT).returncode
    if code:
        atomic_json(B/'EVIDENCE_CONSUMPTION_FAILURE.json',dict(code=code));notify('R10固定报告消费故障，原始评估保留，不伪称完整验收。');return
    atomic_json(B/'COMPLETE_TERMINAL_NOTICE.json',dict(passed=True,formal_runs=6,last_update_each=32,
        environment_steps=1572864,optimizer_steps=6144,comparison_sha256=sha(B/'FINAL_COMPARISON.json'),old_continuous_resume=False))
    notify('R10固定三配对seed比较完成；等待完整只读诊断消费，再据父点/保持/尾部选择后继，不择峰。')


if __name__=='__main__':complete() if len(sys.argv)>1 else owner()
