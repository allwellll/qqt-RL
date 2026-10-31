"""Bounded R12 DAgger behavior-distribution owner; old queues stay paused."""
import os,sys,json,time,subprocess,pickle
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
R=Path(__file__).resolve().parents[1]
B=R/'runs/dagger_rl_20261024/r12_greedy_dagger_states'
read=lambda p:json.loads(Path(p).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def authorize():
    a=read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert a['authorized_by']=='explicit_user_instruction' and not a['old_continuous_resume']
    assert a['bounded_protocol_sha256']==sha(B/'PROTOCOL.json')
    assert a['old_global_pause_sha256']==sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    d=read(B/'SCIENTIFIC_SELECTION.json')
    assert d['execute_greedy_auxiliary_source'] and d['protocol_sha256']==sha(B/'PROTOCOL.json')
    assert d['mechanism_analysis_sha256']==sha(R/'runs/dagger_rl_20261024/r11_native_item_mechanism/MECHANISM_ANALYSIS.json')
    assert not (B/'USER_STOP.json').exists(),'safe atomic user stop'


def notify(message):
    print(message,flush=True);atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=message,unix=time.time()))
    try:subprocess.run(['jdme-notify','send',message],timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.TimeoutExpired):pass


def owner():
    unique=lock(B/'owner.lock');authorize()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',CUBLAS_WORKSPACE_CONFIG=':4096:8',JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
    try:
        for stage,marker in [('smoke','TECHNICAL_SMOKE.json'),('formal','TRAINING_COMPLETE.json'),('bank','NATIVE_BANK_COMPLETE.json')]:
            if (B/marker).exists():continue
            authorize()
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip():
                authorize();atomic_json(B/'RESOURCE_WAIT.json',dict(stage=stage,no_preempt=True));time.sleep(5)
            atomic_json(B/'OWNER_STAGE.json',dict(stage=stage,unix=time.time()))
            module='scripts.dagger_rl_r12_bank' if stage=='bank' else 'scripts.dagger_rl_r12_train'
            with (B/f'{stage}.log').open('a') as stream:
                child=subprocess.Popen([sys.executable,'-m',module]+([] if stage=='bank' else [stage]),cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                    with (B/'GPU_ACTIVITY.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),stage=stage,devices=[[int(x.strip()) for x in row.split(',')] for row in raw.strip().splitlines()]))+'\n')
                    time.sleep(5)
                assert child.returncode==0 and (B/marker).exists(),f'{stage} failed exit={child.returncode}'
            notify(f'R12 {stage}固定阶段完成，原子marker已验收；旧队列不恢复。')
        atomic_json(B/'GPU_COMPLETE.json',dict(passed=True,old_continuous_resume=False))
        notify('R12六固定run及Bank GPU预算终态；CPU/JS独立跟随，按预声明完整证据选择后继。')
    except Exception as error:
        atomic_json(B/'TECHNICAL_FAILURE.json',dict(reason=str(error).replace(str(R),'<WORKTREE>'),unix=time.time()))
        notify('R12技术故障，保留双collector原子点与唯一队列；需同线程修复后继续。')
        raise
    finally:atomic_json(B/'OWNER_EXIT.json',dict(unix=time.time(),old_continuous_resume=False))


def complete():
    unique=lock(B/'completion.lock')
    while not all((B/n).exists() for n in ['GPU_COMPLETE.json','CPU_EVALUATION_COMPLETE.json']):
        if (B/'USER_STOP.json').exists():return
        if any((B/n).exists() for n in ['TECHNICAL_FAILURE.json','CPU_TECHNICAL_FAILURE.json']):return
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (B/'analysis.log').open('a') as f:subprocess.run([sys.executable,'-m','scripts.dagger_rl_r12_report'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    atomic_json(B/'COMPLETE_FULL_TERMINAL_NOTICE.json',dict(passed=True,formal_runs=6,last_update_each=32,environment_steps=3145728,optimizer_steps=6144,comparison_sha256=sha(B/'FINAL_COMPARISON.json'),old_continuous_resume=False))
    notify('R12完整固定三配对筛选终态；数据分布/自然完整/安全与共享父比较完成，须据冻结后继问题继续，不择峰。')


if __name__=='__main__':complete() if len(sys.argv)>1 else owner()
