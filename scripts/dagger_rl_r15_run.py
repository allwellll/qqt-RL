"""Bounded carry-mask screen; full R14 evidence required to launch."""
import os,sys,time,subprocess
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
R=Path(__file__).resolve().parents[1]
B=R/'runs/dagger_rl_20261024/r15_carry_mask'
def read(p):
    import json
    return json.loads(Path(p).read_text())
def authorize():
    a=read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert a['authorized_by']=='explicit_user_instruction' and not a['old_continuous_resume']
    assert a['bounded_protocol_sha256']==sha(B/'PROTOCOL.json')
    assert a['old_global_pause_sha256']==sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    selection=read(B/'SCIENTIFIC_SELECTION.json')
    assert selection['execute_carry_mask_question'] is True
    assert selection['R14_evidence_sha256']==sha(R/'runs/dagger_rl_20261024/r14_item_observation/COMPLETE_EVIDENCE_SYNTHESIS.json')
    assert selection['R14_complete_receipt_sha256']==sha(R/'runs/dagger_rl_20261024/r14_item_observation/COMPLETE_EVIDENCE_CONSUMED.json')
    assert selection['protocol_sha256']==sha(B/'PROTOCOL.json')
    assert not (B/'USER_STOP.json').exists(),'safe atomic user stop'
def notify(message):
    print(message,flush=True);atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=message,unix=time.time()))
    try:subprocess.run(['jdme-notify','send',message],timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.TimeoutExpired):pass
def owner():
    unique=lock(B/'owner.lock');authorize();p=read(B/'PROTOCOL.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',CUBLAS_WORKSPACE_CONFIG=':4096:8',JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
    def child(args,stage):
        authorize()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip():
            authorize();atomic_json(B/'RESOURCE_WAIT.json',dict(stage=stage,no_preempt=True));time.sleep(5)
        atomic_json(B/'OWNER_STAGE.json',dict(stage=stage,unix=time.time()))
        with (B/f'{stage.split("/")[0]}.log').open('a') as f:
            process=subprocess.Popen([sys.executable,'-m']+args,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
            while process.poll() is None:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                import json
                with (B/'GPU_ACTIVITY.jsonl').open('a') as g:g.write(json.dumps(dict(unix=time.time(),stage=stage,devices=[[int(x.strip()) for x in row.split(',')] for row in raw.strip().splitlines()]))+'\n')
                time.sleep(5)
            assert process.returncode==0,f'{stage} exit={process.returncode}'
    try:
        if not (B/'TECHNICAL_SMOKE.json').exists():child(['scripts.dagger_rl_r15_train','smoke'],'smoke')
        assert read(B/'TECHNICAL_SMOKE.json')['passed']
        records=[]
        for seed in p['paired_seeds']:
            for arm in p['arms']:
                cp=B/'formal'/f'{arm}_s{seed}'
                if not (cp/'COMPLETE.json').exists():child(['scripts.dagger_rl_r15_train','formal',arm,str(seed)],f'formal/{arm}_s{seed}')
                record=read(cp/'COMPLETE.json');assert record['update']==32
                assert record['state_sha256']==sha(cp/'u032/state.pkl')
                records.append(record)
        atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=3145728,optimizer_steps=6144,old_continuous_resume=False))
        if not (B/'NATIVE_BANK_COMPLETE.json').exists():child(['scripts.dagger_rl_r15_bank'],'bank')
        assert read(B/'NATIVE_BANK_COMPLETE.json')['passed']
        atomic_json(B/'GPU_COMPLETE.json',dict(passed=True,old_continuous_resume=False))
        notify('R15六固定carry-mask run与声明Bank GPU终态；独立CPU跟随，按冻结单变量后继证据选择，不恢复旧队列。')
    except Exception as e:
        atomic_json(B/'TECHNICAL_FAILURE.json',dict(reason=str(e).replace(str(R),'<WORKTREE>'),unix=time.time()))
        notify('R15技术故障；保留双collector原子点，禁止静默续错或恢复旧队列。');raise
    finally:atomic_json(B/'OWNER_EXIT.json',dict(unix=time.time(),old_continuous_resume=False))
def complete():
    unique=lock(B/'completion.lock')
    while not all((B/n).exists() for n in ['GPU_COMPLETE.json','CPU_EVALUATION_COMPLETE.json']):
        if any((B/n).exists() for n in ['USER_STOP.json','TECHNICAL_FAILURE.json','CPU_TECHNICAL_FAILURE.json']):return
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (B/'analysis.log').open('a') as f:subprocess.run([sys.executable,'-m','scripts.dagger_rl_r15_report'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    atomic_json(B/'COMPLETE_FULL_TERMINAL_NOTICE.json',dict(passed=True,formal_runs=6,environment_steps=3145728,optimizer_steps=6144,comparison_sha256=sha(B/'FINAL_COMPARISON.json'),old_continuous_resume=False))
if __name__=='__main__':complete() if len(sys.argv)>1 else owner()
