"""Isolated bounded DAgger/RL owner. No path can dispatch old queues."""
import os
os.environ.setdefault('JAXBOMB_RULE','bun');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import pathlib,sys,json,time,subprocess
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'runs/dagger_rl_20261024/r4_reservoir_candidate'
from scripts.population_run_state import atomic_json,sha,lock
from scripts.carrier_mechanism_run import save

def read(p):return json.loads(pathlib.Path(p).read_text())
def authorize():
 assert (B/'SCIENTIFIC_SELECTION.json').exists(),'R4 scientific selection pending; preparation is not dispatch'
 selection=read(B/'SCIENTIFIC_SELECTION.json')
 assert selection.get('execute_memory_test') is True,'R4 scientific selection declined memory test'
 assert selection['protocol_sha256']==sha(B/'PROTOCOL.json'),'R4 scientific selection protocol mismatch'
 assert selection['r3_comparison_sha256']==sha(R/'runs/dagger_rl_20261024/r3_confirmation/FINAL_COMPARISON.json'),'R4 scientific selection evidence mismatch'
 d=read(B/'SCOPED_GPU_AUTHORIZATION.json')
 assert d['authorized_by']=='explicit_user_instruction' and not d['old_continuous_resume']
 assert sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')==d['old_global_pause_sha256']
 assert sha(B/'PROTOCOL.json')==d['bounded_protocol_sha256']
 assert not (B/'USER_STOP.json').exists(),'safe boundary user stop'
def notify(message):
 print(message,flush=True);atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=message,unix=time.time()))
 try:subprocess.run(['jdme-notify','send',message],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
 except (OSError,subprocess.TimeoutExpired):pass

def launch(stage):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',CUBLAS_WORKSPACE_CONFIG=':4096:8',JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
 module='scripts.dagger_rl_r4_bank' if stage=='bank' else 'scripts.dagger_rl_r4_train'
 return subprocess.Popen([sys.executable,'-m',module]+([] if stage=='bank' else [stage]),cwd=R,env=env,stdout=(B/f'{stage}.log').open('a'),stderr=subprocess.STDOUT)

def owner():
 prior=R/'runs/dagger_rl_20261024/r3_confirmation'
 assert (prior/'GPU_COMPLETE.json').exists() and (prior/'OWNER_EXIT.json').exists(),'R3 fixed GPU budget still active'
 assert (B/'READY_TECHNICAL.json').exists(),'focused technical checks incomplete'
 authorize();unique=lock(B/'owner.lock')
 atomic_json(B/'OWNER.json',dict(unique=True,bounded=True,old_continuous_resume=False,unix=time.time()))
 try:
  for stage,marker in [('audit','ACTION_CHAIN_AUDIT.json'),('smoke','TECHNICAL_SMOKE.json'),('formal','TRAINING_COMPLETE.json'),('bank','NATIVE_BANK_COMPLETE.json')]:
   if (B/marker).exists():continue
   authorize()
   while subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).strip():
    authorize();atomic_json(B/'RESOURCE_WAIT.json',dict(stage=stage,no_preempt=True));time.sleep(5)
   atomic_json(B/'OWNER_STAGE.json',dict(stage=stage,unix=time.time()))
   child=launch(stage);code=child.wait()
   if code or not (B/marker).exists():raise RuntimeError(f'{stage} exit={code}; marker missing={not (B/marker).exists()}')
   notify(f'DAgger/RL阶段终态 {stage}：固定证据{marker}已提交。')
  atomic_json(B/'GPU_COMPLETE.json',dict(passed=True,old_continuous_resume=False,successor='frozen CPU results required before new confirmatory budget'))
  notify('DAgger/RL三seed两臂固定GPU训练完成，CPU/JS独立消费；下一轮须按冻结证据选择，旧队列不恢复。')
 except Exception as e:
  atomic_json(B/'TECHNICAL_FAILURE.json',dict(reason=str(e),unix=time.time()))
  notify(f'DAgger/RL技术阻塞：{e}；最近原子点保留，无旧队列启动。')
 finally:atomic_json(B/'OWNER_EXIT.json',dict(unix=time.time(),old_continuous_resume=False))
if __name__=='__main__':owner()
