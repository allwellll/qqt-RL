"""One finite owner, isolated authorization, atomic evidence and no old-queue resume."""
import os
os.environ.setdefault('JAXBOMB_RULE','bun');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import pathlib,sys,json,time,subprocess,pickle
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'runs/combat_item_20261023/v1'
from scripts.population_run_state import atomic_json,sha,lock
from scripts.carrier_mechanism_run import save

def read(p):return json.loads(pathlib.Path(p).read_text())
def authorize():
 d=read(B/'SCOPED_GPU_AUTHORIZATION.json')
 assert d['authorized_by']=='explicit_user_instruction' and not d['old_continuous_resume']
 assert sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')==d['old_global_pause_sha256']
 assert not (B/'USER_STOP.json').exists()
def notify(message):
 print(message,flush=True);atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=message,unix=time.time()))
 try:subprocess.run(['jdme-notify','send',message],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
 except (OSError,subprocess.TimeoutExpired):pass

def launch(stage,cpu=False):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else '0,1,2,3,4,5,6,7',JAX_PLATFORMS='cpu' if cpu else 'cuda',JAX_PLATFORM_NAME='cpu' if cpu else 'cuda',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',CUBLAS_WORKSPACE_CONFIG=':4096:8',JAX_COMPILATION_CACHE_DIR=str(B/'cache'))
 return subprocess.Popen([sys.executable,'-m','scripts.combat_item_train',stage],cwd=R,env=env,stdout=(B/f'{stage}.log').open('a'),stderr=subprocess.STDOUT)

def owner():
 authorize();unique=lock(B/'owner.lock')
 atomic_json(B/'OWNER.json',dict(unique=True,bounded=True,old_continuous_resume=False))
 for stage,marker in [('smoke','TECHNICAL_SMOKE.json'),('formal','TRAINING_COMPLETE.json'),('bank','NATIVE_BANK_COMPLETE.json')]:
  if (B/marker).exists():continue
  authorize();atomic_json(B/'STATUS.json',dict(stage=stage,unix=time.time()))
  child=launch(stage);code=child.wait()
  if code or not (B/marker).exists():
   atomic_json(B/'TECHNICAL_FAILURE.json',dict(stage=stage,exit_code=code));notify(f'战斗道具实验技术故障：{stage} exit={code}，证据保留；旧队列未恢复。');return
  notify(f'战斗道具实验阶段终态：{stage}固定预算完成，marker={marker}。')
 atomic_json(B/'GPU_COMPLETE.json',dict(passed=True,old_continuous_resume=False,automatic_successor=False))
 notify('战斗道具实验全部预声明GPU训练及Bank评估结束；释放8卡，CPU/JS独立评估继续；旧O5/储备仍暂停。')

if __name__=='__main__':owner()
