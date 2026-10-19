"""Durable unique JE05 CPU supervisor; JE04 never receives a signal or write."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je05_bun_recipe import RUN,ROOT,read,atomic_json,lock,sha,alive,Adopted,child_env,notify,prepare
import subprocess,time

def ours(pid,script):
    try:return alive(pid) and script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def spawn(script,mode,cores=None):
    command=[sys.executable,'scripts/'+script,mode]
    if cores:command=['taskset','-c',cores,*command]
    with (RUN/f'{mode}_supervised_{time.time_ns()}.log').open('a') as log:
        return subprocess.Popen(command,cwd=ROOT,env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def ensure_evaluator():
    marker=RUN/'EVAL_LAUNCH.json'
    if marker.exists() and ours(read(marker)['pid'],'je05_bun_evaluator.py'):return
    child=spawn('je05_bun_evaluator.py','owner','64-79')
    atomic_json(marker,dict(pid=child.pid,cuda_visible_devices='',unique_lock='authority_je05_v3/owner.lock',unix=time.time()))

def main():
    prepare();lease=lock(RUN/'supervisor.lock')
    atomic_json(RUN/'SUPERVISOR_OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',unix=time.time()))
    child=None;attempt=0
    while not (RUN/'EXPERIMENT_COMPLETE.json').exists():
        ensure_evaluator()
        if child is None or child.poll() is not None:
            marker=RUN/'HANDOFF_OWNER.json'
            if marker.exists() and ours(read(marker)['pid'],'je05_bun_recipe.py'):
                child=Adopted(read(marker)['pid'])
            else:
                attempt+=1;child=spawn('je05_bun_recipe.py','handoff','80-87')
                atomic_json(RUN/f'SUPERVISOR_ATTEMPT_{attempt:03d}.json',dict(pid=child.pid,unix=time.time(),technical_retry_only=True))
        atomic_json(RUN/'SUPERVISOR_HEARTBEAT.json',dict(handoff_pid=child.pid,handoff_alive=child.poll() is None,
            evaluator_pid=read(RUN/'EVAL_LAUNCH.json')['pid'],training_complete=(RUN/'TRAINING_COMPLETE.json').exists(),unix=time.time()))
        time.sleep(10)
    atomic_json(RUN/'SUPERVISOR_FINISHED.json',dict(passed=True,experiment_complete_sha256=sha(RUN/'EXPERIMENT_COMPLETE.json'),unix=time.time()))
    notify('supervisor_finished','JE05训练和CPU权威评估完整结束，终态哈希已冻结。')

if __name__=='__main__':main()
