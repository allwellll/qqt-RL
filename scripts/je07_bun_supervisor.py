"""Unique durable JE07 supervisor; GPU and CPU workers recover independently."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je07_bun_recipe import *

def ours(pid,script):
    try:return alive(pid) and script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def spawn(script,mode,cores):
    with (BASE/f'{mode}_supervised_{time.time_ns()}.log').open('a') as log:
        command=['taskset','-c',cores,sys.executable,'scripts/'+script]
        if mode:command.append(mode)
        return subprocess.Popen(command,cwd=ROOT,env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def main():
    check_seal();lease=lock(BASE/'supervisor.lock');atomic_json(BASE/'SUPERVISOR_OWNER.json',dict(pid=os.getpid(),unix=time.time()))
    while not (BASE/'EXPERIMENT_COMPLETE.json').exists():
        for marker,script,mode,cores in [('EVAL_LAUNCH.json','je07_bun_evaluator.py','owner','96-111'),('OWNER_LAUNCH.json','je07_bun_recipe.py','owner','88-95')]:
            if (BASE/'TRAINING_COMPLETE.json').exists() and mode=='owner' and script=='je07_bun_recipe.py':continue
            if (BASE/'authority_je07_v1/OWNER_FINISHED.json').exists() and script=='je07_bun_evaluator.py':continue
            p=read(BASE/marker).get('pid') if (BASE/marker).exists() else None
            if not p or not ours(p,script):
                child=spawn(script,mode,cores);atomic_json(BASE/marker,dict(pid=child.pid,unix=time.time(),cuda_visible_devices=''))
        training=(BASE/'TRAINING_COMPLETE.json').exists();evaluated=(BASE/'authority_je07_v1/EVALUATION_COMPLETE.json').exists()
        atomic_json(BASE/'SUPERVISOR_HEARTBEAT.json',dict(unix=time.time(),training_complete=training,evaluation_complete=evaluated))
        if training and evaluated:
            atomic_json(BASE/'EXPERIMENT_COMPLETE.json',dict(passed=True,training_sha256=sha(BASE/'TRAINING_COMPLETE.json'),evaluation_sha256=sha(BASE/'authority_je07_v1/EVALUATION_COMPLETE.json'),unix=time.time()))
            notify('experiment_complete','JE07正式4096代及无辅助权威/长局评估完成。所有中间点已保留，快筛不代表显著性。')
        time.sleep(10)
    atomic_json(BASE/'SUPERVISOR_FINISHED.json',dict(passed=True,experiment_sha256=sha(BASE/'EXPERIMENT_COMPLETE.json'),unix=time.time()))

if __name__=='__main__':main()
