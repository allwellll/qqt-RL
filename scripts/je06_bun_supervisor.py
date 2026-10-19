"""Unique durable JE06 supervisor; GPU and CPU workers recover independently."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je06_bun_recipe import *

def ours(pid,script):
    try:return alive(pid) and script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def spawn(script,mode,cores):
    with (BASE/f'{mode}_supervised_{time.time_ns()}.log').open('a') as log:
        command=['taskset','-c',cores,sys.executable,'scripts/'+script]
        if mode:command.append(mode)
        return subprocess.Popen(command,cwd=ROOT,env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def main():
    prepare();lease=lock(BASE/'supervisor.lock');atomic_json(BASE/'SUPERVISOR_OWNER.json',dict(pid=os.getpid(),unix=time.time()))
    while not (BASE/'EXPERIMENT_COMPLETE.json').exists():
        for marker,script,mode,cores in [('EVAL_LAUNCH.json','je06_bun_evaluator.py','owner','64-79'),('OWNER_LAUNCH.json','je06_bun_recipe.py','owner','80-87'),('SIDE_LAUNCH.json','je06_bun_side_diagnostics.py',None,'104-111')]:
            if script=='je06_bun_side_diagnostics.py' and (BASE/'side_diagnostics_600_900_v1/OWNER_FINISHED.json').exists():continue
            p=read(BASE/marker).get('pid') if (BASE/marker).exists() else None
            if not p or not ours(p,script):
                child=spawn(script,mode,cores);atomic_json(BASE/marker,dict(pid=child.pid,unix=time.time(),cuda_visible_devices=''))
        atomic_json(BASE/'SUPERVISOR_HEARTBEAT.json',dict(unix=time.time(),training_complete=(BASE/'TRAINING_COMPLETE.json').exists(),smoke_complete=(BASE/'PRODUCTION_SMOKE_COMPLETE.json').exists()))
        time.sleep(10)
    atomic_json(BASE/'SUPERVISOR_FINISHED.json',dict(passed=True,experiment_sha256=sha(BASE/'EXPERIMENT_COMPLETE.json'),unix=time.time()))

if __name__=='__main__':main()
