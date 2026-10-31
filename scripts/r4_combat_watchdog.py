"""Supervise bounded R4 owners; adopt running trainers, never launch them."""
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.population_run_state import atomic_json,lock

BASE=Path('runs/r4_combat_20261031')


def count(module):
    n=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if module.encode() in a and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':n+=1
        except OSError:pass
    return n


def paused():
    return any((BASE/f).exists() for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))


def main():
    guard=lock(BASE/'WATCHDOG.lock');children={}
    configs=[('scripts.r4_combat_owner',BASE,None),
        ('scripts.r4_candidate_eval_owner',BASE/'candidate_evaluation',BASE/'candidate_evaluation/COMPLETE.json'),
        ('scripts.r4_pressure_eval_owner',BASE/'interceptor',BASE/'interceptor/COMPLETE.json')]
    while not paused():
        states={}
        for module,folder,done in configs:
            if done and done.exists():states[module]='COMPLETE';continue
            if module=='scripts.r4_combat_owner' and all((BASE/'formal'/('r4_'+a+'_c000000')/'COMPLETE.json').exists() for a in ('k1','k3')):
                states[module]='COMPLETE';continue
            n=count(module)
            if n>1:states[module]='DUPLICATE_BLOCKED';continue
            if (folder/'TECHNICAL_BLOCKED.json').exists():states[module]='TECHNICAL_BLOCKED_NO_BLIND_RETRY';continue
            if n==0:
                folder.mkdir(parents=True,exist_ok=True)
                with (folder/'WATCHDOG_OWNER.log').open('ab') as log:
                    child=subprocess.Popen([sys.executable,'-m',module],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
                        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[module]=child
                atomic_json(BASE/'watchdog_restarts'/f'{time.time_ns()}.json',dict(status='OWNER_RESTARTED_ADOPT_WORKERS',module=module,
                    training_signals=0,unix=time.time()))
                states[module]='RESTARTED'
            else:states[module]='UNIQUE_OWNER'
            for key,child in list(children.items()):
                if child.poll() is not None:child.wait();del children[key]
        atomic_json(BASE/'WATCHDOG_STATUS.json',dict(status='SUPERVISING',owners=states,unix=time.time()))
        if all(x=='COMPLETE' for x in states.values()):return
        time.sleep(15)
    atomic_json(BASE/'WATCHDOG_STATUS.json',dict(status='USER_PAUSED_NO_RESTART',unix=time.time()))


if __name__=='__main__':main()
