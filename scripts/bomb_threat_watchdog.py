"""Adopt/restart the existing sole BOMB owner; never launches GPU workers."""
import os,subprocess,sys,time
from pathlib import Path
from scripts.population_run_state import lock,atomic_json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def owner_count():
    found=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if b'scripts.bomb_threat_owner' in argv and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':found+=1
        except OSError:pass
    return found


def paused():
    return any((b/n).exists() for b in (BASE,ROOT/'runs/iter_pbt_v1') for n in ('USER_STOP.json','USER_PAUSE.json'))


def main():
    guard=lock(BASE/'WATCHDOG.lock');child=None;stream=None;restarts=0
    while not paused():
        count=owner_count()
        if child is not None and child.poll() is not None:
            stream.close();atomic_json(BASE/'OWNER_EXIT_RECEIPT.json',dict(exit_code=child.returncode,
                workers_not_signalled=True,automatic_adoption_expected=True,unix=time.time()));child=None
        if count==0:
            stream=(BASE/'logs/watchdog_owner.log').open('ab',buffering=0)
            child=subprocess.Popen(['taskset','-c','70-71',sys.executable,'-m','scripts.bomb_threat_owner'],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            restarts+=1
            atomic_json(BASE/'OWNER_RESTART_RECEIPT.json',dict(status='RESTARTED_EXISTING_OWNER',
                workers_adopted_by_persistent_run_locks=True,old_queues_not_consumed=True,restart_count=restarts,unix=time.time()))
        atomic_json(BASE/'WATCHDOG_STATUS.json',dict(status='DUPLICATE_OWNER_BLOCKED' if count>1 else 'RUNNING',
            owner_count=count,owner_module='scripts.bomb_threat_owner',direct_GPU_worker_launches=0,
            old_queues_restarted=False,restarts=restarts,unix=time.time()))
        time.sleep(15)
    atomic_json(BASE/'WATCHDOG_STATUS.json',dict(status='USER_PAUSED_NO_RESTART',direct_GPU_worker_launches=0,unix=time.time()))


if __name__=='__main__':main()
