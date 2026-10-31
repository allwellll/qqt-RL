"""Unique v2 owner supervision; never launches or signals training workers."""
import os
import subprocess
import sys
import time
from scripts.bun_native_v2_owner import OUT,processes,paused
from scripts.population_run_state import atomic_json,lock


def main():
    guard=lock(OUT/'R2_WATCHDOG.lock')
    while True:
        owners=list(processes('scripts.bun_native_v2_owner'))
        if not paused() and not owners:
            with (OUT/'R2_OWNER.log').open('ab') as log:subprocess.Popen([sys.executable,'-m','scripts.bun_native_v2_owner'],
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(OUT/'R2_OWNER_RESTART_RECEIPT.json',dict(status='RESTART_REQUESTED',adopt_live_workers=True,training_worker_signals=0,unix=time.time()))
        atomic_json(OUT/'R2_WATCHDOG_STATUS.json',dict(status='PAUSED_NO_RESTART' if paused() else 'DUPLICATE_BLOCKED' if len(owners)>1 else 'SUPERVISING',
            owner_count=len(owners),direct_training_launches=0,unix=time.time()))
        time.sleep(10)


if __name__=='__main__':main()
