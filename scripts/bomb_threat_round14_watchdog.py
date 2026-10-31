"""Maintain the bounded CPU recheck and passive report; never launches training."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock
from scripts.bomb_threat_round14_report import write_report

BASE=Path('runs/bomb_threat_v1')
OUT=BASE/'fresh256_a3sr_u640'


def active(module):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if module.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def main():
    guard=lock(OUT/'WATCHDOG.lock');child=None;stream=None
    while not (BASE/'USER_STOP_EVAL.json').exists():
        try:write_report()
        except Exception as error:
            atomic_json(OUT/'REPORT_RETRY.json',dict(status='RETRY_REQUIRED',error_type=type(error).__name__,unix=time.time()))
        if (OUT/'COMPLETE.json').exists():
            atomic_json(OUT/'WATCHDOG_STATUS.json',dict(status='FRESH_COMPLETE_REPORT_UPDATED',unix=time.time()));return
        state=json.loads((OUT/'EXECUTION.json').read_text()) if (OUT/'EXECUTION.json').exists() else {}
        if state.get('status')=='BLOCKED':
            atomic_json(OUT/'WATCHDOG_STATUS.json',dict(status='BLOCKED_NO_BLIND_RETRY',error=state,unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify('A3SR_FRESH_BLOCKED',f"A3SR fresh CPU阻塞：{state.get('error_type')}；完整分片保留，GPU训练不停。")
            return
        if child and child.poll() is not None:stream.close();child=None
        if not active('scripts.bomb_threat_a3sr_fresh') and not active('scripts/bomb_threat_fresh256_eval.js'):
            stream=(OUT/'OWNER.log').open('ab',buffering=0)
            child=subprocess.Popen(['taskset','-c','112-115',sys.executable,'-m','scripts.bomb_threat_a3sr_fresh'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(OUT/'OWNER_RESTART_RECEIPT.json',dict(status='RESUMED_ATOMIC_ACCEPTED_SHARDS',training_untouched=True,unix=time.time()))
        atomic_json(OUT/'WATCHDOG_STATUS.json',dict(status='MONITORING_FRESH_AND_REPORT',cpu_only=True,unix=time.time()))
        time.sleep(30)


if __name__=='__main__':main()
