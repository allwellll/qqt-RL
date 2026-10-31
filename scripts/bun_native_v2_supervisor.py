"""Minimal idempotent integration hook for the existing campaign owner."""
import os
import subprocess
import sys
from pathlib import Path

OUT=Path('runs/bun_native_v2_20261030')


def running(name):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if name.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def ensure_running():
    if not (OUT/'TASK_CONSUMED.json').exists():return
    if any((b/f).exists() for b in (OUT,Path('runs/bomb_threat_v1'),Path('runs/iter_pbt_v1'))
           for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json')):return
    modules=[('scripts.bun_native_v2_diagnostic','DIAGNOSTIC_OWNER.log','DIAGNOSTIC_COMPLETE.json','DIAGNOSTIC_BLOCKED.json'),
             ('scripts.bun_native_v2_owner','R2_OWNER.log','UNAUTHORIZED_TERMINAL.json','R2_OWNER_BLOCKED.json'),
             ('scripts.bun_native_v2_eval_owner','R2_EVAL_OWNER.log','UNAUTHORIZED_EVAL_TERMINAL.json','R2_EVAL_BLOCKED.json'),
             ('scripts.bun_native_v2_donor_recovery','DONOR_RECOVERY_OWNER.log','DONOR_STATE_ACCEPTANCE.json','DONOR_RECOVERY_BLOCKED.json')]
    for name,log,done,blocked in modules:
        if (OUT/done).exists() or (OUT/blocked).exists() or running(name):continue
        with (OUT/log).open('ab') as stream:
            subprocess.Popen([sys.executable,'-m',name],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                             env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
