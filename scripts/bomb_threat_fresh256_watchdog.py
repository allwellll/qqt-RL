"""Adopt the bounded fresh CPU owner; no training/GPU launch or score selection."""
import os,json,time,subprocess,sys
from pathlib import Path
from scripts.population_run_state import atomic_json,lock
B=Path('runs/bomb_threat_v1');O=B/'fresh256_slot01_u800'


def active(module=b'scripts.bomb_threat_fresh256'):
    n=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if module in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':n+=1
        except OSError:pass
    return n


def main():
    guard=lock(O/'WATCHDOG.lock');child=None;stream=None
    while not any((B/n).exists() for n in ('USER_STOP_EVAL.json',)):
        if (O/'COMPLETE.json').exists():
            atomic_json(O/'WATCHDOG_STATUS.json',dict(status='COMPLETE_NO_RELAUNCH',unix=time.time()));return
        state=json.loads((O/'EXECUTION.json').read_text()) if (O/'EXECUTION.json').exists() else {}
        if state.get('status')=='BLOCKED':
            atomic_json(O/'WATCHDOG_STATUS.json',dict(status='BLOCKED_NO_BLIND_RETRY',reason=state,unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify('B_FRESH256_BLOCKED',f'B fresh256 CPU复核技术阻塞：{state.get("error_type","unknown")}；原始完整分片保留，训练未停，不盲重跑，见fresh256_slot01_u800/EXECUTION.json。')
            return
        if child and child.poll() is not None:stream.close();child=None
        count=active()
        if count==0:
            # A departed Python owner may leave valid Node shards in flight.
            # Let them atomically finish before resuming; never duplicate them.
            shards=active(b'scripts/bomb_threat_fresh256_eval.js')
            if shards:
                atomic_json(O/'WATCHDOG_STATUS.json',dict(status='ADOPTING_ORPHAN_SHARDS_BEFORE_RESUME',inflight_shards=shards,training_untouched=True,unix=time.time()))
                time.sleep(20);continue
            atomic_json(O/'OWNER_RESTART_RECEIPT.json',dict(status='RESUME_UNEXPECTED_OWNER_EXIT',completed_shards_preserved=True,training_untouched=True,unix=time.time()))
            stream=(O/'OWNER.log').open('ab',buffering=0)
            child=subprocess.Popen([sys.executable,'-m','scripts.bomb_threat_fresh256'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        atomic_json(O/'WATCHDOG_STATUS.json',dict(status='ADOPTED_UNIQUE_OWNER' if count==1 else 'RESTARTING' if count==0 else 'DUPLICATE_BLOCKED',owner_count=count,cpu_only=True,training_launches=0,unix=time.time()))
        if count>1:return
        time.sleep(20)
    atomic_json(O/'WATCHDOG_STATUS.json',dict(status='USER_EVAL_STOP_NO_RESTART',unix=time.time()))

if __name__=='__main__':main()
