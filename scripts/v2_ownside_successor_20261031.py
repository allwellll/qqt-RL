"""Own-side matrix terminal handling; never revives the invalidated selection."""
import os
os.environ.update(V2_SELECTION_OUTPUT='runs/v2_ownside_selection_20261031',
    V2_SELECTION_REPORT='reports/v2_ownside_selection_20261031.md',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import subprocess
import sys
import time
from pathlib import Path
from scripts import v2_fresh_selection_20261031 as core
from scripts.population_run_state import atomic_json,lock,sha


def notify(message):
    subprocess.run(['jdme-notify','send',message],capture_output=True,check=False)


def finish_report():
    B=core.OUT;r=core.read(B/'RESULTS.json');p=core.read(B/'PROTOCOL.json');core.report(r)
    text=core.REPORT.read_text()
    text=text.replace('hard/normal/easy新seed2026314000/2026315000/2026316000，协作hard复用hard同seed',
        'hard/normal/easy独立新seed'+('/'.join(str(p['conditions'][c]['seed']) for c in ('hunter_hard','hunter_normal','hunter_easy')))+
        '；协作hard独立seed'+str(p['conditions']['coop_hard']['seed']))
    text=text.replace('runs/v2_fresh_selection_20261031/RESULTS.json',str(B/'RESULTS.json'))
    text=text.replace('.venv/bin/python -m scripts.v2_fresh_selection_20261031',
        '.venv/bin/python -m scripts.v2_ownside_scheduler_20261031')
    core.REPORT.write_text(text)
    done=core.read(B/'COMPLETE.json');done['report_sha256']=sha(core.REPORT);atomic_json(B/'COMPLETE.json',done)
    return r


def main():
    B=core.OUT;guard=lock(B/'SUCCESSOR.lock')
    while not (B/'COMPLETE.json').exists():
        if (B/'USER_STOP').exists() or (B/'INVALIDATED.json').exists():
            atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='USER_PAUSED_OR_INVALIDATED',unix=time.time()));return
        if (B/'SCHEDULER2_BLOCKED.json').exists():
            atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='BLOCKED_EVALUATION',unix=time.time()));return
        atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='WAIT_OWN_SIDE_6144',
            R3_sp='separate conditional owner consumes same COMPLETE, no duplicate GPU owner',training_untouched=True,unix=time.time()))
        time.sleep(20)
    done=core.read(B/'COMPLETE.json');assert done['games']==6144 and done['results_sha256']==sha(B/'RESULTS.json')
    if not (B/'REPORT_FINALIZED.json').exists():
        r=finish_report();atomic_json(B/'REPORT_FINALIZED.json',dict(status='PASS',report=str(core.REPORT),report_sha256=sha(core.REPORT),unix=time.time()))
        notify('本侧出生新6144完整局复核完成，固定规则winner='+r['selection']['winner']+'；旧错侧结果不混入。报告'+str(core.REPORT)+'；R3-sp条件owner自动验门槛，训练继续。')
    folder=B/'publication';folder.mkdir(exist_ok=True)
    if (folder/'DONE.json').exists():return
    if (folder/'BLOCKED.json').exists():
        atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='BLOCKED_PUBLICATION_PRESERVE_EVIDENCE',receipt=str(folder/'BLOCKED.json'),unix=time.time()));return
    atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='PREPARING_WINNER_REPLAY_AND_CONDITIONAL_RELEASE',unix=time.time()))
    with (folder/'OWNER.log').open('ab') as log:
        result=subprocess.run(['taskset','-c','140-143',sys.executable,'-m','scripts.v2_web_release_20261031'],
            stdout=log,stderr=subprocess.STDOUT)
    state=core.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else {}
    atomic_json(B/'SUCCESSOR_STATUS.json',dict(status='PUBLICATION_RETURNED' if result.returncode==0 else 'BLOCKED_PUBLICATION',
        publication_status=state.get('status'),returncode=result.returncode,training_untouched=True,unix=time.time()))
    if result.returncode:notify('本侧选型后继出现技术阻塞，证据publication/OWNER.log保留；原R3和R3-sp队列独立，不影响训练。')


if __name__=='__main__':main()
