"""Passive round-five arrival watch, owned by the existing campaign scheduler.

Does not change G1, install a patch, start smoke, or allocate a GPU. The MAIN
owner must still accept the actual evidence before any gate can be advanced.
"""
import json
from pathlib import Path
import subprocess
import time
from scripts.population_run_state import atomic_json,sha,lock

B=Path('runs/rules_v2_critic_20261027/v1')
ROUND=Path('runs/authorized_campaign_20261026/side_cx_results/parity/round5')


def arrival_evidence(root=ROUND):
    root=Path(root);marker=root/'COMPLETE.json'
    if not marker.exists():return dict(ready=False,reason='round5 COMPLETE not copied yet')
    complete=json.loads(marker.read_text())
    if complete.get('status')!='COMPLETE':return dict(ready=False,reason='round5 not terminal')
    wanted=('PATCH_CANDIDATE.diff','VALIDATION.json','CANDIDATE_FROZEN.json','RESIDUAL_DIAGNOSTICS.json')
    by_name={Path(r['path']).name:r for r in complete.get('outputs',[])}
    missing=[];checked=[]
    for name in wanted:
        p=root/name;expected=by_name.get(name,{}).get('sha256')
        if not p.exists() or expected is None:
            missing.append(name);continue
        if sha(p)!=expected:return dict(ready=False,reason='copied candidate artifact SHA mismatch',file=name)
        checked.append(dict(path=str(p),sha256=expected))
    fresh=next((name for name in ('fresh2_result.json','fresh_result.json') if (root/name).exists()),None)
    if fresh is None:missing.append('fresh2_result.json')
    elif fresh not in by_name or sha(root/fresh)!=by_name[fresh]['sha256']:
        return dict(ready=False,reason='fresh2 result SHA unavailable or mismatched')
    else:checked.append(dict(path=str(root/fresh),sha256=sha(root/fresh)))
    return dict(ready=not missing,reason='candidate available; MAIN G1 acceptance still required' if not missing else 'copy incomplete',
                missing=missing,checked_outputs=checked,complete_sha256=sha(marker),
                gate_passed=False,no_GPU_action=True)


def notify(kind,details):
    atomic_json(B/'notifications'/f'{time.time_ns()}_{kind}.json',dict(kind=kind,details=details,unix=time.time()))


def main():
    guard=lock(B/'round5_watch.lock');last_notification=0.;last_resource=0.;free_since=None
    atomic_json(B/'ROUND5_WATCH_STARTED.json',dict(existing_campaign_owner=True,CPU_only=True,
        may_start_GPU=False,may_modify_parity=False,unix=time.time()))
    while True:
        receipt=arrival_evidence()
        atomic_json(B/'ROUND5_WATCH_PROGRESS.json',dict(**receipt,completed_games=0,unix=time.time()))
        if receipt['ready']:
            atomic_json(B/'G1_CANDIDATE_AVAILABLE.json',dict(**receipt,unix=time.time()))
            notify('ROUND5_READY_FOR_MAIN_ACCEPTANCE',receipt)
            atomic_json(B/'ROUND5_WATCH_COMPLETE.json',dict(passed=True,gate_passed=False,
                no_GPU_launched=True,evidence_sha256=sha(B/'G1_CANDIDATE_AVAILABLE.json'),unix=time.time()))
            return
        now=time.time()
        if now-last_resource>=60:
            last_resource=now
            try:
                output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True,timeout=15)
                samples=[tuple(float(x) for x in line.split(',')) for line in output.splitlines()]
                free=len(samples)==8 and all(memory==util==0 for memory,util in samples)
                free_since=(free_since or now) if free else None
                if free_since and now-free_since>=600 and now-last_notification>=600:
                    notify('GPU_IDLE_GT_10_MIN',dict(reason='G1 orchestrator block; round5 evidence pending',
                        idle_seconds=now-free_since,new_GPU_tasks_started=0))
                    last_notification=now
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                atomic_json(B/'WATCH_GPU_OBSERVATION_ERROR.json',dict(error=type(exc).__name__,unix=now))
        time.sleep(20)


if __name__=='__main__':main()
