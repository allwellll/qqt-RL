"""Restart only the authorized campaign owner after a recorded crash."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.population_run_state import atomic_json, lock, sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/authorized_campaign_20261026'


def owner_processes():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try: args=entry.joinpath('cmdline').read_bytes().decode().split('\0')
        except OSError: continue
        if len(args)>2 and Path(args[0]).name.startswith('python') and args[1:3]==['-m','scripts.authorized_experiment_campaign'] and len(args)>3 and args[3]=='owner':
            found.append(entry)
    return found


def user_paused():
    return any((BASE/name).exists() for name in ('USER_STOP.json','USER_PAUSE.json'))


def run():
    BASE.mkdir(parents=True,exist_ok=True)
    guard=lock(BASE/'watchdog.lock')
    atomic_json(BASE/'WATCHDOG.json',dict(unique=True,active=True,
        purpose='restart exact authorized campaign owner only',started_unix=time.time(),
        pause_files=['USER_STOP.json','USER_PAUSE.json'],gpu_idle_alert_seconds=300))
    backoff=5
    while True:
        if (BASE/'COMPLETE.json').exists():
            done=json.loads((BASE/'COMPLETE.json').read_text())
            if done.get('passed'):
                atomic_json(BASE/'WATCHDOG_COMPLETE.json',dict(passed=True,unix=time.time(),
                    reason='all campaign jobs actually complete; no relaunch'))
                return
        if user_paused():
            time.sleep(5); continue
        owners=owner_processes()
        if len(owners)>1:
            atomic_json(BASE/'WATCHDOG_DUPLICATE_OWNER.json',dict(detected=True,count=len(owners),unix=time.time()))
            time.sleep(5);continue
        if owners:
            backoff=5;time.sleep(2);continue
        started=time.time()
        incident=dict(schema='qqt.authorized-campaign-watchdog-restart/v1',
            last_status_sha256=sha(BASE/'EXECUTION_STATUS.json') if (BASE/'EXECUTION_STATUS.json').exists() else None,
            last_status_mtime=(BASE/'EXECUTION_STATUS.json').stat().st_mtime if (BASE/'EXECUTION_STATUS.json').exists() else None,
            campaign_sha256=sha(BASE/'CAMPAIGN.json') if (BASE/'CAMPAIGN.json').exists() else None,
            reason='exact owner process absent; inspect latest owner failure receipt and reconcile committed work',
            unix=started)
        atomic_json(BASE/'watchdog_incidents'/f'{time.time_ns()}.json',incident)
        log=(BASE/'owner.log').open('a')
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONPATH='.')
        process=subprocess.Popen([sys.executable,'-m','scripts.authorized_experiment_campaign','owner'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=process.wait()
        log.close()
        atomic_json(BASE/'watchdog_exits'/f'{time.time_ns()}.json',dict(
            started_unix=started,finished_unix=time.time(),exit_code=code,
            status_sha256=sha(BASE/'EXECUTION_STATUS.json') if (BASE/'EXECUTION_STATUS.json').exists() else None,
            pause_seen=user_paused(),restart_backoff_seconds=backoff))
        if not user_paused():
            time.sleep(backoff);backoff=min(backoff*2,60)


if __name__=='__main__': run()
