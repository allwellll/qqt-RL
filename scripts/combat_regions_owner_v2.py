"""One bounded 1280-update owner with smoke/fresh-restore/formal stages."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock

B = Path('runs/combat_regions_independent_20261104')
T = B / 'training'
MODULE = 'scripts.combat_regions_train_v2'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    guard = lock(T / 'OWNER.lock')
    request = read(B / 'LAUNCH_AUTHORIZATION.json')
    assert request['authorized_by'] == 'cx-v37-kill-route-learning'
    assert request['predecessor_saved_and_exited'] is True
    cpu = read(B / 'CPU_VALIDATION.json'); assert cpu['status'] == 'PASS'
    current = read(T / 'CURRENT.json'); p = read(current['protocol'])
    run = current['run']; out = Path(p['namespace']) / run
    out.mkdir(parents=True, exist_ok=True)
    for stage, flag, receipt in (('SMOKE', '--smoke', 'SMOKE.json'),
            ('RESTORE', '--verify-restore', 'RESTORE.json'), ('FORMAL', None, 'COMPLETE.json')):
        if (out / receipt).exists():
            continue
        if any((T / n).exists() for n in ('USER_PAUSE.json', 'USER_STOP.json', 'STOP_FOR_REBOOT.json')):
            atomic_json(T / 'OWNER_STATUS.json', dict(status='PAUSED', stage=stage, unix=time.time()))
            return
        argv = ['taskset', '-c', '72-79', sys.executable, '-u', '-m', MODULE,
                current['protocol'], run, str(p['seed'])] + ([flag] if flag else [])
        env = dict(os.environ, JAXBOMB_DEVICE_OVERRIDE='0', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
        atomic_json(T / 'OWNER_STATUS.json', dict(status='RUNNING', stage=stage, module=MODULE,
            command=['<PYTHON>', '-m', MODULE, current['protocol'], run, str(p['seed'])] + ([flag] if flag else []),
            gpu=0, cpu_affinity='72-79', unix=time.time()))
        with (out / 'worker.log').open('ab') as stream:
            child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
        if code or not (out / receipt).exists():
            paused = any((T / n).exists() for n in ('USER_PAUSE.json', 'USER_STOP.json', 'STOP_FOR_REBOOT.json'))
            atomic_json(T / 'OWNER_STATUS.json', dict(status='PAUSED' if paused else 'TECHNICAL_BLOCKED',
                stage=stage, returncode=code, log=str(out / 'worker.log'), unix=time.time()))
            return
    atomic_json(T / 'OWNER_STATUS.json', dict(status='COMPLETE_1280', checkpoint=read(out / 'COMPLETE.json')['checkpoint'], unix=time.time()))


if __name__ == '__main__':
    main()
