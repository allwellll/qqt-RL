"""Bounded exact R1 state reconstruction, never a new candidate experiment."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.bun_native_v2_owner import read,processes,OUT,paused,notification

BASE=OUT/'donor_restore_v1'
RUN='restore_slot03_u2760'
MODULE='scripts.bomb_threat_r1_continue_train'


def main():
    guard=lock(BASE/'OWNER.lock');folder=BASE/'formal'/RUN
    folder.mkdir(parents=True,exist_ok=True)
    explicit_fork=(OUT/'DONOR_FORK_AUTHORIZATION.json').exists()
    acceptance=OUT/('DONOR_U2760_RECONSTRUCTION_ACCEPTANCE.json' if explicit_fork else 'DONOR_STATE_ACCEPTANCE.json')
    if acceptance.exists():return
    for phase,marker,flag in [('SMOKE','SMOKE.json','--smoke'),('RESTORE','RESTORE.json','--verify-restore'),('FORMAL','COMPLETE.json',None)]:
        if (folder/marker).exists():continue
        while list(processes(MODULE,RUN)):time.sleep(5)
        if (folder/marker).exists():continue
        if paused():return
        command=['taskset','-c','160-163',sys.executable,'-m',MODULE,str(BASE/'PROTOCOL.json'),RUN,'2026183003']
        if flag:command.append(flag)
        with (folder/f'{phase}.log').open('ab') as log:
            subprocess.run(command,env=dict(os.environ,JAXBOMB_DEVICE_OVERRIDE='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (folder/marker).exists(),'state reconstruction phase did not complete'
    old=Path('runs/bomb_threat_v1/formal/bt_slot03_c000036')
    expected=read(old/'NEXT_STEP_u002760.json');actual=read(folder/'NEXT_STEP_u002760.json')
    assert actual==expected,'reconstructed full training state differs from frozen donor next-step digests'
    cp=folder/'u002760';m=read(cp/'COMMITTED.json');original=read(old/'u002760/COMMITTED.json')
    assert sha(cp/'policy.pt')==m['policy_sha256']==original['policy_sha256']
    assert sha(cp/'state.pkl')==m['state_sha256'] and m['metrics']['finite']
    # state.pkl metadata arm/protocol differs, so preserve both byte hashes and
    # prove all actual training-state digests rather than rewriting old marker.
    atomic_json(acceptance,dict(status='PASS',source=m,original_checkpoint=original['checkpoint'],
        old_state_bytehash_unavailable=original['state_sha256'],reconstructed_state_sha256=m['state_sha256'],
        policy_exact=True,next_step_exact=True,Adam_exact=True,env_aux_RNG_exact=True,ring_exact=True,
        native_RULE='oldR1stock-window',additional_reconstruction_updates=40,not_candidate_training=True,
        parent_protocol='runs/bomb_threat_v1/protocols/bt_slot03_c000036.json',parent_run='bt_slot03_c000036',unix=time.time()))
    notification('DONOR_RECOVERY_PASS','原slot03 u2760完整训练状态限定重建通过：Actor SHA、next-step/Adam/env/aux/RNG/ring精确通过；已授权u2721的R2父点不随此切换，R1继续。',dict(acceptance=str(acceptance)))


if __name__=='__main__':
    try:main()
    except Exception as e:
        blocked='DONOR_U2760_RECONSTRUCTION_BLOCKED.json' if (OUT/'DONOR_FORK_AUTHORIZATION.json').exists() else 'DONOR_RECOVERY_BLOCKED.json'
        atomic_json(OUT/blocked,dict(status='BLOCKED',error_type=type(e).__name__,reason='inspect donor_restore_v1 phase log; no substitute parent and no R1 changes',unix=time.time()))
        explicit=(OUT/'DONOR_FORK_AUTHORIZATION.json').exists()
        message=('旧u2760历史状态重建遇到技术阻塞，证据已保存；已授权并验收的u2721 R2起点保持不变，R1/R2训练继续。'
                 if explicit else 'R2父点完整状态重建遇到技术阻塞，证据已保存，R1训练不受影响；不伪称原Adam/env已恢复。')
        notification('DONOR_U2760_RECONSTRUCTION_BLOCKED' if explicit else 'DONOR_RECOVERY_BLOCKED',message,dict(receipt=str(OUT/blocked)))
        raise
