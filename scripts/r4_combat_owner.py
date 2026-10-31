"""Unique bounded R4 owner; claims only released cards, never signals trainers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.population_run_state import atomic_json, lock, sha

BASE = Path('runs/r4_combat_20261031')
WORKER = 'scripts.r4_combat_train'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def notify(key, message, **evidence):
    path = BASE/'notifications'/(key+'.json')
    if path.exists():
        return
    atomic_json(path, dict(message=message, **evidence, unix=time.time()))
    result = subprocess.run(['jdme-notify', 'send', message], capture_output=True)
    atomic_json(path, dict(message=message, **evidence, notify_returncode=result.returncode, unix=time.time()))


def active(run):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            words = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if WORKER.encode() in words and run.encode() in words:
            return True
    return False


def free_cards():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    return [int(a) for a, b in (line.split(',') for line in raw.splitlines()) if int(b) == 0]


def prepare():
    parent = read(Path('runs/r3_items_route_20261030/protocols/r3_a_ownside_c000001.json'))
    cp = Path('runs/r3_items_route_20261030/formal/r3_a_ownside_c000001/u004002')
    cm = read(cp/'COMMITTED.json')
    assert sha(cp/'state.pkl') == cm['state_sha256'] and sha(cp/'policy.pt') == cm['policy_sha256']
    amendment = BASE/'protocols/r4_combat_v1_amendment03_independent_activation.json'
    value = dict(status='FROZEN', authorization='current user: K1/K3 need not wait for K2/kill arena',
        K1=dict(curriculum_fraction=.3,lessons=[7,12],lesson_assignment='fixed per environment reset key, half each',
            empty_hand_clear_brick=True,bank_conditional_probability=.25/.7,reward='R3 unchanged'),
        K3=dict(base='K1', selfplay_fraction=.55, weights=[.05*.45/.7,.35*.45/.7,.30*.45/.7,.275,.55/3,.55/6],
            method='retain bot and current/ring/fixed conditional proportions'),
        interceptor=dict(status='separate upcoming version after JS/JAX parity',
            no_retroactive_pool_or_carrier_reward_change=True),
        initialization='parent Actor/Critic/Adam/ring exact; fresh actual natural/Bank episodes and independent RNG',
        budget_updates=640,budget_new_environment_steps=167772160,save_every=40,eval_every=80,
        source_checkpoint=str(cp),source_policy_sha256=cm['policy_sha256'])
    if amendment.exists():
        assert read(amendment) == value
    else:
        atomic_json(amendment, value)
    for arm, seed in [('K1',2026331001),('K3',2026331003)]:
        path = BASE/'protocols'/('r4_'+arm.lower()+'_c000000.json')
        if path.exists():
            continue
        p = dict(parent)
        run = 'r4_'+arm.lower()+'_c000000'
        spec = dict(next(iter(parent['arms'].values())),curriculum_fraction=.3)
        if arm == 'K3':
            spec['opponent_weights'] = value['K3']['weights']
        p.update(id='R4-COMBAT-'+arm,namespace=str(BASE/'formal'),namespace_id='R4-COMBAT-20261031',
            arms={run:spec},sources={},policy_source={k:cm[k] for k in ('checkpoint','policy_sha256','update')},
            continuation_source=cm,chunk_start_update=4002,chunk_updates=640,
            environment_steps_per_run=167772160,r1_seed=seed,round14_collector_seed=seed,
            A1_weights=spec['opponent_weights'],r2_same_version=True,spawn_side_migration=True,
            worker_module=WORKER,activation_amendment=str(amendment),r4_arm=arm,
            r1_coefficient_request=str(BASE/'controls'/('unused_'+arm+'_attack.json')),
            r1_item_coefficient_request=str(BASE/'controls'/('unused_'+arm+'_legacy_items.json')),
            r3_item_coefficient_request=str(BASE/'controls'/('unused_'+arm+'_items.json')))
        files = list(parent['source_sha256']) + ['scripts/r4_combat_collector.py','scripts/r4_combat_train.py']
        p['source_sha256'] = {name:sha(Path(name)) for name in files}
        atomic_json(path,p)
    atomic_json(BASE/'K1_K3_REGISTRATION.json',dict(status='REGISTERED',protocols=[str(BASE/'protocols'/('r4_'+arm+'_c000000.json')) for arm in ('k1','k3')],
        preferred_gpu=3,avoid_reserved_cards=[0,5],prior_sp2='runs/r3_selfplay_ramp_20261031/seed02/ROLLBACK_SAFE_SAVE.json',
        inherited_completed_smoke=False,unix=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    BASE.mkdir(exist_ok=True,parents=True)
    if args.prepare:
        prepare();return
    guard=lock(BASE/'owner.lock');children={}
    while True:
        paused=any((BASE/x).exists() for x in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))
        state={}
        for arm in ('k1','k3'):
            run='r4_'+arm+'_c000000';out=BASE/'formal'/run;out.mkdir(exist_ok=True,parents=True)
            if active(run):
                progress=read(out/'PROGRESS.json',{})
                state[arm]=dict(status='WORKER_ACTIVE',progress=progress.get('iterations'),stage=read(out/'CYCLE_STAGE.json'))
                if progress and progress['last'].get('finite'):
                    first=BASE/(arm.upper()+'_FIRST_UPDATE.json')
                    if not first.exists():
                        row=progress['last'];assert min(row['update_l2'].values())>0
                        atomic_json(first,dict(status='PASS',row=row,formal_update=True,smoke_is_real_on_policy=True,unix=time.time()))
                        notify(arm+'_FIRST_UPDATE','R4 '+arm.upper()+'真实首更新：u'+str(row['new_update'])+'，262144 on-policy步，Actor/Critic非零且finite；冷周期'+str(round(row['cycle_seconds'],2))+'秒。',receipt=str(first))
                continue
            child=children.pop(run,None)
            if child:
                rc=child.wait()
                if rc:
                    atomic_json(out/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',stage=read(out/'LAUNCH.json')['stage'],returncode=rc,log=str(out/'worker.log'),unix=time.time()))
                    notify(run+'_BLOCKED','R4 '+arm.upper()+'技术阻塞，已保留checkpoint及worker.log，不重置或重跑通过步骤。',receipt=str(out/'TECHNICAL_BLOCKED.json'))
            if (out/'COMPLETE.json').exists():
                state[arm]=dict(status='COMPLETE',receipt=str(out/'COMPLETE.json'))
                notify(run+'_COMPLETE','R4 '+arm.upper()+'固定640更新预算完成，CPU评估继续。',receipt=str(out/'COMPLETE.json'))
                continue
            if paused or (out/'TECHNICAL_BLOCKED.json').exists():
                state[arm]=dict(status='PAUSED' if paused else 'TECHNICAL_BLOCKED');continue
            stage='SMOKE' if not (out/'SMOKE.json').exists() else 'RESTORE' if not (out/'RESTORE.json').exists() else 'FORMAL'
            claimed={read(BASE/'formal'/('r4_'+a+'_c000000')/'LAUNCH.json',{}).get('gpu')
                     for a in ('k1','k3') if active('r4_'+a+'_c000000')}
            free=[g for g in free_cards() if g not in (0,5) and g not in claimed]
            if not free:
                state[arm]=dict(status='WAIT_EXCLUSIVE_CARD',stage=stage);continue
            gpu=min(free,key=lambda g:(g!=3,g))
            cpus='48-55' if arm=='k1' else '56-63'
            argv=['taskset','-c',cpus,sys.executable,'-m',WORKER,str(BASE/'protocols'/(run+'.json')),run,str(2026331001 if arm=='k1' else 2026331003)]
            if stage!='FORMAL':argv.append('--smoke' if stage=='SMOKE' else '--verify-restore')
            env=dict(os.environ,JAXBOMB_DEVICE_OVERRIDE=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
            with (out/'worker.log').open('a') as stream:
                child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children[run]=child
            atomic_json(out/'LAUNCH.json',dict(stage=stage,gpu=gpu,cpu_affinity=cpus,command=argv[:3]+['<PYTHON>']+argv[4:],unix=time.time()))
            state[arm]=dict(status='LAUNCHED_NOT_YET_UPDATED',stage=stage,gpu=gpu)
            # Let CUDA register the claim before considering another arm.
            time.sleep(4)
        atomic_json(BASE/'EXECUTION_STATUS.json',dict(status='K1_K3_INDEPENDENT_ACTIVATION',arms=state,
            K2='WAIT_COLLECTOR_OPTIMIZER_ACCEPTANCE',K4='WAIT_FULL_PARITY',interceptor='IMPLEMENTING',unix=time.time()))
        if all(v.get('status')=='COMPLETE' for v in state.values()):return
        time.sleep(10)


if __name__=='__main__':main()
