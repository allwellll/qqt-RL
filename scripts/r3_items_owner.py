"""Persistent single R3 owner; adopts live workers and continues640 chunks."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.bun_native_v2_owner import processes

OUT=Path('runs/r3_items_route_20261030')
MODULE='scripts.r3_items_train'


def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default


def paused():
    return (OUT/'RESOURCE_PAUSE.json').exists() or any((d/f).exists() for d in (OUT,Path('runs/bun_native_v2_20261030'),Path('runs/bomb_threat_v1'),Path('runs/iter_pbt_v1'))
        for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))


def notify(key,message,**evidence):
    f=OUT/'notifications'/f'{key}.json'
    if f.exists():return
    atomic_json(f,dict(message=message,**evidence,unix=time.time()))
    with (OUT/'NOTIFY.log').open('ab') as log:subprocess.run(['jdme-notify','send',message],stdout=log,stderr=subprocess.STDOUT,check=False)


def register():
    if (OUT/'R3_SLOTS.json').exists():return
    required=['BASELINE_COMPLETE.json','FOCUSED_ACCEPTANCE.json','PARENT_INPUT_MIGRATION.json','NATIVE_EVENT_FIXTURES.json','EVALUATION_ABI_ACCEPTANCE.json','TECHNICAL_NATIVE_STEP.json']
    for f in required:assert read(OUT/f,{}).get('status') in ('PASS','COMPLETE'),f
    source=read(OUT/'R3_PARENT.json');parent=read(Path(source['pinned_checkpoint'])/'COMMITTED.json')
    assert parent['update']==2722 and sha(Path(parent['checkpoint'])/'state.pkl')==parent['state_sha256']
    base=read('runs/bun_native_v2_20261030/protocols/r2_b_custodyv2_c000000.json')
    recipe=next(iter(base['arms'].values()));slots={}
    sources={str(p):sha(p) for p in Path('scripts').glob('r3_items_*.py')}
    for f in ('scripts/r3_items_metrics.js','scripts/r3_items_inference.js','scripts/r3_items_candidate_eval.js','scripts/r3_items_candidate_metrics.js'):
        sources[f]=sha(Path(f))
    sources['scripts/bun_native_v2_custody_fast.py']=sha(Path('scripts/bun_native_v2_custody_fast.py'))
    for arm,seed,cores in [('a',2026208001,'176-179'),('b',2026208002,'180-183')]:
        name=f'r3_{arm}_c000000';p=copy.deepcopy(base)
        p.update(namespace=str(OUT/'formal'),arms={name:copy.deepcopy(recipe)},continuation_source=parent,chunk_start_update=2722,
            chunk_updates=640,r2_same_version=False,r3_reward_version='r3-events-items-route-chain-v1',r1_seed=seed,round14_collector_seed=seed,
            r3_item_coefficient_request=str(OUT/'controls'/f'{arm}_items.json'),
            r1_coefficient_request=str(OUT/'controls'/f'{arm}_attack.json'),r1_item_coefficient_request=str(OUT/'controls'/f'{arm}_legacy_items.json'))
        p['source_sha256'].update(sources)
        p['environment_steps_per_run']=(2722+640)*262144
        f=OUT/'protocols'/f'{name}.json';atomic_json(f,p)
        slots[arm]=dict(run=name,protocol=str(f),gpu=2,cpu_affinity=cores,seed=seed,chunk=0,worker_module=MODULE)
    atomic_json(OUT/'R3_SLOTS.json',slots)
    atomic_json(OUT/'R3_REGISTRATION.json',dict(status='REGISTERED',slots=slots,common_parent=parent,
        authorization='.hermes/TASK_r3_items_route_chain_20261030.md',amendment=str(OUT/'AMENDMENT02_EVENT_SETTLEMENT_AND_MIGRATION.json'),
        steps_per_update=262144,new_steps_per_chunk=167772160,updates_per_chunk=640,source_sha256=sources,
        inherited_reward_and_opponents=True,no_legacy_queue_resume=True,unix=time.time()))
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write('\nR3正式注册：两路seed2026208001/2026208002，GPU2并行、CPU176–179/180–183；共同父修订b首原子u2722，'
         '每路640更新=167772160新增真实步，smoke2更新计入预算，独立restore后首正式u2725。env2048×128、mb8192、epoch1、LR7.5e−5、entropy.005、DAgger/KL0与R2-b相同。'
         '新增25平面/37维可见输入，Actor/Critic/Adam旧行保留、新行零；自然对局/真实Bank重建和RNG重键，旧R1/R2不变。'
         '奖励包、30tick终局账本尾段、来源和hash见 `runs/r3_items_route_20261030/R3_REGISTRATION.json` 及 amendment02；非单一子项消融，不以短点效果门禁停健康训练。\n')


def launch(entry,phase):
    folder=OUT/'formal'/entry['run'];folder.mkdir(parents=True,exist_ok=True)
    if entry.get('spawn_side_version'):
        focused=read('runs/v2_ownside_selection_20261031/JAX_FOCUSED_ACCEPTANCE.json')
        assert focused['status']=='PASS' and focused['wrong_sides']==0
        assert focused['source_sha256']['scripts/bun_spawn_side.py']==sha(Path('scripts/bun_spawn_side.py'))
    module=entry.get('worker_module',MODULE)
    if module=='scripts.r3_items_fast2_train':
        evidence=Path('runs/bun_native_v2_20261030/custody_performance_v2')
        verified=read(evidence/'PARITY_PROFILE.json')
        matched=read(evidence/'collector_compare/probe_c/COMPLETE.json')
        assert verified['status']=='PASS' and verified['phi_and_cache_and_observations_bitwise_exact']
        assert all(sha(Path(f))==h for f,h in verified['source_sha256'].items())
        assert matched['status']=='PASS' and matched['all_outputs_digest_exact']
        if phase=='FORMAL':
            restored=read(folder/'R3_PERFORMANCE2_RESTORE.json',{})
            assert all(restored.get(k) for k in ('passed','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
            authorization=folder/'ATOMIC_RESOURCE_RESUME.json'
            resume=read(authorization,{})
            if resume.get('status')=='AUTHORIZED':
                atomic_json(authorization,dict(resume,status='CONSUMED',consumed_by='verified performance02 formal launch',unix=time.time()))
    command=['taskset','-c',entry['cpu_affinity'],sys.executable,'-m',module,entry['protocol'],entry['run'],str(entry['seed'])]
    if phase=='SMOKE':command+=['--smoke']
    if phase in ('RESTORE','PERFORMANCE_RESTORE'):command+=['--verify-restore']
    with (folder/f'{phase}.log').open('ab') as log:
        child=subprocess.Popen(command,env=dict(os.environ,JAXBOMB_DEVICE_OVERRIDE=str(entry['gpu'])),stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    atomic_json(folder/'LAUNCH.json',dict(status='LAUNCHED_NOT_ACCEPTED',run=entry['run'],phase=phase,gpu=entry['gpu'],cpu_affinity=entry['cpu_affinity'],
        command=['<PYTHON>','-m',module,entry['protocol'],entry['run'],str(entry['seed'])]+command[9:],unix=time.time()))
    return child


def acceptance(entry):
    folder=OUT/'formal'/entry['run'];p=read(entry['protocol']);u=p['chunk_start_update']+3
    f=folder/f'u{u:06d}/COMMITTED.json';dest=folder/'FIRST_FORMAL_UPDATE.json'
    if dest.exists() or not f.exists():return
    cp=f.parent;r=read(f);m=r['metrics']
    if m.get('user_paused'):
        m=next(row for row in map(json.loads,(folder/'updates.jsonl').read_text().splitlines()) if row['new_update']==u)
    assert m['finite'] and min(m['update_l2'].values())>0
    assert m['actual_environment_steps_this_update']==262144 and sum(m['exposure']['r3_positive_pay'])>0
    assert sha(cp/'state.pkl')==r['state_sha256'] and sha(cp/'policy.pt')==r['policy_sha256']
    assert read(folder/'RESTORE.json')['next_step_exact']
    import pickle
    import numpy as np
    actor=pickle.load((cp/'policy.pt').open('rb'))
    new_row_norms=dict(token=float(np.linalg.norm(actor['tok'][0][216:])),
        global_input=float(np.linalg.norm(actor['state_w'][24:])))
    assert min(new_row_norms.values())>0,'new visible input rows never received a real optimizer update'
    atomic_json(dest,dict(status='PASS',source=r,restore=read(folder/'RESTORE.json'),finite=True,
        actor_critic_nonzero=True,r3_pay=m['exposure']['r3_pay'],new_input_update_norms=new_row_norms,environment_steps_this_update=262144,
        cycle_seconds=m['cycle_seconds'],env_per_second=262144/m['cycle_seconds'],unix=time.time()))
    notify(entry['run']+'_FIRST_UPDATE',f"R3 {entry['run']} 正式u{u}，262144真实on-policy步，finite、Actor/Critic非零，8类R3事件账本进入collector/GAE，原子SHA和完整恢复PASS；R1/R2继续。",receipt=str(dest))


def successor(arm,entry,done):
    p=read(entry['protocol']);assert done['update']==p['chunk_start_update']+640
    cp=Path(done['checkpoint']);assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    name=f"r3_{arm}_c{entry['chunk']+1:06d}";recipe=next(iter(p['arms'].values()))
    p.update(arms={name:recipe},continuation_source=done,chunk_start_update=done['update'],r2_same_version=True,
        environment_steps_per_run=(done['update']+640)*262144)
    from scripts.bun_spawn_side_successor import configure
    updated,p=configure('R3',arm,dict(entry,run=name,chunk=entry['chunk']+1),p)
    name=updated['run']
    f=OUT/'protocols'/f'{name}.json';atomic_json(f,p)
    notify(entry['run']+'_COMPLETE',f"R3 {entry['run']} 640更新chunk原子完成u{done['update']}，接续{name}；评估异步，无发布。",source=done,successor=name)
    return dict(updated,run=name,protocol=str(f),chunk=entry['chunk']+1)


def main():
    guard=lock(OUT/'R3_OWNER.lock');children={};register();last_resource=0
    while True:
        for run,p in list(children.items()):
            if p.poll() is not None:del children[run]
        if time.time()-last_resource>120:
            from scripts.r3_items_resources import refresh
            refresh();last_resource=time.time()
        if paused():
            atomic_json(OUT/'TRAIN_EXECUTION_STATUS.json',dict(status='PAUSE_GATE',unix=time.time()));time.sleep(5);continue
        from scripts.r3_resume2110_20261031 import allocate_controls
        allocate_controls(__import__(__name__,fromlist=['OUT']))
        slots=read(OUT/'R3_SLOTS.json');statuses={}
        for arm,entry in slots.items():
            folder=OUT/'formal'/entry['run'];acceptance(entry)
            if read(folder/'COMPLETE.json'):
                from scripts.bun_spawn_side_successor import enabled
                ownside=read(entry['protocol']).get('spawn_side_version')
                old_gate_invalidated=enabled() and not ownside
                if not old_gate_invalidated and any(read(OUT/'boundary_blocks'/f'{key}.json',{}).get('status')=='BLOCK' for key in (entry['run'],f'r3_{arm}'+('_ownside' if ownside else ''))):
                    statuses[arm]='SAFETY_BLOCKED_AT_COMPLETE_CHUNK';continue
                entry=slots[arm]=successor(arm,entry,read(folder/'COMPLETE.json'));atomic_json(OUT/'R3_SLOTS.json',slots);folder=OUT/'formal'/entry['run']
            if read(folder/'TECHNICAL_FAILURE.json'):
                statuses[arm]='TECHNICAL_BLOCKED';continue
            if list(processes(entry.get('worker_module',MODULE),entry['run'])):statuses[arm]='WORKER_PRESENT';continue
            if entry.get('worker_module')=='scripts.r3_items_fast2_train':
                verified=read(folder/'R3_PERFORMANCE2_RESTORE.json',{})
                if all(verified.get(k) for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact')):
                    if not (folder/'RESTORE.json').exists():atomic_json(folder/'RESTORE.json',verified)
                    if entry.get('performance_restore_pending'):
                        entry['performance_restore_pending']=False;atomic_json(OUT/'R3_SLOTS.json',slots)
            phase='SMOKE' if not (folder/'SMOKE.json').exists() else 'RESTORE' if not (folder/'RESTORE.json').exists() else 'FORMAL'
            if entry.get('performance_restore_pending'):phase='PERFORMANCE_RESTORE'
            # Serialize cold qualification, not the formal training itself.
            # This avoids two simultaneous XLA compilations stealing host
            # resources from already healthy reference workers.
            if arm=='b' and not any((OUT/'formal').glob('r3_a_*/FIRST_FORMAL_UPDATE.json')):
                statuses[arm]='WAIT_A_FIRST_FORMAL_COLD_RESOURCE_GUARD';continue
            last=read(folder/'LAUNCH.json',{})
            if last.get('phase')==phase:
                if time.time()-last['unix']<20:continue
                resume=read(folder/'ATOMIC_RESOURCE_RESUME.json',{})
                if phase=='FORMAL' and resume.get('status')=='AUTHORIZED':
                    cp=Path(resume['checkpoint']);cm=read(cp/'COMMITTED.json')
                    assert cm['metrics'].get('user_paused')
                    assert sha(cp/'state.pkl')==cm['state_sha256']==resume['state_sha256']
                    assert sha(cp/'policy.pt')==cm['policy_sha256']==resume['policy_sha256']
                    assert (folder/f"NEXT_STEP_u{cm['update']:06d}.json").exists()
                    atomic_json(folder/'ATOMIC_RESOURCE_RESUME.json',dict(resume,status='CONSUMED',unix=time.time()))
                    children[entry['run']]=launch(entry,phase);statuses[arm]='ATOMIC_RESOURCE_RESUME';continue
                atomic_json(folder/'TECHNICAL_FAILURE.json',dict(status='BLOCKED',phase=phase,reason='worker exited before phase acceptance; inspect phase log',unix=time.time()))
                notify(entry['run']+'_BLOCKED',f"R3 {entry['run']} {phase}技术阻塞，证据已保存，R1/R2不受影响。",log=str(folder/f'{phase}.log'))
                statuses[arm]='TECHNICAL_BLOCKED';continue
            children[entry['run']]=launch(entry,phase);statuses[arm]='LAUNCHED_'+phase
        atomic_json(OUT/'TRAIN_EXECUTION_STATUS.json',dict(status='ACTIVE',slots=slots,phases=statuses,unix=time.time()))
        time.sleep(5)


def watchdog():
    guard=lock(OUT/'R3_WATCHDOG.lock')
    while True:
        owners=[p for p in processes('scripts.r3_items_owner') if b'--watchdog' not in (p/'cmdline').read_bytes().split(b'\0')]
        if not paused() and not owners:
            with (OUT/'OWNER.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m','scripts.r3_items_owner'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(OUT/'WATCHDOG_RESTART.json',dict(status='OWNER_RESTART_REQUESTED',live_training_adopted=True,unix=time.time()))
        if not paused() and not list(processes('scripts.r3_items_eval_owner')) and not (OUT/'EVAL_BLOCKED.json').exists():
            with (OUT/'EVAL_OWNER.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m','scripts.r3_items_eval_owner'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watchdog',action='store_true');args=parser.parse_args()
    if args.watchdog:watchdog()
    else:
        try:main()
        except Exception:
            atomic_json(OUT/'OWNER_BLOCKED.json',dict(status='BLOCKED',reason='inspect OWNER.log; live workers never signalled',unix=time.time()));raise
