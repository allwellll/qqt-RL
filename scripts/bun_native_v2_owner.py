"""Durable bounded-chunk R2 queue; never touches old R2/O5 or R1 workers."""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('runs/bun_native_v2_20261030')
FRESH=Path('runs/bomb_threat_v1/r1_fresh_publish_1805_20261030')
MODULE='scripts.bun_native_v2_train'


def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def paused():
    return any((p/n).exists() for p in (OUT,Path('runs/bomb_threat_v1'),Path('runs/iter_pbt_v1'))
               for n in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))


def processes(module,run=None):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if module.encode() in a and (run is None or run.encode() in a) and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':
                yield p
        except OSError:pass


def notification(name,message,data):
    f=OUT/'notifications'/f'{name}.json'
    if f.exists():return
    atomic_json(f,dict(message=message,**data,unix=time.time()))
    subprocess.run(['jdme-notify','send',message],check=False)


def gates():
    needed={'DIAGNOSTIC_COMPLETE.json':'COMPLETE','parity/event_v3/ACCEPTANCE.json':'PASS','REWARD_WIRING.json':'PASS','COLLECTOR_WIRING.json':'PASS','DONOR_STATE_ACCEPTANCE.json':'PASS',
            'FOCUSED_TESTS.json':'PASS'}
    missing=[p for p,s in needed.items() if read(OUT/p,{}).get('status')!=s]
    return missing


def donor():
    approved=read(OUT/'DONOR_FORK_AUTHORIZATION.json')
    if approved:
        accepted=read(OUT/'DONOR_STATE_ACCEPTANCE.json')
        assert accepted['status']=='PASS' and accepted['authorization_sha256']==sha(OUT/'DONOR_FORK_AUTHORIZATION.json')
        assert accepted['source']['update']==approved['explicit_parent_update']==2721
        return accepted['source'],dict(method='steer_explicit_u2721_fork',
            authorization=str(OUT/'DONOR_FORK_AUTHORIZATION.json'),
            authorization_sha256=sha(OUT/'DONOR_FORK_AUTHORIZATION.json'),
            source_protocol=approved['parent_protocol'],parent_run=approved['parent_run'],
            recovery_acceptance=str(OUT/'DONOR_STATE_ACCEPTANCE.json'),not_original_u2760_reconstruction=True)
    selected=read(FRESH/'FINAL_SELECTION.json')
    if selected:
        # Publication eligibility and R1 improvement are fixed by the existing
        # predeclared paired protocol; don't pick a new seed peak here.
        label=selected['winner']
        if label=='baseline':
            # Not beating the website is different from having no R1 fresh
            # data. Use the prospective donor-only tie rule, never a new peak.
            order=['slot07_u2960','slot00_u2920','slot03_u2760']
            label=max(order,key=lambda k:selected['comparison'][k]['P1_sum'])
        if label:
            model=selected['models'][label]
            cp=Path(model['checkpoint']);source=read(cp/'COMMITTED.json')
            recovered=read(OUT/'DONOR_STATE_ACCEPTANCE.json')
            if not (cp/'state.pkl').exists() and recovered:
                assert recovered['status']=='PASS' and recovered['original_checkpoint']==str(cp) and recovered['policy_exact']
                return recovered['source'],dict(method='fixed_R1_fresh_winner_exact_state_reconstruction',
                    selection=str(FRESH/'FINAL_SELECTION.json'),selection_sha256=sha(FRESH/'FINAL_SELECTION.json'),label=label,
                    source_protocol=recovered['parent_protocol'],parent_run=recovered['parent_run'],recovery_acceptance=str(OUT/'DONOR_STATE_ACCEPTANCE.json'))
            return source,dict(method='fixed_R1_fresh_winner' if selected['publish'] else 'fixed_R1_highest_P1_sum_no_web_improvement_claim',
                selection=str(FRESH/'FINAL_SELECTION.json'),selection_sha256=sha(FRESH/'FINAL_SELECTION.json'),label=label)
    slots=read('runs/bomb_threat_v1/SLOTS.json')
    run=slots['7']['run'];folder=Path('runs/bomb_threat_v1/formal')/run
    for cp in reversed(sorted(folder.glob('u*/COMMITTED.json'))):
        m=read(cp)
        if (cp.parent/'state.pkl').exists() and (cp.parent/'policy.pt').exists():
            return m,dict(method='explicit_slot07_latest_atomic_fallback',no_fresh_R1_winner=bool(selected))
    raise RuntimeError('no retained complete R1 donor')


def register():
    existing=read(OUT/'R2_REGISTRATION.json')
    if existing:return existing
    locked=read(OUT/'R2_DONOR_LOCK.json')
    if locked:source,selection=locked['source'],locked['selection']
    else:
        source,selection=donor()
        atomic_json(OUT/'R2_DONOR_LOCK.json',dict(source=source,selection=selection,unix=time.time()))
    cp=Path(source['checkpoint'])
    assert sha(cp/'state.pkl')==source['state_sha256'] and sha(cp/'policy.pt')==source['policy_sha256']
    source_run=selection.get('parent_run',cp.parent.name)
    parent=read(selection.get('source_protocol',Path('runs/bomb_threat_v1/protocols')/f'{source_run}.json'))
    # Only the existing progress/BANK_ONLY R1 recipe has the audited R2 ABI.
    assert not parent.get('r1_preserved_legacy',False),'preserved-reward parent needs explicit ABI adapter'
    from scripts.bomb_threat_round15 import pin
    if str(cp).startswith('runs/bomb_threat_v1/formal/'):
        pin([str(cp)],'native-v2 frozen common R2 parent')
    else:
        # Recovery and all new R2 states are outside the authorized legacy
        # retention deletion scope; retain explicitly without invalid v1 pins.
        atomic_json(OUT/'STATE_RETENTION_PINS.json',dict(checkpoints=[dict(checkpoint=str(cp),reason='common R2 parent exact reconstruction')]))
    rows={}
    a_gpu=0
    for i,(arm,gpu,cores,seed,anti) in enumerate((('a',a_gpu,'160-163',2026207001,False),
            ('b',4,'164-167',2026207001,True),('c',7,'168-171',2026207002,True))):
        name=f'r2_{arm}_c000000';p=copy.deepcopy(parent)
        spec=copy.deepcopy(parent['arms'][source_run])
        p.update(id='BUN-NATIVE-V2-R2',namespace=str(OUT/'formal'),arms={name:spec},
            continuation_source=source,policy_source={k:source[k] for k in ('checkpoint','policy_sha256','update')},
            chunk_start_update=source['update'],chunk_updates=640,r1_seed=seed,round14_collector_seed=seed,
            rule_version='bun-native-v2',r2_antifarm=anti,r2_same_version=False,worker_module=MODULE,
            r2_arm=arm,r2_chunk=0,r1_same_version_continuation=False,
            r1_coefficient_request=str(OUT/'no_automatic_coefficient_change.json'),
            r1_item_coefficient_request=str(OUT/'no_automatic_item_coefficient_change.json'),
            r2_parent_selection=selection,source_sha256={})
        for f in ('scripts/bun_native_v2_train.py','scripts/bun_native_v2_collector.py','scripts/bun_native_v2_reward.py',
                  'scripts/bun_native_v2_runtime.py','web/sim-bun-native-v2.js'):
            p['source_sha256'][f]=sha(Path(f))
        # Keep all original sealed recipe dependencies in addition to v2 code.
        p['source_sha256'].update(parent['source_sha256'])
        p['source_sha256']['scripts/bun_native_v2_train.py']=sha(Path('scripts/bun_native_v2_train.py'))
        f=OUT/'protocols'/f'{name}.json';atomic_json(f,p)
        rows[arm]=dict(run=name,protocol=str(f),gpu=gpu,cpu_affinity=cores,seed=seed,antifarm=anti,chunk=0)
    result=dict(status='REGISTERED',arms=rows,common_parent=source,selection=selection,unix=time.time())
    atomic_json(OUT/'R2_REGISTRATION.json',result)
    atomic_json(OUT/'R2_SLOTS.json',rows)
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write(f"\n## BUN-NATIVE-V2-R2 三路注册\n\n共同父 `{source['checkpoint']}`，policy `{source['policy_sha256']}`，state `{source['state_sha256']}`，选择 `{selection['method']}`。a/b seed2026207001，c seed2026207002；b/c新增0.3击杀塑形/敌取包−0.05，终局不折扣。原Adam/Actor/Critic/ring保留，自然对局重建与真实Bank恢复、独立RNG；{640*262144}真实环境步/chunk，640更新，40更新原子存盘，DAgger/KL=0，R1各路继续。GPU{a_gpu}/4/7与R1共卡，CPU160–171独立核段；逐run配置/来源/hash见 `runs/bun_native_v2_20261030/protocols/`。\n")
    return result


def launch(entry,phase):
    folder=OUT/'formal'/entry['run'];folder.mkdir(parents=True,exist_ok=True)
    if entry.get('spawn_side_version'):
        focused=read('runs/v2_ownside_selection_20261031/JAX_FOCUSED_ACCEPTANCE.json')
        assert focused['status']=='PASS' and focused['wrong_sides']==0
        assert focused['source_sha256']['scripts/bun_spawn_side.py']==sha(Path('scripts/bun_spawn_side.py'))
    if entry.get('r2_reward_version')=='custody-graded-conservative-v2':
        accepted=read(OUT/'custody_v2_tests/ACCEPTANCE.json')
        assert accepted['status']=='PASS' and not accepted['failures']
        assert all(sha(Path(f))==h for f,h in accepted['source_sha256'].items())
    env=dict(os.environ,JAXBOMB_DEVICE_OVERRIDE=str(entry['gpu']))
    module=entry.get('worker_module',MODULE)
    if module in ('scripts.bun_native_v2_custody_fast_train','scripts.bun_native_v2_custody_fast2_train'):
        evidence='custody_performance_v2' if module.endswith('fast2_train') else 'custody_performance_v1'
        verified=read(OUT/evidence/'PARITY_PROFILE.json')
        assert verified['status']=='PASS' and verified['phi_and_cache_and_observations_bitwise_exact']
        assert all(sha(Path(f))==h for f,h in verified['source_sha256'].items())
        if evidence=='custody_performance_v2':
            matched=read(OUT/evidence/'collector_compare/probe_c/COMPLETE.json')
            assert matched['status']=='PASS' and matched['all_outputs_digest_exact'] and matched['speedup']>=1.10
            if phase=='FORMAL':
                restored=read(folder/'PERFORMANCE2_RESTORE.json',{})
                assert all(restored.get(k) for k in ('passed','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
                authorization=folder/'ATOMIC_PAUSE_RESUME_AUTHORIZATION.json'
                resume=read(authorization,{})
                if resume.get('status')=='AUTHORIZED':
                    atomic_json(authorization,dict(resume,status='CONSUMED',consumed_by='verified performance02 formal launch',unix=time.time()))
    command=['taskset','-c',entry['cpu_affinity'],sys.executable,'-m',module,entry['protocol'],entry['run'],str(entry['seed'])]
    if phase=='SMOKE':command+=['--smoke']
    if phase in ('RESTORE','PERFORMANCE_RESTORE'):command+=['--verify-restore']
    with (folder/f'{phase}.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    # Store reproducible neutral command, never machine identity or PID.
    atomic_json(folder/'LAUNCH.json',dict(status='LAUNCHED_NOT_ACCEPTED',phase=phase,run=entry['run'],gpu=entry['gpu'],
        cpu_affinity=entry['cpu_affinity'],command=['<PYTHON>','-m',module,entry['protocol'],entry['run'],str(entry['seed'])]+command[9:],unix=time.time()))
    return child


def next_entry(arm,entry,done):
    p=read(entry['protocol']);assert done['update']==p['chunk_start_update']+640
    cp=Path(done['checkpoint'])
    assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    chunk=entry['chunk']+1;prefix=entry.get('run_prefix',f'r2_{arm}');name=f'{prefix}_c{chunk:06d}'
    recipe=p['arms'][entry['run']]
    p.update(arms={name:recipe},continuation_source=done,chunk_start_update=done['update'],r2_same_version=True,r2_chunk=chunk)
    from scripts.bun_spawn_side_successor import configure
    updated,p=configure('R2',arm,dict(entry,run=name,chunk=chunk),p)
    name=updated['run']
    f=OUT/'protocols'/f'{name}.json';atomic_json(f,p)
    notification(entry['run']+'_COMPLETE',f"native-v2 {entry['run']} 完整640更新chunk结束，u{done['update']}，原子SHA验收；同配置接续{name}，v2评估异步。",dict(run=entry['run'],terminal=done,successor=name))
    return dict(updated,run=name,protocol=str(f),chunk=chunk)


def accept_update(arm,entry):
    folder=OUT/'formal'/entry['run']
    p=read(entry['protocol'])
    u=p['chunk_start_update']+3
    cp=folder/f'u{u:06d}'
    receipt=read(cp/'COMMITTED.json')
    dest=folder/'FIRST_FORMAL_UPDATE.json'
    if dest.exists() or not receipt:return
    m=receipt['metrics']
    if m.get('user_paused'):
        # An atomic pause can resave the first formal checkpoint. Keep the
        # actual update evidence rather than interpreting pause metadata as it.
        rows=[json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
        m=next(row for row in rows if row['new_update']==u)
    assert m['finite'] and min(m['update_l2'].values())>0
    assert m['actual_environment_steps_this_update']==p['common']['envs_per_device']*p['common']['rollout_ticks']
    assert sha(cp/'state.pkl')==receipt['state_sha256'] and sha(cp/'policy.pt')==receipt['policy_sha256']
    atomic_json(dest,dict(status='PASS',arm=arm,atomic=receipt,update_metrics=m,smoke_2_updates_reused=True,
        env_per_second=m['actual_environment_steps_this_update']/m['cycle_seconds'],unix=time.time()))
    notification(entry['run']+'_FIRST_UPDATE',f"bun-native-v2 {entry['run']} 正式u{u}：finite，Actor/Critic非零，262144真实环境步，{m['actual_environment_steps_this_update']/m['cycle_seconds']:.0f} env/s，原子state/policy SHA验收；parity6完整局PASS，v2单独榜，R1继续。",dict(acceptance=str(dest)))


def recover_terminal(entry):
    folder=OUT/'formal'/entry['run']
    if (folder/'COMPLETE.json').exists() or list(processes(entry.get('worker_module',MODULE),entry['run'])):return
    p=read(entry['protocol']);target=p['chunk_start_update']+p['chunk_updates']
    cp=folder/f'u{target:06d}';m=read(cp/'COMMITTED.json')
    token=folder/f'NEXT_STEP_u{target:06d}.json'
    if not m or not token.exists():return
    assert m['metrics']['finite'] and min(m['metrics']['update_l2'].values())>0
    assert m['update']==target and sha(cp/'state.pkl')==m['state_sha256'] and sha(cp/'policy.pt')==m['policy_sha256']
    atomic_json(folder/'COMPLETE.json',dict(m,actual_environment_steps=m['environment_steps'],actual_optimizer_steps=m['optimizer_steps']))
    atomic_json(folder/'RECOVERED_TERMINAL.json',dict(status='COMPLETE_FROM_VERIFIED_FINAL_ATOMIC',checkpoint=m,
        next_step_token_sha256=sha(token),worker_exited=True,unix=time.time()))


def capacity_pause_at_complete_chunk(arm, entry):
    """Honor a per-arm allocation decision only after the frozen chunk exits."""
    request_file=OUT/'controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json'
    request=read(request_file,{})
    target=request.get('arms',{}).get(arm)
    if request.get('status')!='AUTHORIZED' or target is None:return None
    assert target['run']==entry['run'],'capacity pause must bind the frozen current run'
    assert sha(Path(request['selection']))==request['selection_sha256']
    p=read(entry['protocol'])
    assert target['target_update']==p['chunk_start_update']+p['chunk_updates']
    folder=OUT/'formal'/entry['run']
    done=read(folder/'COMPLETE.json')
    if not done:
        atomic_json(OUT/'capacity_pause_requests'/f'{arm}.json',dict(
            status='CONSUMED_WAIT_COMPLETE_CHUNK',run=entry['run'],target_update=target['target_update'],
            request_sha256=sha(request_file),training_workers_signalled=False,unix=time.time()))
        return None  # A save at u+40 is not a complete640-update chunk.
    if list(processes(entry.get('worker_module',MODULE),entry['run'])):
        return 'WAIT_COMPLETE_CHUNK_WORKER_EXIT'
    assert done['update']==target['target_update']
    cp=Path(done['checkpoint']);token=folder/f"NEXT_STEP_u{done['update']:06d}.json"
    assert sha(cp/'state.pkl')==done['state_sha256']
    assert sha(cp/'policy.pt')==done['policy_sha256']
    probe=read(token)
    assert all(probe.get(k) for k in ('digest','Adam_digest','full_environment_digest','ring_digest'))
    required=('params','adam','states','aux','keys','combat_ring','combat_ring_cursor')
    # JAX arrays in pickle may initialize CUDA during unpickling. Schema
    # acceptance belongs in a fresh CPU-only process, never in this scheduler.
    code=('import pickle,sys;'
          'p=pickle.load(open(sys.argv[1],"rb"));'
          'required=("params","adam","states","aux","keys","combat_ring","combat_ring_cursor");'
          'assert all(k in p for k in required),"incomplete resumable training state";'
          'assert "actor" in p["params"] and "critic" in p["params"];'
          'assert p["update"]==int(sys.argv[2])')
    subprocess.run([sys.executable,'-c',code,str(cp/'state.pkl'),str(done['update'])],check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        stdout=subprocess.DEVNULL)
    receipt=OUT/'capacity_pauses'/f'{arm}.json'
    atomic_json(receipt,dict(status='PAUSED_COMPLETE_CHUNK',arm=arm,run=entry['run'],gpu=entry['gpu'],
        source=done,full_state_fields=list(required),next_step_token_sha256=sha(token),
        request_sha256=sha(request_file),selection_sha256=request['selection_sha256'],
        training_worker_exited=True,training_workers_signalled=False,checkpoints_deleted=0,
        restore_protocol=entry['protocol'],successor_blocked=True,unix=time.time()))
    entry.update(suspended='SP_CAPACITY_AUTHORIZED_20261031',capacity_pause_receipt=str(receipt))
    notification(entry['run']+'_CAPACITY_PAUSE',
        f"R2-{arm}完整chunk u{done['update']}原子结束，Actor/Critic/Adam/env/aux/RNG/ring及hash验收；停止自动接续，GPU{entry['gpu']}按授权交R3-sp。未向训练worker发信号、未删checkpoint。",
        dict(receipt=str(receipt)))
    return 'PAUSED_COMPLETE_CHUNK_FOR_SP'


def main():
    guard=lock(OUT/'R2_OWNER.lock')
    children={};sample_time=0
    while True:
        for key,p in list(children.items()):
            if p.poll() is not None:del children[key]
        if paused():
            atomic_json(OUT/'R2_EXECUTION_STATUS.json',dict(status='USER_PAUSED',worker_atomic_stop_requested=True,unix=time.time()))
            time.sleep(10);continue
        missing=gates()
        if missing:
            atomic_json(OUT/'R2_EXECUTION_STATUS.json',dict(status='WAITING_TECHNICAL_AND_DIAGNOSTIC',missing=missing,
                R1_untouched=True,unix=time.time()));time.sleep(10);continue
        registration=register();slots=read(OUT/'R2_SLOTS.json');statuses={}
        for arm,entry in slots.items():
            if entry.get('suspended'):
                statuses[arm]='SUSPENDED_'+entry['suspended'];continue
            folder=OUT/'formal'/entry['run']
            accept_update(arm,entry)
            recover_terminal(entry)
            capacity_status=capacity_pause_at_complete_chunk(arm,entry)
            if capacity_status:
                statuses[arm]=capacity_status
                atomic_json(OUT/'R2_SLOTS.json',slots)
                continue
            done=read(folder/'COMPLETE.json')
            if done:
                entry=slots[arm]=next_entry(arm,entry,done);atomic_json(OUT/'R2_SLOTS.json',slots)
                folder=OUT/'formal'/entry['run']
            if read(folder/'TECHNICAL_FAILURE.json'):
                notification(entry['run']+'_BLOCKED','native-v2 '+entry['run']+' 技术阻塞，已保存证据；R1不受影响，需查技术日志。',dict(run=entry['run'],status='BLOCKED'))
                statuses[arm]='BLOCKED';continue
            if list(processes(entry.get('worker_module',MODULE),entry['run'])):
                statuses[arm]='WORKER_PRESENT_NOT_ACCEPTANCE';continue
            progress=read(folder/'PROGRESS.json')
            smoke=read(folder/'SMOKE.json')
            restore=read(folder/'RESTORE.json')
            performance_receipt=entry.get('performance_restore_receipt','PERFORMANCE_RESTORE.json')
            if not restore and entry.get('worker_module') in ('scripts.bun_native_v2_custody_fast_train','scripts.bun_native_v2_custody_fast2_train'):
                verified=read(folder/performance_receipt,{})
                if all(verified.get(k) for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact')):
                    # New chunks have no historical RESTORE receipt. Publish
                    # the same independently verified result under the ABI
                    # name the unchanged formal worker consumes.
                    atomic_json(folder/'RESTORE.json',verified);restore=verified
            phase='SMOKE' if not smoke else 'RESTORE' if not restore else 'FORMAL'
            if entry.get('performance_restore_pending'):
                performance=read(folder/performance_receipt,{})
                if performance.get('passed') and performance.get('next_step_exact') and performance.get('Adam_exact') and performance.get('full_env_aux_RNG_exact'):
                    entry['performance_restore_pending']=False
                    atomic_json(OUT/'R2_SLOTS.json',slots)
                else:phase='PERFORMANCE_RESTORE'
            # Never endlessly retry a failed smoke or compile failure.
            last=read(folder/'LAUNCH.json',{})
            if last.get('phase')==phase and (folder/f'{phase}.log').exists():
                if time.time()-last['unix']<20:continue
                authorization=read(folder/'ATOMIC_PAUSE_RESUME_AUTHORIZATION.json',{})
                if authorization.get('status')=='AUTHORIZED' and phase=='FORMAL':
                    checkpoint=Path(authorization['checkpoint'])
                    saved=read(checkpoint/'COMMITTED.json',{})
                    assert saved.get('metrics',{}).get('user_paused')
                    assert sha(checkpoint/'state.pkl')==saved['state_sha256']==authorization['state_sha256']
                    assert sha(checkpoint/'policy.pt')==saved['policy_sha256']==authorization['policy_sha256']
                    assert (folder/f"NEXT_STEP_u{saved['update']:06d}.json").exists()
                    atomic_json(folder/'ATOMIC_PAUSE_RESUME_AUTHORIZATION.json',dict(authorization,status='CONSUMED',unix=time.time()))
                    children[entry['run']]=launch(entry,phase);statuses[arm]='AUTHORIZED_ATOMIC_RESUME';continue
                atomic_json(folder/'TECHNICAL_FAILURE.json',dict(status='BLOCKED',reason='worker exited before phase acceptance; inspect phase log',phase=phase,unix=time.time()))
                continue
            children[entry['run']]=launch(entry,phase);statuses[arm]='LAUNCHED_'+phase
        atomic_json(OUT/'R2_EXECUTION_STATUS.json',dict(status='ACTIVE',slots=slots,phases=statuses,unix=time.time()))
        # These subprocesses are read-only and do not reserve device memory.
        if time.time()-sample_time>15:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True)
            with (OUT/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),rows=text.splitlines()))+'\n')
            sample_time=time.time()
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(OUT/'R2_OWNER_BLOCKED.json',dict(status='BLOCKED',error_type=type(e).__name__,reason='inspect R2_OWNER.log; R1 training untouched',unix=time.time()))
        raise
