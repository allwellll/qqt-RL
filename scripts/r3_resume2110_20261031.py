"""Versioned 21:10 resume through the existing, uniquely locked SP owner."""
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

AUTH=Path('runs/r3_resume_publish_2110_20261031/AUTHORIZATION.json')

def consume(owner):
    request_file=owner.OUT/'RESUME_2110.json'
    request=owner.read(request_file,{})
    if request.get('status')=='CONSUMED':
        audit_progress(owner,request)
    if request.get('status') not in ('AUTHORIZED','RESTORING'):
        return False
    assert sha(AUTH)==request['authorization_sha256']
    entry=owner.read(owner.OUT/'REGISTRATION.json')['entry']
    folder=owner.OUT/'formal'/entry['run']
    if list(owner.processes(owner.WORKER,entry['run'])):
        return True
    archive=owner.OUT/'attempts/resume2110';archive.mkdir(parents=True,exist_ok=True)
    if request['status']=='AUTHORIZED':
        if owner.compute_app_count(entry['gpu']):
            atomic_json(owner.OUT/'STATUS.json',dict(status='WAIT_2110_EXCLUSIVE_CARD',gpu=entry['gpu'],unix=time.time()))
            return True
        cp=Path(request['checkpoint']);cm=owner.read(cp/'COMMITTED.json')
        assert sha(cp/'state.pkl')==cm['state_sha256']
        assert sha(cp/'policy.pt')==cm['policy_sha256']
        assert (folder/f"NEXT_STEP_u{cm['update']:06d}.json").exists()
        for name in ('LAUNCH.json','RESTORE.json'):
            path=folder/name
            if path.exists():path.rename(archive/('BEFORE_'+name))
        owner.launch(entry,'RESTORE')
        request.update(status='RESTORING',launched_unix=time.time())
        atomic_json(request_file,request)
        return True
    restored=owner.read(folder/'RESTORE.json')
    if not restored:
        if time.time()-request['launched_unix']<30:return True
        raise RuntimeError('21:10 restore exited without acceptance; inspect RESTORE.log')
    assert restored['checkpoint']==request['checkpoint']
    assert all(restored.get(k) for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
    (folder/'LAUNCH.json').rename(archive/'RESUME_RESTORE_LAUNCH.json')
    atomic_json(owner.OUT/'RESUME_2110_RESTORE_ACCEPTANCE.json',dict(status='PASS',restore=restored,sha256=sha(folder/'RESTORE.json'),unix=time.time()))
    for name in ('RESOURCE_PAUSE.json','RESOURCE_CAPACITY_BLOCK.json','OWNER_BLOCKED.json'):
        path=owner.OUT/name
        if path.exists():path.rename(archive/('BEFORE_'+name))
    state=dict(baseline=owner.reference_snapshot(),strikes={},last_update={},unix=time.time(),resume_authorization_sha256=sha(AUTH))
    root=owner.OUT.parent if owner.OUT.name=='seed02' else owner.OUT
    for filename,key in [('AMENDMENT05_RECLAIM_ACTIVE_REFERENCES.json','baseline_authorization_sha256'),('AMENDMENT06_COLD_REFERENCE_EXCLUSION.json','baseline_correction_sha256')]:
        if (root/filename).exists():state[key]=sha(root/filename)
    atomic_json(owner.OUT/'RESOURCE_GUARD.json',state)
    request.update(status='CONSUMED',unix=time.time())
    atomic_json(request_file,request)
    owner.notify('RESUME2110_RESTORE_PASS',f"R3-sp seed{entry['seed']}从u{request['start_update']}完整恢复PASS；55%不升70，剩{request['remaining_updates']}更新。历史bot回退证据保留，由新授权覆盖。",receipt=str(request_file))
    return False

def audit_progress(owner,request):
    entry=owner.read(owner.OUT/'REGISTRATION.json')['entry']
    folder=owner.OUT/'formal'/entry['run']
    target=owner.OUT/'RESUME_2110_FIRST_UPDATE.json'
    if target.exists() or not (folder/'updates.jsonl').exists():return
    rows=owner.tail_rows(folder/'updates.jsonl',700)
    first=next((r for r in rows if r['new_update']>request['start_update']),None)
    if not first:return
    assert first['finite'] and min(first['update_l2'].values())>0
    assert first['actual_environment_steps_this_update']==262144
    source=owner.read(Path(request['checkpoint'])/'COMMITTED.json')
    assert first['environment_steps']==source['environment_steps']+262144
    assert first['optimizer_steps']==source['optimizer_steps']+32
    receipt=dict(status='PASS',run=entry['run'],gpu=entry['gpu'],source_checkpoint=request['checkpoint'],
        source_state_sha256=source['state_sha256'],source_policy_sha256=source['policy_sha256'],
        new_update=first['new_update'],new_environment_steps=262144,new_optimizer_steps=32,
        metrics=first,env_per_second=262144/first['cycle_seconds'],includes_cold_compile=True,
        restore_acceptance_sha256=sha(owner.OUT/'RESUME_2110_RESTORE_ACCEPTANCE.json'),unix=time.time())
    atomic_json(target,receipt)
    owner.notify('RESUME2110_FIRST_UPDATE',f"R3-sp seed{entry['seed']}恢复首正式u{first['new_update']}：262144真实步、32 Adam步，Actor/Critic非零、finite；冷周期{first['cycle_seconds']:.2f}s，完整恢复通过，继续55%原预算。",receipt=str(target))

def superseded_bot_gate(owner,path):
    request=owner.read(owner.OUT/'RESUME_2110.json',{})
    prior=request.get('superseded_bot_gates',{}).get(str(path))
    if prior:
        assert sha(path)==prior,'historical bot evidence changed'
        return True
    if request.get('status')=='CONSUMED':
        # Backlogged CPU evaluation of a pre-resume checkpoint must retain its
        # adverse result without re-consuming the historical stop instruction.
        try:update=int(path.parent.name.rsplit('_u',1)[1])
        except (ValueError,IndexError):return False
        if update<=request['start_update']:
            request.setdefault('superseded_bot_gates',{})[str(path)]=sha(path)
            atomic_json(owner.OUT/'RESUME_2110.json',request)
            return True
    return False

def allocate_controls(owner):
    """Release old R3 boundary blocks only after the reserved card is empty."""
    if owner.OUT!=Path('runs/r3_items_route_20261030'):return
    queue_file=AUTH.parent/'R3_ALLOCATION_QUEUE.json'
    queue=owner.read(queue_file,{})
    if not queue:return
    slots=owner.read(owner.OUT/'R3_SLOTS.json')
    from scripts.r3_selfplay_owner_20261031 import compute_app_count
    for arm in ('a','b'):
        item=queue[arm]
        if item.get('status')=='CONSUMED':continue
        folder=Path('runs/bomb_threat_v1/formal')/item['waiting_run']
        done=owner.read(folder/'COMPLETE.json')
        if not done or compute_app_count(item['gpu']):continue
        retired=Path(done['checkpoint'])
        assert sha(retired/'state.pkl')==done['state_sha256']
        assert sha(retired/'policy.pt')==done['policy_sha256']
        assert (folder/f"NEXT_STEP_u{done['update']:06d}.json").exists()
        source=Path(item['source']);cm=owner.read(source/'COMMITTED.json')
        assert cm['update']==4002 and sha(source/'state.pkl')==cm['state_sha256']
        assert sha(source/'policy.pt')==cm['policy_sha256']
        entry=slots[arm]
        assert owner.read(owner.OUT/'formal'/entry['run']/'COMPLETE.json')['update']==4002
        archive=AUTH.parent/'control_boundary_blocks';archive.mkdir(exist_ok=True)
        for key in (entry['run'],f'r3_{arm}_ownside'):
            path=owner.OUT/'boundary_blocks'/f'{key}.json'
            if path.exists() and owner.read(path).get('status')=='BLOCK':
                atomic_json(archive/path.name,owner.read(path))
                atomic_json(path,dict(status='SUPERSEDED_BY_2110_RESUME',authorization=str(AUTH),previous_sha256=sha(archive/path.name),unix=time.time()))
        entry.update(gpu=item['gpu'],cpu_affinity='176-179' if arm=='a' else '180-183')
        item.update(status='CONSUMED',exclusive_card_verified=True,unix=time.time())
        owner.notify('RESUME2110_'+arm,f"R3-{arm}独占GPU{item['gpu']}安全交接：旧BT完整chunk已退出，从u4002按原配方续640更新；CPU评估不阻塞。",source=cm)
    atomic_json(owner.OUT/'R3_SLOTS.json',slots)
    atomic_json(queue_file,queue)
