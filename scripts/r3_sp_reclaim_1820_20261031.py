"""Consume the explicit 18:20 authorization without replaying earned updates."""
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha


def consume_resume(out):
    from scripts import r3_selfplay_owner_20261031 as owner
    request=owner.read(out/'RECLAIM_RESUME.json',{})
    acceptance=owner.read(out/'RECLAIM_FIRST_FORMAL_ACCEPTANCE.json',{})
    if request.get('status')=='CONSUMED' and not acceptance.get('next_step_token_verified'):
        entry=owner.read(out/'REGISTRATION.json')['entry'];folder=out/'formal'/entry['run']
        file=folder/'u003128/COMMITTED.json'
        token=folder/'NEXT_STEP_u003128.json'
        if file.exists() and token.exists():
            cm=owner.read(file);m=cm['metrics'];before=owner.read(Path(request['checkpoint'])/'COMMITTED.json')
            assert m['finite'] and min(m['update_l2'].values())>0
            assert m['actual_environment_steps_this_update']==262144
            assert cm['environment_steps']==before['environment_steps']+262144
            assert cm['optimizer_steps']==before['optimizer_steps']+32
            assert sha(file.parent/'state.pkl')==cm['state_sha256'] and sha(file.parent/'policy.pt')==cm['policy_sha256']
            atomic_json(out/'RECLAIM_FIRST_FORMAL_ACCEPTANCE.json',dict(status='PASS',source=cm,
                new_environment_steps=262144,new_optimizer_steps=32,
                next_step_token_verified=True,next_step_token_sha256=sha(token),
                env_per_second=262144/m['cycle_seconds'],includes_cold_compile=True,unix=time.time()))
            owner.notify('RECLAIM1820_FIRST_FORMAL','R3-sp主seed恢复后首正式u3128已验收：262144真实步、32 Adam步、Actor/Critic非零、finite及原子SHA通过；GPU2第二seed按同冻结合同接续。',receipt=str(out/'RECLAIM_FIRST_FORMAL_ACCEPTANCE.json'))
        return False
    if request.get('status') not in ('AUTHORIZED','RESTORING'):return False
    entry=owner.read(out/'REGISTRATION.json')['entry']
    folder=out/'formal'/entry['run'];cp=Path(request['checkpoint'])
    if list(owner.processes(owner.WORKER,entry['run'])):return True
    if request['status']=='AUTHORIZED':
        cm=owner.read(cp/'COMMITTED.json')
        assert cm['update']==3127
        assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
        assert (folder/'NEXT_STEP_u003127.json').exists()
        assert owner.compute_app_count(entry['gpu'])==0,'resume card has an unexpected compute application'
        archive=out/'attempts/reclaim1820';archive.mkdir(parents=True,exist_ok=True)
        for name in ('LAUNCH.json','RESTORE.json'):
            path=folder/name
            if path.exists():path.rename(archive/('PRE_RESUME_'+name))
        owner.launch(entry,'RESTORE')
        request.update(status='RESTORING',launched_unix=time.time())
        atomic_json(out/'RECLAIM_RESUME.json',request)
        return True
    restored=owner.read(folder/'RESTORE.json')
    if not restored:
        if time.time()-request['launched_unix']<30:return True
        raise RuntimeError('u3127 restore exited without receipt; inspect RESTORE.log')
    assert restored['checkpoint']==str(cp)
    assert all(restored.get(k) for k in ('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact'))
    archive=out/'attempts/reclaim1820'
    (folder/'LAUNCH.json').rename(archive/'RESUME_RESTORE_LAUNCH.json')
    atomic_json(out/'RECLAIM_RESTORE_ACCEPTANCE.json',dict(status='PASS',restore=restored,sha256=sha(folder/'RESTORE.json'),unix=time.time()))
    for name in ('RESOURCE_PAUSE.json','RESOURCE_CAPACITY_BLOCK.json'):
        path=out/name
        if path.exists():path.rename(archive/('PRE_RESUME_'+name))
    request.update(status='CONSUMED',remaining_updates=553,terminal_update=3680,unix=time.time())
    atomic_json(out/'RECLAIM_RESUME.json',request)
    owner.notify('RECLAIM1820_RESTORE_PASS','R3-sp主seed u3127完整恢复通过，剩553更新接续；旧暂停和u3120 bot回退证据保留，后续新点继续执行护栏。',receipt=str(out/'RECLAIM_RESTORE_ACCEPTANCE.json'))
    return False
