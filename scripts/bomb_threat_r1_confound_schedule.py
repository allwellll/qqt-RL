"""Authorized slot03 switch only after the in-flight formal chunk completes."""
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read,pin
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';OUT=BASE/'iter2/r1_attack_20261030/items_v2/conservation_v3'
MODULE='scripts.bomb_threat_r1_preserved_train'
FILES=['scripts/bomb_threat_r1_preserved_train.py','scripts/bomb_threat_r1_preserved_collector.py','scripts/bomb_threat_r1_preserved_reward.py']


def apply_switch(slot,entry,chunk):
    if slot!=3 or entry.get('r1_preserved_legacy') or not entry.get('r1_items_applied') or entry.get('r1_smoke_fallback'):return None
    ready=read(OUT/'READY.json',{})
    if ready.get('status')!='PASS':return None
    previous=BASE/'formal'/entry['run'];done=read(previous/'COMPLETE.json',{})
    if not done:return None
    assert sha(ROOT/done['checkpoint']/'state.pkl')==done['state_sha256']
    assert sha(ROOT/done['checkpoint']/'policy.pt')==done['policy_sha256']
    p=read(BASE/'protocols'/f'{entry["run"]}.json');name=f'bt_slot03_c{chunk:06d}'
    recipe=dict(p['arms'][entry['run']]);recipe['r1_arm']='PRESERVED_LEGACY_PLUS_R1_CAP010'
    start=done['update'];p.update(arms={name:recipe},continuation_source=done,sources={str(p['r1_seed']):done},chunk_start_update=start,chunk_updates=640,
        environment_steps_per_run=(start+640)*262144,chunk=chunk,r1_preserved_legacy=True,namespace_id='A3SR-R1-PRESERVED-LEGACY-V3',worker_module=MODULE,
        reward_version='legacy5-cap060-plus-new-R1-shared-cap010-v3',r1_authorization=str((OUT/'AMENDMENT05.json').relative_to(ROOT)),
        pbt_lineage=dict(source=done,Actor_Critic_Adam_env_aux_RNG_ring_exact=True,reward_transition='existing episodes use old ledger until natural reset; new episodes use versioned reward'))
    p['source_sha256']=dict(p['source_sha256'],**ready['source_sha256'])
    assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items())
    atomic_json(BASE/'protocols'/f'{name}.json',p)
    atomic_json(OUT.parent/'handoffs'/f'{name}.json',dict(status='BOUNDARY_HANDOFF',previous_run=entry['run'],successor=name,slot=3,source=done,unix=time.time()))
    atomic_json(OUT/'HANDOFF.json',dict(status='READY_FOR_SMOKE',previous=entry['run'],successor=name,source=done,no_worker_signals=True,unix=time.time()))
    atomic_json(BASE/'formal'/entry['run']/'REWARD_REVISION_TERMINAL.json',dict(status='ORIGINAL_RECIPE_FIXED640_COMPLETE_NOT_CORRECTED_ARM',successor=name,source=done))
    pin([str((BASE/'formal'/name/f'u{u:06d}').relative_to(ROOT)) for u in range(start+40,start+641,40)],'approved preserved-legacy revision')
    return dict(entry,run=name,chunk=chunk,source=done,r1_preserved_legacy=True,r1_arm=recipe['r1_arm'],supersedes=entry['run'])


def fallback(slot,entry,chunk):
    if entry.get('r1_smoke_fallback'):
        prior=BASE/'formal'/entry['run'];done=read(prior/'COMPLETE.json',{})
        if not done:return None
        p=read(BASE/'protocols'/f'{entry['run']}.json');name=f'bt_slot{slot:02d}_c{chunk:06d}';recipe=p['arms'][entry['run']]
        p.update(arms={name:recipe},continuation_source=done,sources={str(p['round14_collector_seed']):done},chunk_start_update=done['update'],chunk_updates=320,environment_steps_per_run=(done['update']+320)*262144,chunk=chunk,round14_rekey=False)
        atomic_json(BASE/'protocols'/f'{name}.json',p)
        return dict(entry,run=name,chunk=chunk,source=done)
    if not entry.get('r1_items_applied'):return None
    previous=BASE/'formal'/entry['run']
    if not read(previous/'TECHNICAL_FAILURE.json') or read(previous/'RESTORE.json',{}).get('passed'):return None
    # Explicit fallback authorization: ordinary old-rule A3SR, never run the
    # failing new-rule state under incompatible old physical definitions.
    source=read(OUT.parent.parent/'PROTOCOL.json')['source'];donor=Path(source['checkpoint']).parent.name
    p=read(BASE/'protocols'/f'{donor}.json');recipe=dict(p['arms'][donor]);name=f'bt_slot{slot:02d}_c{chunk:06d}'
    for key in ('round14_authorization','round15_authorization','round16_authorization','a3sr_promotion_authorization'):
        if key in p:p['ancestor_'+key]=p.pop(key)
    seed=read(OUT.parent.parent/'PROTOCOL.json')['seeds'][str(slot)]
    p.update(arms={name:recipe},namespace_id='R1-TECHNICAL-FALLBACK-A3SR',sources={str(seed):source},continuation_source=source,
        chunk_start_update=source['update'],chunk_updates=320,environment_steps_per_run=(source['update']+320)*262144,
        slot=slot,chunk=chunk,round14_collector_seed=seed,round14_rekey=True,worker_module='scripts.bomb_threat_a3sr_resume_train',
        r1_fallback_authorization='.hermes/STEER_r1_confound_20261030_1440.md',pbt_lineage=dict(source=source,rng='independent target seed after donor next-step check',fallback=True))
    p['source_sha256']['scripts/bomb_threat_a3sr_resume_train.py']=sha(ROOT/'scripts/bomb_threat_a3sr_resume_train.py')
    assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items())
    atomic_json(BASE/'protocols'/f'{name}.json',p)
    atomic_json(previous/'EXCLUDED_TECHNICAL.json',dict(status='EXCLUDED_NEVER_RESUME',reason='authorized next-smoke failure fallback',successor=name,unix=time.time()))
    atomic_json(OUT/'fallbacks'/f'{name}.json',dict(status='A3SR_FALLBACK_READY',previous=entry['run'],run=name,source=source,updates=320,not_corrected_R1_evidence=True,unix=time.time()))
    return dict(entry,run=name,chunk=chunk,source=source,r1_smoke_fallback=True,r1_items_applied=False,r1_preserved_legacy=False,role='AUTHORIZED_A3SR_FALLBACK')


def refresh_ready():
    wiring=read(OUT/'WIRING.json',{})
    if wiring.get('status')!='PASS':
        atomic_json(OUT/'EXECUTION_STATUS.json',dict(status='CPU_WIRING_PENDING_CURRENT_GPU_CHUNK_CONTINUES',instruction='.hermes/STEER_r1_confound_20261030_1440.md',boundary_update=2960,training_worker_signals=0,unix=time.time()))
        return
    log=OUT/'FOCUSED_FINAL.log'
    if not log.exists() or '[100%]' not in log.read_text() or 'FAILED' in log.read_text():return
    target=OUT/'READY.json'
    if not target.exists():
        seal={f:sha(ROOT/f) for f in FILES}
        atomic_json(target,dict(status='PASS',source_sha256=seal,wiring_sha256=sha(OUT/'WIRING.json'),test_log_sha256=sha(log),tests=18,
            gates=['legacy settlement exact','new additions cap010','void and no ledger crossover','actual native growth reward projection','old episode env/reward nextstep exact'],
            next='at original slot03 u2960 COMPLETE atomically fork namespace; smoke2 fresh restore then640 updates',unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('R1_CONFOUND_READY','slot03保留旧系数版CPU门槛PASS：18项测试、双座位真实奖励接线及旧中局next-step等价；已排原正式chunk u2960后接续，训练不中断。')
    done=read(OUT/'HANDOFF.json',{})
    atomic_json(OUT/'EXECUTION_STATUS.json',dict(status='HANDED_OFF' if done else 'READY_WAIT_CURRENT_FORMAL_ATOMIC_END',handoff=done,boundary_update=2960,ready_sha256=sha(target),unix=time.time()))
