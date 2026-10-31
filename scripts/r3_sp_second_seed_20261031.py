"""Consume the conditional u4000 handoff only at a complete R3 chunk."""
import copy
import json
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha

OUT=Path('runs/r3_selfplay_ramp_20261031')
R3=Path('runs/r3_items_route_20261030')
EVAL=R3/'evaluation_ownside_v1'
NS=OUT/'seed02'


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def scores(doc):
    modes=doc['metrics']['holdout'];out={}
    assert set(modes)=={'greedy','categorical'}
    for seat in ('P0','P1'):
        assert all(modes[m][seat]['games']==64 for m in modes)
        out[seat]=sum(modes[m][seat]['win_minus_loss'] for m in modes)
    return out


def eligible_arms(baselines,observations):
    return [arm for arm in ('a','b') if all(
        max(scores(doc)[seat] for doc in observations[arm])<=baselines[arm][seat]
        for seat in ('P0','P1'))]


def freeze():
    path=OUT/'SECOND_SEED_PROTOCOL_1610.json'
    if path.exists():return read(path)
    baseline={};files={}
    for arm in ('a','b'):
        values=[]
        for u in range(3440,3761,80):
            f=EVAL/f'r3_{arm}_ownside_c000001_u{u:06d}_holdout/COMPLETE.json'
            d=read(f);assert d and d['status']=='COMPLETE'
            values.append(scores(d));files[str(f)]=sha(f)
        baseline[arm]={seat:max(v[seat] for v in values) for seat in ('P0','P1')}
    p=dict(status='FROZEN',authorization='user 2026-10-31 16:10 steering',baseline_highs=baseline,
           baseline_files_sha256=files,required_updates=[3840,3920,4000],
           plateau='no new high in either seat aggregate across the two separately preserved action modes; both modes64 games per seat',
           selection='among plateau arms, weakest u4000 aggregate P1; tie a; retain other original R3 as control',
           boundary='complete frozen640-update chunk only, never mid-update signal',seed=2026323002,
           source='same frozen R3-a u3040 as SP1; retain Actor/Critic/Adam/ring, rebuild natural env and RNG with independent seed',
           new_updates=640,new_environment_steps=167772160,namespace=str(NS),
           resource='reuse yielded R3 worker lane; current a/b share GPU2, therefore SP2 also shares with remaining R3, never claim whole-card release',
           guards='BT15%, R2/R3 5%, three disjoint5-update windows; same SP1 protocol; no new reward or opponent mix',unix=time.time())
    atomic_json(path,p);return p


def status(value):
    atomic_json(OUT/'SECOND_SEED_CONDITIONAL_1610.json',dict(value,unix=time.time()))


def second_seed_metadata(original, seed, namespace, entry):
    value=copy.deepcopy(original)
    value.update(seed=seed,experiment_id='R3-SP-SEED02-20261031',namespace=str(namespace))
    value['resource']=dict(value['resource'],gpu=entry['gpu'],training_cpu_affinity=entry['cpu_affinity'])
    return value


def advance():
    # Called by the existing CPU service; it never becomes a second GPU owner.
    try:
        guard=lock(OUT/'SECOND_SEED_DISPATCH.lock')
    except BlockingIOError:return
    if (OUT/'SECOND_SEED_BLOCKED.json').exists():return
    p=freeze()
    from scripts import r3_selfplay_owner_20261031 as owner
    if owner.paused():status(dict(status='USER_PAUSED'));return
    if (NS/'REGISTRATION.json').exists():
        if not list(owner.processes('scripts.r3_sp_seed2_owner_20261031')):
            with (NS/'WATCHDOG.log').open('ab') as log:
                subprocess.Popen(['.venv/bin/python','-m','scripts.r3_sp_seed2_owner_20261031','--watchdog'],
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        status(dict(status='SECOND_SEED_OWNER_ACTIVE_CHECK_FORMAL_RECEIPT',namespace=str(NS)));return
    reclaim=read(OUT/'RECLAIM_1820_AUTHORIZATION.json',{})
    override=reclaim.get('override_wait_u4000_holdout') is True
    observations={};files={};missing=[]
    for arm in ('a','b'):
        observations[arm]=[]
        for u in p['required_updates']:
            matches=list(EVAL.glob(f'r3_{arm}_*_u{u:06d}_holdout/COMPLETE.json'))
            if not matches:missing.append([arm,u]);continue
            assert len(matches)==1,'ambiguous ownside holdout identity'
            f=matches[0];d=read(f);assert d['status']=='COMPLETE';scores(d)
            observations[arm].append(d);files[str(f)]=sha(f)
    if missing and not override:status(dict(status='WAIT_U4000_COMPLETE_HOLDOUT',missing=missing,original_R3_not_paused=True));return
    arms=['a','b'] if override else eligible_arms(p['baseline_highs'],observations)
    if not arms:status(dict(status='NO_YIELD_NEW_HOLDOUT_HIGH',files=files));return
    decision=read(OUT/'SECOND_SEED_CAPACITY_DECISION.json')
    slots=read(R3/'R3_SLOTS.json')
    if not decision:
        arm='a' if override else min(arms,key=lambda arm:(scores(observations[arm][-1])['P1'],arm))
        entry=slots[arm];protocol=read(entry['protocol'])
        decision=dict(status='WAIT_COMPLETE_CHUNK',arm=arm,entry=entry,
                      target_update=protocol['chunk_start_update']+protocol['chunk_updates'],
                      evidence_files_sha256=files,baseline_highs=p['baseline_highs'],unix=time.time())
        # Existing owner consumes this durable gate before creating a successor.
        path=R3/'boundary_blocks'/f'r3_{arm}_ownside.json'
        existing=read(path)
        assert override or not existing or existing.get('status')!='BLOCK','preexisting safety block requires separate review'
        if not override:atomic_json(path,dict(status='BLOCK',reason='USER_AUTHORIZED_SP_SECOND_SEED_CAPACITY',
                             target_update=decision['target_update'],decision=str(OUT/'SECOND_SEED_CAPACITY_DECISION.json'),unix=time.time()))
        atomic_json(OUT/'SECOND_SEED_CAPACITY_DECISION.json',decision)
        message=('R3-sp第二seed：18:20授权已覆盖等待u4000 holdout，R3-a/b冻结u4002，GPU2原生辅助评估安全让卡后由新seed独占。' if override else
                 'R3-sp第二seed：u4000完整holdout无新高条件满足，请求较弱原R3线完整chunk后让出worker槽，保留另一对照；共GPU2。')
        owner.notify('SECOND_SEED_YIELD_REGISTERED',message,decision=str(OUT/'SECOND_SEED_CAPACITY_DECISION.json'))
    # If the owner crossed its prior boundary just before consuming our gate,
    # bind the already running successor and wait for its complete budget.
    current=read(R3/'R3_SLOTS.json')[decision['arm']]
    if current['run']!=decision['entry']['run']:
        protocol=read(current['protocol'])
        decision.update(entry=current,target_update=protocol['chunk_start_update']+protocol['chunk_updates'],
                        boundary_race_adopted_without_worker_signal=True,unix=time.time())
        atomic_json(OUT/'SECOND_SEED_CAPACITY_DECISION.json',decision)
    entry=decision['entry'];folder=R3/'formal'/entry['run'];done=read(folder/'COMPLETE.json')
    if not done or list(owner.processes(entry.get('worker_module','scripts.bun_spawn_side_train'),entry['run'])):
        status(dict(status='WAIT_COMPLETE_CHUNK_WORKER_EXIT',decision=decision));return
    assert done['update']==decision['target_update']
    cp=Path(done['checkpoint'])
    assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    assert (folder/f"NEXT_STEP_u{done['update']:06d}.json").exists()
    if (OUT/'RESOURCE_PAUSE.json').exists() and not override:status(dict(status='WAIT_SP1_RESOURCE_OR_BOT_BLOCK_RESOLUTION',decision=decision));return
    gpu=2 if override else entry['gpu']
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in inventory.splitlines()}
    if free[gpu]<48*1024:status(dict(status='WAIT_YIELDED_GPU_CAPACITY',gpu=gpu,free_MiB=free[gpu],decision=decision));return
    known=0 if override else sum(len(list(owner.processes(e.get('worker_module','scripts.bun_spawn_side_train'),e['run'])))
              for a,e in slots.items() if a!=decision['arm'] and e['gpu']==gpu)
    if owner.compute_app_count(gpu)>known:
        status(dict(status='WAIT_UNKNOWN_COMPUTE_APP_NO_SIGNAL',gpu=gpu,decision=decision));return
    # New seed namespace retains the common clean parent, never the weaker donor.
    registration=copy.deepcopy(read(OUT/'REGISTRATION.json'));oldp=read(registration['entry']['protocol'])
    name='r3_sp55_seed2_c000000';recipe=copy.deepcopy(next(iter(oldp['arms'].values())))
    oldp.update(namespace=str(NS/'formal'),namespace_id='R3-SP-SEED02-20261031',arms={name:recipe},
                r1_seed=p['seed'],round14_collector_seed=p['seed'],chunk_start_update=3040,chunk_updates=640,
                environment_steps_per_run=3680*262144)
    for key in ('r1_coefficient_request','r1_item_coefficient_request','r3_item_coefficient_request'):
        source=Path(oldp[key]);target=NS/'controls'/source.name
        if source.exists():atomic_json(target,read(source))
        oldp[key]=str(target)
    oldp['source_sha256']['scripts/r3_sp_seed2_train_20261031.py']=sha(Path('scripts/r3_sp_seed2_train_20261031.py'))
    target=NS/'protocols'/f'{name}.json';atomic_json(target,oldp)
    for filename in ('PROTOCOL.json','PARENT_GATE.json','CPU_RUNTIME_ACCEPTANCE.json','FIXED_HISTORY_EVALUATION_V1.json'):
        value=read(OUT/filename)
        if filename=='PROTOCOL.json':value=second_seed_metadata(value,p['seed'],NS,dict(entry,gpu=gpu,cpu_affinity='176-183' if override else entry['cpu_affinity']))
        atomic_json(NS/filename,value)
    registration.update(entry=dict(run=name,protocol=str(target),gpu=gpu,cpu_affinity='176-183' if override else entry['cpu_affinity'],seed=p['seed']),
                        status='REGISTERED_SECOND_SEED_NOT_YET_UPDATED',protocol_sha256=sha(target),exclusive_gpu=override,
                        placement_authorization='SECOND_SEED_PROTOCOL_1610.json',unix=time.time())
    atomic_json(NS/'RESOURCE_GUARD.json',dict(baseline=owner.reference_snapshot(),strikes={},last_update={},unix=time.time()))
    atomic_json(NS/'REGISTRATION.json',registration)
    atomic_json(OUT/'SECOND_SEED_HANDOFF.json',dict(status='YIELDED_COMPLETE_CHUNK_SHA_PASS_SECOND_SEED_REGISTERED',
                  donor_terminal=done,gpu=gpu,exclusive_gpu=override,other_original_R3_retained=True,
                  training_workers_signalled=False,protocol_sha256=sha(target),namespace=str(NS),unix=time.time()))
    with Path('docs/training_experiment_ledger.md').open('a') as ledger:
        ledger.write('\nR3-SP第二seed2026323002条件接续：`runs/r3_selfplay_ramp_20261031/SECOND_SEED_HANDOFF.json`；同u3040干净父/Actor-Critic-Adam-ring，独立seed自然重建env/RNG，640更新，另一原R3对照保留。让出原R3 worker槽，GPU2仍与剩余原R3共卡，不称独占。完整smoke/fresh restore和首正式另验。\n')
    status(dict(status='SECOND_SEED_REGISTERED_WATCHDOG_STARTS_NEXT_TICK',namespace=str(NS)))


def safe_advance():
    try:advance()
    except Exception as error:
        atomic_json(OUT/'SECOND_SEED_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()))
        from scripts import r3_selfplay_owner_20261031 as owner
        owner.notify('SECOND_SEED_TECHNICAL_BLOCKED','R3-sp第二seed条件编排技术阻塞，收据SECOND_SEED_BLOCKED.json；第一seed及原训练未发信号。')
