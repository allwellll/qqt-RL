"""Consumed by the existing sole owner; no separate training controller."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def switch_request(slot,entry):
    decision=read(BASE/'iter2/SWITCH_DECISION.json',{})
    if decision.get('status')!='AUTHORIZED_READY' or str(slot) not in decision.get('slots',{}):return None
    if entry.get('iter2_arm') or not entry.get('run'):return None
    return dict(status='ITER2_AUTHORIZED_SWITCH_AT_ATOMIC_40',slot=slot,decision='runs/bomb_threat_v1/iter2/SWITCH_DECISION.json',
        target_arm=decision['slots'][str(slot)],old_run=entry['run'])


def make_protocol(slot,chunk,source,recipe,length=320):
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json');name=f'bt_slot{slot:02d}_c{chunk:06d}'
    path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        existing=read(path)
        assert existing['arms'][name]==recipe and existing.get('continuation_source')==source,'refusing existing protocol mutation'
        return path
    start=source['update'] if source else 0
    policy_source=recipe.get('fallback_policy_source',config['policy_source'])
    hashes=dict(config['source_sha256'])
    module='scripts.bomb_threat_iter2_train'
    if recipe.get('iter2_fallback'):
        module='scripts.bomb_threat_iter2_fallback_train'
        hashes['scripts/bomb_threat_iter2_fallback_train.py']=sha(ROOT/'scripts/bomb_threat_iter2_fallback_train.py')
    activation=read(BASE/'iter2/DIAGNOSTICS_ACTIVATION.json',{})
    if activation.get('status')=='READY_AT_NEXT_CHUNK':
        proof=ROOT/activation['parity_receipt']
        assert sha(proof)==activation['parity_sha256'] and read(proof)['status']=='PASS'
        module=module.replace('_train','_diagnostics_train')
        for file in ('scripts/bomb_threat_passive_diagnostics.py','scripts/bomb_threat_collector_diagnostics_v1.py',module.replace('.','/')+'.py'):
            assert sha(ROOT/file)==activation['source_sha256'][file]
            hashes[file]=activation['source_sha256'][file]
    if recipe.get('entropy_fixed_005'):
        authorization=read(BASE/'iter2/STEER1320_AUTHORIZATION.json',{})
        assert authorization.get('status')=='AUTHORIZED_READY' and slot==7 and recipe['iter2_arm']=='A2'
        assert source, 'entropy restoration requires full current state'
        module='scripts.bomb_threat_iter2_fixed_entropy_train'
        hashes['scripts/bomb_threat_iter2_fixed_entropy_train.py']=sha(ROOT/'scripts/bomb_threat_iter2_fixed_entropy_train.py')
    if recipe.get('threat_multiplier')==2.5:
        assert slot==1 and source and recipe['iter2_arm']=='A3'
        from scripts.bomb_threat_decision2040 import OUT
        assert read(OUT/'SLOT01_THREAT_REDUCTION_REQUEST.json',{}).get('status')=='AUTHORIZED_TRIGGERED'
        module='scripts.bomb_threat_threat25_train'
        hashes['scripts/bomb_threat_threat25_train.py']=sha(ROOT/'scripts/bomb_threat_threat25_train.py')
    migration=bool(source and read(BASE/'protocols'/(Path(source['checkpoint']).parent.name+'.json'),{}).get('a3s_to_p0_amendment03'))
    if migration:
        module='scripts.bomb_threat_a3s_to_p0_train'
        for file in ('scripts/bomb_threat_a3s_to_p0_train.py','scripts/bomb_threat_a3s_to_p0_collector.py'):
            hashes[file]=sha(ROOT/file)
    value=dict(namespace='runs/bomb_threat_v1/formal',common=config['common'],arms={name:recipe},
        sources={str(2026112900+slot):source or policy_source},source_sha256=hashes,
        environment_steps_per_run=(start+length)*262144,chunk_start_update=start,chunk_updates=length,
        continuation_source=source,policy_source=policy_source,initialization_decision=config['initialization_decision'],
        worker_module=module,spawn_manifest=config['spawn_manifest'],
        A1_weights=config['A1_weights'],fixed_actor_source=config['fixed_actor_source'],
        rule_version='stock-window-v1',reward_version='bomb-threat-iter2-'+recipe['iter2_arm'],
        training_reward_variant='bomb-threat-iter2-'+recipe['iter2_arm'],slot=slot,chunk=chunk,
        iter2_arm=recipe['iter2_arm'],initialization='policy-only plus new Critic/Adam/env/RNG with20update Critic warmup')
    if migration:value.update(a3s_to_p0_amendment03=True,initialization='full-state continuation; natural in-flight seat drain; new episodes P0')
    if activation.get('status')=='READY_AT_NEXT_CHUNK':
        value['passive_diagnostics_version']='masked-live-collector-v1'
        value['passive_diagnostics_authorization']='.hermes/STEER_20261029_1120.md'
    if recipe.get('iter2_fallback'):
        value.update(iter2_fallback=True,reward_version='bomb-threat-v1',training_reward_variant='bomb-threat-v1')
    if recipe.get('entropy_fixed_005'):
        value.update(entropy_restored_1320=True,initialization='full latest Actor/Critic/Adam/env/aux/RNG/ring continuation; fixed entropy0.005; no Critic warmup',
            entropy_restore_authorization='.hermes/STEER_20261029_1320.md')
    if path.exists():assert read(path)==value,'refusing existing protocol mutation'
    else:atomic_json(path,value)
    return path


def apply_entropy_restore(slot,entry,done,chunk):
    """13:20 decision supersedes generic A2 rollback; preserve complete state."""
    if slot!=7 or entry.get('iter2_arm')!='A2' or entry.get('entropy_restored_1320'):return None
    decision=read(BASE/'iter2/STEER1320_AUTHORIZATION.json',{})
    a2=decision.get('A2',{})
    if decision.get('status')!='AUTHORIZED_READY' or entry['run']!=a2.get('boundary_run'):return None
    out=BASE/'formal'/entry['run']
    complete=read(out/'COMPLETE.json',{})
    if not complete or done['update']!=a2['boundary_update'] or complete.get('state_sha256')!=done['state_sha256']:return None
    cp=ROOT/done['checkpoint']
    assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    old=read(BASE/'protocols'/f"{entry['run']}.json")['arms'][entry['run']]
    recipe=dict(old,entropy_fixed_005=True,entropy_restore_authorization='.hermes/STEER_20261029_1320.md')
    path=make_protocol(slot,chunk,done,recipe)
    atomic_json(out/'STEER1320_ENTROPY_HANDOFF.json',dict(status='FULL_STATE_ENTROPY_RESTORE_AT_CHUNK_BOUNDARY',
        source=done,successor=path.stem,only_changed='entropy schedule to fixed0.005',
        preserved=['Actor','Critic','Adam','env','aux','RNG','ring','counters'],
        training_worker_signals=0,critic_reinitialized=False,unix=time.time()))
    return dict(entry,run=path.stem,chunk=chunk,source=done,entropy_restored_1320=True,role='iter2_A2_fixed_entropy005',old_boundary=done['checkpoint'])


def round1_manages_slot(slot,entry):
    """Keep the 14:00 A2/A1 decisions ahead of older ITER2 rollback rules."""
    decision=read(BASE/'iter2/DECISION_ROUND1.json',{})
    if decision.get('status')!='AUTHORIZED':return False
    arm=entry.get('iter2_arm')
    return (slot==7 and arm=='A2') or (slot==4 and arm=='A1')


def round1_a1_eval_marker(checkpoint):
    """Return the immutable dev evaluation marker for a checkpoint path."""
    cp=Path(checkpoint)
    return BASE/'eval'/f'{cp.parent.name}_{cp.name}'/'NATURAL_COMPLETE.json'


def round1_a1_gate(checkpoint,evaluation,commit,checked_at_chunk):
    """Apply the user-defined A1 u960 dev threshold without reinterpreting it."""
    if not commit or evaluation.get('policy_sha256')!=commit.get('policy_sha256'):
        raise ValueError('A1 u960 dev evaluation does not match the frozen checkpoint')
    greedy=evaluation.get('natural_js_full2400',{}).get('greedy',{})
    if greedy.get('games')!=64:
        raise ValueError('A1 u960 dev greedy evaluation is incomplete')
    trigger=(greedy['bombs_per_game']<20 or greedy['zero_bomb_games']>=15)
    return dict(status='A3_SWITCH_REQUIRED' if trigger else 'CONTINUE_A1',checkpoint=checkpoint,
        policy_sha256=commit['policy_sha256'],greedy=greedy,trigger=trigger,
        rule='bombs_per_game<20 OR zero_bomb_games>=15',checked_at_chunk=checked_at_chunk,unix=time.time())


def round1_boundary_disposition(slot,gate,switch_created):
    """Resolve the round-1 boundary without falling through older rollback rules."""
    if switch_created:return 'SWITCH'
    if slot==7:return 'BLOCK'
    if slot==4 and gate.get('status')=='CONTINUE_A1':return 'CONTINUE'
    if slot==4:return 'WAIT'
    return 'UNMANAGED'


def _latest_a3_policy_source(target_slot):
    """Choose the newest committed Actor from active A3 slot01, without its state."""
    slots=read(BASE/'SLOTS.json',{})
    parent=slots.get('1',{})
    if parent.get('iter2_arm')!='A3':raise ValueError('A3 slot01 parent is not active')
    run=parent['run'];out=BASE/'formal'/run
    commits=sorted(out.glob('u*/COMMITTED.json'),key=lambda p:int(p.parent.name[1:]))
    if commits:
        cm=read(commits[-1],{})
    else:
        # The parent may already have advanced to a fresh continuation protocol
        # at the same owner tick. Its persisted source is the newest completed
        # atomic checkpoint until the successor commits its first update.
        source=parent.get('source')
        if not source:raise ValueError('A3 parent has no atomic checkpoint or continuation source')
        source_commit=ROOT/source['checkpoint']/'COMMITTED.json'
        cm=read(source_commit,{})
        if not cm or cm.get('checkpoint')!=source['checkpoint']:
            raise ValueError('A3 parent continuation source commit is missing or mismatched')
        if cm.get('policy_sha256')!=source.get('policy_sha256') or cm.get('update')!=source.get('update'):
            raise ValueError('A3 parent continuation source commit metadata mismatch')
        run=Path(source['checkpoint']).parent.name
    cp=ROOT/cm['checkpoint']
    if not cm.get('passed') or sha(cp/'policy.pt')!=cm['policy_sha256']:
        raise ValueError('A3 parent policy SHA failed atomic verification')
    source={k:cm[k] for k in ('checkpoint','policy_sha256','update')}
    lineage=dict(parent_run=run,parent_slot=1,parent_seed=2026112901,
        parent_checkpoint=cm['checkpoint'],parent_update=cm['update'],
        parent_policy_sha256=cm['policy_sha256'],parent_state_sha256=cm.get('state_sha256'),
        target_slot=target_slot,target_seed=int(read(BASE/'PROTOCOL.json')['seeds'][str(target_slot)]),
        copied=['Actor policy only'],
        reinitialized=['Critic','Adam','environment','auxiliary state','RNG','combat ring'],
        critic_warmup_updates=20)
    return source,lineage


def apply_round1_a3_seed(slot,entry,done,chunk):
    """At an atomic chunk boundary fork a fresh A3 seed from current slot01 Actor."""
    if not round1_manages_slot(slot,entry):return None
    old_run=entry['run'];old_out=BASE/'formal'/old_run
    old_protocol=read(BASE/'protocols'/f'{old_run}.json',{})
    complete=read(old_out/'COMPLETE.json',{})
    required=int(old_protocol.get('chunk_start_update',0))+int(old_protocol.get('chunk_updates',0))
    if not complete or done.get('update')!=required or complete.get('state_sha256')!=done.get('state_sha256'):
        return None
    cp=ROOT/done['checkpoint']
    if sha(cp/'state.pkl')!=done['state_sha256'] or sha(cp/'policy.pt')!=done['policy_sha256']:
        raise ValueError('ITER2 round1 source chunk failed atomic SHA verification')

    reason='A2_STOP_TO_A3_SECOND_SEED' if slot==7 else 'A1_U960_GATE_TO_A3_THIRD_SEED'
    if slot==4:
        gate=read(BASE/'iter2/ROUND1_A1_U960_DECISION.json',{})
        if gate.get('status')=='CONTINUE_A1':return None
        target=gate.get('checkpoint','runs/bomb_threat_v1/formal/bt_slot04_c000015/u000960')
        marker=round1_a1_eval_marker(target)
        if not marker.exists():
            atomic_json(BASE/'iter2/ROUND1_A1_U960_DECISION.json',dict(status='WAITING_FOR_U960_DEV_EVAL',
                checkpoint=target,checked_at_chunk=old_run,last_atomic=done,unix=time.time()))
            return None
        evaluation=read(marker,{})
        target_commit=ROOT/target/'COMMITTED.json'
        cm=read(target_commit,{})
        gate=round1_a1_gate(target,evaluation,cm,old_run)
        atomic_json(BASE/'iter2/ROUND1_A1_U960_DECISION.json',gate)
        trigger=gate['trigger']
        if not trigger:return None
        reason='A1_U960_GATE_TO_A3_THIRD_SEED'

    source,lineage=_latest_a3_policy_source(slot)
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json',{})
    recipe=dict(config['arms']['A3'],fallback_policy_source=source)
    target_chunk=max(chunk, int(chunk))
    path=make_protocol(slot,target_chunk,None,recipe,length=int(config.get('chunk_updates',320)))
    proto=read(path,{})
    lineage.update(target_run=path.stem,decision='.hermes/DECISION_iter2_round1_20261029_1400.md',reason=reason)
    proto.update(pbt_parent=source['checkpoint'],pbt_decision='.hermes/DECISION_iter2_round1_20261029_1400.md',
        pbt_lineage=lineage,round1_role=reason,supersedes_run=old_run)
    atomic_json(path,proto)
    atomic_json(old_out/'ITER2_ROUND1_SUPERSEDED.json',dict(status=reason,last_atomic=done,
        successor=path.stem,pbt_parent=source['checkpoint'],pbt_lineage=lineage,
        formal_updates_discarded=0,training_worker_signals=0,unix=time.time()))
    new_entry=dict(run=path.stem,chunk=target_chunk,gpu=slot,source=source,
        role='iter2_A3_second_seed' if slot==7 else 'iter2_A3_third_seed',iter2_arm='A3',
        old_boundary=done['checkpoint'],pbt_parent=source['checkpoint'],pbt_lineage=lineage,
        pbt_decision='.hermes/DECISION_iter2_round1_20261029_1400.md',round1_decision=True,
        supersedes=old_run)
    out=BASE/'formal'/path.stem;out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'PBT_HANDOFF.json',dict(status='PENDING_FRESH_INIT',lineage=lineage,
        source_policy_sha256=source['policy_sha256'],critic_reinitialized=True,adam_reinitialized=True,
        environment_rng_reinitialized=True,critic_warmup_updates=20,unix=time.time()))
    receipt=dict(status='A3_SEED_PROTOCOL_READY',decision='.hermes/DECISION_iter2_round1_20261029_1400.md',
        source_run=old_run,last_atomic=done,successor=path.stem,pbt_lineage=lineage,
        protocol=str(path.relative_to(ROOT)),protocol_sha256=sha(path),formal_updates_discarded=0,
        training_worker_signals=0,unix=time.time())
    atomic_json(BASE/'iter2/ROUND1_LAST_HANDOFF.json',receipt)
    return new_entry


def rollback_recipe(recipe,source):
    """The task authorizes original settings, not another tuning experiment."""
    return dict(recipe,curriculum_fraction=.25,opponent_weights=[.2,.25,.2,.15,.1,.1],
        entropy_coefficient=.005,iter2_fallback=True,
        fallback_policy_source={k:source[k] for k in ('checkpoint','policy_sha256','update')})


def apply_rollback(slot,entry,done,chunk):
    """Only called with an absent worker and an atomically completed chunk."""
    if entry.get('iter2_arm')=='A3S':return None  # own frozen640 seat verdict
    if slot==1 and (BASE/'a3s/extension1280/MONITOR_PROTOCOL.json').exists():return None  # latest20:40 threat-only intervention supersedes broad recipe rollback
    proto=read(BASE/'protocols'/f"{entry['run']}.json",{})
    recipe=proto.get('arms',{}).get(entry['run'],{})
    if not entry.get('iter2_arm') or recipe.get('iter2_fallback') or recipe.get('entropy_fixed_005'):return None
    decision1320=read(BASE/'iter2/STEER1320_AUTHORIZATION.json',{})
    if slot==7 and decision1320.get('status')=='AUTHORIZED_READY':return None
    from scripts.bomb_threat_iter2_analysis import refresh
    basis=refresh()
    decision=basis.get('rollback_decisions',{}).get(str(slot),{})
    if decision.get('status')!='RESTORE_ORIGINAL_RECIPE':return None
    early=decision.get('reason')=='STEER1120_A3_EARLY'
    if done['update']<640 and not (slot==1 and done['update']>=320 and early):return None
    cp=ROOT/done['checkpoint']
    assert sha(cp/'policy.pt')==done['policy_sha256']
    assert sha(cp/'state.pkl')==done['state_sha256']
    restored=rollback_recipe(recipe,done)
    path=make_protocol(slot,chunk,None,restored)
    receipt=dict(status='AUTHORIZED_ORIGINAL_RECIPE_RESTORED',decision=decision,
        basis='runs/bomb_threat_v1/iter2/DEV_BUCKETS_AND_ROLLBACK.json',
        basis_sha256=sha(BASE/'iter2/DEV_BUCKETS_AND_ROLLBACK.json'),last_atomic=done,
        successor=path.stem,worker_absent=True,training_worker_signals=0,
        initialization='latest committed Actor, new Critic/Adam/env/RNG plus20-update warmup',unix=time.time())
    atomic_json(BASE/'formal'/entry['run']/'ITER2_RECIPE_ROLLBACK.json',receipt)
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=restored['fallback_policy_source'],
        role='iter2_original_recipe_fallback',iter2_arm=entry['iter2_arm'],iter2_fallback=True,
        old_boundary=done['checkpoint'])


def apply_switch(slot,entry,chunk):
    request=switch_request(slot,entry)
    if not request:return None
    old=BASE/'formal'/entry['run'];commits=sorted(old.glob('u*/COMMITTED.json'))
    if not commits:return None
    cm=read(commits[-1]);assert cm['update']%40==0
    assert sha(commits[-1].parent/'policy.pt')==cm['policy_sha256']
    assert sha(commits[-1].parent/'state.pkl')==cm['state_sha256']
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json');arm=request['target_arm'];recipe=config['arms'][arm]
    path=make_protocol(slot,chunk,None,recipe)
    atomic_json(old/'ITER2_SUPERSEDED_ATOMIC.json',dict(status='USER_AUTHORIZED_RECIPE_SWITCH',last_atomic=cm,
        successor=path.stem,old_worker_absent=True,training_worker_signals=0,source_actor_is_original_u960=True,
        not_resuming_old_slot_params=True,unix=time.time()))
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=config['policy_source'],role='iter2_'+arm,
        iter2_arm=arm,old_boundary=cm['checkpoint'],switch_request=request)


def formal_update_recorded(update,stage,protocol):
    return stage=='formal' and update>protocol.get('chunk_start_update',0)+2


def report(slots,force=False):
    """Two-hour durable report and first formal-update notification."""
    from scripts.iter_pbt_owner import notify
    collapse=BASE/'a3s/amendment03/GUARD_STATUS.json'
    if (BASE/'a3s/amendment03/PROTOCOL.json').exists() and (not collapse.exists() or time.time()-collapse.stat().st_mtime>60):
        from scripts.bomb_threat_slot04_guard2140 import refresh as refresh_collapse
        refresh_collapse()
    monitor=BASE/'a3s/extension1280/SLOT01_RED_OBSERVATION.json'
    if (BASE/'a3s/extension1280/MONITOR_PROTOCOL.json').exists() and (not monitor.exists() or time.time()-monitor.stat().st_mtime>60):
        from scripts.bomb_threat_decision2040 import refresh as refresh2040
        refresh2040()
    if (BASE/'iter2/STEER1320_AUTHORIZATION.json').exists():
        receipt=BASE/'iter2/STEER1320_EXECUTION.json'
        if not receipt.exists() or time.time()-receipt.stat().st_mtime>60:
            from scripts.bomb_threat_steer1320 import refresh
            refresh()
    if (BASE/'iter2/amendment05_steer1120_diagnostics_a3.json').exists():
        receipt=BASE/'iter2/STEER1120_EXECUTION.json'
        if not receipt.exists() or time.time()-receipt.stat().st_mtime>60:
            from scripts.bomb_threat_steer1120_report import refresh as steer_refresh
            steer_refresh()
    analysis_path=BASE/'iter2/DEV_BUCKETS_AND_ROLLBACK.json'
    if not analysis_path.exists() or time.time()-analysis_path.stat().st_mtime>60:
        from scripts.bomb_threat_iter2_analysis import refresh
        refresh()
    analysis=read(analysis_path,{})
    active=[]
    for slot,entry in slots.items():
        if not entry.get('iter2_arm'):continue
        out=BASE/'formal'/entry['run'];progress=read(out/'PROGRESS.json',{})
        warm=read(out/'WARMUP_PROGRESS.json',{});launch=read(out/'LAUNCH.json',{})
        record=dict(slot=slot,run=entry['run'],arm=entry['iter2_arm'],stage=launch.get('stage'),
            fallback=bool(entry.get('iter2_fallback')),
            formal_update=progress.get('iterations',0),warm_updates=len(warm.get('rows',[])),
            init=read(out/'INIT.json',{}),warmup_gate=read(out/'WARMUP_GATE.json',{}),last=progress.get('last',{}))
        active.append(record)
        protocol=read(BASE/'protocols'/f"{entry['run']}.json",{})
        if formal_update_recorded(record['formal_update'],launch.get('stage'),protocol) and record['last'].get('finite') and all(record['last'].get('update_l2',{}).get(k,0)>0 for k in ('actor','critic')):
            last=record['last'];initialization=('完整state/Adam/env/RNG续训、固定熵0.005，无新增Critic预热' if entry.get('entropy_restored_1320') else 'Critic预热与恢复通过')
            notify(entry['run']+'_FIRST_FORMAL_UPDATE',
                f'ITER2 {entry["iter2_arm"]} slot{slot} 已正式u{record["formal_update"]}，Actor/Critic非零、finite，{262144/last["cycle_seconds"]:.0f} env/s；{initialization}。',
                run=entry['run'],update=record['formal_update'],environment_steps=last['environment_steps'])
    atomic_json(BASE/'iter2/TRAINING_STATUS.json',dict(status='RUNNING' if active else 'WAITING_ATOMIC_SWITCH',arms=active,unix=time.time()))
    status=read(BASE/'iter2/ROUND_REPORT_STATUS.json',{})
    request=read(BASE/'iter2/FORCE_REPORT_REQUEST.json',{})
    force=force or request.get('unix',0)>status.get('unix',0)
    if not force and time.time()<status.get('next_due_unix',0):return
    if (BASE/'fresh256_slot01_u800/gatefix_1945/TASK_CONSUMED.json').exists():
        from scripts.bomb_threat_fresh256_observation import main as observe
        observe()
    n=int(status.get('round',1))+1
    # Dedicated decision report owns round5; periodic generic report starts6.
    if n==5 and (BASE/'iter2/STEER1320_AUTHORIZATION.json').exists():n=6
    while (ROOT/f'reports/bomb_threat_v1_iter2_round{n}_20261029.md').exists():n+=1
    path=ROOT/f'reports/bomb_threat_v1_iter2_round{n}_20261029.md'
    text=[f'# ITER2 第{n}轮（20261029）','',
        '起点阻塞已由10:00决定解决：u960 Actor保留，新Critic/Adam/env/RNG，不恢复已授权删除的历史state。u960 holdout16/46/2、胜减负+14、零泡14，未触发改用u360。原始state审计收据保留但不再是阻塞。',
        '', '三arm共享policy SHA `77a06184ab5e704f021cf750f9c834e2b2e0374549abcc4347fea27b4ef31a4c`；slot04 A1分布、slot07 A2熵退火、slot01 A3三项×5/cap.60。20次Critic预热仅A1基础奖励分布；Actor逐位冻结、Adam Actor counter=0。预热门槛EV≥.3或预声明末5步平台；未通过就明确BLOCKED，不悄延长。u0 Actor lr0，u1=目标/40，u40到原lr。',
        '', '|slot/arm|run|阶段|Critic预热|正式u|finite|Actor/Critic L2|env/s|', '|---|---|---|---:|---:|---|---|---:|']
    for r in active:
        v=r['last'];delta=v.get('update_l2',{})
        label=r['arm']+('/原配方回退' if r['fallback'] else '')
        text.append(f'|{r["slot"]}/{label}|{r["run"]}|{r["stage"]}|{r["warm_updates"]}|{r["formal_update"]}|{v.get("finite")}|{delta.get("actor")}/{delta.get("critic")}|{262144/v["cycle_seconds"] if v else 0:.1f}|')
    text+=['','首640正式更新为两轮比较点，各arm167,772,160正式on-policy环境tick，另列5,242,880预热tick；每40保存/评argmax、categorical及τ.5。320更新chunk由同一owner接续，用户暂停优先。旧slot在40更新已提交边界退出，训练worker信号0；slot00/03/05/06不变。',
        '', 'D1/D2矩阵仍异步：`runs/bomb_threat_v1/iter2/D12_PROGRESS.json`和各条件COMPLETE；固定slot06为u2880，禁止把最新或单seed最高分当最好。D3仅JAX hunter tier3，与JS环境/PRNG/对手实现混杂，不能单因素归因。诊断详细表暂见round1（10:00起点决定覆盖其中旧起点阻塞文字）。',
        '', '信号审计32/32完整、真实账本守恒；该rule3小样本无撤销/拒绝暴露，不能证明其他分布无异常。A3实际支付/净奖励<30%且低于终局/击杀/交付channel[2,9,61]，每40更新复核；影子按类归因不等于原生独立支付。未修改网页、BEST、奖励终局项及旧运行路径。',
        '', '200更新分桶见下表；不存在的数据不填0。640更新比较使用dev u440–640（含u640），与训练墙钟范围内slot00/03等槽加权均值比较，各至少3点。三项严格低于才在下一chunk原子终态恢复原配方；缺评估只记录PENDING并续320更新，不为报告停健康训练。回退另建namespace，保留当前Actor，按10:00约定新Critic/Adam/env/RNG与20步预热，不冒称恢复旧完整state。复现入口 `.venv/bin/python -m scripts.bomb_threat_iter2_train <protocol> <run> <seed>`，GPU由现有owner分配，不能手动重复启动。',
        '', '|arm/200更新桶|已评点|胜-负均值|击杀均值|泡/局均值|零泡均值|','|---|---:|---:|---:|---:|---:|']
    for b in analysis.get('buckets',[]):
        m=b['means'];text.append(f'|{b["arm"]}/{b["formal_update_range"]}|{b["points"]}|{m["win_minus_loss"]:.3f}|{m["causal_kills"]:.3f}|{m["bombs_per_game"]:.3f}|{m["zero_bomb_games"]:.3f}|')
    if not analysis.get('buckets'):text.append('|待首40更新完整dev评估|0|—|—|—|—|')
    text+=['','640更新状态：`'+json.dumps(analysis.get('rollback_decisions',{}),ensure_ascii=False)+'`。',
        '', 'D1/D2/D3完整异步结果与限制：`reports/bomb_threat_v1_iter2_diagnostics_20261029.md`；表格中τ.5独立行仅诊断，不作argmax闸门。']
    if (BASE/'fresh256_slot01_u800/gatefix_1945/TASK_CONSUMED.json').exists():
        observation=read(BASE/'fresh256_slot01_u800/RECENT6_AND_SLOT04.json',{})
        rate=observation.get('slot01_kill_efficiency',{})
        text+=['','19:45修订：暂不PBT复制，slot00/03保留对照；A3S固定640严格分seat dev/holdout判定。',
            'slot01击杀/百泡（真实放置数分母）与00/03匹配对照：'+rate.get('status','PENDING')+'。只观察，连续3点低于70%上报，无自动回退。明细见`reports/bomb_threat_v1_iter2_round8_fresh256_20261029.md`。']
        text+=['','|slot01 checkpoint|击杀/真实放泡|击杀/百泡|00|03|合并对照|低于70%|','|---|---|---:|---:|---:|---:|---|']
        for r in rate.get('points',[])[-6:]:
            text.append(f"|{r['checkpoint']}|{r['causal_kills']}/{r['actual_bombs']}|{r['kills_per100_actual_bombs']}|{r['controls_per100'].get('0')}|{r['controls_per100'].get('3')}|{r['pooled_control_per100']}|{r['below70percent']}|")
    path.write_text('\n'.join(text)+'\n');atomic_json(BASE/'iter2/ROUND_REPORT_STATUS.json',dict(round=n,report=str(path.relative_to(ROOT)),report_sha256=sha(path),next_due_unix=time.time()+7200,unix=time.time()))
    notify('ITER2_ROUND'+str(n),'ITER2阶段报告已更新；起点按10:00新Critic预热协议执行，训练/CPU诊断异步继续。',report=str(path.relative_to(ROOT)))
