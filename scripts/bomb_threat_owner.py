"""Unique BOMB-THREAT owner, atomic old-chunk handoff and asynchronous eval."""
import os,json,time,sys,subprocess
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
from scripts.iter_pbt_owner import read,resources,CPU,notify
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';OLD=ROOT/'runs/iter_pbt_v1'
GPUS=[0,1,3,4,5,6,7]
PARENT_GPUS={0,1,3,4,7}


def causal_lineage_limits(run_name,base=None):
    """Map the current branch's run names to each fork's maximum source update."""
    base=BASE if base is None else base
    result={};name=run_name;limit=float('inf')
    while name and name not in result:
        result[name]=limit
        proto=read(base/'protocols'/(name+'.json'),{})
        sources=proto.get('sources',{})
        if not sources:break
        source=next(iter(sources.values()));cp=Path(source['checkpoint'])
        if cp.parent.name==name:break
        name=cp.parent.name;limit=int(source.get('update',cp.name[1:]))
    return result


def pbt_source_metadata(source,recipe,slot,target_seed):
    """Bind a PBT fork to the exact source run's rule/reward semantics."""
    if not recipe.get('zero_stock_shaping'):
        return dict(pbt_rekey_source=False,zero_shaping_weights=[])
    from scripts.bomb_threat_pbt import (NATIVE_ONLY_ZERO_STOCK_SHAPING,
        ZERO_SHAPING_WEIGHTS,training_reward_variant)
    run=Path(source['checkpoint']).parent.name
    source_path=BASE/'protocols'/(run+'.json')
    source_protocol=read(source_path,{})
    if not source_protocol:
        raise ValueError('PBT source must have a local frozen protocol for reward-version validation')
    config=read(BASE/'PROTOCOL.json')
    source_variant=training_reward_variant(source_protocol,run)
    if source_protocol.get('rule_version')!=config['rule_version'] or source_protocol.get('reward_version')!=config['reward_version']:
        raise ValueError('PBT parent rule/reward version differs from the current frozen campaign')
    if tuple(recipe.get('zero_shaping_weights',()))!=ZERO_SHAPING_WEIGHTS or \
       any(recipe.get(k)!=0 for k in ('threat_weight','topology_weight','chain_weight')):
        raise ValueError('PBT fork must explicitly set threat/topology/chain shaping weights to zero')
    source_seed=int(next(iter(source_protocol.get('sources',{}))))
    target_variant=NATIVE_ONLY_ZERO_STOCK_SHAPING
    return dict(source_run_protocol=str(source_path.relative_to(ROOT)),
        source_protocol_sha256=sha(source_path),source_reward_version=source_protocol['reward_version'],
        source_rule_version=source_protocol['rule_version'],source_training_reward_variant=source_variant,
        training_reward_variant=target_variant,pbt_rekey_source=source_variant!=target_variant,
        zero_shaping_weights=list(ZERO_SHAPING_WEIGHTS),pbt_parent=source['checkpoint'],
        pbt_decision=recipe.get('pbt_decision','runs/bomb_threat_v1/PBT_COPY_DECISION_20261029_0700.json'),
        pbt_lineage=dict(parent_run=run,parent_checkpoint=source['checkpoint'],parent_update=source['update'],
            parent_seed=source_seed,target_slot=slot,target_seed=target_seed,
            source_state_sha256=source['state_sha256'],source_policy_sha256=source['policy_sha256'],
            source_training_reward_variant=source_variant,target_training_reward_variant=target_variant,
            copied=['actor','critic','adam','environment','auxiliary_state_except_zeroed_credit_accumulators','combat_ring','combat_ring_cursor'],
            rng='rekeyed with target slot seed' if source_variant!=target_variant else 'continued unchanged from existing zero-shaping lineage'))

def worker_source_files(config,is_pbt):
    """Return the exact worker ABI inputs for both formal and prewarm runs."""
    worker=(['scripts/bomb_threat_pbt_train.py','scripts/bomb_threat_pbt.py']
        if is_pbt else ['scripts/bomb_threat_long_train.py'])
    return config['source_files']+worker+['scripts/bomb_threat_schedule.py']

def workers():
    found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if not any(m in a for m in (b'scripts.bomb_threat_reboot_worker',b'scripts.bomb_threat_train',b'scripts.bomb_threat_long_train',b'scripts.bomb_threat_pbt_train',b'scripts.bomb_threat_iter2_train',b'scripts.bomb_threat_iter2_fallback_train',b'scripts.bomb_threat_iter2_diagnostics_train',b'scripts.bomb_threat_iter2_fallback_diagnostics_train',b'scripts.bomb_threat_iter2_fixed_entropy_train',b'scripts.bomb_threat_seat5050_train',b'scripts.bomb_threat_threat25_train',b'scripts.bomb_threat_a3s_to_p0_train',b'scripts.bomb_threat_pbt2350_train',b'scripts.bomb_threat_a3sr_train',b'scripts.bomb_threat_a3sr_resume_train',b'scripts.bomb_threat_r1_train',b'scripts.bomb_threat_r1_progress_train',b'scripts.bomb_threat_r1_preserved_train',b'scripts.bomb_threat_r1_continue_train',b'scripts.bomb_threat_slot01_copy07_train')) or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            protocol=next(x.decode() for x in a if x.startswith(b'runs/bomb_threat_v1/protocols/'))
            name=Path(protocol).stem
            if name in found:
                # A fork-before-exec helper can expose the same argv briefly;
                # the durable per-run worker.lock remains authoritative.
                continue
            found[name]=int(p.name)
        except (OSError,StopIteration):pass
    return found

def make_protocol(g,chunk,source,recipe,remaining_updates=None):
    if recipe.get('iter2_arm')=='A3SR':
        if source and source['update']>=640:
            from scripts.bomb_threat_round14 import make_protocol as continuation
            return continuation(g,chunk,source,recipe,320 if remaining_updates is None else remaining_updates)
        from scripts.bomb_threat_a3sr_schedule import make_protocol as a3sr_protocol
        return a3sr_protocol(g,chunk,source,recipe,320 if remaining_updates is None else remaining_updates)
    if source:
        from scripts.bomb_threat_pbt2350 import continue_protocol as pbt2350_continue
        continued=pbt2350_continue(g,chunk,source,recipe,320 if remaining_updates is None else remaining_updates)
        if continued:return continued
    if recipe.get('iter2_arm')=='A3S':
        from scripts.bomb_threat_seat5050_schedule import make_protocol as seat_protocol
        return seat_protocol(g,chunk,source,recipe,320 if remaining_updates is None else remaining_updates)
    if recipe.get('iter2_arm'):
        from scripts.bomb_threat_iter2_schedule import make_protocol as iter2_protocol
        return iter2_protocol(g,chunk,source,recipe,320 if remaining_updates is None else remaining_updates)
    name=f'bt_slot{g:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        existing=read(path);previous=next(iter(existing['sources'].values()))
        if (previous['checkpoint'],previous['state_sha256']) != (source['checkpoint'],source['state_sha256']):
            raise ValueError('Refusing to reuse a protocol with different source ancestry: '+name)
        if existing['arms'][name]!=recipe:raise ValueError('Refusing recipe overwrite: '+name)
        return path
    config=read(BASE/'PROTOCOL.json');seed=config['seeds'][str(g)];start=source['update']
    from scripts.bomb_threat_schedule import chunk_updates
    length=chunk_updates(config) if remaining_updates is None else int(remaining_updates)
    if length<=0:raise ValueError('Recovery must preserve positive remaining budget')
    is_pbt=bool(recipe.get('zero_stock_shaping'))
    files=worker_source_files(config,is_pbt)
    source_reward=pbt_source_metadata(source,recipe,g,config['seeds'][str(g)])
    value=dict(namespace='runs/bomb_threat_v1/formal',common=config['common'],arms={name:recipe},
        sources={str(seed):source},source_sha256={f:sha(ROOT/f) for f in files},
        environment_steps_per_run=(start+length)*262144,chunk_start_update=start,chunk_updates=length,
        preserve_source_environment=True,worker_module='scripts.bomb_threat_pbt_train' if is_pbt else 'scripts.bomb_threat_long_train',
        prewarm_acceptance=(f'runs/bomb_threat_v1/prewarm_pbt/slot{g:02d}_c{chunk:06d}/ACCEPTANCE.json' if is_pbt else f'runs/bomb_threat_v1/prewarm/slot{g:02d}_v1/ACCEPTANCE.json'),
        spawn_manifest=config['training_spawn_manifest'],warm_baseline_env_per_second=20255.36,
        fixed_actor_source=read(ROOT/'runs/combat_mix_v1/PROTOCOL.json')['sources']['2026103051'],
        rule_version=config['rule_version'],reward_version=config['reward_version'],slot=g,chunk=chunk,
        **source_reward)
    atomic_json(path,value);return path

def invalid_ancestry(run_name):
    """Return whether a run's source chain contains an explicitly excluded run."""
    seen=set(); name=run_name
    while name and name not in seen:
        seen.add(name)
        out=BASE/'formal'/name
        if read(out/'INVALID_STARTING_POINT.json') or read(out/'EXCLUDED_INVALID.json'):
            return True
        proto=read(BASE/'protocols'/(name+'.json'),{})
        sources=proto.get('sources',{})
        if not sources: return False
        cp=next(iter(sources.values())).get('checkpoint','')
        parts=Path(cp).parts
        try: i=parts.index('formal')
        except ValueError: return False
        if i+1>=len(parts): return False
        parent=parts[i+1]
        if parent==name: return False
        name=parent
    return False

def shaping_pause_request(g):
    from scripts.bomb_threat_iter2_schedule import switch_request
    entry=read(BASE/'SLOTS.json',{}).get(str(g),{})
    pending=switch_request(g,entry) if (BASE/'iter2/SWITCH_DECISION.json').exists() else None
    if pending:return pending
    if entry.get('iter2_arm'):
        # Old branch's shaping alarms cannot stop a newly authorized arm.
        # ITER2's own640-update/relative-control and cap monitors are separate.
        return None
    # Workers call this helper at each saved boundary too.  The owner's
    # in-memory override alone is insufficient: a newly authorized native-only
    # fork must not be stopped by the old shaped lineage's persisted alarm.
    entry=read(BASE/'SLOTS.json',{}).get(str(g),{})
    if entry.get('authorized_shaping_override'):
        proto=read(BASE/'protocols'/f"{entry.get('run','')}.json",{})
        recipe=proto.get('arms',{}).get(entry.get('run'),{})
        decision_path=entry.get('authorized_shaping_override_basis')
        decision=read(ROOT/decision_path,{}) if decision_path else {}
        if (decision.get('status')=='AUTHORIZED' and str(g) in decision.get('slots',{})
                and recipe.get('zero_stock_shaping') is True
                and all(recipe.get(k)==0 for k in ('threat_weight','topology_weight','chain_weight'))):
            return None
    for path in sorted((BASE/'shaping_pause_requests').glob(f'slot{g:02d}_*.json')):
        value=read(path,{})
        if value.get('status')=='SHAPING_PAUSE_REQUIRED' and value.get('slot')==g:
            return value
    return None


def active_shaping_pause(g,entry):
    """Only an explicit, persisted decision may override a shaping pause."""
    return None if entry.get('authorized_shaping_override') else shaping_pause_request(g)

def next_chunk(g):
    vals=[]
    for p in (BASE/'protocols').glob(f'bt_slot{g:02d}_c*.json'):
        try: vals.append(int(p.stem.rsplit('_c',1)[1]))
        except (IndexError,ValueError): pass
    return max(vals,default=-1)+1

def prewarm_running(g,warm_name):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if b'scripts.bomb_threat_prewarm' in a and any(x.endswith(f'/{warm_name}.json'.encode()) for x in a):
                if (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return True
        except OSError:pass
    return False


def prepare_prewarm(g,entry,children):
    formal=read(BASE/'protocols'/f"{entry['run']}.json");recipe=formal['arms'][entry['run']]
    is_pbt=bool(recipe.get('zero_stock_shaping'))
    current=BASE/'formal'/entry['run'];commits=sorted(current.glob('u*/COMMITTED.json'))
    source=read(commits[-1]) if commits else entry.get('source')
    if not source:return
    index=source['update']
    warm_chunk=next_chunk(g) if commits else entry.get('chunk',next_chunk(g))
    warm_name=f'slot{g:02d}_c{warm_chunk:06d}' if is_pbt else f'slot{g:02d}_v1'
    out=(BASE/'prewarm_pbt' if is_pbt else BASE/'prewarm')/warm_name
    if any((out/f).exists() for f in ('ACCEPTANCE.json','FAILURE.json')) or prewarm_running(g,warm_name):return
    if not (ROOT/source['checkpoint']/f'../NEXT_STEP_u{index:06d}.json').exists():return
    config=read(BASE/'PROTOCOL.json');seed=config['seeds'][str(g)]
    files=worker_source_files(config,is_pbt)
    warm_pbt=pbt_source_metadata(source,recipe,g,seed)
    warm_proto=dict(namespace=str((BASE/'prewarm_pbt' if is_pbt else BASE/'prewarm').relative_to(ROOT)),
        common=config['common'],arms={warm_name:recipe},sources={str(seed):source},
        source_sha256={f:sha(ROOT/f) for f in files},environment_steps_per_run=(index+2)*262144,
        chunk_start_update=index,chunk_updates=2,preserve_source_environment=True,
        worker_module='scripts.bomb_threat_pbt_train' if is_pbt else 'scripts.bomb_threat_long_train',
        spawn_manifest=config['training_spawn_manifest'],warm_baseline_env_per_second=20255.36,
        fixed_actor_source=formal['fixed_actor_source'],rule_version='stock-window-v1',
        reward_version=formal.get('reward_version','bomb-threat-v1'),slot=g,**warm_pbt)
    path=BASE/'prewarm_protocols'/f'{warm_name}.json'
    if not path.exists():atomic_json(path,warm_proto)
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),JAXBOMB_DEVICE_OVERRIDE=str(g),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # Prewarm uses the idle training slot's CPU lane; it is gone before formal launch.
    affinity={0:'48-51',1:'52-55',3:'72-79',4:'80-87',5:'64-67',6:'68-71',7:'176-179'}[g]
    stream=(BASE/'logs'/f'prewarm_{warm_name}.log').open('ab',buffering=0)
    p=subprocess.Popen(['taskset','-c',affinity,sys.executable,'-m','scripts.bomb_threat_prewarm',str(path.relative_to(ROOT))],
        env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    children['prewarm_'+warm_name]=(p,stream)
    atomic_json(out/'LAUNCH.json',dict(status='RUNNING',gpu=g,cpu_affinity=affinity,
        overlapping_training_run=entry['run'],source_checkpoint=source['checkpoint'],formal=False,unix=time.time()))


def retention_recovery_source(protocol,out,root=None):
    """Recover current work from its kept state, without the removed old source.

    A new namespace avoids rewriting frozen worker code/protocols. Total budget,
    Adam, RNG, environment and ring remain at the latest committed point.
    """
    root=ROOT if root is None else Path(root)
    sources=protocol.get('sources',{})
    if not sources:return None
    old_source_exists=all((root/s['checkpoint']/'state.pkl').exists() for s in sources.values())
    commits=sorted(out.glob('u*/COMMITTED.json'))
    if not commits:
        if old_source_exists:return None
        raise ValueError('Missing retained state for interrupted current run')
    cp=commits[-1].parent;cm=read(commits[-1],{})
    source_update=max(int(s.get('update',Path(s['checkpoint']).name[1:])) for s in sources.values())
    real=cm.get('metrics',{}).get('finite') and all(cm.get('metrics',{}).get('update_l2',{}).get(k,0)>0 for k in ('actor','critic'))
    # Rebase an interrupted formal run even if its old source happens to still
    # exist: the periodic cleaner may delete that source during worker import.
    # Fresh/smoke/restore stages have not yet passed a real formal commit.
    if old_source_exists and (not real or cm['update']<=source_update):return None
    if not cm.get('passed') or sha(cp/'state.pkl')!=cm['state_sha256']:
        raise ValueError('Retained recovery state failed atomic SHA verification')
    from scripts.checkpoint_policy_artifact import verify_policy
    verify_policy(cp)
    if not (out/f"NEXT_STEP_u{cm['update']:06d}.json").exists():
        raise ValueError('Retained state has no completed next-step parity token')
    remaining=protocol['environment_steps_per_run']//262144-cm['update']
    if remaining<=0:raise ValueError('Budget already finished without COMPLETE receipt')
    return cm,remaining


def main():
    guard=lock(BASE/'OWNER.lock');slots=read(BASE/'SLOTS.json',{});children={}
    while True:
        live=workers();samples=resources();free={x['gpu'] for x in samples if not x['occupied']}
        from scripts.r1_shared_dispatch import shared_free_slots
        free |= shared_free_slots(slots, ROOT/'runs/bun_native_v2_20261030/R2_SLOTS.json')
        paused=any((b/f).exists() for b in (BASE,OLD) for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))
        gate=read(OLD/'BOMB_THREAT_HANDOFF.json',{})
        slots_dirty=False
        for g in GPUS:
            if g not in gate.get('armed_slots',[]):continue
            entry=slots.get(str(g))
            from scripts.slot01_copy07_handoff_20261031 import apply_switch as copy07_switch
            replacement=copy07_switch(g,entry,live,next_chunk(g)) if g==1 else None
            if replacement:
                slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                continue
            from scripts.bomb_threat_r1_stop2255 import consume as consume_r1_stop2255
            if consume_r1_stop2255(g,entry,live):continue
            if not paused and entry:
                from scripts.bomb_threat_reboot_recovery import dispatch as reboot_dispatch
                if reboot_dispatch(g,entry,children,live,free,CPU[g]):continue
            if entry and invalid_ancestry(entry.get('run','')):
                entry['invalid_starting_point']=True
                entry['invalid_reason']='source ancestry contains an excluded latest-derived run'
                slots_dirty=True
            if entry and entry.get('invalid_starting_point') and entry['run'] not in live:
                retired=BASE/'formal'/entry['run'];retired.mkdir(parents=True,exist_ok=True)
                commits=sorted(retired.glob('u*/COMMITTED.json'))
                atomic_json(retired/'EXCLUDED_INVALID.json',dict(
                    reason=entry['invalid_reason'],never_resume=True,
                    last_atomic=read(commits[-1]) if commits else None,
                    worker_absent=True,unix=time.time()))
                slots.pop(str(g),None);entry=None;slots_dirty=True
            if not entry:
                # DECISION_bomb_threat_v1_init hard contract: five parent
                # lanes and two u1120 control lanes; never use D/latest PBT.
                if g in PARENT_GPUS:
                    pp=ROOT/'runs/authorized_campaign_20261026/scale_env_batch_pilot_v1/formal/env2048_mb8192_s2026103051/u000160'
                else:
                    pp=OLD/'formal/slot01_c000024/u001120'
                cm=read(pp/'COMMITTED.json')
                if not cm or not cm.get('passed'):continue
                assert sha(pp/'state.pkl')==cm['state_sha256']
                source=cm
                recipe=dict(lr=7.5e-5,entropy_coefficient=.005,combat_shaping=0.,curriculum_fraction=.25,opponent_weights=[.2,.25,.2,.15,.1,.1])
                chunk=next_chunk(g)
                path=make_protocol(g,chunk,source,recipe)
                entry=slots[str(g)]=dict(run=path.stem,chunk=chunk,gpu=g,source=source,
                    old_boundary=source['checkpoint'],role='mandated_parent' if g in PARENT_GPUS else 'mandated_u1120')
                atomic_json(BASE/'SLOTS.json',slots)
                out=BASE/'formal'/path.stem;out.mkdir(parents=True,exist_ok=True)
                atomic_json(out/'BOMB_THREAT_HANDOFF_RECEIPT.json',dict(successor=path.stem,atomic_state_sha256=source['state_sha256'],worker_exit_before_launch=True,unix=time.time()))
            name=entry['run'];out=BASE/'formal'/name
            if (BASE/'iter2/SWITCH_DECISION.json').exists():
                from scripts.bomb_threat_iter2_schedule import switch_request
                switch=switch_request(g,entry)
                if switch and entry.get('authorized_shaping_override'):
                    # Existing workers retain their imported helper code. Its
                    # persisted override must be withdrawn so the already-
                    # supported40-update request can actually be consumed.
                    entry['authorized_shaping_override']=False
                    entry['iter2_pending_switch']=switch
                    atomic_json(BASE/'SLOTS.json',slots)
            if entry.get('invalid_starting_point') and not read(out/'COMPLETE.json'):
                continue
            if name in live:
                if not paused and not entry.get('iter2_arm'):prepare_prewarm(g,entry,children)
                continue
            if not paused:
                if entry.get('r1_smoke_fallback'):
                    from scripts.bomb_threat_r1_confound_schedule import fallback
                    replacement=fallback(g,entry,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        continue
                from scripts.bomb_threat_r1_confound_schedule import apply_switch as confound_switch
                replacement=confound_switch(g,entry,next_chunk(g))
                if replacement:
                    slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                    notify(name+'_PRESERVED_LEGACY','slot03已完成原正式chunk；完整state接续新版旧奖励保留+独立新项CAP.10，先smoke和fresh restore。')
                    continue
                from scripts.bomb_threat_r1_items_schedule import apply_switch as item_switch
                replacement=item_switch(g,entry,next_chunk(g))
                if replacement:
                    slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                    notify(name+'_R1_ITEMS_BOUNDARY','新版R1原生道具规则在安全边界接管；独立namespace，先smoke和fresh恢复，旧对照不变。')
                    continue
            if not paused:
                from scripts.bomb_threat_a3sr_schedule import apply_switch as a3sr_switch
                replacement=a3sr_switch(g,entry,next_chunk(g))
                if replacement:
                    slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                    notify(name+'_A3SR_SWITCH','发布验收完成；slot05旧线已原子停止，开始A3SR新namespace，20次Critic预热、2更新smoke和完整恢复，固定640正式更新。')
                    continue
            if not paused and (BASE/'iter2/SWITCH_DECISION.json').exists():
                from scripts.bomb_threat_iter2_schedule import apply_switch
                replacement=apply_switch(g,entry,next_chunk(g))
                if replacement:
                    entry=slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                    name=entry['run'];out=BASE/'formal'/name;out.mkdir(parents=True,exist_ok=True)
                    notify(name+'_ITER2_SWITCH',f'ITER2 {entry["iter2_arm"]} 接管slot{g:02d}，旧worker已在40更新原子边界退出；开始新Critic预热，未向训练发信号。')
            shaping_pause=active_shaping_pause(g,entry)
            if shaping_pause:
                commits=sorted(out.glob('u*/COMMITTED.json'))
                receipt=BASE/'shaping_pause_requests'/f'slot{g:02d}_ATOMIC_PAUSED.json'
                if not receipt.exists():
                    atomic_json(receipt,dict(status='PAUSED_AT_ATOMIC_CHUNK_BOUNDARY',run=name,
                        last_atomic=read(commits[-1]) if commits else None,request=shaping_pause,
                        worker_absent=True,coefficients_unchanged=True,awaiting_orchestrator=True,unix=time.time()))
                    notify(name+'_SHAPING_PAUSED',f'BOMB-THREAT {name} 连续两点评估放泡上升而击杀/胜局不升；已在原子chunk边界阻止下一塑形chunk，等待编排。')
                continue
            failure=read(out/'TECHNICAL_FAILURE.json')
            if failure and entry.get('r1_items_applied') and not entry.get('r1_same_version_continuation'):
                from scripts.bomb_threat_r1_confound_schedule import fallback
                replacement=fallback(g,entry,next_chunk(g))
                if replacement:
                    slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                    notify(name+'_SMOKE_FALLBACK','R1新smoke再失败，封存失败namespace；按授权退回冻结共同父A3SR原配方320更新，修复另行版本化。')
                    continue
            if failure:
                # A smoke OOM is a transient resource collision, not a rule
                # failure. Preserve the log and retry once at the same atomic
                # source when the physical slot is actually free.
                logtxt=(BASE/'logs').glob(f'{name}_*.log')
                oom=any('RESOURCE_EXHAUSTED' in p.read_text(errors='replace') for p in logtxt)
                retry=read(out/'RETRY_SMOKE.json')
                retry_attempt=read(out/'RETRY_SMOKE_ATTEMPT.json')
                if oom and not retry:
                    atomic_json(out/'RETRY_SMOKE.json',dict(reason='transient GPU collision during smoke',original_failure=failure,unix=time.time()))
                    failure=None
                elif oom and retry and not retry_attempt and not read(out/'SMOKE.json'):
                    failure=None
                else: continue
            if read(out/'COMPLETE.json'):
                done=read(out/'COMPLETE.json')
                from scripts.bomb_threat_r1_schedule import apply_switch as r1_switch, manages as r1_manages
                if paused and r1_manages(g,entry):continue
                from scripts.bomb_threat_r1_schedule import revision_hold, TARGETS as r1_targets
                if g in r1_targets and revision_hold() and not entry.get('r1_items_applied'):
                    atomic_json(out/'ITEM_REVISION_STAGE_BLOCKED.json',dict(status='HOLD_AT_COMPLETE_BOUNDARY',last_atomic=done,unix=time.time()))
                    continue
                if not paused:
                    replacement=r1_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_R1_BOUNDARY','R1在完整chunk原子保存退出后接管；共同slot05冻结完整state，独立RNG，640更新；先技术smoke和fresh恢复。')
                        continue
                    if r1_manages(g,entry):
                        from scripts.bomb_threat_r1_continue_schedule import apply_continue
                        replacement=apply_continue(g,entry,done,next_chunk(g))
                        if replacement:
                            slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                            notify(name+'_R1_CONTINUE_1805',f'R1 slot{g:02d} 从u{done["update"]}完整state同配置续640更新；RNG/Adam/env/aux/ring不重置，先smoke与fresh恢复；持久停止优先。')
                        continue
                from scripts.bomb_threat_a3sr_promotion import apply_switch as promotion_switch, manages as promotion_manages
                if not paused:
                    replacement=promotion_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_A3SR_PROMOTED','用户授权A3SR推广：旧chunk完整提交后全state接管，smoke/独立恢复后正式训练；评估不阻塞。')
                        continue
                    if promotion_manages(g,entry):continue
                from scripts.bomb_threat_round16 import apply_switch as round16_switch, terminal as round16_terminal
                if not paused:
                    replacement=round16_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_ROUND16_SWITCH','round16 GPU5从自身u1280完整state续训固定320更新；配方/RNG不变。')
                        continue
                if round16_terminal(entry,done):continue
                from scripts.bomb_threat_round15 import apply_switch as round15_switch, terminal as round15_terminal
                if not paused:
                    replacement=round15_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_ROUND15_SWITCH',f'round15 slot07完整chunk边界复制已验收A3SR donor u{replacement["source"]["update"]}；旧state保留，独立RNG，固定320更新。')
                        continue
                    round15_auth=read(BASE/'iter2/a3sr/round15/AUTHORIZATION.json',{})
                    if g==7 and name==round15_auth.get('slot07_boundary_run'):
                        # Wait at the committed boundary for an admitted donor;
                        # never continue the retired branch or weaken admission.
                        continue
                if round15_terminal(entry,done):continue
                from scripts.bomb_threat_round14 import apply_switch as round14_switch,terminal as round14_terminal
                if not paused:
                    auth=read(BASE/'iter2/a3sr/round14/AUTHORIZATION.json',{})
                    missing_donor=(g==7 and auth.get('status')=='AUTHORIZED' and
                        name==auth.get('slot07_boundary_run') and
                        not (ROOT/auth['donor']['checkpoint']/'state.pkl').exists())
                    if missing_donor:
                        blocked=BASE/'iter2/a3sr/round14/SLOT07_COPY_BLOCKED.json'
                        if not blocked.exists():
                            atomic_json(blocked,dict(status='BLOCKED',required_source=auth['donor']['checkpoint'],
                                required_state_sha256=auth['donor']['state_sha256'],old_last_atomic=done,
                                reason='concurrent retention deleted donor after slot05 accepted source; no exact replica verified',
                                substitute_checkpoint_used=False,training_worker_signals=0,unix=time.time()))
                            notify('ROUND14_COPY_BLOCKED','slot07精确复制阻塞：u640 donor state被并发旧清理器删除；不擅用u642，旧线/其余训练继续，报告已落盘。')
                    replacement=None if missing_donor else round14_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_ROUND14_SWITCH',f'round14 slot{g:02d}完整原子边界接管A3SR u640，旧state保留；完整状态恢复至1280，配方不变。')
                        continue
                if round14_terminal(entry,done):continue
                from scripts.bomb_threat_a3sr_schedule import terminal as a3sr_terminal
                if not entry.get('round14_applied') and a3sr_terminal(entry,done):continue
                if not paused:
                    from scripts.bomb_threat_pbt2350 import apply_switch as pbt2350_switch
                    replacement=pbt2350_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_PBT2350_SWITCH','slot04完整chunk已原子结束，按23:50决定复制slot07 c23最新完整state；独立slot04 RNG，无Critic预热，开始smoke/恢复/正式续训。')
                        continue
                    if (BASE/'a3s/amendment03/PROTOCOL.json').exists():
                        from scripts.bomb_threat_slot04_guard2140 import apply_switch as collapse_switch
                        replacement=collapse_switch(g,entry,done,next_chunk(g))
                        if replacement:
                            slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                            notify(name+'_AMENDMENT03_SWITCH','slot04授权连续3点护栏触发；完整chunk原子边界保留完整state/seed转P0 A3，原在途P1局自然终局后全部P0。')
                            continue
                    if (BASE/'a3s/extension1280/MONITOR_PROTOCOL.json').exists():
                        from scripts.bomb_threat_decision2040 import terminal_reversion
                        replacement=terminal_reversion(g,entry,done,next_chunk(g))
                        if replacement:
                            slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                            notify(name+'_A3S1280_FAIL',f'A3S slot{g:02d}固定1280严格判定FAIL；完整终态保留，按20:40决定切回原A3配方。')
                            continue
                    from scripts.bomb_threat_seat5050_schedule import apply_switch as seat_switch,terminal as seat_terminal
                    if seat_terminal(g,entry,done):continue
                    replacement=None if entry.get('role')=='A3S1280_FAIL_TO_ORIGINAL_A3' or entry.get('pbt2350_applied') or entry.get('iter2_arm')=='A3SR' else seat_switch(g,entry,done,next_chunk(g))
                    if replacement:
                        entry=slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_A3S_SWITCH',f'A3S slot{g:02d}在u{done["update"]}完整chunk原子边界接管；0更新丢弃，Actor来自slot01，20次Critic预热后seat50/50训练。',checkpoint=done)
                        continue
                if entry.get('invalid_starting_point'):
                    atomic_json(out/'INVALID_STARTING_POINT.json',dict(reason='DECISION requires parent/u1120; this provisional latest-start run is excluded',checkpoint=done.get('checkpoint'),state_sha256=done.get('state_sha256'),unix=time.time()))
                    # Retire this evidence-only protocol and allocate a fresh
                    # mandated-source protocol on the same physical slot.
                    slots.pop(str(g),None);atomic_json(BASE/'SLOTS.json',slots)
                    continue
                chunk_length=read(BASE/'protocols'/f'{name}.json').get('chunk_updates',80)
                notify(name+'_COMPLETE',f'BOMB-THREAT chunk终态 {name}；{chunk_length}次更新、{chunk_length*262144:,}步，原子保存，接续下一chunk。',checkpoint=done)
                recipe=read(BASE/'protocols'/f'{name}.json')['arms'][name]
                if (BASE/'a3s/extension1280/MONITOR_PROTOCOL.json').exists():
                    from scripts.bomb_threat_decision2040 import boundary_recipe
                    recipe=boundary_recipe(g,recipe,done)
                # The round-1 decision supersedes the prior A2 fixed-entropy
                # continuation.  Consume only after an atomic chunk; the
                # technical smoke/restore namespace is retained as evidence.
                round1_active=(entry.get('iter2_arm') in ('A1','A2') and
                    read(BASE/'iter2/DECISION_ROUND1.json',{}).get('status')=='AUTHORIZED')
                if round1_active:
                    from scripts.bomb_threat_iter2_schedule import apply_round1_a3_seed,round1_boundary_disposition
                    replacement=apply_round1_a3_seed(g,entry,done,next_chunk(g))
                    gate=read(BASE/'iter2/ROUND1_A1_U960_DECISION.json',{}) if g==4 else {}
                    disposition=round1_boundary_disposition(g,gate,bool(replacement))
                    if replacement:
                        entry=slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_ROUND1_A3_SWITCH',f'ITER2 round1 已在{done["update"]}原子边界将slot{g:02d}切换为A3新seed；Actor来自slot01最新原子点，Critic/Adam/env/RNG重置并预热20步。',checkpoint=done)
                        continue
                    if disposition in ('BLOCK','WAIT'):
                        atomic_json(BASE/'iter2'/f'ROUND1_SLOT{g:02d}_BOUNDARY_WAIT.json',dict(
                            status='WAITING_FOR_ROUND1_AUTHORIZED_SUCCESSOR',slot=g,run=name,
                            last_atomic=done,gate=gate,legacy_iter2_rollback_skipped=True,
                            reason='A2 must not launch another chunk' if g==7 else 'A1 may continue only after explicit CONTINUE_A1',
                            training_worker_signals=0,unix=time.time()))
                        continue
                if entry.get('iter2_arm') and entry.get('iter2_arm')!='A3SR' and not round1_active:
                    from scripts.bomb_threat_iter2_schedule import apply_entropy_restore
                    replacement=apply_entropy_restore(g,entry,done,next_chunk(g))
                    if replacement:
                        entry=slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_FIXED_ENTROPY1320',f'ITER2 slot07已在u{done["update"]}完整chunk原子终态停熵退火；保留完整状态，下一namespace固定熵0.005，依次smoke/restore/正式续训。')
                        continue
                    from scripts.bomb_threat_iter2_schedule import apply_rollback
                    replacement=apply_rollback(g,entry,done,next_chunk(g))
                    if replacement:
                        entry=slots[str(g)]=replacement;atomic_json(BASE/'SLOTS.json',slots)
                        notify(name+'_ITER2_ROLLBACK',f'ITER2 slot{g:02d} 满足授权dev回退条件，已在chunk原子终态恢复原配方；下一namespace从当前Actor加新Critic预热，不向训练发信号。')
                        continue
                override=read(out/'CONTINUATION_SOURCE_OVERRIDE.json',{})
                if override:
                    done=override['source']
                    assert sha(ROOT/done['checkpoint']/'state.pkl')==done['state_sha256']
                    recipe=override.get('recipe',recipe)
                chunk=next_chunk(g)
                path=make_protocol(g,chunk,done,recipe)
                entry.update(run=path.stem,chunk=chunk,source=done);atomic_json(BASE/'SLOTS.json',slots)
                name=entry['run'];out=BASE/'formal'/name
            path=BASE/'protocols'/f'{name}.json';out.mkdir(parents=True,exist_ok=True)
            if not paused:
                old_proto=read(path,{})
                recovery=None if entry.get('iter2_arm') else retention_recovery_source(old_proto,out)
                if recovery:
                    source,remaining=recovery;previous_name=name
                    path=make_protocol(g,next_chunk(g),source,old_proto['arms'][name],remaining_updates=remaining)
                    new_proto=read(path);new_proto['prewarm_acceptance']=old_proto['prewarm_acceptance']
                    new_proto['retention_recovery_of']=previous_name
                    atomic_json(path,new_proto)
                    atomic_json(out/'RETENTION_RECOVERY_HANDOFF.json',dict(status='CONTINUING_IN_NEW_NAMESPACE',
                        successor=path.stem,last_atomic=source,remaining_updates=remaining,
                        submitted_updates_not_repeated=True,unsaved_updates='unknown; at most save cadence',unix=time.time()))
                    entry.update(run=path.stem,chunk=new_proto['chunk'],source=source)
                    atomic_json(BASE/'SLOTS.json',slots)
                    name=entry['run'];out=BASE/'formal'/name;out.mkdir(parents=True,exist_ok=True)
            args=[str(path.relative_to(ROOT)),name,str(read(BASE/'PROTOCOL.json')['seeds'][str(g)])]
            proto=read(path,{});module=proto.get('worker_module','scripts.bomb_threat_train')
            if module=='scripts.bomb_threat_r1_train':
                from scripts.bomb_threat_r1_schedule import revision_hold
                if revision_hold():
                    atomic_json(out/'ITEM_REVISION_STAGE_BLOCKED.json',dict(status='HOLD',reason='new item progress specification; next stage not launched',unix=time.time()))
                    continue
            if proto.get('prewarm_acceptance'):
                warm=read(ROOT/proto['prewarm_acceptance'],{})
                if warm.get('status')!='PASS':
                    # A failed PBT prewarm is immutable evidence.  Retry once
                    # under a fresh protocol so the corrected handoff order
                    # and source-file hashes are consumed; never rerun the
                    # failed warm directory in place.
                    warm_failure=read(ROOT/proto['prewarm_acceptance'].replace('/ACCEPTANCE.json','/FAILURE.json'),{})
                    if warm_failure and int(entry.get('prewarm_retry_count',0))<1:
                        old_run=name; old_warm=proto['prewarm_acceptance']
                        retry_count=int(entry.get('prewarm_retry_count',0))+1
                        atomic_json(out/'PREWARM_RETRY_HANDOFF.json',dict(
                            status='REPLACED_BY_FRESH_PROTOCOL',old_run=old_run,
                            old_prewarm=old_warm,failure=warm_failure,
                            reason='donor parity was checked after target combat_mix remap',
                            corrected_worker='scripts.bomb_threat_pbt_train',
                            retry_count=retry_count,unix=time.time()))
                        recipe=proto['arms'][name]
                        chunk=next_chunk(g)
                        path=make_protocol(g,chunk,entry['source'],recipe)
                        entry.update(run=path.stem,chunk=chunk,prewarm_retry_count=retry_count,
                            retry_of=old_run)
                        slots[str(g)]=entry
                        atomic_json(BASE/'SLOTS.json',slots)
                        continue
                    if not paused:prepare_prewarm(g,entry,children)
                    continue
                from scripts.bomb_threat_schedule import prewarm_fingerprint
                if warm['fingerprint']!=prewarm_fingerprint(proto):
                    atomic_json(out/'TECHNICAL_FAILURE.json',dict(reason='prewarm fingerprint mismatch',unix=time.time()))
                    continue
                stage='formal'
            elif not (out/'SMOKE.json').exists():args+=['--smoke'];stage='smoke'
            elif not read(out/'SMOKE.json').get('passed'):continue
            elif not (out/'RESTORE.json').exists():args+=['--verify-restore'];stage='restore'
            else:
                stage='formal'
                if not gate.get('all_slots_released'):
                    gate.update(armed_slots=GPUS,all_slots_released=True,technical_smoke=name,restore_sha256=sha(out/'RESTORE.json'))
                    atomic_json(OLD/'BOMB_THREAT_HANDOFF.json',gate)
            if g==3 and not proto.get('r1_preserved_legacy') and stage=='formal':
                hold=read(BASE/'iter2/r1_attack_20261030/items_v2/conservation_v3/STAGE_HOLD.json',{})
                if hold.get('status')=='PREPARING' and time.time()-hold['created_unix']<hold['max_seconds']:
                    atomic_json(out/'CONFOUND_STAGE_HOLD.json',dict(status='WAITING_VERSIONED_REWARD_BEFORE_FORMAL',last_atomic=read(sorted(out.glob('u*/COMMITTED.json'))[-1]),unix=time.time()))
                    continue
            if paused or g not in free:continue
            log=(BASE/'logs'/f'{name}_{stage}.log');log.parent.mkdir(parents=True,exist_ok=True)
            if stage=='smoke' and read(out/'RETRY_SMOKE.json') and not read(out/'RETRY_SMOKE_ATTEMPT.json'):
                atomic_json(out/'RETRY_SMOKE_ATTEMPT.json',dict(retry_of=read(out/'TECHNICAL_FAILURE.json'),unix=time.time()))
            stream=log.open('ab',buffering=0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),JAXBOMB_DEVICE_OVERRIDE=str(g),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            if module in ('scripts.bomb_threat_r1_progress_train','scripts.bomb_threat_r1_preserved_train','scripts.bomb_threat_r1_continue_train'):
                # The versioned rule installer imports JAX before generated
                # worker main; backend/cache config must already be present.
                env.update(JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',JAXBOMB_RULE='bun',
                    JAX_COMPILATION_CACHE_DIR=str(BASE/'iter2/r1_attack_20261030/items_v2/cache'),
                    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0')
            p=subprocess.Popen(['taskset','-c',CPU[g],sys.executable,'-m',module]+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            children[name]=(p,stream)
            pbt_launch={}
            if proto.get('zero_shaping_weights'):
                pbt_launch=dict(pbt_parent=proto.get('pbt_parent'),pbt_decision=proto.get('pbt_decision'),
                    lineage=proto.get('pbt_lineage'),zero_shaping_weights=proto['zero_shaping_weights'],
                    source_reward_version=proto.get('source_reward_version'),target_reward_version=proto.get('reward_version'),
                    source_rule_version=proto.get('source_rule_version'),target_rule_version=proto.get('rule_version'),
                    source_training_reward_variant=proto.get('source_training_reward_variant'),
                    target_training_reward_variant=proto.get('training_reward_variant'),
                    source_protocol_sha256=proto.get('source_protocol_sha256'),
                    pbt_handoff_receipt='PBT_HANDOFF.json',prewarm_acceptance=proto.get('prewarm_acceptance'))
            atomic_json(out/'LAUNCH.json',dict(stage=stage,gpu=g,protocol=str(path.relative_to(ROOT)),unix=time.time(),**pbt_launch))
        for name,(p,stream) in list(children.items()):
            if p.poll() is None:continue
            stream.close();del children[name]
            if p.returncode and not name.startswith('prewarm_'):
                atomic_json(BASE/'formal'/name/'TECHNICAL_FAILURE.json',dict(exit_code=p.returncode,unix=time.time()))
                notify(name+'_FAILED',f'BOMB-THREAT技术阻塞 {name}，日志和完整原子点已保留；其他训练不受影响。')
        from scripts.bomb_threat_pbt2350 import refresh as pbt2350_acceptance
        pbt2350_acceptance()
        from scripts.bomb_threat_a3sr_schedule import refresh as a3sr_acceptance
        a3sr_acceptance()
        from scripts.bomb_threat_round14 import refresh as round14_acceptance
        round14_acceptance()
        from scripts.bomb_threat_a3sr_promotion import refresh as promotion_acceptance
        promotion_acceptance()
        from scripts.bomb_threat_r1_schedule import refresh as r1_acceptance
        r1_acceptance()
        from scripts.bomb_threat_r1_items_schedule import refresh as item_acceptance
        item_acceptance()
        from scripts.bomb_threat_r1_publish_pipeline_1805 import ensure_running as r1_publish_successor
        r1_publish_successor()
        from scripts.bun_native_v2_supervisor import ensure_running as native_v2_successor
        native_v2_successor()
        from scripts.bomb_threat_r1_candidate_confirm import ensure_running as candidate_confirm
        candidate_confirm()
        from scripts.bomb_threat_r1_monitor1440 import refresh as r1_monitor1440
        r1_monitor1440()
        from scripts.bomb_threat_r1_confound_schedule import refresh_ready as confound_ready
        confound_ready()
        from scripts.bomb_threat_round16 import refresh as round16_acceptance
        round16_acceptance()
        from scripts.bomb_threat_round15 import refresh as round15_acceptance
        round15_acceptance()
        from scripts.bomb_threat_a3sr_verdict import refresh as a3sr_verdict
        a3sr_verdict()
        # Only cadence40 checkpoints request full games. Boundary1/2 are
        # retained as technical evidence and never consume the ability queue.
        for cp in BASE.glob('formal/*/u*/COMMITTED.json'):
            r=read(cp)
            if r.get('metrics',{}).get('fresh_parent'):continue
            run_name=cp.parent.parent.name
            if invalid_ancestry(run_name) or (cp.parent.parent/'EXCLUDED_EVALUATION.json').exists():continue
            if r['update']%40:continue
            name=run_name+'_'+cp.parent.name
            q=BASE/'eval_requests'/f'{name}.json'
            if not q.exists():
                proto=read(BASE/'protocols'/(run_name+'.json'),{})
                extra=dict(iter2_arm=proto['iter2_arm']) if proto.get('iter2_arm') else {}
                atomic_json(q,dict(name=name,checkpoint=r['checkpoint'],split='dev',rule_version='stock-window-v1',reward_version=proto.get('reward_version','bomb-threat-v1'),**extra))
        if slots_dirty:atomic_json(BASE/'SLOTS.json',slots)
        if (BASE/'a3s/AUTHORIZATION.json').exists():
            from scripts.bomb_threat_seat5050_acceptance import refresh as seat_acceptance
            try:seat_acceptance()
            except (FileNotFoundError,ValueError,AssertionError) as error:
                atomic_json(BASE/'a3s/ACCEPTANCE_RETRY.json',dict(status='RETRY_REQUIRED',
                    error_type=type(error).__name__,training_unaffected=True,unix=time.time()))
        if (BASE/'iter2/SWITCH_DECISION.json').exists():
            from scripts.bomb_threat_iter2_schedule import report as iter2_report
            try:
                iter2_report(slots)
            except FileNotFoundError as error:
                # A concurrently rewritten gate marker may vanish between
                # queue refresh and report aggregation. Keep the GPU owner
                # alive and retry on the next loop.
                atomic_json(BASE/'iter2/REPORT_RETRY_REQUIRED.json',dict(
                    status='RETRY_REQUIRED',error_type='FileNotFoundError',
                    missing_basename=Path(error.filename).name if getattr(error,'filename',None) else None,
                    training_unaffected=True,unix=time.time()))
        atomic_json(BASE/'STATUS.json',dict(status='PAUSED' if paused else 'RUNNING',slots=slots,active=list(live),gpu_samples=samples,unix=time.time(),dedicated_eval_gpu=2))
        time.sleep(5)

if __name__=='__main__':main()
