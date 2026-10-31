"""Consume authorized A3SR promotion only at complete chunk boundaries."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read,pin
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'
OUT=BASE/'iter2/a3sr/promotion_20261030';AUTH=OUT/'AUTHORIZATION.json'
INSTRUCTION='.hermes/DECISION_promote_a3sr_20261030.md'
TARGETS=(0,3,4,7)


def prepare():
    if AUTH.exists():return read(AUTH)
    slots=read(BASE/'SLOTS.json');donor_run=slots['5']['run']
    assert read(BASE/'protocols'/f'{donor_run}.json')['arms'][donor_run]['iter2_arm']=='A3SR'
    candidates=sorted((BASE/'formal'/donor_run).glob('u*/COMMITTED.json'),reverse=True)
    # Pin before reading payload hashes; select newest complete retained state,
    # independent of dev/holdout scores.
    pin([str(p.parent.relative_to(ROOT)) for p in candidates], 'A3SR promotion latest atomic source selection')
    source=next(read(p) for p in candidates if (p.parent/'state.pkl').exists())
    cp=ROOT/source['checkpoint']
    for f,k in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:assert sha(cp/f)==source[k]
    assert source['metrics']['finite'] and min(source['metrics']['update_l2'].values())>0
    value=dict(status='AUTHORIZED',instruction=INSTRUCTION,instruction_sha256=sha(ROOT/INSTRUCTION),
        boundary_runs={str(g):slots[str(g)]['run'] for g in TARGETS},donor=source,
        donor_selection='latest retained complete atomic state at authorization consumption; frozen common donor, no score selection',
        target_seeds={str(g):2026112900+g for g in TARGETS},target_slots=list(TARGETS),
        chunk_updates=320,continuing_slot05=True,unchanged_slots=[1,6],
        budget='320 updates per chunk, automatically continue authorized A3SR lanes until user pause or new decision',
        RNG='donor next-step parity before independent per-slot rekey; self-continuations preserve RNG',unix=time.time())
    atomic_json(OUT/'AMENDMENT01_ATOMIC_DONOR_AND_GATE_REFERENCE.json',dict(status='FROZEN',
        instruction=INSTRUCTION,common_donor=source['checkpoint'],
        source_choice='newest at authorization consumption, shared by four replacements; no peak selection',
        rolling_gate='compare latest4 complete holdout cadence40 points with preceding nonoverlapping4 on same lineage, byseat; fewerthan8 gives MISSING and requests holdout, not failure',
        delivery_drop='strictly greaterthan50percent relative to preceding4 mean; zero reference cannot trigger',
        declines='W-L decrease strictly>6, unilateral-zero orself-death increases strictly>3; no contemporary control gates',
        preexisting_B='retain historical B audits unchanged; promotion decision overrides B as a training admission gate',
        web='P1sum W-L aboveu800 and eachtier no less thanu800 minus5; reportonly, no auto publish',unix=time.time()))
    atomic_json(AUTH,value);return value


def make_protocol(slot,chunk,source,rekey):
    donor=Path(source['checkpoint']).parent.name;template=read(BASE/'protocols'/f'{donor}.json')
    recipe=template['arms'][donor];assert recipe['iter2_arm']=='A3SR'
    name=f'bt_slot{slot:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        p=read(path);assert p['continuation_source']==source and p['arms'][name]==recipe
        assert p['round14_rekey']==rekey;return path
    assert all(sha(ROOT/f)==h for f,h in template['source_sha256'].items())
    proto=dict(template)
    for key in ('round14_authorization','round15_authorization','round16_authorization'):
        if key in proto:proto['ancestor_'+key]=proto.pop(key)
    seed=2026112900+slot;start=source['update']
    proto.update(namespace_id='A3SR-promoted-20261030',arms={name:recipe},sources={str(seed):source},
        continuation_source=source,chunk_start_update=start,chunk_updates=320,
        environment_steps_per_run=(start+320)*262144,slot=slot,chunk=chunk,
        worker_module='scripts.bomb_threat_a3sr_resume_train',
        source_protocol=str((BASE/'protocols'/f'{donor}.json').relative_to(ROOT)),
        source_protocol_sha256=sha(BASE/'protocols'/f'{donor}.json'),
        a3sr_promotion_authorization=str(AUTH.relative_to(ROOT)),round14_rekey=rekey,
        round14_collector_seed=template['round14_collector_seed'],
        initialization='complete Actor/Critic/Adam/env/aux/ring source parity; independent target RNG only on fork; no warmup',
        pbt_lineage=dict(donor=source['checkpoint'],donor_update=start,target_slot=slot,target_seed=seed,
            state_sha256=source['state_sha256'],policy_sha256=source['policy_sha256'],
            rng='independent after donor parity' if rekey else 'self continuation unchanged',
            copied=['Actor','Critic','Adam','env','aux','ring','ring_cursor','fixed_actor','warmup_rows','counters'],
            critic_warmup_updates=0,formal_updates_discarded=0))
    atomic_json(path,proto);return path


def rolling_gate(points):
    """Only a complete previous4/latest4 comparison can block healthy training."""
    if len(points)<8:return dict(status='MISSING',missing=['eight complete same-lineage holdout cadence40 points'],failures=[])
    previous,current=points[-8:-4],points[-4:]
    failures=[];means={}
    for seat in ('P0','P1'):
        means[seat]={}
        for field in ('win_minus_loss','zero_bomb_games','self_deaths','delivery'):
            old=sum(p['by_seat'][seat][field] for p in previous)/4
            new=sum(p['by_seat'][seat][field] for p in current)/4
            means[seat][field]=dict(previous=old,current=new,change=new-old)
            bad=(old-new>=6 if field=='win_minus_loss' else new-old>=3 if field in ('zero_bomb_games','self_deaths') else old>0 and new<=old*.5)
            if bad:failures.append(seat+'_'+field)
    return dict(status='BLOCK' if failures else 'PASS',missing=[],failures=failures,means=means)


def holdout_update_floor(latest):
    """Keep eight saved cadence points throughout the next40-update interval."""
    return int(latest)//40*40-280


def final_acceptance(records,donor):
    selected={}
    for record in records:
        if record['slot'] in TARGETS:selected.setdefault(record['slot'],record)
    if set(selected)!=set(TARGETS):
        return dict(status='MISSING',missing_slots=sorted(set(TARGETS)-set(selected)))
    digests=[]
    for record in selected.values():
        handoff=record['source_handoff']
        assert record['finite'] and min(record['update_l2'].values())>0
        assert record['smoke']['passed'] and record['restore']['passed']
        assert record['source']['state_sha256']==donor['state_sha256']
        assert handoff['donor_next_step_exact_before_rekey']
        assert handoff['Actor_Critic_Adam_env_aux_ring_exact']
        assert handoff['RNG_rekeyed'] and handoff['critic_warmup_updates']==0
        assert handoff['formal_updates_discarded']==0
        assert handoff['target_rng_digest']!=handoff['source_rng_digest']
        digests.append(handoff['target_rng_digest'])
    assert len(set(digests))==len(TARGETS), 'promoted slots must have distinct RNG streams'
    return dict(status='PASS',slots=list(TARGETS),source_checkpoint=donor['checkpoint'],
        state_sha256=donor['state_sha256'],policy_sha256=donor['policy_sha256'],
        first_updates={str(s):dict(run=r['run'],update=r['first_formal_update'],
            update_l2=r['update_l2'],target_rng_digest=r['source_handoff']['target_rng_digest'])
            for s,r in selected.items()},
        full_state_copy_and_source_parity=True,independent_RNG=True,
        critic_warmup_updates=0,training_worker_signals=0,discarded_formal_updates=0)


def manages(slot,entry):
    auth=read(AUTH,{})
    return auth.get('status')=='AUTHORIZED' and (entry.get('a3sr_promotion_applied') or
        slot==5 and entry.get('iter2_arm')=='A3SR' or
        slot in TARGETS and entry['run']==auth['boundary_runs'].get(str(slot)))


def refresh_holdout_gates():
    from scripts.bomb_threat_owner import causal_lineage_limits
    from scripts.bomb_threat_round16_report import argmax
    for slot,entry in read(BASE/'SLOTS.json',{}).items():
        slot=int(slot)
        if not(entry.get('a3sr_promotion_applied') or slot==5 and entry.get('iter2_arm')=='A3SR'):continue
        limits=causal_lineage_limits(entry['run']);points=[]
        current_proto=read(BASE/'protocols'/f"{entry['run']}.json")
        latest=read(BASE/'formal'/entry['run']/'PROGRESS.json',{}).get('iterations',current_proto['chunk_start_update'])
        for run,limit in limits.items():
            # Distinct independent branches must not count as this lane's history.
            if not run.startswith(f'bt_slot{slot:02d}_'):continue
            proto=read(BASE/'protocols'/f'{run}.json',{})
            if proto.get('arms',{}).get(run,{}).get('iter2_arm')!='A3SR':continue
            for cp in (BASE/'formal'/run).glob('u*/COMMITTED.json'):
                update=int(cp.parent.name[1:])
                cm=read(cp)
                if update%40 or update>limit or update<holdout_update_floor(latest) or cm.get('metrics',{}).get('fresh_parent'):continue
                name=f'{run}_{cp.parent.name}_holdout';q=BASE/'eval_requests'/f'{name}.json'
                # Existing immutable requests are never rewritten.
                if not q.exists():atomic_json(q,dict(name=name,checkpoint=cm['checkpoint'],split='holdout',
                    rule_version='stock-window-v1',reward_version=proto['reward_version'],iter2_arm='A3SR',
                    reason='authorized A3SR same-line rolling4 holdout monitor; no paired contemporaneous admission'))
                if len(list((BASE/'eval'/name).glob('natural_greedy_*.json.outcome.json')))==8:
                    value=argmax(name)
                    if value:points.append((update,value))
        points=[v for _,v in sorted(points,key=lambda row:row[0])]
        result=rolling_gate(points)
        result.update(slot=slot,run=entry['run'],points=[p['name'] for p in points[-8:]],
            action='at next complete chunk boundary block continuation and notify; no automatic source/weight change' if result['status']=='BLOCK' else 'continue training; missing assessments asynchronous',unix=time.time())
        atomic_json(OUT/'rolling_gates'/f'slot{slot:02d}.json',result)


def apply_switch(slot,entry,done,chunk):
    auth=read(AUTH,{})
    if auth.get('status')!='AUTHORIZED':return None
    continuing=entry.get('a3sr_promotion_applied') or slot==5 and entry.get('iter2_arm')=='A3SR'
    first=slot in TARGETS and entry['run']==auth['boundary_runs'].get(str(slot)) and not entry.get('a3sr_promotion_applied')
    if not(first or continuing):return None
    # Owner calls this only once COMPLETE exists and workers() reports absent.
    out=BASE/'formal'/entry['run'];complete=read(out/'COMPLETE.json',{})
    old=read(BASE/'protocols'/f"{entry['run']}.json")
    assert complete.get('state_sha256')==done['state_sha256']
    assert done['update']==old['chunk_start_update']+old['chunk_updates']
    monitor=read(OUT/'rolling_gates'/f'slot{slot:02d}.json',{})
    from scripts.bomb_threat_a3sr_lineage_watch import boundary_block
    lineage_stop=boundary_block(slot)
    if continuing and (monitor.get('status')=='BLOCK' or lineage_stop):
        receipt=dict(status='BLOCKED_AT_COMPLETE_CHUNK',last_atomic=done,monitor=monitor,unix=time.time())
        if lineage_stop:receipt['post_publish_lineage_watch']=lineage_stop
        atomic_json(OUT/'blocked'/f'{entry["run"]}.json',receipt)
        from scripts.iter_pbt_owner import notify
        cause='发布后JS确认或rolling观察' if lineage_stop else '四点holdout均值'
        notify('A3SR_ROLLING_BLOCK_'+entry['run'],f'A3SR slot{slot:02d}{cause}触发批准停止条件；已完整chunk原子保存，阻止下一chunk，等待编排方决定，不自动回退。')
        return None
    source=auth['donor'] if first else done
    pin([done['checkpoint'],source['checkpoint']], 'A3SR promotion safe complete boundary and source')
    for cm in (done,source):
        for f,k in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:assert sha(ROOT/cm['checkpoint']/f)==cm[k]
    path=make_protocol(slot,chunk,source,rekey=first)
    receipt=dict(status='PROTOCOL_READY',target_slot=slot,successor=path.stem,source=source,
        old_last_atomic=done,old_chunk_complete=True,training_worker_signals=0,
        RNG_rekey=first,protocol=str(path.relative_to(ROOT)),protocol_sha256=sha(path),unix=time.time())
    atomic_json(OUT/'handoffs'/f'{path.stem}.json',receipt)
    atomic_json(out/'A3SR_PROMOTION_SUCCESSOR.json',receipt)
    from scripts.iter_pbt_owner import notify
    notify(entry['run']+'_A3SR_CHUNK_COMPLETE',
        f"A3SR接续边界 {entry['run']} u{done['update']} 已完整原子保存、worker退出；state SHA {done['state_sha256']}；下一项 {path.stem}，无训练信号、无更新丢失。")
    return dict(entry,run=path.stem,chunk=chunk,source=source,iter2_arm='A3SR',
        role='A3SR_promoted' if slot in TARGETS else 'A3SR_original_continuing',
        a3sr_promotion_applied=True,a3sr_promotion_first_run=entry.get('a3sr_promotion_first_run',path.stem),
        supersedes=entry['run'],pbt_lineage=read(path)['pbt_lineage'])


def refresh():
    auth=read(AUTH,{})
    if not auth:return
    from scripts.bomb_threat_a3sr_lineage_watch import ensure_running
    ensure_running()
    records=[]
    cadence=OUT/'HOLDOUT_REFRESH.json'
    if time.time()-read(cadence,{}).get('unix',0)>60:
        refresh_holdout_gates()
        atomic_json(cadence,dict(status='UPDATED',unix=time.time()))
    for p in sorted((OUT/'handoffs').glob('*.json')):
        handoff=read(p);name=handoff['successor'];out=BASE/'formal'/name
        target=OUT/'first_updates'/f'{name}.json'
        if target.exists():records.append(read(target));continue
        smoke=read(out/'SMOKE.json',{});restore=read(out/'RESTORE.json',{})
        if not(smoke.get('passed') and restore.get('passed')):continue
        rows=[json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r['new_update']>handoff['source']['update']+2]
        if not rows:continue
        r=rows[0];assert r['finite'] and min(r['update_l2'].values())>0
        value=dict(status='PASS',run=name,slot=handoff['target_slot'],source=handoff['source'],
            first_formal_update=r['new_update'],finite=True,update_l2=r['update_l2'],
            environment_steps=r['environment_steps'],env_per_second=262144/r['cycle_seconds'],
            smoke=smoke,restore=restore,source_handoff=read(out/'SOURCE_HANDOFF.json'),unix=time.time())
        atomic_json(target,value);atomic_json(BASE/'notifications'/f'A3SR_PROMOTION_{name}.json',value)
        from scripts.iter_pbt_owner import notify
        notify('A3SR_PROMOTION_'+name,f"A3SR推广slot{value['slot']:02d} {name}：源u{handoff['source']['update']} stateSHA {handoff['source']['state_sha256']}；正式u{r['new_update']} finite、Actor/Critic非零，smoke/fresh恢复通过，无Critic预热。")
        records.append(value)
    completed=sorted({v['slot'] for v in records if v['slot'] in TARGETS})
    value=dict(status='ALL_FOUR_FORMAL_ACCEPTED' if completed==list(TARGETS) else 'WAITING_FULL_CHUNK_BOUNDARIES',
        target_slots=list(TARGETS),accepted_slots=completed,source=auth['donor'],
        receipt_files=[str(p.relative_to(ROOT)) for p in sorted((OUT/'first_updates').glob('*.json'))],unix=time.time())
    atomic_json(OUT/'EXECUTION_STATUS.json',value)
    acceptance=final_acceptance(records,auth['donor'])
    if acceptance['status']=='PASS':
        acceptance['unix']=time.time()
        atomic_json(OUT/'FINAL_ACCEPTANCE.json',acceptance)
    consumed=read(OUT/'TASK_CONSUMED.json',{})
    consumed.update(status=value['status'],accepted_slots=completed,target_slots=list(TARGETS),
        training_continues=True,worker_signals=0,execution_status=str((OUT/'EXECUTION_STATUS.json').relative_to(ROOT)),unix=time.time())
    atomic_json(OUT/'TASK_CONSUMED.json',consumed)
    slots=read(BASE/'SLOTS.json',{})
    text=['# A3SR推广执行与简化门禁 — 20261030','',
        '执行 `.hermes/DECISION_promote_a3sr_20261030.md`。推广只在当前完整chunk提交并退出后接管，不打断在途训练；原A3SR推广FAIL留作历史，由本决定明确覆盖；不改网页/BEST。','',
        f"状态 **{value['status']}**；已验收新线首正式更新槽位{completed}，目标{list(TARGETS)}。",'',
        f"共同donor `{auth['donor']['checkpoint']}`；state SHA `{auth['donor']['state_sha256']}`，policy SHA `{auth['donor']['policy_sha256']}`。授权消费时最新完整点，无按分数选点；先pin再验SHA。",'',
        '完整复制Actor/Critic/Adam/env/aux/ring/计数器；先donor next-step parity，再各自target seed独立RNG rekey；无Critic预热。slot05保留自己完整状态与RNG，320更新chunk连续接续；slot01/06不变。配方/奖励/学习率/对手比例/解码完全沿用A3SR，DAgger/KL仍0。','',
        '| slot | 当前run | 当前update | smoke | fresh恢复 | 新线首正式update | Actor/Critic L2 |','|---|---|---:|---|---|---:|---|']
    for slot in (0,1,3,4,5,6,7):
        entry=slots[str(slot)];name=entry['run'];out=BASE/'formal'/name
        row=read(out/'PROGRESS.json',{})
        accepted=[r for r in records if r['slot']==slot]
        first=accepted[0] if accepted else {}
        text.append(f"|{slot}|{name}|{row.get('iterations','MISSING')}|{read(out/'SMOKE.json',{}).get('passed','pending')}|{read(out/'RESTORE.json',{}).get('passed','pending')}|{first.get('first_formal_update','pending')}|{first.get('update_l2','pending')}|")
    text+=['','## 滚动holdout门禁','',
        '同物理槽新A3SR谱系，只看最近4个完整40倍数holdout点与前4点的分seat均值。胜负差下降≥6、单方零泡/自伤增加≥3、交付事件均值下降≥一半才阻断下一完整chunk；其他差异按噪声观察。缺8点为MISSING并异步补最近8点，不用旧同期对照或B挡住健康训练。交付使用原生逐局交付事件，非终局库存数量。触发时保留完整末state、通知等回退来源，不擅改参数。','',
        '| slot | 状态 | 已有完整点 | 失败项 | 缺数据 |','|---|---|---:|---|---|']
    for path in sorted((OUT/'rolling_gates').glob('*.json')):
        r=read(path);text.append(f"|{r['slot']}|{r['status']}|{len(r['points'])}|{r['failures']}|{r['missing']}|")
    text+=['','门禁解释及预算 `promotion_20261030/AMENDMENT01_ATOMIC_DONOR_AND_GATE_REFERENCE.json`，等号边界按原决定“以上”由amendment02纠正；每次完整交接和首正式更新SHA收据分别在handoffs、first_updates。网页仅当同u800 seed fresh256的P1三档W−L总和高于u800且每档≥u800−5时上报编排方；不自动发布。08:40同seed矩阵及08:15独立新seed复核并行继续。','']
    (ROOT/'reports/bomb_threat_v1_a3sr_promotion_20261030.md').write_text('\n'.join(text))

    if completed==list(TARGETS) and not (OUT/'TASK_NOTIFICATION.json').exists():
        atomic_json(OUT/'TASK_NOTIFICATION.json',value)
        from scripts.iter_pbt_owner import notify
        notify('A3SR_PROMOTION_ALL_FOUR','A3SR已在完整chunk边界推广至slot00/03/04/07，四路smoke/独立恢复及首正式finite/ActorCritic非零均验收；slot05继续，slot01/06不变，fresh评估异步。')

if __name__=='__main__':prepare();refresh()
