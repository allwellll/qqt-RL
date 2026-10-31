"""Consume R1 in the existing sole owner, only after complete native chunks."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read,pin
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'
OUT=BASE/'iter2/r1_attack_20261030';TARGETS=(0,3,4,7)


def revision_hold():
    return read(OUT/'ITEM_PROGRESS_REVISION_HOLD.json',{}).get('status')=='HOLD'


def manages(slot,entry):
    return bool(entry.get('r1_applied'))


def make_protocol(slot,chunk):
    frozen=read(OUT/'PROTOCOL.json');source=frozen['source'];donor=Path(source['checkpoint']).parent.name
    template=read(BASE/'protocols'/f'{donor}.json');oldrecipe=template['arms'][donor]
    name=f'bt_slot{slot:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        p=read(path);assert p.get('namespace_id')=='A3SR-ATTACK-R1-20261030';return path
    arm=frozen['arms'][str(slot)];seed=frozen['seeds'][str(slot)]
    recipe=dict(oldrecipe,r1_attack=arm['attack'],r1_bank_fix=True,r1_arm='BANK_ONLY' if not arm['attack'] else 'ATTACK_SELFPLAY45' if slot==4 else 'ATTACK')
    weights=[.05,.25,.25,.20,.15,.10] if slot==4 else [.05,.35,.30,.15,.10,.05]
    recipe['opponent_weights']=weights
    hashes=dict(template['source_sha256'])
    assert all(sha(ROOT/f)==h for f,h in hashes.items()),'old frozen ABI changed'
    files=['scripts/bomb_threat_r1_collector.py','scripts/bomb_threat_r1_reward.py','scripts/bomb_threat_r1_train.py','jax_bomb/bun_safety.py','jax_bomb/je07_native_items.py']
    final_seal=read(OUT/'WORKER_SOURCE_SEAL_FINAL02.json')
    assert all(sha(ROOT/f)==expected for f,expected in final_seal['files'].items()),'R1 frozen worker seal changed'
    hashes.update({f:sha(ROOT/f) for f in files})
    p=dict(template)
    for key in ('round14_authorization','round15_authorization','round16_authorization','a3sr_promotion_authorization'):
        if key in p:p['ancestor_'+key]=p.pop(key)
    p.update(namespace_id='A3SR-ATTACK-R1-20261030',arms={name:recipe},sources={str(seed):source},
        source_sha256=hashes,continuation_source=source,chunk_start_update=source['update'],chunk_updates=640,
        environment_steps_per_run=(source['update']+640)*262144,slot=slot,chunk=chunk,
        worker_module='scripts.bomb_threat_r1_train',r1_attack=arm['attack'],r1_seed=seed,
        A1_weights=weights,r1_bank_manifest=str((OUT/'BANK_FROZEN.json').relative_to(ROOT)),
        r1_bank_manifest_sha256=sha(OUT/'BANK_FROZEN.json'),
        r1_coefficient_request=str((OUT/'coefficient_requests'/f'slot{slot:02d}.json').relative_to(ROOT)),
        reward_version='r1-attack-v1-cap030' if arm['attack'] else template['reward_version'],
        r1_authorization=str((OUT/'PROTOCOL.json').relative_to(ROOT)),
        pbt_lineage=dict(donor=source['checkpoint'],donor_update=source['update'],target_slot=slot,target_seed=seed,
            state_sha256=source['state_sha256'],policy_sha256=source['policy_sha256'],
            copied=['Actor','Critic','Adam','native_env','native_aux','ring','ring_cursor','fixed_actor','counters'],
            migrated=['new shaping history reset' if arm['attack'] else 'old shaping history preserved','new opponent kind allocation','independent RNG'],
            critic_warmup_updates=0,formal_updates_discarded=0))
    atomic_json(path,p)
    pin([str((BASE/'formal'/name/f'u{u:06d}').relative_to(ROOT)) for u in range(source['update'],source['update']+641,40)],'R1 fixed budget checkpoints; independent restore and evaluation')
    return path


def apply_switch(slot,entry,done,chunk):
    if slot not in TARGETS:return None
    if revision_hold():return None
    if entry.get('r1_applied'):
        if not (OUT/'terminal'/f'{entry["run"]}.json').exists():
            cp=ROOT/done['checkpoint'];assert sha(cp/'state.pkl')==done['state_sha256']
            assert sha(cp/'policy.pt')==done['policy_sha256']
            p=read(BASE/'protocols'/f'{entry["run"]}.json');assert done['update']==p['chunk_start_update']+640
            receipt=dict(status='COMPLETE_FIXED_640',slot=slot,run=entry['run'],last_atomic=done,new_updates=640,new_environment_steps=167772160,
                next_action='bounded R1 complete, await explicit next scientific budget; do not fall through to old A3SR automatic loop',unix=time.time())
            atomic_json(OUT/'terminal'/f'{entry["run"]}.json',receipt)
            from scripts.iter_pbt_owner import notify
            notify('R1_TERMINAL_'+entry['run'],f'R1 slot{slot:02d}固定640次更新完成，新增167772160环境步，末点SHA已核验；CPU评估异步继续，不自动扩展预算。')
        return None
    ready=read(OUT/'READY_FOR_BOUNDARY.json',{})
    if ready.get('status')!='PASS':return None
    for marker,expected in ready['prerequisites'].items():assert sha(OUT/marker)==expected,'R1 technical gate hash changed'
    if entry.get('iter2_arm')!='A3SR':return None
    out=BASE/'formal'/entry['run'];p=read(BASE/'protocols'/f'{entry["run"]}.json')
    assert read(out/'COMPLETE.json')['state_sha256']==done['state_sha256']
    assert done['update']==p['chunk_start_update']+p['chunk_updates']
    # Complete frozen pre-R1 u2400 only where it naturally exists. No R1 point
    # may be silently substituted by the independent observer.
    path=make_protocol(slot,chunk);proto=read(path);source=proto['continuation_source']
    atomic_json(OUT/'handoffs'/f'{path.stem}.json',dict(status='BOUNDARY_HANDOFF',previous_run=entry['run'],previous_final=done,
        successor=path.stem,slot=slot,source=source,training_worker_signals=0,unix=time.time()))
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=source,role='R1_ATTACK_FIXED640',iter2_arm='A3SR',r1_applied=True,r1_arm=proto['arms'][path.stem]['r1_arm'],
        old_boundary=done['checkpoint'],supersedes=entry['run'],authorized_shaping_override=True)


def monitor_gate(points):
    """Frozen comparable first point; two consecutive bad points latch half."""
    if len(points)<3:return dict(status='MISSING',reason='requires baseline and two followups')
    ref=points[0];last=points[-2:]
    badempty=all(p['empty_bomb_fraction']-ref['empty_bomb_fraction']>=.10 and p['kills']<=ref['kills'] for p in last)
    badself=any(p['self_harm']-ref['self_harm']>=3 for p in last)
    return dict(status='HALVE' if badempty or badself else 'OBSERVE',empty_increase_without_kills=badempty,self_harm_increase=badself)


def refresh():
    if not (OUT/'PROTOCOL.json').exists():return
    successor=read(OUT/'items_v2/EXECUTION_STATUS.json',{})
    if revision_hold() and successor:
        receipt=dict(status='SUPERSEDED_BY_NATIVE_ITEMS_V2',
            successor_status=successor.get('status'),
            successor='runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2/EXECUTION_STATUS.json',
            old_reward_formal_launch_blocked=True,old_artifacts_preserved=True,unix=time.time())
        atomic_json(OUT/'EXECUTION_STATUS.json',receipt)
        atomic_json(OUT/'TASK_CONSUMED.json',receipt)
        return
    rows=[]
    for hand in sorted((OUT/'handoffs').glob('*.json')):
        h=read(hand);run=h['successor'];out=BASE/'formal'/run;slot=h['slot'];p=read(BASE/'protocols'/f'{run}.json')
        if (out/'EXCLUDED_TECHNICAL.json').exists():continue
        commits=sorted(out.glob('u*/COMMITTED.json'));latest=read(commits[-1]) if commits else None
        update=read(out/'PROGRESS.json',{}).get('iterations',h['source']['update'])
        first=OUT/'first_updates'/f'{run}.json'
        if not first.exists() and read(out/'SMOKE.json',{}).get('passed') and read(out/'RESTORE.json',{}).get('passed'):
            updates=[json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines()]
            formal=[r for r in updates if r['new_update']>h['source']['update']+2]
            if formal:
                r=formal[0];assert r['finite'] and min(r['update_l2'].values())>0
                atomic_json(first,dict(status='PASS',run=run,slot=slot,first_formal_update=r['new_update'],finite=True,
                    update_l2=r['update_l2'],env_per_second=262144/r['cycle_seconds'],cycle_seconds=r['cycle_seconds'],
                    environment_steps=r['environment_steps'],actual_environment_steps_this_update=r['actual_environment_steps_this_update'],
                    source_handoff=read(out/'SOURCE_HANDOFF.json'),smoke=read(out/'SMOKE.json'),fresh_restore=read(out/'RESTORE.json'),unix=time.time()))
                from scripts.iter_pbt_owner import notify
                notify('R1_FIRST_'+run,f"R1 {run} 正式u{r['new_update']} finite，Actor/Critic L2 {r['update_l2']}；{262144/r['cycle_seconds']:.0f} env/s，smoke和fresh完整恢复通过。")
        for cp in commits:
            cm=read(cp)
            if cm['update']%40:continue
            for split in ('dev','holdout'):
                name=f'{run}_{cp.parent.name}'+('_holdout' if split=='holdout' else '')
                req=BASE/'eval_requests'/f'{name}.json'
                if not req.exists():atomic_json(req,dict(name=name,checkpoint=cm['checkpoint'],split=split,rule_version='stock-window-v1',reward_version=p['reward_version'],
                    iter2_arm='A3SR',r1_metrics=True,r1_arm=p['arms'][run]['r1_arm'],reason='fixed40 R1 cadence, same native JS both action modes; nonblocking'))
        from scripts.bomb_threat_round16_report import argmax
        from scripts.bomb_threat_a3sr_promotion import rolling_gate
        holdout=[]
        for cp in commits:
            if int(cp.parent.name[1:])%40:continue
            observed=argmax(f'{run}_{cp.parent.name}_holdout')
            if observed:holdout.append(dict(update=int(cp.parent.name[1:]),metrics=observed))
        observed_gate=rolling_gate([x['metrics'] for x in holdout])
        atomic_json(OUT/'holdout_observation'/f'{run}.json',dict(status='OBSERVATION_ONLY',same_lineage_rolling_gate=observed_gate,
            points=holdout,recent4_by_seat={seat:{field:sum(x['metrics']['by_seat'][seat][field] for x in holdout[-4:])/len(holdout[-4:])
                for field in ('win_minus_loss','zero_bomb_games','self_deaths','delivery')} for seat in ('P0','P1')} if holdout else {},
            fixed_budget_not_stopped_by_missing_evaluation=True,web_replacement='requires separately confirmed same-protocol three-tier fresh256 P1 standard; no automatic publish',unix=time.time()))
        # Adaptive opposition is OBSERVATION ONLY, never changes weights.
        exposures=[]
        if (out/'updates.jsonl').exists():exposures=[json.loads(l)['exposure'] for l in (out/'updates.jsonl').read_text().splitlines()]
        wins=sum(e.get('opponent_win',[0]*6)[2] for e in exposures);draws=sum(e.get('opponent_draw',[0]*6)[2] for e in exposures);losses=sum(e.get('opponent_loss',[0]*6)[2] for e in exposures);total=wins+draws+losses
        adaptive=total>0 and wins/total>=.5 and draws/total<=.4
        if adaptive and not (OUT/'adaptive_gate'/f'{run}.json').exists():
            atomic_json(OUT/'adaptive_gate'/f'{run}.json',dict(status='NOTIFY_ONLY',run=run,wins=wins,draws=draws,losses=losses,automatic_weight_changes=False,unix=time.time()))
            from scripts.iter_pbt_owner import notify
            notify('R1_ADAPTIVE_'+run,'R1 hard对手滚动chunk达到观察门槛；仅通知编排方，尚未调自对弈比例。')
        points=[]
        for cp in commits:
            name=f'{run}_{cp.parent.name}'
            summaries=[]
            for f in sorted((BASE/'eval'/name).glob('natural_greedy_*.json.r1_metrics.json')):
                extra=read(f)
                if extra.get('version')!='r1-passive-bomb-lifetime-v2' or extra.get('source_sha256')!=sha(ROOT/'scripts/bomb_threat_r1_eval.js'):continue
                summaries.extend(extra['episodes'])
            if len(summaries)!=64 or len({x['seed'] for x in summaries})!=64 or not all(x['done'] for x in summaries):continue
            bomb=sum(x['learner']['resolved_bombs'] for x in summaries);empty=sum(x['learner']['empty_bombs'] for x in summaries)
            points.append(dict(update=int(cp.parent.name[1:]),empty_bomb_fraction=empty/bomb if bomb else 0,kills=sum(x['learner']['kills'] for x in summaries),self_harm=sum(x['learner']['self_harm'] for x in summaries)))
        gate=monitor_gate(points);atomic_json(OUT/'anti_hack'/f'{run}.json',dict(**gate,run=run,points=points,unix=time.time()))
        if p['r1_attack'] and gate['status']=='HALVE':
            target=ROOT/p['r1_coefficient_request']
            if not target.exists():
                atomic_json(target,dict(protocol_id='A3SR-ATTACK-R1-20261030',factor=.5,gate=gate,points=points,apply='next atomic40 boundary; no training stop',unix=time.time()))
                from scripts.iter_pbt_owner import notify
                notify('R1_ANTI_HACK_'+run,'R1防hack红线触发；已请求下一40更新原子边界将R1系数减半，训练不停，原始证据保留。')
        rows.append(dict(slot=slot,run=run,update=update,new_updates=update-h['source']['update'],latest_atomic=latest,
            smoke=read(out/'SMOKE.json',{}).get('passed'),restore=read(out/'RESTORE.json',{}).get('passed'),first_formal_accepted=first.exists(),complete=(out/'COMPLETE.json').exists(),anti_hack=gate))
    done=[r for r in rows if r['complete']]
    ready=(OUT/'READY_FOR_BOUNDARY.json').exists()
    status='ALL_FOUR_COMPLETE_CPU_EVAL_PENDING' if len(done)==4 else 'RUNNING' if rows else 'WAITING_COMPLETE_CHUNK_BOUNDARIES' if ready else 'WAITING_TECHNICAL_GATES'
    atomic_json(OUT/'EXECUTION_STATUS.json',dict(status=status,arms=rows,target_slots=list(TARGETS),unix=time.time()))
    atomic_json(OUT/'TASK_CONSUMED.json',dict(status=status,instruction='.hermes/TASK_a3sr_attack_r1_20261030.md',
        CPU_prerequisites_passed=ready,first_formal_accepted_slots=[r['slot'] for r in rows if r['first_formal_accepted']],
        training_worker_signals=0,old_queues_resumed=False,publication=False,unix=time.time()))

if __name__=='__main__':refresh()
