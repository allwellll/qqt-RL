"""A3SR bounded640-update successor, consumed by the existing unique owner."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';B=BASE/'iter2/a3sr'


def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def make_protocol(slot,chunk,source,recipe,length=320):
    assert slot==5 and recipe['iter2_arm']=='A3SR'
    frozen=read(B/'PROTOCOL.json');assert frozen['status']=='AUTHORIZED_READY'
    start=source['update'] if source else 0
    assert 0<=start<640 and start+length<=640
    name=f'bt_slot{slot:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        value=read(path);assert value['arms'][name]==recipe and value['continuation_source']==source
        return path
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json')
    hashes=dict(config['source_sha256']);hashes.update(frozen['versioned_sources'])
    value=dict(namespace='runs/bomb_threat_v1/formal',namespace_id='A3SR-v1',common=config['common'],arms={name:recipe},
        sources={str(2026112905):source or recipe['fallback_policy_source']},source_sha256=hashes,
        environment_steps_per_run=(start+length)*262144,chunk_start_update=start,chunk_updates=length,
        continuation_source=source,policy_source=recipe['fallback_policy_source'],initialization_decision=frozen['authorization'],
        worker_module='scripts.bomb_threat_a3sr_train',spawn_manifest=config['spawn_manifest'],
        A1_weights=config['A1_weights'],fixed_actor_source=config['fixed_actor_source'],
        rule_version='stock-window-v1',reward_version='bomb-threat-iter2-A3',training_reward_variant='bomb-threat-iter2-A3',
        slot=slot,chunk=chunk,iter2_arm='A3SR',learner_seat_fraction=.5,a3sr_protocol=str((B/'PROTOCOL.json').relative_to(ROOT)),
        initialization='latest A3 slot01 Actor only;new Critic/Adam/env/aux/RNG/ring;20 warmup;40 Actor ramp')
    atomic_json(path,value);return path


def apply_switch(slot,entry,chunk):
    frozen=read(B/'PROTOCOL.json',{})
    if slot!=5 or entry.get('iter2_arm')=='A3SR' or frozen.get('status')!='AUTHORIZED_READY':return None
    release=read(B/'PUBLICATION_PRIORITY_PAUSE.json',{})
    assert release.get('status')=='RELEASED_AFTER_PUBLICATION_VERIFIED'
    assert read(ROOT/release['publication_receipt'])['status']=='COMPLETED'
    out=BASE/'formal'/entry['run'];old=read(out/'SHAPING_PAUSED.json',{}) or read(out/'COMPLETE.json',{})
    if not old:return None
    retained=[p for p in sorted(out.glob('u*/COMMITTED.json'),reverse=True) if (p.parent/'state.pkl').exists()]
    assert retained,'old slot05 has no retained atomic state'
    done=read(retained[0]);assert done['passed']
    for f,k in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:assert sha(ROOT/done['checkpoint']/f)==done[k]
    from scripts.bomb_threat_iter2_schedule import _latest_a3_policy_source
    source,lineage=_latest_a3_policy_source(slot)
    cfg=read(BASE/'iter2/TRAINING_PROTOCOL.json');recipe=dict(cfg['arms']['A3'],iter2_arm='A3SR',
        fallback_policy_source=source,learner_seat_fraction=.5)
    path=make_protocol(slot,chunk,None,recipe)
    lineage.update(target_run=path.stem,reason='authorized reset-only A3SR after publication',formal_updates_discarded=0)
    atomic_json(B/'HANDOFF.json',dict(status='PROTOCOL_READY',protocol=str(path.relative_to(ROOT)),protocol_sha256=sha(path),
        lineage=lineage,old_run=entry['run'],old_last_atomic=done,worker_signals=0,unix=time.time()))
    atomic_json(out/'A3SR_SUPERSEDED.json',dict(status='SAFE_ATOMIC_REPLACEMENT',last_atomic=done,successor=path.stem,worker_signals=0))
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=source,iter2_arm='A3SR',role='A3SR_reset_randomization',
        old_boundary=done['checkpoint'],supersedes=entry['run'],pbt_lineage=lineage)


def terminal(entry,done):
    if entry.get('iter2_arm')!='A3SR' or done['update']<640:return False
    assert done['update']==640
    target=B/'TERMINAL.json'
    if not target.exists():
        for split in ('dev','holdout'):
            name=f"{entry['run']}_u000640"+('_holdout' if split=='holdout' else '')
            q=BASE/'eval_requests'/f'{name}.json'
            if not q.exists():atomic_json(q,dict(name=name,checkpoint=done['checkpoint'],split=split,
                rule_version='stock-window-v1',reward_version='bomb-threat-iter2-A3',iter2_arm='A3SR',reason='A3SR fixed640 terminal native map seat verdict'))
        result=dict(status='FIXED_BUDGET_COMPLETE_EVALUATION_ASYNC',run=entry['run'],last_atomic=done,
            formal_updates=640,formal_environment_steps=167772160,warmup_updates=20,worker_signals=0,
            successor='stop A3SR; native dev+holdout and9 fixed P1 brick-separated observations; no automatic promotion',unix=time.time())
        atomic_json(target,result);atomic_json(BASE/'notifications/A3SR_TERMINAL.json',result)
        from scripts.iter_pbt_owner import notify
        notify('A3SR_TERMINAL','A3SR固定640更新完整原子保存结束，训练停止扩展；原生地图dev/holdout异步评估，未晋升。')
    return True


def refresh():
    handoff=read(B/'HANDOFF.json',{})
    if not handoff or (B/'FIRST_FORMAL_ACCEPTANCE.json').exists():return
    name=Path(handoff['protocol']).stem;out=BASE/'formal'/name
    smoke=read(out/'SMOKE.json',{});restore=read(out/'RESTORE.json',{})
    if not smoke.get('passed') or not restore.get('passed'):return
    rows=[json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()] if (out/'updates.jsonl').exists() else []
    formal=[r for r in rows if r['new_update']>2]
    if not formal:return
    row=formal[0];assert row['finite'] and min(row['update_l2'].values())>0
    result=dict(status='PASS',run=name,first_post_restore_update=row['new_update'],finite=True,update_l2=row['update_l2'],
        environment_steps=row['environment_steps'],formal_environment_steps=row['new_update']*262144,
        env_per_second=262144/row['cycle_seconds'],smoke=smoke,restore=restore,lineage=handoff['lineage'],
        exposure=row['exposure'],worker_signals=0,unix=time.time())
    atomic_json(B/'FIRST_FORMAL_ACCEPTANCE.json',result);atomic_json(BASE/'notifications/A3SR_FIRST_FORMAL.json',result)
    from scripts.iter_pbt_owner import notify
    notify('A3SR_FIRST_FORMAL',f'A3SR正式u{row["new_update"]} finite，两网非零，{result["env_per_second"]:.0f}env/s；2更新smoke≥18k和完整fresh恢复通过，固定640预算。')
