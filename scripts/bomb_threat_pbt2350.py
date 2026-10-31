"""23:50 authorized slot04 full-state replacement, consumed by sole owner."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'
AUTH=BASE/'iter2/pbt2350/AMENDMENT.json'


def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def apply_switch(slot,entry,done,chunk):
    auth=read(AUTH,{})
    if slot!=4 or auth.get('status')!='AUTHORIZED_AT_FULL_CHUNK' or entry['run']!=auth.get('boundary_run'):return None
    old=read(BASE/'protocols'/f"{entry['run']}.json")
    required=old['chunk_start_update']+old['chunk_updates']
    complete=read(BASE/'formal'/entry['run']/'COMPLETE.json',{})
    assert done['update']==required and complete.get('state_sha256')==done['state_sha256']
    for file,key in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
        assert sha(ROOT/done['checkpoint']/file)==done[key]
    donor=auth['donor_run'];donor_proto=read(BASE/'protocols'/f'{donor}.json')
    commits=sorted((BASE/'formal'/donor).glob('u*/COMMITTED.json'),reverse=True)
    # Latest complete atomic payload still retained, never select by score.
    candidates=[read(p) for p in commits if (p.parent/'state.pkl').exists() and (p.parent.parent/f'NEXT_STEP_{p.parent.name}.json').exists()]
    assert candidates,'authorized c23 donor has no complete retained state'
    source=candidates[0]
    assert source['passed'] and source['metrics']['finite']
    for file,key in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
        assert sha(ROOT/source['checkpoint']/file)==source[key]
    assert donor_proto['iter2_arm']=='A3'
    name=f'bt_slot{slot:02d}_c{chunk:06d}'
    path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        proto=read(path);assert proto.get('pbt2350_authorization')==str(AUTH.relative_to(ROOT))
    else:
        proto=dict(donor_proto)
        recipe=dict(next(iter(donor_proto['arms'].values())))
        lineage=dict(parent_run=donor,parent_checkpoint=source['checkpoint'],parent_update=source['update'],
            parent_seed=2026112907,target_slot=4,target_seed=2026112904,
            source_state_sha256=source['state_sha256'],source_policy_sha256=source['policy_sha256'],
            copied=['Actor','Critic','Adam','env','aux','ring','ring_cursor','fixed_actor','warmup_rows','counters'],
            RNG='independent target seed, applied after donor next-step parity',critic_warmup_updates=0,
            replaced_run=entry['run'],replaced_last_atomic=done['checkpoint'])
        proto.update(arms={name:recipe},sources={'2026112904':source},continuation_source=source,
            chunk_start_update=source['update'],chunk_updates=320,environment_steps_per_run=(source['update']+320)*262144,
            slot=4,chunk=chunk,worker_module='scripts.bomb_threat_pbt2350_train',pbt2350_rekey=True,pbt2350_collector_seed=2026112907,
            pbt2350_authorization=str(AUTH.relative_to(ROOT)),pbt_lineage=lineage,
            initialization='full donor Actor/Critic/Adam/env/aux/ring, independent target RNG; no warmup')
        proto['source_sha256']=dict(proto['source_sha256'])
        for f in ('scripts/bomb_threat_pbt2350.py','scripts/bomb_threat_pbt2350_train.py'):
            proto['source_sha256'][f]=sha(ROOT/f)
        atomic_json(path,proto)
    atomic_json(BASE/'formal'/entry['run']/'PBT2350_SUPERSEDED.json',dict(status='COMPLETE_CHUNK_PBT_REPLACEMENT',
        last_atomic=done,successor=name,lineage=proto['pbt_lineage'],formal_updates_discarded=0,training_worker_signals=0,unix=time.time()))
    atomic_json(AUTH.parent/'HANDOFF.json',dict(status='PROTOCOL_READY',protocol=str(path.relative_to(ROOT)),
        protocol_sha256=sha(path),lineage=proto['pbt_lineage'],unix=time.time()))
    return dict(entry,run=name,chunk=chunk,source=source,role='A3_PBT2350_from_slot07c23',
        pbt2350_applied=True,pbt_lineage=proto['pbt_lineage'],supersedes=entry['run'])


def refresh():
    handoff=read(AUTH.parent/'HANDOFF.json',{})
    if not handoff:return
    proto=read(ROOT/handoff['protocol']);name=Path(handoff['protocol']).stem;out=BASE/'formal'/name
    accepted=AUTH.parent/'FIRST_FORMAL_ACCEPTANCE.json'
    if accepted.exists():return
    smoke=read(out/'SMOKE.json',{});restore=read(out/'RESTORE.json',{});pbt=read(out/'PBT_HANDOFF.json',{})
    rows=[]
    if (out/'updates.jsonl').exists():
        for line in (out/'updates.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    formal=[r for r in rows if r['new_update']>proto['chunk_start_update']+2]
    if not (formal and smoke.get('passed') and restore.get('passed') and pbt.get('status')=='PASS'):return
    row=formal[0];assert row['finite'] and min(row['update_l2'].values())>0
    result=dict(status='PASS',run=name,first_post_restore_update=row['new_update'],finite=True,
        update_l2=row['update_l2'],environment_steps=row['environment_steps'],env_per_second=262144/row['cycle_seconds'],
        checkpoint=smoke['checkpoint'],fresh_restore=restore,pbt_handoff=pbt,worker_signals=0,unix=time.time())
    atomic_json(accepted,result)
    atomic_json(BASE/'notifications/PBT2350_FIRST_FORMAL.json',result)
    from scripts.iter_pbt_owner import notify
    notify('PBT2350_FIRST_FORMAL',f'slot04 PBT正式续训u{row["new_update"]}，finite且Actor/Critic真实非零，fresh恢复通过，{result["env_per_second"]:.0f}env/s；无预热、旧线完整原子保存。')


def continue_protocol(slot,chunk,source,recipe,length=320):
    parent=read(BASE/'protocols'/(Path(source['checkpoint']).parent.name+'.json'),{})
    if not parent.get('pbt2350_authorization'):return None
    name=f'bt_slot{slot:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        assert read(path)['continuation_source']==source
        return path
    value=dict(parent,arms={name:recipe},sources={str(2026112900+slot):source},continuation_source=source,
        chunk_start_update=source['update'],chunk_updates=length,chunk=chunk,pbt2350_rekey=False,
        environment_steps_per_run=(source['update']+length)*262144,
        initialization='full target-state continuation, including independently keyed target RNG; no warmup')
    atomic_json(path,value);return path
