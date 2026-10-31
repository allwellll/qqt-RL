"""08:15 bounded slot05 self-continuation; consumed by the sole training owner."""
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bomb_threat_round15 import read,pin
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/bomb_threat_v1'
OUT=BASE/'iter2/a3sr/round16'
AUTH=OUT/'AUTHORIZATION.json'
INSTRUCTION='.hermes/STEER_gpu5_holdout_20261030_0815.md'

def prepare():
    cp='runs/bomb_threat_v1/formal/bt_slot05_c000028/u001280'
    pin([cp,'runs/bomb_threat_v1/formal/bt_slot05_c000028/u001200',
         'runs/bomb_threat_v1/formal/bt_slot07_c000029/u001280'], 'round16 continuation/holdout/PBT preservation')
    cm=read(ROOT/cp/'COMMITTED.json')
    for f,k in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
        assert sha(ROOT/cp/f)==cm[k]
    assert cm['metrics']['finite'] and min(cm['metrics']['update_l2'].values())>0
    value=dict(status='AUTHORIZED',instruction=INSTRUCTION,instruction_sha256=sha(ROOT/INSTRUCTION),
        boundary_run='bt_slot05_c000028',source=cm,target_slot=5,target_seed=2026112905,
        collector_seed=2026112905,chunk_updates=320,end_update=1600,unix=time.time())
    if AUTH.exists():
        assert read(AUTH)['source']==cm
    else:atomic_json(AUTH,value)
    return read(AUTH)

def make_protocol(auth,chunk):
    source=auth['source'];donor=Path(source['checkpoint']).parent.name
    template=read(BASE/'protocols'/f'{donor}.json');recipe=template['arms'][donor]
    assert recipe['iter2_arm']=='A3SR'
    name=f'bt_slot05_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        old=read(path)
        assert old['continuation_source']==source and old['arms'][name]==recipe
        return path
    assert all(sha(ROOT/f)==h for f,h in template['source_sha256'].items())
    proto=dict(template)
    for key in ('round14_authorization','round15_authorization'):
        if key in proto:proto['ancestor_'+key]=proto.pop(key)
    proto.update(namespace_id='A3SR-round16-slot05',arms={name:recipe},
        sources={str(auth['target_seed']):source},continuation_source=source,
        chunk_start_update=source['update'],chunk_updates=320,environment_steps_per_run=1600*262144,
        slot=5,chunk=chunk,worker_module='scripts.bomb_threat_a3sr_resume_train',
        source_protocol=str((BASE/'protocols'/f'{donor}.json').relative_to(ROOT)),
        source_protocol_sha256=sha(BASE/'protocols'/f'{donor}.json'),
        round16_authorization=str(AUTH.relative_to(ROOT)),round14_collector_seed=auth['collector_seed'],
        round14_rekey=False,initialization='complete self state continuation; no RNG reset or warmup',
        pbt_lineage=dict(donor=source['checkpoint'],state_sha256=source['state_sha256'],
            policy_sha256=source['policy_sha256'],target_slot=5,target_seed=auth['target_seed'],
            rng='continued unchanged',copied=['Actor','Critic','Adam','env','aux','RNG','ring','counters'],
            critic_warmup_updates=0,formal_updates_discarded=0))
    atomic_json(path,proto)
    return path

def apply_switch(slot,entry,done,chunk):
    auth=read(AUTH,{})
    if slot!=5 or entry.get('round16_applied') or entry['run']!=auth.get('boundary_run'):return None
    assert chunk==29 and done['checkpoint']==auth['source']['checkpoint']
    path=make_protocol(auth,chunk)
    atomic_json(OUT/'SLOT05_HANDOFF.json',dict(status='PROTOCOL_READY',protocol=str(path.relative_to(ROOT)),
        source=done,worker_signals=0,unix=time.time()))
    return dict(entry,run=path.stem,chunk=chunk,source=done,round16_applied=True,
        round16_end_update=1600,supersedes=entry['run'])

def terminal(entry,done):
    if not entry.get('round16_applied'):return False
    assert done['update']==entry['round16_end_update']
    if not (OUT/'SLOT05_TERMINAL.json').exists():
        value=dict(status='FIXED320_COMPLETE',run=entry['run'],last_atomic=done,
                   next='await explicit extension',unix=time.time())
        atomic_json(OUT/'SLOT05_TERMINAL.json',value)
        atomic_json(BASE/'notifications/ROUND16_SLOT05_TERMINAL.json',value)
        from scripts.iter_pbt_owner import notify
        notify('ROUND16_SLOT05_TERMINAL',f"round16 slot05 u{done['update']}固定320更新完成并原子保存；等待扩展，其他训练不停。")
    return True

def refresh():
    handoff=read(OUT/'SLOT05_HANDOFF.json',{})
    target=OUT/'SLOT05_FIRST_FORMAL.json'
    if not handoff or target.exists():return
    out=BASE/'formal'/Path(handoff['protocol']).stem
    smoke,restore=read(out/'SMOKE.json',{}),read(out/'RESTORE.json',{})
    if not(smoke.get('passed') and restore.get('passed')):return
    import json
    rows=[json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['new_update']>handoff['source']['update']+2]
    if not rows:return
    r=rows[0];assert r['finite'] and min(r['update_l2'].values())>0
    value=dict(status='PASS',run=out.name,first_post_restore_update=r['new_update'],
        finite=True,update_l2=r['update_l2'],environment_steps=r['environment_steps'],
        env_per_second=262144/r['cycle_seconds'],smoke=smoke,restore=restore,
        source_handoff=read(out/'SOURCE_HANDOFF.json'),unix=time.time())
    atomic_json(target,value);atomic_json(BASE/'notifications/ROUND16_SLOT05_FIRST_FORMAL.json',value)
    from scripts.iter_pbt_owner import notify
    notify('ROUND16_SLOT05_FIRST_FORMAL',f"GPU5 round16 {out.name} u{r['new_update']} finite、Actor/Critic非零，完整state恢复通过；{value['env_per_second']:.0f}env/s。")

if __name__=='__main__':prepare();refresh()
