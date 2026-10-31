"""A3S dispatch by the existing unique owner, only at complete chunk boundaries."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def read(p,default=None):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default


def budget_limit():
    auth=read(BASE/'a3s/extension1280/AUTHORIZATION.json',{})
    if auth.get('status')!='AUTHORIZED':return 640
    assert sha(Path(auth['instruction']))==auth['instruction_sha256']
    assert auth['formal_updates_per_seed']==1280
    return 1280


def make_protocol(slot,chunk,source,recipe,length=320):
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json');frozen=read(BASE/'a3s/PROTOCOL.json')
    name=f'bt_slot{slot:02d}_c{chunk:06d}';path=BASE/'protocols'/f'{name}.json'
    if path.exists():
        value=read(path);assert value['arms'][name]==recipe and value['continuation_source']==source
        return path
    start=int(source['update']) if source else 0
    limit=budget_limit()
    assert start in range(0,limit,320) and start+length<=limit,'A3S authorized fixed budget, no automatic overrun'
    policy_source=recipe['fallback_policy_source'];hashes=dict(config['source_sha256'])
    for file in frozen['versioned_sources']:
        actual=sha(ROOT/file)
        if actual!=frozen['versioned_sources'][file]:
            amend=read(BASE/'a3s/extension1280/AUTHORIZATION.json',{})
            assert amend.get('status')=='AUTHORIZED' and amend.get('source_sha256',{}).get(file)==actual,file
        hashes[file]=actual
    value=dict(namespace='runs/bomb_threat_v1/formal',namespace_id='A3S-v1',common=config['common'],arms={name:recipe},
        sources={str(2026112900+slot):source or policy_source},source_sha256=hashes,
        environment_steps_per_run=(start+length)*262144,chunk_start_update=start,chunk_updates=length,
        continuation_source=source,policy_source=policy_source,initialization_decision=config['initialization_decision'],
        worker_module='scripts.bomb_threat_seat5050_train',spawn_manifest=config['spawn_manifest'],
        A1_weights=config['A1_weights'],fixed_actor_source=config['fixed_actor_source'],
        rule_version='stock-window-v1',reward_version='bomb-threat-iter2-A3',training_reward_variant='bomb-threat-iter2-A3',
        slot=slot,chunk=chunk,iter2_arm='A3S',learner_seat_fraction=.5,seat5050_protocol='runs/bomb_threat_v1/a3s/PROTOCOL.json',
        initialization='latest A3 slot01 Actor only; new Critic/Adam/env/aux/RNG/ring;20 warmup;40 Actor ramp')
    atomic_json(path,value);return path


def apply_switch(slot,entry,done,chunk):
    auth=read(BASE/'a3s/AUTHORIZATION.json',{})
    if slot not in (4,7) or entry.get('iter2_arm')=='A3S' or auth.get('status')!='AUTHORIZED_READY':return None
    out=BASE/'formal'/entry['run'];complete=read(out/'COMPLETE.json',{})
    if not complete or done.get('state_sha256')!=complete.get('state_sha256'):return None
    cp=ROOT/done['checkpoint']
    assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    from scripts.bomb_threat_iter2_schedule import _latest_a3_policy_source
    source,lineage=_latest_a3_policy_source(slot)
    cfg=read(BASE/'iter2/TRAINING_PROTOCOL.json');recipe=dict(cfg['arms']['A3'],iter2_arm='A3S',
        fallback_policy_source=source,learner_seat_fraction=.5)
    path=make_protocol(slot,chunk,None,recipe)
    lineage.update(target_run=path.stem,decision='.hermes/DECISION_seat5050_20261029_1720.md',
        reason='A3S_PHYSICAL_SEAT5050_AT_COMPLETE_CHUNK',formal_updates_discarded=0)
    proto=read(path);proto.update(pbt_lineage=lineage,supersedes_run=entry['run']);atomic_json(path,proto)
    atomic_json(out/'A3S_SUPERSEDED.json',dict(status='COMPLETE_CHUNK_TO_A3S',last_atomic=done,successor=path.stem,
        lineage=lineage,formal_updates_discarded=0,training_worker_signals=0,unix=time.time()))
    target=BASE/'formal'/path.stem;target.mkdir(exist_ok=True)
    atomic_json(target/'PBT_HANDOFF.json',dict(status='PENDING_FRESH_INIT',lineage=lineage,unix=time.time()))
    atomic_json(BASE/'a3s'/f'SLOT{slot:02d}_HANDOFF.json',dict(status='A3S_PROTOCOL_READY',old_run=entry['run'],
        source=source,last_atomic=done,successor=path.stem,lineage=lineage,protocol_sha256=sha(path),
        formal_updates_discarded=0,training_worker_signals=0,unix=time.time()))
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=source,iter2_arm='A3S',role='A3S_seat5050',
        old_boundary=done['checkpoint'],pbt_lineage=lineage,supersedes=entry['run'])


def terminal(slot,entry,done):
    if entry.get('iter2_arm')!='A3S':return False
    limit=budget_limit()
    if done['update']<limit:return False
    # Async CPU can finish while other seven lanes continue. No unconstrained
    # A3S chunk may slip past640 while its mandatory verdict is not available.
    for split in ('dev','holdout'):
        name=f"{entry['run']}_u{limit:06d}"+('_holdout' if split=='holdout' else '')
        q=BASE/'eval_requests'/f'{name}.json'
        if not q.exists():atomic_json(q,dict(name=name,checkpoint=done['checkpoint'],split=split,
            rule_version='stock-window-v1',reward_version='bomb-threat-iter2-A3',iter2_arm='A3S',reason='A3S frozen640 terminal seat verdict'))
    atomic_json(BASE/'a3s'/f'SLOT{slot:02d}_TERMINAL.json',dict(status='FIXED_BUDGET_COMPLETE_AWAITING_SEAT_VERDICT',
        last_atomic=done,worker_absent=True,formal_updates=limit,formal_environment_steps=limit*262144,
        successor='no promotion; strict dev+holdout seat verdict; at1280 FAIL stop A3S and restore A3 original recipe',unix=time.time()))
    from scripts.bomb_threat_seat5050_verdict import refresh
    refresh(slot,entry,done)
    return True
