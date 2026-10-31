"""20:40 authorized passive monitor and exact full-chunk interventions."""
import json,time,statistics
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';OUT=BASE/'a3s/extension1280'


def read(p,default=None):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default


def evaluate_points(records,cutoff):
    """P0 primary as specified by the red P0 trend; all64 reported alongside."""
    points=[]
    records=sorted(records,key=lambda r:r['update'])
    for i,r in enumerate(records):
        if r['train_unix']<=cutoff:continue
        history=records[max(0,i-8):i];m=r['P0']
        means={k:statistics.mean(h['P0'][k] for h in history) if len(history)==8 else None for k in ('bombs_per_game','causal_kills','win_minus_loss')}
        complete=len(history)==8 and [h['update'] for h in history]+[r['update']]==list(range(r['update']-320,r['update']+1,40))
        qualifies=m['bombs_per_game']>means['bombs_per_game']+10 and m['causal_kills']<=means['causal_kills'] if complete else None
        points.append(dict(**r,previous8=means,history_checkpoints=[h['checkpoint'] for h in history],qualifies=qualifies))
    pairs=[points[i-1:i+1] for i in range(1,len(points)) if points[i]['update']-points[i-1]['update']==40 and all(p['qualifies'] is True for p in points[i-1:i+1])]
    return dict(status='REDUCE_THREAT_AT_NEXT_FULL_CHUNK' if pairs else 'RED_OBSERVE',points=points,trigger_pairs=pairs)


def refresh():
    auth=read(OUT/'MONITOR_PROTOCOL.json',{})
    if auth.get('status')!='AUTHORIZED':return {}
    records=[]
    from scripts.bomb_threat_gate_report import recompute_natural
    for marker in (BASE/'eval').glob('bt_slot01_*/NATURAL_COMPLETE.json'):
        doc=read(marker);cp=ROOT/doc['checkpoint'];update=int(cp.name[1:])
        if doc.get('split')!='dev' or doc.get('seed')!=2026122900 or update%40:continue
        cfg=read(BASE/'protocols'/(cp.parent.name+'.json'),{})
        if cfg.get('iter2_arm')!='A3':continue
        cm=read(cp/'COMMITTED.json',{})
        if not cm.get('passed') or cm.get('policy_sha256')!=doc['policy_sha256']:continue
        if cm['metrics']['unix']<=auth['monitor_from_unix']-10000:continue
        _,metrics=recompute_natural(marker.parent);m=metrics['greedy']
        records.append(dict(update=update,checkpoint=doc['checkpoint'],train_unix=cm['metrics']['unix'],marker=str(marker.relative_to(ROOT)),marker_sha256=sha(marker),P0=m['by_seat']['P0'],all64=m))
    dedup={r['update']:r for r in sorted(records,key=lambda r:r['train_unix'])}
    result=evaluate_points(list(dedup.values()),auth['monitor_from_unix']);result.update(protocol_sha256=sha(OUT/'MONITOR_PROTOCOL.json'),unix=time.time(),automatic_scope='only threat x5 to x2.5 at completed chunk, other terms unchanged; no PBT copy')
    atomic_json(OUT/'SLOT01_RED_OBSERVATION.json',result)
    request=OUT/'SLOT01_THREAT_REDUCTION_REQUEST.json'
    if result['trigger_pairs'] and not request.exists():
        atomic_json(request,dict(status='AUTHORIZED_TRIGGERED',trigger=result['trigger_pairs'][0],source_sha256=sha(OUT/'SLOT01_RED_OBSERVATION.json'),target_threat_multiplier=2.5,topology_multiplier=5.,chain_multiplier=5.,cap=.6,unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('SLOT01_2040_TWO_POINT_TRIGGER','slot01按20:40决定连续两点P0放泡>前8点均值+10且击杀不升；下个完整chunk仅威胁×5降×2.5，其他不动。')
    # Complete the original640 verdict asynchronously without gating extension.
    from scripts.bomb_threat_seat5050_verdict import refresh as verdict
    for slot in (4,7):
        receipt=read(BASE/'a3s'/f'SLOT{slot:02d}_TERMINAL.json',{})
        if receipt.get('last_atomic',{}).get('update')==640:
            done=receipt['last_atomic'];verdict(slot,dict(run=Path(done['checkpoint']).parent.name),done)
    return result


def boundary_recipe(slot,recipe,done):
    request=read(OUT/'SLOT01_THREAT_REDUCTION_REQUEST.json',{})
    if slot!=1 or recipe.get('iter2_arm')!='A3' or recipe.get('threat_multiplier')==2.5 or request.get('status')!='AUTHORIZED_TRIGGERED':return recipe
    assert done['update']%320==0 or (ROOT/done['checkpoint']).parent.joinpath('COMPLETE.json').exists()
    atomic_json(OUT/'SLOT01_REDUCTION_BOUNDARY.json',dict(status='FULL_STATE_CONTINUATION',source=done,only_changed='threat coefficient .10 to .05; topology .20 chain .05 cap .60 unchanged',no_critic_reset=True,training_signals=0,unix=time.time()))
    return dict(recipe,threat_multiplier=2.5)


def terminal_reversion(slot,entry,done,chunk):
    """After1280 strict FAIL, restore original P0 A3 init convention once."""
    if entry.get('iter2_arm')!='A3S' or done['update']!=1280:return None
    from scripts.bomb_threat_seat5050_verdict import refresh as verdict
    result=verdict(slot,entry,done)
    if result['status']!='FAIL':return None
    cp=ROOT/done['checkpoint']
    assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json');source={k:done[k] for k in ('checkpoint','policy_sha256','update')}
    recipe=dict(config['arms']['A3'],fallback_policy_source=source)
    from scripts.bomb_threat_iter2_schedule import make_protocol
    path=make_protocol(slot,chunk,None,recipe)
    atomic_json(OUT/f'SLOT{slot:02d}_RESTORE_A3.json',dict(status='AUTHORIZED_A3S1280_FAIL_TO_ORIGINAL_A3',verdict=result,
        source=done,successor=path.stem,original_recipe=recipe,initialization='final Actor retained; original A3 policy-only init20Critic warmup; new env/aux/RNG/ring, no fabricated carry',
        reason='mixed-seat auxiliary state cannot be silently reinterpreted as P0',worker_signals=0,unix=time.time()))
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=source,iter2_arm='A3',role='A3S1280_FAIL_TO_ORIGINAL_A3',old_boundary=done['checkpoint'])
