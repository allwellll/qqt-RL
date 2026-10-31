"""Authorized amendment03: passive raw-evidence guard, full-chunk handoff only."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';OUT=BASE/'a3s/amendment03'

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def evaluate_points(records):
    rows=sorted((r for r in records if r['update']>=880 and r['update']%40==0),key=lambda r:r['update'])
    points=[]
    for r in rows:
        zero=r['P0']['zero_bomb_games']+r['P1']['zero_bomb_games']
        points.append(dict(**r,unilateral_zero_total=zero,qualifies=r['P0']['win_minus_loss']<=0 and zero>=12))
    triples=[points[i-2:i+1] for i in range(2,len(points)) if all(r['qualifies'] for r in points[i-2:i+1]) and [r['update'] for r in points[i-2:i+1]]==list(range(points[i]['update']-80,points[i]['update']+1,40))]
    return dict(status='SWITCH_AT_NEXT_FULL_CHUNK' if triples else 'OBSERVE',trigger=bool(triples),points=points,triples=triples)

def transition_scope(slot,arm,update,complete):return slot==4 and arm=='A3S' and update>=960 and update%320==0 and complete

def dev_greedy(out):
    """Accept exactly64 completed argmax games; categorical completion is independent."""
    from scripts.bomb_threat_eval import summarize
    req=read(BASE/'eval_requests'/(out.name+'.json'),{})
    if req.get('split','dev')!='dev' or not req.get('checkpoint'):return None
    cp=ROOT/req['checkpoint'];cm=read(cp/'COMMITTED.json',{})
    if not cm.get('passed'):return None
    model=out/'model/model.json'
    if not model.exists():return None
    assert read(model)['meta']['checkpoint_sha256']==cm['policy_sha256']
    mh=sha(model);rows=[];files={}
    for start in range(0,64,8):
        p=out/f'natural_greedy_{start:03d}.json';side=out/(p.name+'.outcome.json')
        if not p.exists() or not side.exists():return None
        a=read(p);d=read(side)
        assert a['model_json_sha256']==mh and a['rule_version']==d['rule_version']=='stock-window-v1'
        assert d['measurement_version']==2 and d['engine']=='production_JS' and not d['teacher'] and d['mode']=='greedy'
        assert a['simulator_sha256']==d['simulator_sha256']==sha(ROOT/'web/sim-stock-window-v1.js')
        eps=d['episodes'];primary=a['results']['natural_full2400']['episodes']
        assert len(eps)==len(primary)==8
        for e,r in zip(primary,eps):
            assert r['done'] and 0<r['ticks']<=2400 and e['seed']==r['seed'] and e['team']==r['seat'] and e['ticks']==r['ticks']
            assert e['causal_kill']==r['causal_kills'][r['seat']]
        rows+=eps;files[str(p.relative_to(ROOT))]=sha(p);files[str(side.relative_to(ROOT))]=sha(side)
    assert {r['seed'] for r in rows}==set(range(2026122900,2026122964))
    assert all(r['seat']==(r['seed']-2026122900)%2 for r in rows)
    return dict(update=cm['update'],checkpoint=req['checkpoint'],policy_sha256=cm['policy_sha256'],model_sha256=mh,files=files,**summarize(rows))

def refresh():
    protocol=read(OUT/'PROTOCOL.json',{})
    if protocol.get('status')!='AUTHORIZED':return {}
    assert sha(ROOT/protocol['instruction'])==protocol['instruction_sha256']
    rows=[]
    for out in (BASE/'eval').glob('bt_slot04_*_u*'):
        if out.name.endswith('_holdout'):continue
        name,u=out.name.rsplit('_u',1)
        if not u.isdigit() or int(u)<880 or int(u)%40:continue
        cfg=read(BASE/'protocols'/(name+'.json'),{})
        if cfg.get('iter2_arm')!='A3S':continue
        g=dev_greedy(out)
        if not g:continue
        assert g['games']==64 and all(g['by_seat'][s]['games']==32 for s in ['P0','P1'])
        rows.append(dict(update=g['update'],checkpoint=g['checkpoint'],P0=g['by_seat']['P0'],P1=g['by_seat']['P1'],policy_sha256=g['policy_sha256'],model_sha256=g['model_sha256'],files=g['files']))
    result=evaluate_points(rows);result.update(unix=time.time(),protocol_sha256=sha(OUT/'PROTOCOL.json'))
    atomic_json(OUT/'GUARD_STATUS.json',result)
    if result['trigger'] and not (OUT/'SWITCH_REQUEST.json').exists():
        atomic_json(OUT/'SWITCH_REQUEST.json',dict(status='AUTHORIZED_TRIGGERED',trigger=result['triples'][0],unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('SLOT04_AMENDMENT03_TRIGGER','slot04 A3S从u880起连续3 dev点P0胜-负≤0且单方零泡≥12；下一完整chunk切P0 A3，完整state/seed保留，slot07不变。')
    return result

def apply_switch(slot,entry,done,chunk):
    if slot!=4 or entry.get('iter2_arm')!='A3S':return None
    refresh()
    request=read(OUT/'SWITCH_REQUEST.json',{})
    complete=read(BASE/'formal'/entry['run']/'COMPLETE.json',{})
    if request.get('status')!='AUTHORIZED_TRIGGERED' or not transition_scope(slot,'A3S',done['update'],complete.get('state_sha256')==done.get('state_sha256')):return None
    cp=ROOT/done['checkpoint']
    assert sha(cp/'state.pkl')==done['state_sha256'] and sha(cp/'policy.pt')==done['policy_sha256']
    config=read(BASE/'iter2/TRAINING_PROTOCOL.json');recipe=dict(config['arms']['A3'],fallback_policy_source={k:done[k] for k in ['checkpoint','policy_sha256','update']})
    from scripts.bomb_threat_iter2_schedule import make_protocol
    # Dedicated worker implements lossless in-flight seat drain to P0 resets.
    path=make_protocol(slot,chunk,done,recipe)
    proto=read(path);proto.update(worker_module='scripts.bomb_threat_a3s_to_p0_train',initialization='full current state preserved; in-flight physical seat preserved until natural terminal; all new episodes P0; no warmup/reset',a3s_to_p0_amendment03=True)
    for f in ['scripts/bomb_threat_a3s_to_p0_train.py','scripts/bomb_threat_a3s_to_p0_collector.py']:
        proto['source_sha256'][f]=sha(ROOT/f)
    atomic_json(path,proto)
    receipt=dict(status='FULL_STATE_A3S_TO_P0_AT_COMPLETE_CHUNK',source=done,successor=path.stem,seed=2026112904,preserved=['Actor','Critic','Adam','env','aux','RNG','ring','counters'],seat_migration='existing P1 episodes finish naturally with their own ledgers; new episodes all P0; transient P1 exposure reported',worker_signals=0,discarded_updates=0,unix=time.time())
    atomic_json(OUT/'HANDOFF.json',receipt)
    return dict(run=path.stem,chunk=chunk,gpu=slot,source=done,iter2_arm='A3',role='A3S_AMENDMENT03_TO_P0_FULLSTATE',old_boundary=done['checkpoint'])
