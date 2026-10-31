"""Frozen640 A3S seat verdict; independent CPU eval never changes old gates."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def read(p,default=None):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default


def compare(candidate,control):
    result={};missing=[]
    for split in ('dev','holdout'):
        c=candidate.get(split,{}).get('by_seat',{});r=control.get(split,{}).get('by_seat',{})
        if any(not x.get(s) or x[s].get('games')!=32 for x in (c,r) for s in ('P0','P1')):
            missing.append(split);continue
        diff={s:(c[s]['win']-c[s]['loss'])-(r[s]['win']-r[s]['loss']) for s in ('P0','P1')}
        zero=sum(c[s]['zero_bomb_games'] for s in ('P0','P1'));rz=sum(r[s]['zero_bomb_games'] for s in ('P0','P1'))
        conditions=dict(P1_win_minus_loss_plus5=diff['P1']>=5,P0_win_minus_loss_drop_at_most3=diff['P0']>=-3,
            total_single_zero_not_increased=zero<=rz)
        result[split]=dict(passed=all(conditions.values()),conditions=conditions,win_minus_loss_difference=diff,
            candidate_single_zero=zero,control_single_zero=rz)
    return dict(status='PENDING' if missing else 'PASS' if all(x['passed'] for x in result.values()) else 'FAIL',
        split_results=result,missing_splits=missing,comparison='frozen terminal; each split separately; no peak selection')


def request(cp,arm,split):
    p=Path(cp);name=f'{p.parent.name}_{p.name}'+('_holdout' if split=='holdout' else '')
    q=BASE/'eval_requests'/f'{name}.json'
    if not q.exists():atomic_json(q,dict(name=name,checkpoint=cp,split=split,rule_version='stock-window-v1',
        reward_version='bomb-threat-iter2-A3',iter2_arm=arm,reason='A3S frozen640 seat comparison'))
    return name


def refresh(slot,entry,done):
    out=BASE/'a3s' if done['update']==640 else BASE/'a3s'/f'verdict{done["update"]}';out.mkdir(parents=True,exist_ok=True);frozen=out/f'SLOT{slot:02d}_COMPARATOR.json'
    if not frozen.exists():
        # Predeclared latest control commit no later than target terminal time.
        # This point is selected by time, never its evaluation score.
        cutoff=done['metrics']['unix'];candidates=[]
        for path in (BASE/'formal').glob('bt_slot01_*/u*/COMMITTED.json'):
            r=read(path,{})
            if r.get('passed') and r.get('metrics',{}).get('unix',float('inf'))<=cutoff and r.get('update',0)%40==0:
                proto=read(BASE/'protocols'/(path.parent.parent.name+'.json'),{})
                if proto.get('iter2_arm')=='A3':candidates.append(r)
        if not candidates:return dict(status='PENDING',reason='control atomic checkpoint unavailable')
        cp=max(candidates,key=lambda x:x['metrics']['unix']);assert sha(ROOT/cp['checkpoint']/'policy.pt')==cp['policy_sha256']
        atomic_json(frozen,dict(status='FROZEN',selection='latest slot01 A3 atomic cadence<=A3S terminal training time',
            cutoff_unix=cutoff,checkpoint=cp['checkpoint'],policy_sha256=cp['policy_sha256'],update=cp['update'],unix=time.time()))
    control=read(frozen);metrics={'candidate':{},'control':{}};sources={}
    from scripts.bomb_threat_gate_report import recompute_natural
    for label,cp,arm in [('candidate',done['checkpoint'],'A3S'),('control',control['checkpoint'],'A3')]:
        for split in ('dev','holdout'):
            name=request(cp,arm,split);p=BASE/'eval'/name
            if (p/'NATURAL_COMPLETE.json').exists():
                marker,m=recompute_natural(p)
                expected=done['policy_sha256'] if label=='candidate' else control['policy_sha256']
                assert marker['policy_sha256']==expected and marker['split']==split
                metrics[label][split]=m['greedy'];sources[label+'_'+split]=sha(p/'NATURAL_COMPLETE.json')
    v=compare(metrics['candidate'],metrics['control']);v.update(slot=slot,run=entry['run'],checkpoint=done['checkpoint'],
        comparator=control,metrics=metrics,source_sha256=sources,decision='.hermes/DECISION_seat5050_20261029_1720.md',unix=time.time())
    atomic_json(out/f'SLOT{slot:02d}_VERDICT.json',v)
    if v['status'] in ('PASS','FAIL'):
        notification=BASE/'notifications'/f'A3S_SLOT{slot:02d}_U{done["update"]}_{v["status"]}.json'
        if not notification.exists():
            atomic_json(notification,v)
            from scripts.iter_pbt_owner import notify
            notify(f'A3S_SLOT{slot:02d}_VERDICT',f'A3S slot{slot:02d}固定640更新dev+holdout分座位判定{v["status"]}，完整原子点保留；未晋升BEST/网页。')
    return v
