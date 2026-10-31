"""Passive 200-update buckets and the authorized 640-update rollback test."""
import json,time
from pathlib import Path
import statistics
from scripts.population_run_state import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def read(p,default=None):
    # Files are atomically replaced by the eval owner; an exists/read race must
    # be treated as temporarily unavailable evidence, not as a GPU-owner crash.
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default


def comparison(arm_rows,controls):
    fields=('win_minus_loss','causal_kills','bombs_per_game')
    if len(arm_rows)<3 or any(len(controls.get(s,[]))<3 for s in ('0','3')):
        return dict(status='PENDING_MATCHED_DEV_DATA',arm_points=len(arm_rows),control_points={s:len(controls.get(s,[])) for s in ('0','3')})
    a={k:statistics.mean(r['metrics'][k] for r in arm_rows) for k in fields}
    # Balance the two unchanged control slots, rather than overweighting the
    # slot whose CPU evaluations happened to finish first.
    c={k:statistics.mean(statistics.mean(r['metrics'][k] for r in controls[s]) for s in ('0','3')) for k in fields}
    worse={k:a[k]<c[k] for k in fields}
    return dict(status='RESTORE_ORIGINAL_RECIPE' if all(worse.values()) else 'CONTINUE',arm_mean=a,
        control_equal_slot_mean=c,strictly_worse=worse,arm_points=len(arm_rows),control_points={s:len(controls[s]) for s in ('0','3')},
        inference='operational dev rollback condition, not paired causal proof or BEST selection')


def a3_early_observation(arm_rows, records):
    """11:20 authorization; three cadence points in u200..320, not a peak test.

    Latest preceding control per slot within 600s of the arm's training time;
    slots 00/03/06 are equally weighted. Missing controls never count as zero.
    """
    points=[]
    for arm in sorted(arm_rows,key=lambda r:r['update']):
        if arm['update'] not in (200,240,280,320):continue
        controls={}
        for slot in ('0','3','6'):
            eligible=[r for r in records if r['slot']==slot and not r.get('arm') and r.get('train_unix')
                and arm.get('train_unix') and 0<=arm['train_unix']-r['train_unix']<=600]
            if eligible:controls[slot]=max(eligible,key=lambda r:r['train_unix'])
        mean=statistics.mean(r['metrics']['win_minus_loss'] for r in controls.values()) if len(controls)==3 else None
        met=arm['metrics']
        points.append(dict(update=arm['update'],loss=met['loss'],win_minus_loss=met['win_minus_loss'],
            control_equal_slot_mean=mean,control_markers={s:r.get('marker') for s,r in controls.items()},
            qualifies=met['loss']>=6 and met['win_minus_loss']<mean-5 if mean is not None else None))
    trigger=[]
    for i in range(2,len(points)):
        tail=points[i-2:i+1]
        if [p['update'] for p in tail] not in ([200,240,280],[240,280,320]):continue
        if all(p['qualifies'] is True for p in tail):trigger=tail;break
    status='RESTORE_ORIGINAL_RECIPE' if trigger else ('PENDING_A3_EARLY_DEV' if len(points)<3 or any(p['qualifies'] is None for p in points) else 'CONTINUE')
    return dict(status=status,points=points,trigger_points=trigger,
        rule='u200..320 three consecutive cadence40 dev greedy points: loss>=6 and win-loss<equal_slot_control-5',
        control_slots=['0','3','6'],max_preceding_control_age_seconds=600,
        execution='next completed atomic chunk; no in-flight worker signal; u640 rule retained')


def refresh():
    records=[]
    for marker in (BASE/'eval').glob('*/NATURAL_COMPLETE.json'):
        d=read(marker,{})
        if d.get('split')!='dev' or d.get('seed')!=2026122900 or d.get('measurement_version')!=2 or not d.get('checkpoint'):continue
        cp=ROOT/d['checkpoint'];name=cp.parent.name
        if not name.startswith('bt_slot'):continue
        if any((p/f).exists() for p in (marker.parent,cp.parent) for f in ('EXCLUDED_INVALID.json','EXCLUDED_EVALUATION.json','INVALID_STARTING_POINT.json')):continue
        proto=read(BASE/'protocols'/(name+'.json'),{});cm=read(cp/'COMMITTED.json',{});m=d['natural_js_full2400']['greedy']
        if not cm.get('passed') or cm.get('policy_sha256')!=d.get('policy_sha256'):continue
        metrics=dict(m);metrics['win_minus_loss']=m['win']-m['loss']
        records.append(dict(slot=str(proto.get('slot',int(name[7:9]))),arm=proto.get('iter2_arm'),
            fallback=bool(proto.get('iter2_fallback')),entropy_restored_1320=bool(proto.get('entropy_restored_1320')),
            update=cm.get('update',int(cp.name[1:])),
            train_unix=cm.get('metrics',{}).get('unix'),metrics=metrics,
            checkpoint=d['checkpoint'],marker=str(marker.relative_to(ROOT)),marker_sha256=sha(marker)))
    results={};buckets=[]
    for slot,arm in [('4','A1'),('7','A2'),('1','A3')]:
        rows=[r for r in records if r['slot']==slot and r['arm']==arm and not r['fallback'] and not r['entropy_restored_1320']]
        for bucket in sorted({max(0,(r['update']-1)//200) for r in rows}):
            rr=[r for r in rows if max(0,(r['update']-1)//200)==bucket]
            buckets.append(dict(slot=slot,arm=arm,formal_update_range=[bucket*200+1,(bucket+1)*200],points=len(rr),
                means={k:statistics.mean(r['metrics'][k] for r in rr) for k in ('win_minus_loss','causal_kills','bombs_per_game','zero_bomb_games')},
                markers=[r['marker'] for r in rr]))
        tail=[r for r in rows if 440<=r['update']<=640 and r['train_unix']]
        if not any(r['update']==640 for r in tail):
            results[slot]=dict(status='PENDING_U640_DEV',tail_points=len(tail));continue
        lo=min(r['train_unix'] for r in tail);hi=max(r['train_unix'] for r in tail)
        controls={s:[r for r in records if r['slot']==s and not r['arm'] and r['train_unix'] and lo<=r['train_unix']<=hi] for s in ('0','3')}
        results[slot]=dict(comparison(tail,controls),formal_window=[440,640],train_unix_range=[lo,hi],
            arm_markers=[r['marker'] for r in tail],control_markers={s:[r['marker'] for r in rr] for s,rr in controls.items()})
    a3_rows=[r for r in records if r['arm']=='A3' and not r['fallback']]
    early=a3_early_observation(a3_rows,records)
    if early['status']=='RESTORE_ORIGINAL_RECIPE':
        results['1']=dict(status='RESTORE_ORIGINAL_RECIPE',reason='STEER1120_A3_EARLY',early_observation=early,
            u640_comparison=results.get('1'))
    watch={}
    authorization=read(BASE/'iter2/STEER1320_AUTHORIZATION.json',{})
    if authorization.get('status')=='AUTHORIZED_READY':
        from scripts.bomb_threat_steer1320 import a3_bomb_kill_observation
        watch=a3_bomb_kill_observation(a3_rows,records)
        results['1']=dict(status='RESTORE_ORIGINAL_RECIPE' if watch['status']=='RESTORE_ORIGINAL_RECIPE' else 'CONTINUE',
            reason='STEER1320_A3_BOMB_KILL' if watch['status']=='RESTORE_ORIGINAL_RECIPE' else 'STEER1320_CONTINUE_A3',
            original_u640_comparison=results.get('1'),bomb_kill_observation=watch)
    a3_points=[]
    for r in sorted(a3_rows,key=lambda r:r['update']):
        g=read(ROOT/r['marker'].rsplit('/',1)[0]/'GATE_METRICS.json',{})
        hack=g.get('hack_monitor',{})
        a3_points.append(dict(update=r['update'],checkpoint=r['checkpoint'],marker=r['marker'],
            loss=r['metrics'].get('loss'),self_deaths=r['metrics'].get('self_deaths'),
            draw_kill_difference=r['metrics'].get('draw_kill_difference'),
            threat_paid_fraction=hack.get('new_shaping_reward_fraction_total',hack.get('threat_reward_fraction_total'))))
    value=dict(status='PASSIVE_DEV_ANALYSIS',rollback_decisions=results,buckets=buckets,
        a3_early_observation=early,a3_points=a3_points,a3_bomb_kill_observation=watch,
        sparse_or_missing_data_never_zero_filled=True,holdout_used_for_tuning=False,unix=time.time())
    atomic_json(BASE/'iter2/DEV_BUCKETS_AND_ROLLBACK.json',value);return value


if __name__=='__main__':refresh()
