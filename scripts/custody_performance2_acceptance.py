"""Observe restored real PPO updates, separately from micro/collector probes."""
import json
import statistics
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

BASE = Path('runs/bun_native_v2_20261030')
OUT = BASE / 'custody_performance_v2'
R3 = Path('runs/r3_items_route_20261030')
_VERIFIED_FILES = {}


def verify_file(path, expected):
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, expected)
    if _VERIFIED_FILES.get(path) != signature:
        assert sha(path) == expected
        _VERIFIED_FILES[path] = signature


def read(path):
    return json.loads(Path(path).read_text())


def rows(run):
    return update_rows(BASE / 'formal' / run / 'updates.jsonl')


def update_rows(path):
    # A writer may be between write() and newline/flush. Do not hide malformed
    # completed records, or fail a healthy worker over an unfinished last row.
    return [json.loads(line) for line in Path(path).read_text().splitlines(keepends=True)
            if line.endswith('\n')]


def warm_rows(history, boundary, *, activated=None):
    after = [r for r in history if r['new_update'] > boundary][2:]
    if activated is not None:
        # Exclude an update that straddled control activation. Its end timestamp
        # alone does not establish that it consumed the new scheduling mode.
        after = [r for r in after if r['unix'] - r['cycle_seconds'] >= activated]
    return after[-5:]


def first_atomic(folder, boundary, receipt):
    restore = read(folder / receipt)
    assert all(restore.get(k) for k in ('passed', 'fresh_process', 'next_step_exact',
        'Adam_exact', 'full_env_aux_RNG_exact', 'ring_exact'))
    cp = folder / f'u{boundary+1:06d}'
    if not (cp / 'COMMITTED.json').exists():
        return None
    if not (folder / f'NEXT_STEP_u{boundary+1:06d}.json').exists():
        return None
    committed = read(cp / 'COMMITTED.json')
    verify_file(cp / 'state.pkl', committed['state_sha256'])
    verify_file(cp / 'policy.pt', committed['policy_sha256'])
    row = next(r for r in update_rows(folder / 'updates.jsonl') if r['new_update'] == boundary + 1)
    assert row['finite'] and min(row['update_l2'].values()) > 0
    assert row['actual_environment_steps_this_update'] == 262144
    return dict(restore=restore, checkpoint=str(cp), committed=committed,
        update=boundary+1, actual_environment_steps=row['actual_environment_steps_this_update'],
        actor_critic_update_l2=row['update_l2'], finite=True)


def related_throughput():
    ready=OUT/'ATOMIC_SWITCH_READY.json'
    if not ready.exists():return None
    switch=read(ready);measurements={};r3ready=True;first_updates={}
    control=R3/'controls/GPU_STAGGER.json'
    schedule=read(control) if control.exists() else None
    r3slots=read(R3/'R3_SLOTS.json')
    for arm,entry in r3slots.items():
        folder=R3/'formal'/entry['run'];f=folder/'updates.jsonl'
        history=update_rows(f)
        after=[r for r in history if r['new_update']>switch['R3_states'][arm]['update']]
        first_updates['R3_'+arm]=first_atomic(folder,switch['R3_states'][arm]['update'],'R3_PERFORMANCE2_RESTORE.json') if (folder/'R3_PERFORMANCE2_RESTORE.json').exists() else None
        warm=warm_rows(history,switch['R3_states'][arm]['update'])
        if len(warm)<3:r3ready=False;continue
        assert all(r['finite'] and min(r['update_l2'].values())>0 for r in warm)
        measurements['R3_'+arm]=dict(run=entry['run'],gpu=entry['gpu'],points=[r['new_update'] for r in warm],
            cycle_seconds=statistics.median(r['cycle_seconds'] for r in warm),
            rollout_seconds=statistics.median(r['rollout_seconds'] for r in warm),
            env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in warm),
            stagger_wait_seconds=statistics.median(r.get('stagger_wait_seconds',0) for r in warm),
            throughput_includes_stagger_wait=True)
        if schedule:
            post=warm_rows(history,switch['R3_states'][arm]['update'],activated=schedule['unix'])
            measurements['R3_'+arm]['post_stagger']=dict(points=[r['new_update'] for r in post],
                status='MEASURED' if len(post)>=3 else 'WAITING_3_POST_CONTROL_UPDATES',
                cycle_seconds=statistics.median(r['cycle_seconds'] for r in post) if post else None,
                env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in post) if post else None,
                stagger_wait_seconds=statistics.median(r.get('stagger_wait_seconds',0) for r in post) if post else None)
    legacy=read('runs/bomb_threat_v1/SLOTS.json')
    for arm in ('4','7'):
        entry=legacy[arm];f=Path('runs/bomb_threat_v1/formal')/entry['run']/'updates.jsonl'
        history=update_rows(f)[-5:]
        measurements['R1_slot'+arm]=dict(run=entry['run'],points=[r['new_update'] for r in history],
            cycle_seconds=statistics.median(r['cycle_seconds'] for r in history),
            env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in history),
            last_update_age_seconds=time.time()-history[-1]['unix'])
    for arm,entry in read(BASE/'R2_SLOTS.json').items():
        if arm not in ('b','c'):continue
        history=rows(entry['run']);warm=warm_rows(history,switch['states'][arm]['update'])
        folder=BASE/'formal'/entry['run']
        first_updates['R2_'+arm]=first_atomic(folder,switch['states'][arm]['update'],'PERFORMANCE2_RESTORE.json') if (folder/'PERFORMANCE2_RESTORE.json').exists() else None
        if len(warm)<3:continue
        measurements['R2_'+arm]=dict(run=entry['run'],points=[r['new_update'] for r in warm],
            cycle_seconds=statistics.median(r['cycle_seconds'] for r in warm),
            env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in warm))
    if r3ready and all(e['gpu']==2 for e in r3slots.values()) and any(measurements['R3_'+a]['cycle_seconds']>60 for a in ('a','b')) and not control.exists():
        atomic_json(control,dict(mode='whole_update_mutex',gpu=2,authorization=str(OUT/'R3_FAST2_AUTHORIZATION.json'),
            reason='post-fast2>=3warm formal updates, sharedGPU2 median cycle>60s; stagger preserves R1/R2 throughput',
            measured={a:measurements['R3_'+a] for a in ('a','b')},waiting_counted_in_cycle_seconds=True,unix=time.time()))
        with (OUT/'ACCEPTANCE_NOTIFY.log').open('ab') as log:
            subprocess.run(['jdme-notify','send','R3新custody切换后GPU2共卡暖态仍超过60秒，已按用户授权开启整更新错峰；不迁移到R1/R2共享卡，不改奖励/步数/Adam，等待时间计入有效env/s。'],stdout=log,stderr=subprocess.STDOUT,check=False)
    result=dict(status='WARM_OBSERVATION' if r3ready else 'WAITING_R3_WARM_UPDATES',measurements=measurements,
        first_formal_updates=first_updates,R3_stagger=read(control) if control.exists() else None,unix=time.time())
    atomic_json(OUT/'SIX_LINE_THROUGHPUT.json',result)
    if len(first_updates)==4 and all(first_updates.values()):
        status=read(OUT/'SWITCH_STATUS.json')
        status.update(status='SWITCH_APPLIED_WARM_MEASURED' if len(measurements)==6 else 'SWITCH_APPLIED_FIRST_UPDATES_VERIFIED',
            first_updates=first_updates,latest_throughput=str(OUT/'SIX_LINE_THROUGHPUT.json'),unix=time.time())
        atomic_json(OUT/'SWITCH_STATUS.json',status)
    return result


def refresh():
    ready = OUT / 'ATOMIC_SWITCH_READY.json'
    if not ready.exists():
        return None
    switch, slots = read(ready), read(BASE / 'R2_SLOTS.json')
    ref = rows(slots['a']['run'])
    ref = [r for r in ref if r['new_update'] > switch['states']['a']['update']][2:][-5:]
    if len(ref) < 5:
        return None
    reference = statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in ref)
    arms = {}
    for arm in ('b', 'c'):
        entry = slots[arm]
        folder = BASE / 'formal' / entry['run']
        rpath = folder / 'PERFORMANCE2_RESTORE.json'
        if not rpath.exists():
            return None
        restore = read(rpath)
        assert all(restore.get(k) for k in ('passed', 'fresh_process', 'next_step_exact', 'Adam_exact', 'full_env_aux_RNG_exact', 'ring_exact'))
        at = switch['states'][arm]['update']
        history = rows(entry['run'])
        after = [r for r in history if r['new_update'] > at]
        warm = after[2:][-5:]
        if len(warm) < 5:
            return None
        assert all(r['finite'] and min(r['update_l2'].values()) > 0 for r in after)
        atomic = folder / f'u{at+1:06d}'
        if not (atomic / 'COMMITTED.json').exists():
            return None
        cp = read(atomic / 'COMMITTED.json')
        verify_file(atomic / 'state.pkl', cp['state_sha256'])
        verify_file(atomic / 'policy.pt', cp['policy_sha256'])
        speed = statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in warm)
        arms[arm] = dict(run=entry['run'], restore=restore, first_new_atomic=cp,
            formal_updates=len(after), new_real_environment_steps=sum(r['actual_environment_steps_this_update'] for r in after),
            cold_updates=[dict(update=r['new_update'],cycle_seconds=r['cycle_seconds'],rollout_seconds=r['rollout_seconds']) for r in after[:2]],
            warm_points=[r['new_update'] for r in warm],env_per_second=speed,
            cycle_seconds=statistics.median(r['cycle_seconds'] for r in warm),
            rollout_seconds=statistics.median(r['rollout_seconds'] for r in warm),
            fraction_of_R2a=speed/reference,target_accepted=speed/reference>=.70,finite_actor_critic_nonzero=True)
    result = dict(status='PASS' if all(v['target_accepted'] for v in arms.values()) else 'TARGET_NOT_MET',
        R2a_env_per_second=reference,R2a_points=[r['new_update'] for r in ref],arms=arms,
        source='actual formal PPO update logs, not diagnostic rollout steps',
        shared_context_limit='b/c still share GPU4/7 with R1; a hasGPU3; no isolated hardware-causality claim',unix=time.time())
    atomic_json(OUT / 'FULL_UPDATE_ACCEPTANCE.json',result)
    return result


def notify_six_lines(side):
    if not side or len(side['measurements']) != 6 or not all(side['first_formal_updates'].values()):
        return
    measurements=side['measurements']
    staggered=bool(side['R3_stagger'])
    if staggered and any(measurements['R3_'+a].get('post_stagger',{}).get('status')!='MEASURED' for a in ('a','b')):
        return
    marker=OUT/('SIX_LINES_STAGGER_NOTIFIED.json' if staggered else 'SIX_LINES_NOTIFIED.json')
    if marker.exists():return
    values={name:(value['post_stagger'] if staggered and name.startswith('R3_') else value)
            for name,value in measurements.items()}
    message='custody第二版原子切换后正式暖态：'+', '.join(
        '%s %.0f env/s (%.1fs/update)'%(name,values[name]['env_per_second'],values[name]['cycle_seconds'])
        for name in ('R2_b','R2_c','R3_a','R3_b','R1_slot4','R1_slot7'))
    message+='；四路完整恢复和首新原子SHA通过，奖励/预算不变。'+('R3已错峰，等待计入有效吞吐。' if staggered else 'R3共卡未触发>60秒错峰阈值。')
    atomic_json(marker,dict(message=message,metrics=values,source=str(OUT/'SIX_LINE_THROUGHPUT.json'),unix=time.time()))
    with (OUT/'ACCEPTANCE_NOTIFY.log').open('ab') as log:
        subprocess.run(['jdme-notify','send',message],stdout=log,stderr=subprocess.STDOUT,check=False)
    with Path('reports/custody_performance_20261031.md').open('a') as f:
        f.write('\n## 六路正式暖态吞吐\n\n'+message+'\n')


def main():
    guard=lock(OUT/'ACCEPTANCE.lock')
    while True:
        switch=OUT/'SWITCH_STATUS.json'
        if switch.exists() and read(switch)['status'] in ('NOT_DEPLOYED_NO_FULL_COLLECTOR_BENEFIT','BLOCKED'):
            return
        side=related_throughput()
        notify_six_lines(side)
        result=refresh()
        if result:
            marker=OUT/(result['status']+'_NOTIFIED.json')
            if not marker.exists():
                b,c=result['arms']['b'],result['arms']['c']
                message='custody第二版正式更新验收%s：b %.0f/c %.0f env/s，为a的%.1f%%/%.1f%%；各>=5暖态点、Actor/Critic非零、完整恢复和首新原子SHA已核验。70%%目标%s，原奖励/预算不变。'%(result['status'],b['env_per_second'],c['env_per_second'],100*b['fraction_of_R2a'],100*c['fraction_of_R2a'],'通过' if result['status']=='PASS' else '未达到，健康训练继续，须按共卡/host证据继续诊断')
                atomic_json(marker,dict(message=message,acceptance_sha256=sha(OUT/'FULL_UPDATE_ACCEPTANCE.json'),unix=time.time()))
                with (OUT/'ACCEPTANCE_NOTIFY.log').open('ab') as log:
                    subprocess.run(['jdme-notify','send',message],stdout=log,stderr=subprocess.STDOUT,check=False)
                with Path('reports/custody_performance_20261031.md').open('a') as f:
                    f.write('\n## 第二版正式暖态验收\n\n'+message+'\n\n完整实数、cold/warm点、恢复和state/policy SHA：`'+str(OUT/'FULL_UPDATE_ACCEPTANCE.json')+'`。\n')
            if result['status']=='PASS' and side and side['status']=='WARM_OBSERVATION' and not side['R3_stagger']:
                return
        time.sleep(15)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        atomic_json(OUT/'ACCEPTANCE_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),training_untouched=True,unix=time.time()))
        raise
