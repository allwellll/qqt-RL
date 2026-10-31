"""Passive acceptance and authorized R2-a resource guard; no GPU launcher."""
import json
import os
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'runs/slot01_copy07_gpu3_20261031'
R = ROOT / 'runs/bun_native_v2_20261030'
BT = ROOT / 'runs/bomb_threat_v1'


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def rows(path, count=25):
    if not path.exists():
        return []
    # Do not read every historical training row for a live throughput probe.
    with path.open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 1600000))
        data = f.read().splitlines()
    parsed = []
    for line in data[-count-2:]:
        try:
            value = json.loads(line)
        except ValueError:
            continue  # first partial line or concurrently-written final row
        if value.get('finite') and value.get('cycle_seconds', 0) > 0:
            parsed.append(value)
    return parsed[-count:]


def notify(key, message):
    dest = B / 'notifications' / (key + '.json')
    if read(dest, {}).get('returncode') == 0:
        return
    result = subprocess.run(['jdme-notify', 'send', message], capture_output=True, text=True)
    atomic_json(dest, dict(returncode=result.returncode, message=message, unix=time.time()))


def accept(root, run, source, dest):
    if dest.exists():
        return read(dest)
    out = root / 'formal' / run
    start = source['update']
    smoke, restore = read(out/'SMOKE.json', {}), read(out/'RESTORE.json', {})
    if not smoke.get('passed') or not restore.get('passed'):
        return None
    candidates = [r for r in rows(out/'updates.jsonl', 250) if r['new_update'] > start+2]
    if not candidates:
        return None
    first = candidates[0]
    assert first['new_update'] == start+3
    assert all(first['update_l2'][x] > 0 for x in ['actor', 'critic'])
    cp = out / f"u{first['new_update']:06d}"
    committed = read(cp/'COMMITTED.json')
    if not committed:
        return None
    assert committed['update'] == first['new_update']
    assert sha(cp/'state.pkl') == committed['state_sha256']
    assert sha(cp/'policy.pt') == committed['policy_sha256']
    token = out / f"NEXT_STEP_u{first['new_update']:06d}.json"
    if not token.exists():
        return None
    result = dict(status='PASS', run=run, source=source, first_update=first,
        target_atomic=committed, next_step_token_sha256=sha(token),
        env_per_second=first['actual_environment_steps_this_update']/first['cycle_seconds'],
        smoke_sha256=sha(out/'SMOKE.json'), restore_sha256=sha(out/'RESTORE.json'),
        source_handoff=read(out/'SOURCE_HANDOFF.json'), unix=time.time())
    atomic_json(dest, result)
    notify(dest.stem, f"21:10任务 {run} 首个正式u{first['new_update']}：finite、Actor/Critic非零、完整state/policy SHA及next-step token验收，{result['env_per_second']:.0f} env/s；冷启动点不冒充warm吞吐。")
    return result


def resource_guard(accepted):
    if not accepted:
        return
    baseline = read(B/'REFERENCE_BASELINES.json')['baselines']
    state = read(B/'RESOURCE_GUARD.json', dict(last={}, strikes={}))
    table = read(R/'R2_SLOTS.json')
    own = table['a']
    own_rows = rows(R/'formal'/own['run']/'updates.jsonl')
    if not own_rows or own_rows[-1]['new_update'] < accepted['source']['update']+8:
        return
    comparisons = {}
    for key, base in baseline.items():
        root, slot = key.rsplit(':', 1)
        table_name = 'SLOTS.json' if root.endswith('bomb_threat_v1') else 'R2_SLOTS.json'
        entry = read(ROOT/root/table_name)[slot]
        history = rows(ROOT/root/'formal'/entry['run']/'updates.jsonl', 18)
        if len(history) < 6:
            continue
        from scripts.r3_selfplay_owner_20261031 import reference_window_snapshot
        value = reference_window_snapshot(history)
        phase = str(value['update']%2)
        fixed = base['phase_baselines'][phase]
        ratio = value['env_per_second']/fixed['env_per_second']
        minimum = .85 if root.endswith('bomb_threat_v1') else .95
        last = state['last'].get(key, base['update'])
        if value['update']-last >= 5:
            state['strikes'][key] = state['strikes'].get(key, 0)+1 if ratio < minimum else 0
            state['last'][key] = value['update']
        comparisons[key] = dict(run=entry['run'], ratio=ratio, minimum_ratio=minimum,
            env_per_second=value['env_per_second'], baseline_env_per_second=fixed['env_per_second'],
            window_updates=value['window_updates'], strikes=state['strikes'].get(key,0))
    failed = [k for k,v in comparisons.items() if v['strikes'] >= 3]
    state.update(status='BOUNDARY_PAUSE_REQUESTED' if failed else 'OBSERVING',
                 comparisons=comparisons, unix=time.time())
    atomic_json(B/'RESOURCE_GUARD.json', state)
    if failed and not (B/'R2A_RESOURCE_PAUSE_REQUEST.json').exists():
        p = read(ROOT/own['protocol'])
        target = p['chunk_start_update']+p['chunk_updates']
        # Existing owner consumes this per-arm gate; healthy unrelated arms
        # receive no global pause, worker signal, reward or optimizer change.
        selection = B/'R2A_RESOURCE_PAUSE_REQUEST.json'
        atomic_json(selection, dict(status='AUTHORIZED_GUARD_TRIGGERED', references=failed,
            comparisons=comparisons, boundary_run=own['run'], target_update=target,
            instruction='.hermes/STEER_slot01_stop_copy07_gpu3_20261031_2110.md', unix=time.time()))
        request_path = R/'controls/SP_CAPACITY_PAUSE_AFTER_CHUNK.json'
        request = read(request_path)
        request['arms']['a'] = dict(run=own['run'], target_update=target, gpu=3)
        request.update(selection=str(selection.relative_to(ROOT)), selection_sha256=sha(selection))
        atomic_json(request_path, request)
        notify('R2A_RESOURCE_PAUSE', 'R2-a资源参照连续三个5更新窗口越界，已请求仅R2-a完整chunk原子边界暂停；其他训练不受影响，详见RESOURCE_GUARD.json。')


def slot07_monitor():
    files = list((BT/'eval').glob('bt_slot07*_holdout/NATURAL_COMPLETE.json'))
    files = sorted(files, key=lambda p: (int(p.parent.name.split('_u')[1].split('_')[0]), p.parent.name))[-3:]
    if len(files) != 3:
        return
    points = []
    for f in files:
        m = read(f)
        points.append(dict(name=f.parent.name, modes={mode:{k:g.get(k) for k in
            ['games','win','draw','loss','causal_kills','deaths','zero_bomb_games','draw_kill_difference','bombs_per_game']}
            for mode,g in m['natural_js_full2400'].items()}, marker_sha256=sha(f)))
    alarms=[]
    for mode in ['greedy','categorical']:
        values=[p['modes'].get(mode,{}) for p in points]
        if all(v.get('deaths',0)>20 for v in values) and all(values[i+1].get('causal_kills',0)<=values[i].get('causal_kills',0) for i in range(2)):
            alarms.append(mode)
    atomic_json(B/'SLOT07_LIVE_MONITOR.json',dict(status='ALARM' if alarms else 'OBSERVING',
        modes_separate=True,points=points,alarms=alarms,not_a_promotion_gate=True,unix=time.time()))
    for mode in alarms:
        notify('SLOT07_DEATH_ALARM_'+mode+'_'+points[-1]['name'],
            f"slot07 {mode} 连续3个完整holdout点死亡>20且击杀未上升；已报警，不改奖励或闸门。")


def main():
    os.chdir(ROOT)
    guard=lock(B/'MONITOR.lock')
    samples=0
    while True:
        r2=read(B/'R2A_RESUME_AUTHORIZATION.json')
        e=read(R/'R2_SLOTS.json')['a']
        superseded=(B/'R2A_SUPERSEDED_2110.json').exists()
        a=None if superseded else accept(R,e['run'],r2['source'],B/'R2A_FIRST_FORMAL_UPDATE.json')
        h=read(B/'COPY_HANDOFF.json',{})
        copied=accept(BT,Path(h['protocol']).stem,h['source'],B/'SLOT01_FIRST_FORMAL_UPDATE.json') if h else None
        if not superseded:resource_guard(a)
        slot07_monitor()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,power.draw,memory.used','--format=csv,noheader'],text=True)
        with (B/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),rows=gpu.splitlines()))+'\n')
        samples+=1
        complete=bool((a or superseded) and copied)
        atomic_json(B/'EXECUTION_STATUS.json',dict(status='COMPLETED_MONITORING' if complete else 'WAIT_ATOMIC_BOUNDARY_AND_GPU_ACCEPTANCE',
            slot01_boundary_update=read(B/'TASK_CONSUMED.json')['target_update'],
            slot01_stopped=(B/'SLOT01_STOPPED.json').exists(),slot01_first_formal=bool(copied),
            r2a_first_formal=bool(a),gpu_samples=samples,unix=time.time()))
        if complete and not (B/'TASK_DONE.json').exists():
            atomic_json(B/'TASK_DONE.json',dict(status='COMPLETED',
                slot01_acceptance_sha256=sha(B/'SLOT01_FIRST_FORMAL_UPDATE.json'),
                r2a_acceptance_sha256=sha(B/'R2A_FIRST_FORMAL_UPDATE.json') if a else None,
                r2a_allocation_superseded=superseded,
                gate_report='runs/slot01_copy07_gpu3_20261031/SLOT07_B_GATE_OFFICIAL.json',unix=time.time()))
            message=('slot01旧chunk停止并全state复制slot07，独立seed正式更新已验；R2-a GPU3正式续训被较新R3-sp授权覆盖，未冒称完成。' if superseded else '21:10指令全部验收：slot01旧chunk停止并全state复制slot07，独立seed正式更新已验；GPU3 R2-a smoke/恢复/正式更新通过。')
            notify('TASK_COMPLETE',message+'B闸门、R2-b诊断及吞吐证据见reports/slot01_copy07_gpu3_20261031.md。')
        time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(B/'MONITOR_BLOCKED.json',dict(status='BLOCKED',error=type(error).__name__,reason=str(error),unix=time.time()))
        raise
