"""One-shot, evidence-gated atomic update handoff; never signals workers."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.bun_native_v2_owner import processes

BASE = Path('runs/bun_native_v2_20261030')
OUT = BASE / 'custody_performance_v2'
R3 = Path('runs/r3_items_route_20261030')


def read(path):
    return json.loads(Path(path).read_text())


def status(stage, **extra):
    atomic_json(OUT / 'SWITCH_STATUS.json', dict(status=stage, unix=time.time(), **extra))


def notify(message):
    atomic_json(OUT / 'SWITCH_NOTIFICATION.json', dict(message=message, unix=time.time()))
    with (OUT / 'NOTIFY_SWITCH.log').open('ab') as log:
        subprocess.run(['jdme-notify', 'send', message], stdout=log, stderr=subprocess.STDOUT, check=False)


def workers(slots, default):
    return [p for entry in slots.values() for p in processes(entry.get('worker_module', default), entry['run'])]


def latest(folder):
    paths = sorted(folder.glob('u*/COMMITTED.json'))
    assert paths
    receipt = read(paths[-1])
    cp = paths[-1].parent
    assert receipt['metrics'].get('user_paused'), 'not a cooperative atomic pause'
    assert sha(cp / 'state.pkl') == receipt['state_sha256']
    assert sha(cp / 'policy.pt') == receipt['policy_sha256']
    assert (folder / f"NEXT_STEP_u{receipt['update']:06d}.json").exists()
    rows = [json.loads(line) for line in (folder / 'updates.jsonl').read_text().splitlines()]
    assert rows[-1]['new_update'] == receipt['update'], 'latest effective update not saved'
    return receipt


def stop_owner(module):
    owners = [p for p in processes(module) if b'--watchdog' not in (p / 'cmdline').read_bytes().split(b'\0')]
    assert len(owners) == 1, 'expected unique owner'
    os.kill(int(owners[0].name), signal.SIGTERM)
    # Watchdogs retain their own exclusive locks and adopt/restart controllers.


def main():
    guard = lock(OUT / 'SWITCH.lock')
    if (OUT / 'ATOMIC_SWITCH_READY.json').exists():
        saved = read(OUT / 'ATOMIC_SWITCH_READY.json')
        pause = BASE / 'USER_PAUSE.json'
        if pause.exists() and sha(pause) == saved.get('pause_sha256'):
            pause.replace(OUT / 'TEMPORARY_PAUSE_CONSUMED.json')
        status('FULL_RESTORE_AND_FORMAL_UPDATE_PENDING', recovered_completed_handoff=True)
        return
    comparison = OUT / 'collector_compare/probe_c/COMPLETE.json'
    began = time.time()
    while not comparison.exists():
        status('WAITING_MATCHED_COLLECTOR', training_untouched=True)
        assert list(processes('scripts.custody_collector_compare')), 'matched collector exited without receipt'
        assert time.time() - began < 3600, 'matched collector exceeded bounded qualification deadline'
        time.sleep(5)
    matched = read(comparison)
    assert matched['status'] == 'PASS' and matched['all_outputs_digest_exact']
    if matched['speedup'] < 1.10:
        status('NOT_DEPLOYED_NO_FULL_COLLECTOR_BENEFIT', comparison=str(comparison), speedup=matched['speedup'], training_untouched=True)
        notify('custody第二版不部署：同状态collector虽逐输出SHA一致，但收益未达10%，保留第一版和健康训练；70%目标仍未达到，详见性能报告。')
        return
    parity = read(OUT / 'PARITY_PROFILE.json')
    assert parity['status'] == 'PASS' and parity['phi_and_cache_and_observations_bitwise_exact']
    assert all(sha(Path(f)) == h for f, h in parity['source_sha256'].items())
    # Finish the two already-started R3 qualifications before a shared
    # namespace pause can affect them. Never repeat a completed smoke.
    began = time.time()
    while not all((R3 / 'formal' / e['run'] / 'FIRST_FORMAL_UPDATE.json').exists()
                  for e in read(R3 / 'R3_SLOTS.json').values()):
        status('WAITING_R3_EXISTING_QUALIFICATION', training_untouched=True)
        assert time.time() - began < 2400, 'R3 qualification blocked; no pause requested'
        time.sleep(5)
    roots = (BASE, R3, Path('runs/bomb_threat_v1'), Path('runs/iter_pbt_v1'))
    pause = BASE / 'USER_PAUSE.json'
    for root in roots:
        for name in ('USER_PAUSE.json','USER_STOP.json','STOP_FOR_REBOOT.json'):
            gate=root/name
            if gate==pause and gate.exists():
                assert read(gate).get('owner')=='CUSTODY-PERFORMANCE-02','user pause supersedes handoff'
            else:
                assert not gate.exists(),'user pause supersedes handoff'
    assert not (R3 / 'RESOURCE_PAUSE.json').exists()
    slots, r3slots = read(BASE / 'R2_SLOTS.json'), read(R3 / 'R3_SLOTS.json')
    assert all(slots[a]['worker_module'] == 'scripts.bun_native_v2_custody_fast_train' for a in ('b', 'c'))
    assert (OUT/'R3_FAST2_AUTHORIZATION.json').exists()
    if (OUT/'HOTSWITCH_AUTHORIZATION.json').exists():
        atomic_json(OUT/'HOTSWITCH_AUTHORIZATION_BEFORE_R3_EXTENSION.json',read(OUT/'HOTSWITCH_AUTHORIZATION.json'))
    atomic_json(OUT / 'HOTSWITCH_AUTHORIZATION.json', dict(status='AUTHORIZED_EVIDENCE_GATE_PASSED',
        authorization='user custody performance followup; next safe atomic update', comparison_sha256=sha(comparison),
        reward_batch_optimizer_budget_unchanged=True, scope='R2b/c and R3a/b exact custody code; a resumes identical complete state', unix=time.time()))
    if not pause.exists():
        atomic_json(pause, dict(status='TEMPORARY_PERFORMANCE_ATOMIC_HANDOFF', owner='CUSTODY-PERFORMANCE-02',
            reason='loaded workers only expose namespace cooperative gate; R1 untouched, a/R3 saved then resumed without update loss', unix=time.time()))
    pause_hash = sha(pause)
    status('COOPERATIVE_ATOMIC_SAVE_REQUESTED', training_worker_signals=0)
    began = time.time()
    while workers(slots, 'scripts.bun_native_v2_train') or workers(r3slots, 'scripts.r3_items_train'):
        assert time.time() - began < 1200, 'cooperative save exceeded deadline; gate preserved, inspect live stage'
        time.sleep(5)
    snapshots = {a: latest(BASE / 'formal' / e['run']) for a, e in slots.items()}
    r3snapshots = {a: latest(R3 / 'formal' / e['run']) for a, e in r3slots.items()}
    assert sha(pause) == pause_hash, 'pause ownership changed'
    stop_owner('scripts.bun_native_v2_owner')
    stop_owner('scripts.r3_items_owner')
    for a, entry in slots.items():
        folder = BASE / 'formal' / entry['run']
        cp = snapshots[a]
        if a in ('b', 'c'):
            protocol = read(entry['protocol'])
            previous = entry['protocol']
            protocol['performance_amendment'] = str(OUT / 'HOTSWITCH_AUTHORIZATION.json')
            protocol.update(worker_module='scripts.bun_native_v2_custody_fast2_train',
                custody_performance_version='sparse-fields-eight-sweep-exact-v2',
                custody_performance_parity=str(OUT / 'PARITY_PROFILE.json'), performance_switch_source=cp)
            for f in ('scripts/custody_distance_candidate.py', 'scripts/bun_native_v2_custody_fast2.py', 'scripts/bun_native_v2_custody_fast2_train.py'):
                protocol['source_sha256'][f] = sha(Path(f))
            path = BASE / 'protocols' / f"{entry['run']}_performance02.json"
            atomic_json(path, protocol)
            entry.update(protocol=str(path), previous_performance_protocol=previous,
                worker_module='scripts.bun_native_v2_custody_fast2_train', performance_restore_pending=True,
                performance_restore_receipt='PERFORMANCE2_RESTORE.json', performance_atomic_source=cp)
        auth = folder / 'ATOMIC_PAUSE_RESUME_AUTHORIZATION.json'
        if auth.exists():
            atomic_json(folder / 'ATOMIC_PAUSE_RESUME_AUTHORIZATION_PRE_PERF02.json', read(auth))
        atomic_json(auth, dict(status='AUTHORIZED', checkpoint=cp['checkpoint'], state_sha256=cp['state_sha256'],
            policy_sha256=cp['policy_sha256'], reason='exact performance02 handoff; no repeated smoke', unix=time.time()))
    for a, entry in r3slots.items():
        folder = R3 / 'formal' / entry['run']
        cp = r3snapshots[a]
        protocol=read(entry['protocol']);previous=entry['protocol']
        protocol.update(worker_module='scripts.r3_items_fast2_train',
            custody_performance_version='sparse-fields-eight-sweep-exact-v2',
            custody_performance_parity=str(OUT/'PARITY_PROFILE.json'),
            performance_switch_source=cp,performance_amendment=str(OUT/'R3_FAST2_AUTHORIZATION.json'))
        for f in ('scripts/custody_distance_candidate.py','scripts/bun_native_v2_custody_fast2.py','scripts/r3_items_fast2_train.py'):
            protocol['source_sha256'][f]=sha(Path(f))
        path=R3/'protocols'/f"{entry['run']}_performance02.json";atomic_json(path,protocol)
        entry.update(protocol=str(path),previous_performance_protocol=previous,
            worker_module='scripts.r3_items_fast2_train',performance_restore_pending=True,
            performance_atomic_source=cp)
        auth = folder / 'ATOMIC_RESOURCE_RESUME.json'
        if auth.exists():
            atomic_json(folder / 'ATOMIC_RESOURCE_RESUME_PRE_PERF02.json', read(auth))
        atomic_json(auth, dict(status='AUTHORIZED', checkpoint=cp['checkpoint'], state_sha256=cp['state_sha256'],
            policy_sha256=cp['policy_sha256'], reason='R3 same exact custody performance02 handoff, frozen R3 reward/inputs retained', unix=time.time()))
    atomic_json(BASE / 'R2_SLOTS.json', slots)
    atomic_json(R3 / 'R3_SLOTS.json', r3slots)
    assert sha(pause) == pause_hash
    atomic_json(OUT / 'ATOMIC_SWITCH_READY.json', dict(status='VERIFIED_READY', states=snapshots, R3_states=r3snapshots,
        pause_sha256=pause_hash, all_last_updates_saved=True, updates_discarded=0, training_worker_signals=0, repeated_smoke=False, unix=time.time()))
    pause.replace(OUT / 'TEMPORARY_PAUSE_CONSUMED.json')
    status('FULL_RESTORE_AND_FORMAL_UPDATE_PENDING', live_owners='watchdogs restart unique controllers',
        states=snapshots, R3_states=r3snapshots, whole_update_target_ratio=.70)
    notify('custody第二版同状态collector收益和全输出SHA门槛通过，已在安全更新点完整保存，R2-b/c与R3-a/b同步切换；R1不停，a原配置完整恢复，无丢更新/重复smoke。四路先独立恢复再正式更新，70%及R3共卡60秒门槛待实测。')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('BLOCKED', error_type=type(error).__name__, reason=str(error), no_training_worker_signals=True)
        notify('custody第二版接续阻塞，已落盘SWITCH_STATUS；未向训练worker发信号，若原子暂停已设置则保留闸门，立即检查。')
        raise
