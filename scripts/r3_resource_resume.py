"""Consume the corrected co-tenant reference after the bounded profile ends."""
import json
import os
import signal
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.bun_native_v2_owner import processes
from scripts.r3_items_owner import OUT,read,notify

PROFILE=Path('runs/bun_native_v2_20261030/custody_performance_v1/collector_probe/probe_c/COMPLETE.json')


def main():
    guard=lock(OUT/'RESOURCE_RESUME.lock')
    while not PROFILE.exists():
        if not list(processes('scripts.custody_collector_profile')):
            atomic_json(OUT/'RESOURCE_RESUME_BLOCKED.json',dict(status='BLOCKED',reason='bounded custody collector profile exited without completion; inspect its log',unix=time.time()))
            notify('RESOURCE_RESUME_BLOCKED','R3资源恢复阻塞：custody有界collector profile无完成收据；需查日志，R1/R2训练继续。');return
        time.sleep(5)
    assert read(PROFILE)['status']=='COMPLETE'
    while list(processes('scripts.r3_items_train')):time.sleep(5)
    from scripts.r3_items_resources import refresh
    resource=refresh()
    assert all(v['ratio']>=.85 for v in resource['comparisons'].values()),'reference still exceeds15pct slowdown'
    assert not any((d/f).exists() for d in (OUT,Path('runs/bun_native_v2_20261030'),Path('runs/bomb_threat_v1'),Path('runs/iter_pbt_v1'))
        for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json'))
    owners=[p for p in processes('scripts.r3_items_owner') if b'--watchdog' not in (p/'cmdline').read_bytes().split(b'\0')]
    assert len(owners)==1
    os.kill(int(owners[0].name),signal.SIGTERM)
    for _ in range(30):
        if not list(p for p in processes('scripts.r3_items_owner') if b'--watchdog' not in (p/'cmdline').read_bytes().split(b'\0')):break
        time.sleep(1)
    slots=read(OUT/'R3_SLOTS.json');rebind={}
    runtime={'scripts/r3_items_train.py','scripts/r3_items_collector.py','scripts/r3_items_reward.py',
        'scripts/r3_items_runtime.py','scripts/r3_items_observation.py'}
    for arm,entry in slots.items():
        protocol=read(entry['protocol']);old_path=entry['protocol'];old_hash=sha(Path(old_path))
        for f,h in list(protocol['source_sha256'].items()):
            if f.startswith('scripts/r3_items_') and f.endswith('.py') and f not in runtime:
                protocol['source_sha256'].pop(f)
            else:assert sha(Path(f))==h,'frozen runtime changed: '+f
        protocol['operational_amendment']=str(OUT/'AMENDMENT03_RESOURCE_REFERENCE_CONTEXT.json')
        path=OUT/'protocols'/f"{entry['run']}_scheduler01.json";atomic_json(path,protocol)
        entry['protocol']=str(path);rebind[arm]=dict(original=old_path,original_sha256=old_hash,new=str(path),new_sha256=sha(path),
            runtime_unchanged=True,reward_budget_network_optimizer_unchanged=True)
        if arm=='a':
            folder=OUT/'formal'/entry['run'];cp=folder/'u002725';cm=read(cp/'COMMITTED.json')
            assert cm['metrics'].get('user_paused') and (folder/'NEXT_STEP_u002725.json').exists()
            assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
            atomic_json(folder/'ATOMIC_RESOURCE_RESUME.json',dict(status='AUTHORIZED',checkpoint=str(cp),state_sha256=cm['state_sha256'],policy_sha256=cm['policy_sha256'],
                reason='corrected R2 co-tenant reference, all reference ratios>=85pct; no repeat smoke or update',unix=time.time()))
    atomic_json(OUT/'R3_SLOTS.json',slots)
    atomic_json(OUT/'SCHEDULER_REBIND.json',dict(status='OPERATIONAL_ONLY',slots=rebind,
        rationale='training runtime remains sealed; controller/report/evaluator scripts tracked separately so orchestration fixes do not invalidate frozen reward code',
        operational_sha256={f:sha(Path(f)) for f in ['scripts/r3_items_owner.py','scripts/r3_items_resources.py','scripts/r3_items_eval_owner.py']},unix=time.time()))
    (OUT/'RESOURCE_PAUSE.json').replace(OUT/'RESOURCE_PAUSE_CONTEXT_FIXED_RECEIPT.json')
    atomic_json(OUT/'RESOURCE_RESUME_ACCEPTANCE.json',dict(status='AUTHORIZED_ATOMIC_RESUME',all_reference_ratios_ge85pct=True,
        reference_ratios={k:v['ratio'] for k,v in resource['comparisons'].items()},
        source='r3_a u2725 full state/Adam/env/aux/RNG; b not started and retains initial seed/budget',
        profile=str(PROFILE),training_worker_signals=0,updates_discarded=0,smoke_repeated=False,unix=time.time()))
    notify('RESOURCE_CONTEXT_FIXED_RESUME','R3资源误报已修正：原窗口为R2暂停时R1独占卡，不可作共卡基线；正确共卡窗口全部≥85%。a从u2725完整原子state继续，b随后按原seed启动；无重复smoke/丢更新，R1/R2不停。')


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'RESOURCE_RESUME_BLOCKED.json',dict(status='BLOCKED',reason='inspect RESOURCE_RESUME.log; no training worker signals',unix=time.time()));raise
