"""Read-only reference throughput guard; only R3 can be paused by this file."""
import json
import statistics
import time
from pathlib import Path
from scripts.population_run_state import atomic_json

OUT=Path('runs/r3_items_route_20261030')


def references():
    slots=json.loads(Path('runs/bomb_threat_v1/SLOTS.json').read_text())
    a={f'R1_{s}':Path('runs/bomb_threat_v1/formal')/slots[s]['run'] for s in ('0','4','7')}
    for k,v in json.loads(Path('runs/bun_native_v2_20261030/R2_SLOTS.json').read_text()).items():
        a['R2_'+k]=Path('runs/bun_native_v2_20261030/formal')/v['run']
    return a


def sample(folder):
    f=folder/'updates.jsonl'
    if not f.exists():return None
    rows=[json.loads(l) for l in f.read_text().splitlines()]
    warm=[r for r in rows[3:] if r.get('finite') and r.get('cycle_seconds',0)>0][-5:]
    if len(warm)<5:return None
    return dict(run=folder.name,update=warm[-1]['new_update'],points=[r['new_update'] for r in warm],
        env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in warm))


def refresh():
    path=OUT/'RESOURCE_BASELINE.json'
    baseline=json.loads(path.read_text()) if path.exists() else dict(status='COLLECTING_WARM_REFERENCE',references={})
    now={k:s for k,p in references().items() if (s:=sample(p)) is not None}
    for k,s in now.items():
        if k not in baseline['references']:baseline['references'][k]=s
    baseline['status']='REFERENCE_RECORDED';atomic_json(path,baseline)
    prior=OUT/'RESOURCE_MONITOR.json';state=json.loads(prior.read_text()) if prior.exists() else dict(strikes={},last_updates={})
    comparisons={}
    for k,s in now.items():
        b=baseline['references'][k]
        # A new run has a different compilation age. Do not compare its cold
        # startup or stale progress to an old run's warm window.
        if s['run']!=b['run']:continue
        ratio=s['env_per_second']/b['env_per_second'];bad=ratio<.85
        if s['update']-state['last_updates'].get(k,s['update']-5)>=5:
            state['strikes'][k]=state['strikes'].get(k,0)+1 if bad else 0
            state['last_updates'][k]=s['update']
        comparisons[k]=dict(baseline=b,current=s,ratio=ratio,below85pct=bad,consecutive_new_windows=state['strikes'].get(k,0))
    # At least three advancing, nonoverlapping five-update windows are needed.
    failed=[k for k,v in comparisons.items() if v['consecutive_new_windows']>=3]
    state.update(status='BLOCKED_R3_RESOURCE' if failed else 'OBSERVING',comparisons=comparisons,unix=time.time())
    atomic_json(prior,state)
    if failed and not (OUT/'RESOURCE_PAUSE.json').exists():
        atomic_json(OUT/'RESOURCE_PAUSE.json',dict(status='PAUSE_R3_ONLY_AT_SAFE_UPDATE',references=failed,
            reason='reference warm throughput below85pct over advancing observations; correlation, not proven unique cause',unix=time.time()))
        from scripts.r3_items_owner import notify
        notify('RESOURCE_PAUSE','R3资源护栏：既有R1/R2暖态吞吐下降超过15%，已请求仅R3在更新原子边界保存暂停；其他训练不受信号，需排查共卡/CPU瓶颈。')
    return state
