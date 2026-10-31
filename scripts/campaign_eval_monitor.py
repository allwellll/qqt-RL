"""Passive scheduling acceptance; raw counts and missing comparisons stay explicit."""
import json
import os
from pathlib import Path
import statistics
import time
from scripts.population_run_state import atomic_json,sha
from scripts.campaign_eval_resources import row_rate

def warm_rate(history,job):
    """Last continuous working segment, excluding pauses and affinity changes."""
    segment=[];key=None
    for sample in history:
        worker=next((w for w in sample['workers'] if w['job']==job),None)
        if worker is None or worker.get('suspended'):
            segment=[];key=None;continue
        current=(sample.get('scheduler_revision'),tuple(worker.get('cpu_affinity') or []))
        if current!=key:segment=[];key=current
        count=worker.get('completed_games')
        if count is not None and count>0:segment.append((sample['unix'],count))
    if len(segment)<2 or segment[-1][0]-segment[0][0]<120:return None
    return (segment[-1][1]-segment[0][1])/(segment[-1][0]-segment[0][0])*60

def tick(campaign,active,root,base):
    now=time.time();history_path=base/'EVAL_PARALLEL_SAMPLES.jsonl'
    if history_path.exists() and now-history_path.stat().st_mtime<60:return
    config=json.loads((base/'EVAL_PARALLEL_CONFIG.json').read_text())
    snapshots=[]
    for job,process,_ in active.values():
        resource=job['resource']
        if resource=='gpu':continue
        path=root/job.get('progress',job['complete'])
        try:value=json.loads(path.read_text())
        except (OSError,ValueError):value={}
        snapshots.append(dict(job=job['id'],resource=resource,completed_games=value.get('completed_games',0),progress_unix=value.get('unix'),suspended=job.get('resource_guard_suspended',False),cpu_affinity=job.get('cpu_affinity'),cpu_only=True))
    cpu_usage=int(Path('/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage').read_text())/1e9
    sample=dict(unix=now,scheduler_revision=json.dumps(config['limits'],sort_keys=True),cgroup_cpu_seconds=cpu_usage,workers=snapshots,active_gpu=active['gpu'][0]['id'] if 'gpu' in active else None)
    with history_path.open('a') as stream:stream.write(json.dumps(sample)+'\n')
    history=[json.loads(line) for line in history_path.read_text().splitlines()]
    baseline=json.loads((base/'EVAL_PARALLEL_BASELINE.json').read_text())
    before={row['job']:row for row in baseline['active']}
    comparisons=[];rates={}
    for job in campaign['jobs']:
        if job['resource'] not in ('cpu','evaluation','js','analysis'):continue
        output=root/job.get('progress',job['complete']);folder=output.parent
        try:progress=json.loads(output.read_text())
        except (OSError,ValueError):continue
        rate=warm_rate(history,job['id'])
        if rate and rate>0:rates.setdefault(job['resource'],[]).append(rate)
        old=before.get(job['id']);before_rate=None
        if old:
            cache=folder/'batches';files=sorted(cache.glob('*.json'),key=lambda p:p.stat().st_mtime)
            if job['resource']=='js':files=sorted(folder.glob('*/*/*/RESULT.json'),key=lambda p:p.stat().st_mtime)
            files=[p for p in files if p.stat().st_mtime<=baseline['unix']][-16:]
            if len(files)>1:
                elapsed=files[-1].stat().st_mtime-files[0].stat().st_mtime
                if elapsed>0:before_rate=(len(files)-1)*8/elapsed*60
        comparisons.append(dict(job=job['id'],resource=job['resource'],status=job['status'],before_games_per_minute=before_rate,after_warm_games_per_minute=rate,completed_games=progress.get('completed_games'),total_games=progress.get('total_games'),comparison_status='measured' if rate else 'waiting for >=120s after first committed batch'))
    total_cpu=None
    if len(history)>=2:
        a,b=history[-2:];total_cpu=(b['cgroup_cpu_seconds']-a['cgroup_cpu_seconds'])/(b['unix']-a['unix'])
    gpu=[]
    for row in baseline['active']:
        if row['resource']!='gpu':continue
        try:updates=[json.loads(line) for line in (root/row['updates_path']).read_text().splitlines()]
        except (OSError,ValueError):continue
        later=[r for r in updates if r['new_update']>row['update_count']]
        gpu.append(dict(job=row['job'],before_median_env_per_second=row['warm_median_env_per_second'],before_median_cycle_seconds=row['warm_median_cycle_seconds'],after_median_env_per_second=statistics.median(row_rate(r) for r in later) if later else None,after_median_cycle_seconds=statistics.median(r['cycle_seconds'] for r in later) if later else None,after_update_count=len(later),caveat='same historical run; most expansion may postdate its terminal; use calibration for causal scheduling comparison'))
    eta=[]
    for resource in ('cpu','evaluation','js','analysis'):
        jobs=[j for j in campaign['jobs'] if j['resource']==resource and j['status'] not in ('COMPLETE','DELEGATED_SIDE_CX')]
        remaining=0;unknown=[]
        for job in jobs:
            budget=(job.get('budget') or {}).get('games')
            try:p=json.loads((root/job.get('progress',job['complete'])).read_text())
            except (OSError,ValueError):p={}
            if budget is None and job.get('args') and len(job['args'])>=4:
                try:
                    spec=json.loads((root/job['args'][2]).read_text());condition=spec.get('conditions',{}).get(job['args'][3],{});budget=condition.get('games_per_mode',0)*2 or None
                except (OSError,ValueError):pass
            if (job.get('budget') or {}).get('new_games')==0:continue
            if budget is None:unknown.append(job['id'])
            else:remaining+=max(0,budget-p.get('completed_games',0))
        worker_rate=statistics.median(rates[resource]) if rates.get(resource) else None
        concurrency=config['limits'][resource] if resource!='js' else 1
        aggregate=worker_rate*concurrency if worker_rate else None
        eta.append(dict(resource=resource,known_remaining_games=remaining,budget_unknown_jobs=unknown,estimated_aggregate_games_per_minute=aggregate,eta_seconds=remaining/aggregate*60 if aggregate else None,eta_unix=now+remaining/aggregate*60 if aggregate else None,scope='all known queued games including future checkpoints; training/input availability adds delay; no guaranteed deadline'))
    merges=[]
    for path in base.glob('**/PARALLEL_MERGE.json'):
        try:v=json.loads(path.read_text());merges.append(dict(path=str(path.relative_to(root)),sha256=sha(path),passed=v.get('passed'),games=v.get('total_games')))
        except (OSError,ValueError):pass
    atomic_json(base/'EVAL_PARALLEL_ACCEPTANCE.json',dict(schema='qqt.cpu-eval-parallel-acceptance/v1',stage='PARTIAL_ACCEPTANCE_TRAINING_GUARD_ACTIVE',implemented=['unique resource slots and adoption','CPU-only independent affinity groups','native3/checkpoint4/analysis2/JS batches2','cache reuse with model/seed/hash validation','JS safe batch-boundary handoff','same-run GPU throughput guard and automatic CPU reduction'],worker_counts={r:dict(present=sum(w['resource']==r for w in snapshots),working=sum(w['resource']==r and not w['suspended'] for w in snapshots),suspended=sum(w['resource']==r and w['suspended'] for w in snapshots)) for r in ('cpu','evaluation','analysis','js')},js_internal_workers=2,total_cgroup_cpu_cores=total_cpu,cpu_quota_cores=88,evaluation_affinity_budget=config.get('cpu_affinity_budget',27),side_cpu_reservation=config.get('side_cpu_reservation',0),gpu_reserved_cpu_count=12,job_throughput=comparisons,gpu_training_comparisons=gpu,gpu_calibration=campaign.get('eval_gpu_calibration'),gpu_guard_references=campaign.get('eval_gpu_references',{}),eta_by_resource=eta,merge_validation=merges,missing=['full merge awaits each fixed job completion','same-run controlled GPU comparison awaits >=20 warm updates per phase'] if not campaign.get('eval_gpu_calibration',{}).get('complete') else ['full merge awaits each fixed job completion'],zombies=baseline['zombies'],old_queues_resumed=False,science_changed=False,unix=now))
