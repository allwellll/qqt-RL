"""Recompute nonintrusive D timing evidence; do not infer unmeasured phases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'runs/expert_reward_20261025/D_ablation_v1'


def distribution(values):
    values=np.asarray(values,float)
    return dict(n=int(values.size),mean=float(values.mean()) if values.size else None,
        p50=float(np.median(values)) if values.size else None,
        p90=float(np.quantile(values,.9)) if values.size else None,
        minimum=float(values.min()) if values.size else None,
        maximum=float(values.max()) if values.size else None)


def analyze(namespace):
    samples=[json.loads(line) for line in (namespace/'SAMPLES.jsonl').read_text().splitlines()]
    gpu=[]
    for index in range(8):
        rows=[next(g for g in s['gpus'] if g['index']==index) for s in samples]
        util=[g['utilization_percent'] for g in rows]
        gpu.append(dict(index=index,utilization_percent=distribution(util),
            sampled_util_above_10_percent_fraction=float(np.mean(np.asarray(util)>10)),
            sampled_zero_util_fraction=float(np.mean(np.asarray(util)==0)),
            memory_used_mib=distribution([g['memory_used_mib'] for g in rows]),
            power_watts=distribution([g['power_watts'] for g in rows if g['power_watts'] is not None]),
            compute_role_counts={role:sum(role in g['compute_roles'] for g in rows)
                for role in sorted({role for g in rows for role in g['compute_roles']})}))
    roles=sorted({role for s in samples for role in s['process_roles']})
    cpu={role:dict(cpu_percent_one_core=distribution([s['process_roles'].get(role,{}).get('cpu_percent_one_core',0)
        for s in samples[1:]]),rss_bytes=distribution([s['process_roles'].get(role,{}).get('rss_bytes',0) for s in samples]),
        observed_process_count=distribution([s['process_roles'].get(role,{}).get('count',0) for s in samples])) for role in roles}
    historical=[];log_rows={}
    for path in sorted((D/'formal').glob('*/updates.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        # Historical restarted logs can contain duplicate update labels. Keep
        # only the latest occurrence for timing, do not sum duplicate samples.
        rows=list({r['new_update']:r for r in rows}.values())
        log_rows[path.parent.name]={r['new_update']:r for r in rows}
        first=[r for r in rows if r.get('startup_compile_included')]
        warm=[r for r in rows if not r.get('startup_compile_included')]
        total=sum(r['iteration_seconds'] for r in warm)
        rollout=sum(r['rollout_seconds'] for r in warm)
        historical.append(dict(run=path.parent.name,updates=len(rows),
            last_new_update=max((r['new_update'] for r in rows),default=0),
            first_call=[{k:r[k] for k in ('new_update','iteration_seconds','rollout_seconds','steps_per_second')} for r in first],
            warm_iteration_seconds=distribution([r['iteration_seconds'] for r in warm]),
            warm_rollout_seconds=distribution([r['rollout_seconds'] for r in warm]),
            warm_iteration_env_per_second=16384*len(warm)/total if total else None,
            measured_rollout_share=rollout/total if total else None,
            update_plus_aggregation_share=(total-rollout)/total if total else None,
            unmeasured_wall_overhead_included=False))
    unique={}
    for sample in samples:
        st=sample['training']
        if 'unix' in st and 'new_update' in st:
            unique[(st['seed'],st['new_update'])]=st
    observed=sorted(unique.values(),key=lambda s:s['unix'])
    cycles=[]
    for previous,now in zip(observed,observed[1:]):
        if now['seed']!=previous['seed'] or now['new_update']!=previous['new_update']+1:
            continue
        run=f"legacy_s{now['seed']}"
        # This observation is registered for the current legacy seed2853.
        row=log_rows.get(run,{}).get(now['new_update'])
        if row is None:continue
        wall=now['unix']-previous['unix']
        cycles.append(dict(new_update=now['new_update'],wall_seconds=wall,
            measured_iteration_seconds=row['iteration_seconds'],rollout_seconds=row['rollout_seconds'],
            uninstrumented_boundary_seconds=wall-row['iteration_seconds'],
            checkpointed=row['checkpointed'],update_plus_aggregation_seconds=row['iteration_seconds']-row['rollout_seconds']))
    seconds=sum(c['wall_seconds'] for c in cycles)
    missing={
        'compile_only_seconds':'First call includes trace/compile plus actual rollout/update, not a compile-only timer.',
        'pure_GPU_optimizer_seconds':'Existing interval includes CPU replay aggregation, batch transfers, wrapper creation, dispatch, cache lookup and device wait.',
        'rule_bot_vs_env_vs_policy':'All run inside device-side JAX collector; no kernel-level trace attached to the frozen D worker.',
        'device_transfer_only_seconds':'D copies/replicates full pytrees outside/inside measured intervals without dedicated transfer timers.',
        'checkpoint_only_seconds':'Observed update-boundary wall excess includes checkpoint, host telemetry and copies; not an isolated save timer.',
        'CPU_eval_wall_phases':'CPU/Node roles sampled independently; their internal phase timers not present in this passive observation.',
        'exact_GPU_duty_cycle':'One-second nvidia-smi utilization sampling is an activity proxy with possible aliasing, not kernel-duration integration.'}
    return dict(schema='qqt.gpu-efficiency-observation-analysis/v1',samples=len(samples),
        duration_seconds=samples[-1]['elapsed_seconds']-samples[0]['elapsed_seconds'],
        sample_interval_seconds=distribution(np.diff([s['elapsed_seconds'] for s in samples])),
        sampling_command_wall_seconds=distribution([s['sampling_seconds'] for s in samples]),
        gpu=gpu,cpu_process_roles=cpu,historical_updates=historical,observed_cycles=cycles,
        wall_effective_env_per_second=16384*len(cycles)/seconds if seconds else None,
        boundary_overhead_share=sum(c['uninstrumented_boundary_seconds'] for c in cycles)/seconds if seconds else None,
        rollout_wall_share=sum(c['rollout_seconds'] for c in cycles)/seconds if seconds else None,
        update_plus_aggregation_wall_share=sum(c['update_plus_aggregation_seconds'] for c in cycles)/seconds if seconds else None,
        missing_phase_timers={k:dict(value=None,reason=v) for k,v in missing.items()},
        no_training_or_queue_changes=True,
        evidence_sha256=hashlib.sha256((namespace/'SAMPLES.jsonl').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('namespace',type=Path);args=parser.parse_args()
    report=analyze(args.namespace)
    output=args.namespace/'ANALYSIS.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('samples','duration_seconds','wall_effective_env_per_second','boundary_overhead_share')}))


if __name__=='__main__':main()
