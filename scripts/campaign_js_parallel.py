"""Scheduling-only production-JS batch concurrency; frozen evaluator unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.population_run_state import atomic_json,lock,sha

def validate_batch(result,spec,name,start,end):
    assert result['passed'] and result['model_json_sha256']==spec['models'][name]['model_json_sha256']
    episodes=result['results']['natural_full2400']['episodes']
    assert len(episodes)==end-start
    assert [e['seed'] for e in episodes]==list(range(spec['seed_start']+start,spec['seed_start']+end))
    return episodes

def tasks(spec,out):
    return [(name,mode,start,min(start+8,spec['games_per_condition']),out/name/mode/f'{start:04d}'/'RESULT.json')
            for name in spec['models'] for mode in spec['action_modes']
            for start in range(0,spec['games_per_condition'],8)]

def truncation_coverage(episodes):
    available=[e['artificial_truncation'] for e in episodes if isinstance(e.get('artificial_truncation'),bool)]
    return dict(available=len(available),missing=len(episodes)-len(available),flagged=sum(available))

def run_batch(task,spec):
    name,mode,start,end,target=task
    target.parent.mkdir(parents=True,exist_ok=True)
    guard=lock(target.parent/'batch.lock')
    if not target.exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JE10_ROUTE_TASK='natural_full2400',
            JE10_ROUTE_GAMES=str(spec['games_per_condition']),JE10_ROUTE_START=str(start),
            JE10_ROUTE_END=str(end),JE10_ROUTE_SEED_START=str(spec['seed_start']))
        subprocess.run(['node','scripts/expert_validation_js_eval.js',mode,spec['models'][name]['model_json'],str(target),spec['regions']],env=env,check=True)
    value=json.loads(target.read_text());validate_batch(value,spec,name,start,end)
    return task

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('protocol');a=p.parse_args()
    spec=json.loads(Path(a.protocol).read_text());out=Path(spec['namespace']);out.mkdir(parents=True,exist_ok=True)
    guard=lock(out/'owner.lock');global_guard=lock(Path(spec['JS_unique_lock']))
    for file,expected in spec['source_sha256'].items():assert sha(file)==expected,file
    for model in spec['models'].values():
        assert sha(model['model_json'])==model['model_json_sha256']
        assert json.loads(Path(model['model_json']).read_text())['meta']['checkpoint_sha256']==model['policy_sha256']
    all_tasks=tasks(spec,out);cached=[];pending=[]
    for task in all_tasks:
        if task[-1].exists():
            validate_batch(json.loads(task[-1].read_text()),spec,task[0],task[2],task[3]);cached.append(task)
        else:pending.append(task)
    completed=sum(t[3]-t[2] for t in cached)
    def progress(task):
        atomic_json(out/'PROGRESS.json',dict(completed_games=completed,total_games=len(spec['models'])*len(spec['action_modes'])*spec['games_per_condition'],last_model=task[0],action_mode=task[1],unix=time.time(),scheduler='two batch workers; original global indices',cpu_only=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_batch,t,spec) for t in pending]):
            task=future.result();completed+=task[3]-task[2];progress(task)
    rows=[];sources=[];coverage=dict(available=0,missing=0,flagged=0)
    for name,mode,start,end,target in all_tasks:
        episodes=validate_batch(json.loads(target.read_text()),spec,name,start,end)
        rows.extend(dict(model=name,action_mode=mode,episode=e) for e in episodes)
        sources.append(dict(path=str(target),sha256=sha(target),seeds=[e['seed'] for e in episodes]))
        for key,value in truncation_coverage(episodes).items():coverage[key]+=value
    assert len(rows)==len(spec['models'])*len(spec['action_modes'])*spec['games_per_condition']
    atomic_json(out/'SUMMARY.json',dict(protocol=a.protocol,protocol_sha256=sha(a.protocol),rows=rows,engine='production JS web/sim.js',opponent='JS Hunter normal',teacher=False,training_reward_status='not recorded; observable reconstruction is not actual JAX reward',missing_metrics=['own_alive/actionable ticks and complete causal item branch net-benefit require follow-up'],no_peak_selection=True))
    atomic_json(out/'PARALLEL_MERGE.json',dict(passed=True,total_games=len(rows),source_batches=sources,seed_sets_exact=True,duplicate_seeds_within_condition=0,cached_games_reused=sum(t[3]-t[2] for t in cached),artificial_truncation_flag_count=coverage['flagged'],artificial_truncation_field_coverage=coverage,truncation_status='Missing episode flags remain unknown; scheduling adds no external timeout or horizon change',summary_sha256=sha(out/'SUMMARY.json'),unix=time.time()))
    atomic_json(out/'COMPLETE.json',dict(passed=True,total_games=len(rows),summary_sha256=sha(out/'SUMMARY.json'),merge_sha256=sha(out/'PARALLEL_MERGE.json'),scope='Fixed natural JS normal-Hunter matrix only; short combat Hunter and Python Tactical separate'))

if __name__=='__main__':main()
