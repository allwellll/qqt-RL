"""Task-parallel execution of the unchanged eight-route JS protocol."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json
TASKS=['native_death_stock','native_sugar_stock','real_drop','carry_near','carry_mid','carry_far','recovery_threat','natural_full2400']


def merge_shards(shards):
    if len(shards)!=len(TASKS):raise ValueError('missing route task')
    first=shards[0];results={}
    for task,shard in zip(TASKS,shards):
        if not shard.get('passed') or shard['model_json_sha256']!=first['model_json_sha256'] or shard['seeds']!=first['seeds'] or list(shard['results'])!=[task]:raise ValueError('route shard identity mismatch')
        result=shard['results'][task]
        if result['games']!=len(first['seeds']) or [e['seed'] for e in result['episodes']]!=first['seeds']:raise ValueError('route shard denominator mismatch')
        results[task]=result
    return dict(first,results=results,execution='4 parallel independent tasks; identical fixed per-task seeds, simulator, greedy model and task definitions')


def merge_episodes(task,shards):
    first=shards[0];episodes=[]
    for shard in shards:
        if not shard.get('passed') or shard['model_json_sha256']!=first['model_json_sha256'] or list(shard['results'])!=[task]:raise ValueError('episode identity mismatch')
        part=shard['results'][task]
        if part['games']!=len(part['episodes']) or shard['seeds']!=[e['seed'] for e in part['episodes']]:raise ValueError('episode denominator mismatch')
        episodes.extend(part['episodes'])
    episodes.sort(key=lambda e:e['game'])
    games=[e['game'] for e in episodes]
    if games!=list(range(len(episodes))) or len({e['seed'] for e in episodes})!=len(episodes):raise ValueError('duplicate or missing episode shard')
    numeric=[k for k,v in episodes[0].items() if type(v) in (int,float) and k not in ['game','seed','team']]
    sums={k:sum(e[k] for e in episodes) for k in numeric}
    stages={}
    for episode in episodes:
        for phase,values in episode['stage_funnel'].items():
            dest=stages.setdefault(phase,{})
            for key,value in values.items():dest[key]=dest.get(key,0)+value
    result=dict(first['results'][task],games=len(episodes),**sums)
    result.update(stage_funnel=stages,episodes=episodes,
        pickup_episodes=sum(e['pickup']>0 for e in episodes),delivery_episodes=sum(e['delivery']>0 for e in episodes),
        oracle_opportunity_episodes=sum(e['oracle_opportunities']>0 for e in episodes),first_oracle_match_episodes=sum(e['first_oracle_match']>0 for e in episodes))
    return dict(first,results={task:result},seeds=[e['seed'] for e in episodes])


def run(model,outfile,regions):
    outfile=Path(outfile);outfile.parent.mkdir(parents=True,exist_ok=True)
    games=int(os.environ.get('JE10_ROUTE_GAMES','16'))
    def task_job(job):
        task,begin,end=job
        folder=outfile.parent/'route_shards'/task/f'{begin:03d}';folder.mkdir(parents=True,exist_ok=True)
        target=folder/'ROUTE_FUNNEL.json'
        subprocess.run(['node','scripts/je10_route_funnel_current.js',model,str(target),regions],cwd=ROOT,
            env=dict(os.environ,JE10_ROUTE_TASK=task,JE10_ROUTE_START=str(begin),JE10_ROUTE_END=str(end)),check=True,
            stdout=(folder/'route.log').open('a'),stderr=subprocess.STDOUT)
        result=json.loads(target.read_text())
        for pattern in ['REVIEW_FIXED_*.json','fixed-*.trace.json']:
            for path in folder.glob(pattern):shutil.copyfile(path,outfile.parent/path.name)
        return task,result
    jobs=[(task,begin,min(begin+8,games)) for task in TASKS for begin in range(0,games,8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(task_job,jobs))
    shards=[merge_episodes(task,[r for t,r in parts if t==task]) for task in TASKS]
    result=merge_shards(shards);result['execution']='8 independent fixed task/episode CPU shards, identical global seeds and seat indices'
    atomic_json(outfile,result)


if __name__=='__main__':run(*sys.argv[1:])
