"""One CPU-only queue owner for all paired runs and the untouched holdout."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
from scripts.retention_screen import BASE,ARM,declaration,read,env_for
from scripts.population_run_state import lock,atomic_json


def child(run,split,g=None):
    env=env_for(run,True)
    env.update(JE10_ROUTE_PARALLEL='1',JE10_DIAGNOSTIC_REPORT=declaration().get('evaluation_report','reports/retention_evaluation_20261019.md'),JE10_EVAL_SEED='20261019' if split=='dev' else '20261029',
        JE10_ROUTE_GAMES='32' if split=='dev' else '64',
        JE10_ROUTE_SEED_START='2026102100' if split=='dev' else '2026102900',
        JE10_EVAL_NAMESPACE='authority_je10' if split=='dev' else 'authority_retention_test')
    args=[sys.executable,'scripts/retention_evaluate.py',split,str(run.relative_to(ROOT))]
    if g is not None:args.append(str(g))
    return subprocess.run(args,cwd=ROOT,env=env,stdout=(run/(split+'_eval.log')).open('a'),stderr=subprocess.STDOUT).returncode


def evaluate_child(run,split,g):
    from scripts import je10_evaluate as evaluator
    evaluator.BASE=run;evaluator.OUT=run/os.environ['JE10_EVAL_NAMESPACE']
    evaluator.POINTS=declaration()['points'] if split=='dev' else [32]
    evaluator.OUT.mkdir(exist_ok=True)
    manifest=read(ROOT/'data/evaluation/bun_hunter_spawns_v1.json');manifest['seed']=int(os.environ['JE10_EVAL_SEED'])
    atomic_json(evaluator.OUT/'spawns.json',manifest)
    atomic_json(evaluator.OUT/'PROTOCOL.json',dict(split=split,seed=manifest['seed'],route_games=int(os.environ['JE10_ROUTE_GAMES']),route_seed_start=int(os.environ['JE10_ROUTE_SEED_START']),frozen_tasks=True))
    if split=='test' and g==32 and not (evaluator.OUT/'g000000/COMPLETE.json').exists():
        first=ROOT/declaration()['runs'][0]['namespace']
        source=first/'authority_retention_test/g000000'
        target=evaluator.OUT/'g000000'
        if run!=first and (source/'COMPLETE.json').exists():
            if not target.exists():shutil.copytree(source,target)
            atomic_json(target/'SHARED_PARENT_RECEIPT.json',dict(source=str(source.relative_to(ROOT)),same_parent_params=True,independent_samples=False))
        else:evaluator.evaluate(0)
    if not (evaluator.OUT/f'g{g:06d}'/'COMPLETE.json').exists():evaluator.evaluate(g)


def owner():
    unique=lock(BASE/'evaluator.lock')
    # Respect the repository-wide fixed CPU/JS authority lock too.
    import fcntl
    global_lock=(ROOT/'runs/je10_bun_20261018/evaluation.lock').open('a')
    fcntl.flock(global_lock.fileno(),fcntl.LOCK_EX)
    d=declaration();completed=[]
    for spec in d['runs']:
        run=ROOT/spec['namespace']
        for g in d['points']:
            while not (run/'formal'/ARM/f'gen{g:06d}'/'COMMITTED.json').exists():
                atomic_json(BASE/'EVAL_QUEUE.json',dict(completed=completed,running=None,waiting=[spec['namespace'],g],controls_gpu=False));time.sleep(10)
            atomic_json(BASE/'EVAL_QUEUE.json',dict(completed=completed,running=[spec['namespace'],g],controls_gpu=False))
            for attempt in range(3):
                if child(run,'dev',g)==0:break
                atomic_json(BASE/f'EVAL_RETRY_{spec["arm"]}_{spec["seed"]}_{g}_{attempt}.json',dict(retry=attempt,controls_gpu=False))
            else:raise RuntimeError('fixed dev evaluation retries exhausted')
            completed.append([spec['namespace'],g])
    # Holdout is consumed only after every declared development checkpoint.
    # The same parent is evaluated once on holdout seeds, before g32 deltas.
    first=ROOT/d['runs'][0]['namespace']
    if child(first,'test',0):raise RuntimeError('holdout parent evaluation failed')
    for spec in d['runs']:
        run=ROOT/spec['namespace']
        if run!=first:
            source=first/'authority_retention_test/g000000';target=run/'authority_retention_test/g000000'
            if not target.exists():shutil.copytree(source,target)
            atomic_json(target/'SHARED_PARENT_RECEIPT.json',dict(source=str(source.relative_to(ROOT)),same_parent_params=True,independent_samples=False))
        if child(ROOT/spec['namespace'],'test',32):raise RuntimeError('holdout evaluation failed')
    atomic_json(BASE/'EVALUATION_COMPLETE.json',dict(passed=True,completed=completed,holdout_runs=6,unique_owner=True,controls_gpu=False))


if __name__=='__main__':
    if len(sys.argv)==1:owner()
    else:evaluate_child(ROOT/sys.argv[2],sys.argv[1],int(sys.argv[3]))
