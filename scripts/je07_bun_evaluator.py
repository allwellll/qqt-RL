"""Unique CPU authority queue, entirely asynchronous to the eight-device arm."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse
import concurrent.futures
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je07_bun_recipe import *
from scripts import je07_bun_authority as authority
def tasks(policy,target):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=Path(target).parent;model=authority.export(policy,out/'model')
    def run_task(task):
        file=out/f'web_task_{task}.json'
        if not file.exists():
            subprocess.run(['node','scripts/je05_web_tasks.js',str(model),str(file),task,str(RUN/'regions.json')],
                cwd=ROOT,env=child_env('',True),check=True)
        data=read(file);assert data['passed'] and data['native_trap'] and data['native_items']
        assert data['model_json_sha256']==sha(model) and data['results'][task]['games']==16
        return task,data['results'][task]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=dict(pool.map(run_task,['carry_home','near_steal','route_break','full']))
    atomic_json(target,dict(passed=True,policy_sha256=sha(policy),results=results,protocol='je05_native_transport_web_v2',
        cuda_visible_devices='',native_trap=True,native_items=True,screening_only=True,
        opponent='real web Hunter normal; frozen paired seed/spawn task diagnostic',
        shards={f'web_task_{task}.json':sha(out/f'web_task_{task}.json') for task in results}))
OUT=RUN/'authority_je07_v1'

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(namespace=OUT.name,seed=read(authority.SPAWNS)['seed'],spawns_sha256=sha(authority.SPAWNS),
        hunter_games_per_tier=64,hunter_ticks=300,hunter_residual=64,tactical_games=64,bun_task_games=16,
        native_task_seed=2026101131,native_task_ticks=300,native_tasks='web/sim.js nativeTrap/nativeItems; real Hunter normal; separate from legacy JAX task scores',input_encoding='bun24-trap-timer-v2',checkpoints=POINTS,cuda_visible_devices='',
        baseline='selected completed raw parent; freshly evaluated same protocol; assist OFF',screening_only=True,
        source_hashes={f:sha(ROOT/f) for f in authority.FILES+['scripts/je07_bun_evaluator.py','scripts/je07_bun_authority.py']})
    if (OUT/'PROTOCOL.json').exists():assert read(OUT/'PROTOCOL.json')==protocol
    else:atomic_json(OUT/'PROTOCOL.json',protocol)
    if not (OUT/'spawns.json').exists():atomic_json(OUT/'spawns.json',read(authority.SPAWNS))

def evaluate(label,policy,proof):
    setup();out=OUT/label;final=out/'COMPLETE_WITH_BUN_TASKS.json'
    if final.exists():
        d=read(final);assert d['policy_sha256']==sha(policy)
        assert d['authority_sha256']==sha(out/'COMPLETE.json')
        assert all(sha(out/f)==h for f,h in d['additional_shards'].items());return
    out.mkdir(parents=True,exist_ok=True);checkpoint_lease=lock(out/'evaluation.lock')
    start=time.time();main=authority.evaluate(label,policy,proof)
    target=out/'bun_tasks.json'
    if not target.exists():
        with (out/'bun_tasks.log').open('a') as log:
            subprocess.run([sys.executable,'scripts/je07_bun_evaluator.py','tasks','--policy',str(policy),'--out',str(target)],env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,check=True)
    task=read(target);assert task['policy_sha256']==sha(policy)
    assert all(sha(out/f)==h for f,h in task['shards'].items())
    funnel={}
    for tier in ['hard','normal','easy']:
        episodes=[e for start in range(0,64,8) for e in read(out/f'hunter_{tier}_{start:03d}.json')['episodes']]
        rows={name:sum(e['funnel'][name] for e in episodes) for name in episodes[0]['funnel']}
        rows.update(threat=sum(sum(bool(p['threat']) for p in e['attack_trace']['placements']) for e in episodes),
            pressure=sum(sum(bool(p['pressure']) for p in e['attack_trace']['placements']) for e in episodes),
            settled_survival=main['hunter'][tier]['safe_win'])
        funnel[tier]=rows
    atomic_json(out/'FUNNEL.json',dict(by_opponent_family=funnel,scope='passive web primary trace; settled_survival uses full residual; not training proxy'))
    atomic_json(final,dict(passed=True,policy_sha256=sha(policy),authority_sha256=sha(out/'COMPLETE.json'),
        additional_shards={'bun_tasks.json':sha(target),'FUNNEL.json':sha(out/'FUNNEL.json'),**task['shards']},tasks=task['results'],funnel=funnel,
        wall_seconds=time.time()-start))

def active_points():
    return POINTS

def probe(label,policy):
    """Unassisted annealing evidence and separate assisted diagnostics/long tasks."""
    out=OUT/label;model=authority.export(policy,out/'model')
    from scripts.je07_bun_curriculum import PROBE_MANIFEST_SHA
    folder=BASE/'assist_probe_v1'/label;folder.mkdir(parents=True,exist_ok=True)
    for mode in ['off','on']:
        target=folder/f'web_probe_{mode}.json' if mode=='off' else BASE/'assist_diagnostic_v1'/f'{label}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            subprocess.run(['node','scripts/je07_web_probe.js',str(model),str(target),str(BASE/'regions.json'),mode],cwd=ROOT,env=child_env('',True),check=True)
        data=read(target);assert data['model_json_sha256']==sha(model) and data['assist']==(mode=='on')
        assert data['games_per_bucket']==16 and not data['smoke']
        assert data['seed_manifest_sha256']==PROBE_MANIFEST_SHA
        if mode=='off':
            atomic_json(folder/'PROBE_COMPLETE.json',dict(assist=False,games_per_bucket=16,
                near=data['results']['near']['capture_episodes'],mid=data['results']['mid']['capture_episodes'],far=data['results']['far']['capture_episodes'],
                full_capture=data['results']['full']['capture_episodes'],seed_manifest_sha256=PROBE_MANIFEST_SHA,
                policy_sha256=sha(policy),source_sha256=sha(target),screening_only=True,not_a_training_gate=True))
    side=BASE/'side_diagnostics_600_900_v1'/label;side.mkdir(parents=True,exist_ok=True)
    jobs=[(task,horizon) for horizon in [600,900] for task in ['route_break','full']]
    def run(job):
        task,horizon=job;target=side/f'{task}_{horizon}.json'
        if not target.exists():subprocess.run(['node','scripts/je06_web_tasks.js',str(model),str(target),task,str(BASE/'regions.json'),str(horizon)],cwd=ROOT,env=child_env('',True),check=True)
        data=read(target);assert data['model_json_sha256']==sha(model) and data['primary_ticks']==horizon
        return target.name,sha(target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:hashes=dict(pool.map(run,jobs))
    atomic_json(side/'COMPLETE.json',dict(passed=True,assist=False,hashes=hashes,policy_sha256=sha(policy),screening_only=True))

def owner():
    OUT.mkdir(parents=True,exist_ok=True);own=lock(OUT/'owner.lock')
    atomic_json(OUT/'OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',phase='WAIT_PREPARED',unix=time.time()))
    while not (BASE/'PREPARED.json').exists():time.sleep(10)
    setup();atomic_json(OUT/'OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',phase='RUNNING',unix=time.time()))
    baseline=ROOT/read(BASE/'INITIALIZATION.json')['parent']/'policy.pt'
    verify_checkpoint(baseline.parent,read(baseline.parent/'COMMITTED.json')['plan_sha256'])
    assert sha(baseline)==read(BASE/'INITIALIZATION.json')['parent_policy_sha256']
    while True:
        jobs=[('baseline_parent',baseline,dict(policy_sha256=sha(baseline)))]
        for g in active_points():
            cp=resolve_checkpoint(g)
            if cp is not None:
                verify(cp);jobs.append((f'{ARM}_g{g:06d}',cp/'policy.pt',dict(generation=g,committed_sha256=sha(cp/'COMMITTED.json'))))
        completed=[]
        for label,policy,proof in jobs:
            final=OUT/label/'COMPLETE_WITH_BUN_TASKS.json'
            if final.exists():
                evaluate(label,policy,proof)
                if not (BASE/'side_diagnostics_600_900_v1'/label/'COMPLETE.json').exists():probe(label,policy)
                completed.append(label);continue
            with (OUT/f'{label}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'scripts/je07_bun_evaluator.py','checkpoint','--label',label,'--policy',str(policy)],cwd=ROOT,env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(OUT/'ACTIVE.json',dict(pid=child.pid,label=label,unix=time.time()))
                code=child.wait()
            if code:
                atomic_json(OUT/f'TECHNICAL_RETRY_{label}_{time.time_ns()}.json',dict(returncode=code));break
            completed.append(label)
            probe(label,policy)
            if label!='baseline_parent':
                base=read(OUT/'baseline_parent/COMPLETE.json');cur=read(OUT/label/'COMPLETE.json')
                atomic_json(OUT/f'COMPARE_{label}.json',dict(baseline_policy_sha256=sha(baseline),
                    hunter_delta={d:{k:cur['hunter'][d][k]-base['hunter'][d][k] for k in cur['hunter'][d]} for d in ['hard','normal','easy']},
                    task_delta={task:{k:read(final)['tasks'][task][k]-read(OUT/'baseline_parent/COMPLETE_WITH_BUN_TASKS.json')['tasks'][task][k] for k in ['capture_episodes','steal_episodes','death_episodes','self_episodes','trade_episodes','safe_win']} for task in ['carry_home','near_steal','route_break','full']},
                    baseline_complete_sha256=sha(OUT/'baseline_parent/COMPLETE_WITH_BUN_TASKS.json'),current_complete_sha256=sha(final),screening_only=True))
            notify('evaluation_'+OUT.name+'_'+label,'JE07 '+label+' CPU权威评估完成（Hunter64/档仅快筛），GPU训练不受影响。')
        atomic_json(OUT/'QUEUE.json',dict(completed=completed,pending_generations=[g for g in active_points() if f'{ARM}_g{g:06d}' not in completed],unix=time.time()))
        first=[g for g in POINTS if g<=2048]
        if all(f'{ARM}_g{g:06d}' in completed for g in first) and not (OUT/'BLOCK_EVALUATION_COMPLETE_2048.json').exists():
            atomic_json(OUT/'BLOCK_EVALUATION_COMPLETE_2048.json',dict(passed=True,completed=completed,screening_only=True,unix=time.time()))
        if (RUN/'TRAINING_COMPLETE.json').exists() and len(completed)==1+len(active_points()):
            atomic_json(OUT/'EVALUATION_COMPLETE.json',dict(passed=True,completed=completed,hashes={label:sha(OUT/label/'COMPLETE_WITH_BUN_TASKS.json') for label in completed},unix=time.time()))
            atomic_json(OUT/'OWNER_FINISHED.json',dict(passed=True,unix=time.time()));return
        time.sleep(10)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['owner','checkpoint','tasks']);ap.add_argument('--policy');ap.add_argument('--out');ap.add_argument('--label');args=ap.parse_args()
    if args.command=='tasks':tasks(args.policy,Path(args.out))
    elif args.command=='checkpoint':evaluate(args.label,Path(args.policy),{})
    else:owner()
