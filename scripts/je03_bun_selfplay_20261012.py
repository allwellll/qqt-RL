"""Durable nonblocking handoff and conservative native Bun392 self-play."""
import argparse
import copy
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha,lock
from scripts.je03_bun_stage1_20261011 import split_forward,digest,verify_checkpoint,gpu_owners,read
from scripts.je03_bun831_20261011 import save_pickle,child_env
from scripts.bun_checkpoint_state import alive
PARENT=Path(os.environ.get('BUN_SELFPLAY_PARENT_RUN', ROOT/'runs/je03_bun_stage1_20261011/bun392_retention_v2')).resolve()
RUN=Path(os.environ.get('BUN_SELFPLAY_RUN',ROOT/'runs/je03_bun_stage2_20261012/bun392_selfplay12_v2')).resolve()
ARMS=['bun392_selfplay_control','bun392_selfplay_dynamic']
POINTS=[0,8,16,32,64,128,256,512,768,1024,1536,2048,3072,4096]
RETENTION_V2=True
SOURCES=['scripts/je03_bun_selfplay_20261012.py','scripts/je03_bun_selfplay_worker_20261012.py',
         'scripts/je03_bun_selfplay_ops_20261012.py','scripts/je03_bun_selfplay_evaluator_20261012.py',
         'jax_bomb/bun_env.py','jax_bomb/jax_train.py','jax_bomb/jax_net.py',
         'scripts/je03_bun831_evaluator_20261011.py','scripts/je03_bun_evaluator_20261011.py',
         'scripts/je03_bun_stage1_20261011.py','web/sim.js','web/bun_hunter_bot.js']
_SOURCE_CACHE={}

def metrics(directory):
    h=read(directory/'HUNTER_COMPLETE.json')['metrics']
    t=read(directory/'tactical_v2.json')['settled_style_summary']
    tasks=read(directory/'COMPLETE_WITH_BUN_TASKS.json')['tasks']
    return dict(safe=sum(v['safe_win'] for v in h.values()),
                hunter=h,tactical=t,tasks={k:{q:v[q] for q in ['capture_episodes','steal_episodes','return_episodes','death_episodes','self_episodes']} for k,v in tasks.items()},
                death=sum(v['death_episode'] for v in h.values()),
                self=sum(v['self_episodes'] for v in h.values()),
                trade=sum(v['trade_episodes'] for v in h.values()),
                complete_sha256=sha(directory/'COMPLETE_WITH_BUN_TASKS.json'))

def choose(rows):
    """Rank adjacent stable intervals, choose earliest competitive member.
    No pass/fail threshold; missing late CPU results never delays a GPU."""
    if not rows:raise ValueError('no complete source observations')
    keys=sorted(rows)
    def utility(x):
        return (x['safe']+x['tactical']['safe_win']+.5*sum(
            v['capture_episodes']+v['steal_episodes'] for v in x['tasks'].values())
            -.75*x['death']-2*x['self']-2*x['trade'])
    values={g:utility(rows[g]) for g in keys}
    if len(keys)==1:return keys[0],dict(values=values,window=keys)
    intervals=[(min(values[a],values[b])-.25*abs(values[a]-values[b]),a,b)
               for a,b in zip(keys,keys[1:])]
    def tier_support(g):
        h=rows[g].get('hunter')
        return min(h[t]['safe_win'] for t in ('hard','normal')) if h else 0
    _,a,b=max(intervals,key=lambda t:(min(tier_support(t[1]),tier_support(t[2])),t[0],-t[1]))
    # Earliest in the strongest stable interval within 3 utility points.
    chosen=a if values[a]>=values[b]-3 else b
    return chosen,dict(values=values,window=[a,b],intervals=intervals)

def source_evidence():
    rows={}
    for g in [512,768,1024,1536,2048,3072,4096]:
        directory=PARENT/'authority_bun831_v1'/f'bun392_retention_teacher_g{g:06d}'
        if (directory/'COMPLETE_WITH_BUN_TASKS.json').exists():
            if g in _SOURCE_CACHE:
                rows[g]=_SOURCE_CACHE[g];continue
            d=read(directory/'COMPLETE_WITH_BUN_TASKS.json')
            assert d['authority_sha256']==sha(directory/'COMPLETE.json')
            assert all(sha(directory/f)==h for f,h in d['additional_shards'].items())
            cp=PARENT/'formal/bun392_retention_teacher'/f'gen{g:06d}'
            m=verify_checkpoint(cp,sha(PARENT/'plan.json'))
            assert m['policy_sha256']==d['policy_sha256']
            rows[g]=metrics(directory)
            _SOURCE_CACHE[g]=rows[g]
    return rows

def plan():
    parent=read(PARENT/'plan.json')
    r=copy.deepcopy(parent['recipes']['bun392_retention_teacher'])
    r.update(lr=.000075, teacher_kl_coef=.10,
             opponents='idle=0,roam=0,rule_combat=.35,weak=.20,old=.33,recent=.06,history=.06')
    return dict(experiment_id='JE03-BUN392-SELFPLAY12-20261012-v2',
        stage=2,user_authorized=True,arms=ARMS,checkpoints=POINTS,
        generations=4096,seed=2026101201,envs_per_device=1024,steps=128,
        minibatch_per_device=4096,epochs=1,recovery_every=64,
        ticks_per_generation=524288,ticks_per_arm=2147483648,
        gamma=.995,lam=.95,clip=.10,value_coef=.5,ema_decay=.995,
        optimizer_eps=1e-8,hp=1,architecture=parent['architecture'],
        initialization='same selected Stage1 raw actor AND critic; fresh matched Adam/RNG/env; g0 frozen teacher unchanged',
        recipes={a:dict(r) for a in ARMS},
        reward=dict(profile='retention_v2',actual_semantics='frozen current per-lesson override semantics, including existing task rewards; no reward fix bundled with self-play',parent_seal_sha256=sha(PARENT/'SOURCE_SEAL.json')),
        attribution='only .12 opponent slots change fixed Stage1 snapshot -> versioned learner snapshots; control uses identical slot sampling but fixed candidates',
        league_refresh_every=1 if os.environ.get('BUN_SELFPLAY_CPU_TEST')=='1' else 128,league=dict(recent=.06,history=.06,rule=.35,g0=.33,stage1=.20,
        first_version='selected Stage1 actor',refresh='raw actor after completed update every128; history=previous version; no opponent gradient; versions+RNG in full checkpoint',
        roles='learner P0 on even envs/P1 on odd envs; 50:50 both arms; identical paired native reset RNG; roles differ from parent and are shared'),
        teacher=dict(coefficient=.10,direction='current||g0',scope='all learner Actor minibatch rows, masked move x3 ability joint; zero Critic gradient; item head remains trainable'),
        conservative_changes=dict(lr='.0002 -> .000075',clip='.2 -> .1',epochs='2 -> 1 (64 actual updates/gen)',global_gradient_clip=.5),
        selection='first maximize minimum hard/normal safe support across adjacent pair, then stable utility; earliest member within3 utility; available authenticated results; late CPU never blocks launch',
        cpu_evaluation='same Hunter64/tier300+64 residual, Tactical64 native300, Bun tasks16/type + native web item/trap/rescue probes, async;64 is screening only',
        protection='finite/damaged checkpoints/crash only; preserve failure, bounded autorestart same state; no ability blocking gate',
        degradation='three consecutive evaluations with safe <=start-6 OR death >=start+10 OR self >=start+4: atomic stop and new recovery namespace on same cards from initial candidate, frozen slots, lr half; never promote a winner',
        no_cutoff=True,no_start_gate=True)

def prepare(select=True):
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'SOURCE_SEAL.json').exists():verify_seal();return
    p=plan();atomic_json(RUN/'plan.json',p)
    if not select:return
    rows=source_evidence();generation,reason=choose(rows)
    cp=PARENT/'formal/bun392_retention_teacher'/f'gen{generation:06d}'
    m=verify_checkpoint(cp,sha(PARENT/'plan.json'))
    with (cp/'state.pkl').open('rb') as f:saved=pickle.load(f)
    actor={};critic={}
    # Saved pmap replicas remain identical; preserve both trained parameter trees.
    def unrep(q):
        if isinstance(q,dict):return {k:unrep(v) for k,v in q.items()}
        if isinstance(q,tuple):return tuple(unrep(v) for v in q)
        if isinstance(q,list):return [unrep(v) for v in q]
        return np.asarray(q)[0].copy()
    params=unrep(saved['live'][0])
    save_pickle(RUN/'inputs/stage1_actor.pkl',params['actor'])
    save_pickle(RUN/'inputs/stage1_critic.pkl',params['critic'])
    with (PARENT/'inputs/bun392_gen0.pkl').open('rb') as f:g0=pickle.load(f)
    save_pickle(RUN/'inputs/bun392_gen0.pkl',g0)
    atomic_json(RUN/'INITIALIZATION.json',dict(generation=generation,checkpoint=str(cp),
        source_marker_sha256=sha(cp/'COMMITTED.json'),source_actor_sha256=m['policy_sha256'],
        decision=reason,rows=rows,cutover_selection_unix=time.time(),frozen=True))
    atomic_json(RUN/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},
        inputs={f:sha(RUN/'inputs'/f) for f in ['stage1_actor.pkl','stage1_critic.pkl','bun392_gen0.pkl']},
        plan_sha256=sha(RUN/'plan.json')))
    atomic_json(RUN/'PREPARED.json',dict(passed=True,cpu_only=True,unix=time.time()))

def verify_seal():
    d=read(RUN/'SOURCE_SEAL.json')
    assert sha(RUN/'plan.json')==d['plan_sha256']
    assert all(sha(ROOT/f)==h for f,h in d['sources'].items())
    assert all(sha(RUN/'inputs'/f)==h for f,h in d['inputs'].items())

def available(devices):
    owners=gpu_owners()
    return all(not owners[int(i)] for i in devices.split(','))

class AdoptedProcess:
    def __init__(self,pid):self.pid=pid
    def poll(self):return None if alive(self.pid) else 0
    @property
    def returncode(self):return self.poll()

def command(arm,devices,phase='formal',resume=None,cpu=False):
    cmd=[sys.executable,str(ROOT/'scripts/je03_bun_selfplay_worker_20261012.py'),
         '--arm',arm,'--devices',devices,'--phase',phase]
    if resume:cmd+=['--resume',str(resume)]
    if cpu:cmd+=['--cpu']
    return cmd

def arm_owner(arm,devices):
    verify_seal();own=lock(RUN/f'owner_{arm}.lock')
    atomic_json(RUN/f'OWNER_{arm}.json',dict(pid=os.getpid(),devices=devices,unix=time.time()))
    for attempt in range(1,17):
        assert available(devices),'GPU collision; never take occupied cards'
        cps=sorted((RUN/'formal'/arm).glob('gen*/COMMITTED.json'))
        resume=cps[-1].parent if cps else None
        if resume:
            m=verify_checkpoint(resume,sha(RUN/'plan.json'))
            if m['generation']==read(RUN/'plan.json')['generations']:
                atomic_json(RUN/f'TRAINING_COMPLETE_{arm}.json',dict(passed=True,generation=m['generation'],unix=time.time()));return
            events=RUN/'formal'/arm/'events.jsonl'
            if events.exists():
                raw=events.read_text()
                events.with_name(f'events_before_resume_{time.time_ns()}.jsonl').write_text(raw)
                events.write_text(''.join(l+'\n' for l in raw.splitlines() if json.loads(l)['generation']<=m['generation']))
        with (RUN/f'{arm}_attempt{attempt}_{time.time_ns()}.log').open('a') as log:
            child=subprocess.Popen(command(arm,devices,resume=resume),env=child_env(devices),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(RUN/f'LAUNCH_{arm}.json',dict(pid=child.pid,devices=devices,resume=str(resume) if resume else None,unix=time.time()))
            while child.poll() is None:
                sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                with (RUN/f'GPU_{arm}.jsonl').open('a') as f:f.write(json.dumps(dict(pid=child.pid,gpu=sample,unix=time.time()))+'\n')
                time.sleep(5)
        if child.returncode==0:
            d=read(RUN/'formal'/arm/'TRAINER_EXIT.json')
            if d['generation']==read(RUN/'plan.json')['generations']:
                atomic_json(RUN/f'TRAINING_COMPLETE_{arm}.json',d);return
            atomic_json(RUN/f'ATOMIC_EXIT_{arm}.json',d);return
        atomic_json(RUN/f'TECHNICAL_STOP_{arm}_{child.pid}.json',dict(returncode=child.returncode,attempt=attempt,unix=time.time()))
        time.sleep(5)
    raise RuntimeError('bounded restart exhausted')

def regression_points(arm):
    directory=RUN/'authority_stage2_recovery_v2'
    baseline=directory/f'{arm}_g000000'
    if not (baseline/'COMPLETE_WITH_BUN_TASKS.json').exists():return []
    start=metrics(baseline);rows=[]
    for g in POINTS[1:]:
        p=directory/f'{arm}_g{g:06d}'
        if (p/'COMPLETE_WITH_BUN_TASKS.json').exists():rows.append((g,metrics(p)))
    if len(rows)<3:return []
    window=rows[-3:]
    flags=[d['safe']<=start['safe']-6 or d['death']>=start['death']+10 or d['self']>=start['self']+4 for g,d in window]
    return [g for g,d in window] if all(flags) else []

def recovery_namespace(arm,devices):
    """Replace an unhealthy ability trajectory without waiting for another arm."""
    target=RUN/f'recovery_{arm}_{time.time_ns()}'
    target.mkdir();p=read(RUN/'plan.json');p['experiment_id']+='-safety-recovery-'+arm
    for r in p['recipes'].values():r.update(lr=r['lr']/2,dynamic_enabled=False)
    atomic_json(target/'plan.json',p)
    for file in (RUN/'inputs').iterdir():
        with file.open('rb') as f:save_pickle(target/'inputs'/file.name,pickle.load(f))
    atomic_json(target/'INITIALIZATION.json',dict(read(RUN/'INITIALIZATION.json'),recovery_of=str(RUN),reason='three consecutive safety/ability regressions; frozen baseline rollback; no winner selection'))
    atomic_json(target/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},inputs={f.name:sha(f) for f in (target/'inputs').iterdir()},plan_sha256=sha(target/'plan.json')))
    env=dict(child_env('',True),BUN_SELFPLAY_RUN=str(target))
    with (target/f'owner_{arm}.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'arm-owner','--arm',arm,'--devices',devices],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (target/'cpu_owner.log').open('a') as log:
        ev=subprocess.Popen([sys.executable,str(ROOT/'scripts/je03_bun_selfplay_evaluator_20261012.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(namespace=str(target),pid=child.pid,evaluator_pid=ev.pid,devices=devices,unix=time.time())

def cpu_probe():
    """Parallel correctness monitor; never read by the handoff start path."""
    prepare();verify_seal()
    for arm in ARMS:
        for phase,resume in [('cpu_smoke',None),('cpu_replay',RUN/'cpu_smoke'/arm/'gen000001')]:
            with (RUN/f'{phase}_{arm}.log').open('a') as log:
                result=subprocess.run(command(arm,'',phase,resume,cpu=True),
                    env=child_env('',True),stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(RUN/f'TECHNICAL_STOP_{phase}_{arm}_{time.time_ns()}.json',dict(returncode=result.returncode));break
        else:
            a=read(RUN/'cpu_smoke'/arm/'TRAINER_EXIT.json');b=read(RUN/'cpu_replay'/arm/'TRAINER_EXIT.json')
            atomic_json(RUN/f'CPU_RESTORE_MONITOR_{arm}.json',dict(
                passed=a['final_state_digest']==b['final_state_digest'],smoke=a,replay=b,nonblocking=True))

def handoff():
    RUN.mkdir(parents=True,exist_ok=True);own=lock(RUN/'handoff.lock')
    p=plan()
    if not (RUN/'plan.json').exists():atomic_json(RUN/'plan.json',p)
    assigned=read(RUN/'HANDOFF_ASSIGNMENTS.json') if (RUN/'HANDOFF_ASSIGNMENTS.json').exists() else {}
    children={a:AdoptedProcess(d['pid']) for a,d in assigned.items()}
    cpu=AdoptedProcess(read(RUN/'CPU_LAUNCH.json')['pid']) if (RUN/'CPU_LAUNCH.json').exists() else None
    atomic_json(RUN/'HANDOFF_OWNER.json',dict(pid=os.getpid(),no_cutoff=True,nonblocking_checks=True,unix=time.time()))
    while len(assigned)<2:
        rows=source_evidence();g,reason=choose(rows)
        atomic_json(RUN/'PARENT_CURVE_LATEST.json',dict(rows=rows,candidate=g,decision=reason,unix=time.time()))
        for arm,old,devices in zip(ARMS,['bun392_retention_control','bun392_retention_teacher'],['0,1,2,3','4,5,6,7']):
            if arm in assigned:continue
            complete=PARENT/f'TRAINING_COMPLETE_{old}.json'
            exit_marker=PARENT/'formal'/old/'TRAINER_EXIT.json'
            if not complete.exists():continue
            old_launch=read(PARENT/f'LAUNCH_{old}.json')
            old_owner=read(PARENT/f'OWNER_{old}.json')
            if alive(old_launch['pid']) or alive(old_owner['pid']) or not available(devices):continue
            d=read(exit_marker);assert d['passed'] and d['generation']==4096
            verify_checkpoint(PARENT/'formal'/old/'gen004096',sha(PARENT/'plan.json'))
            prepare() # source choice freezes once at first free 4-card group
            with (RUN/f'owner_{arm}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'arm-owner','--arm',arm,'--devices',devices],
                    env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            assigned[arm]=dict(pid=child.pid,devices=devices,parent_exit_sha256=sha(exit_marker),unix=time.time())
            children[arm]=child
            atomic_json(RUN/'HANDOFF_ASSIGNMENTS.json',assigned)
            if cpu is None:
                with (RUN/'cpu_owner.log').open('a') as log:
                    cpu=subprocess.Popen([sys.executable,str(ROOT/'scripts/je03_bun_selfplay_evaluator_20261012.py')],
                        env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(RUN/'CPU_LAUNCH.json',dict(pid=cpu.pid,cuda_visible_devices='',unix=time.time()))
                # Checks execute alongside real GPU updates.
                with (RUN/'cpu_restore_monitor.log').open('a') as log:
                    probe=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'cpu-probe'],
                        env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(RUN/'CPU_RESTORE_LAUNCH.json',dict(pid=probe.pid,nonblocking=True))
        atomic_json(RUN/'HANDOFF_STATE.json',dict(pid=os.getpid(),phase='TRAINING' if len(assigned)==2 else 'WAIT_PARENT_ATOMIC_EXIT',
            assigned=assigned,candidate=g,checks_block_launch=False,unix=time.time()))
        for arm,child in list(children.items()):
            if child.poll() is not None and not (RUN/f'TRAINING_COMPLETE_{arm}.json').exists() and available(assigned[arm]['devices']):
                with (RUN/f'owner_early_retry_{arm}_{time.time_ns()}.log').open('a') as log:
                    children[arm]=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'arm-owner','--arm',arm,'--devices',assigned[arm]['devices']],env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                assigned[arm]['pid']=children[arm].pid;atomic_json(RUN/'HANDOFF_ASSIGNMENTS.json',assigned)
        time.sleep(10)
    # Durable monitor remains sole scheduling writer; CPU result never gates GPUs.
    replacements=read(RUN/'SAFETY_REPLACEMENTS.json') if (RUN/'SAFETY_REPLACEMENTS.json').exists() else {}
    while not all((RUN/f'TRAINING_COMPLETE_{a}.json').exists() or a in replacements for a in ARMS):
        for arm,child in list(children.items()):
            if arm in replacements:continue
            try:window=regression_points(arm)
            except Exception as e:
                atomic_json(RUN/f'TECHNICAL_MONITOR_{time.time_ns()}.json',dict(error=repr(e),arm=arm));window=[]
            if window and child.poll() is None:
                launch=read(RUN/f'LAUNCH_{arm}.json')
                if alive(launch['pid']):
                    os.kill(launch['pid'],signal.SIGTERM)
                    atomic_json(RUN/f'SAFETY_ATOMIC_REPLACEMENT_REQUEST_{arm}.json',dict(window=window,pid=launch['pid'],unix=time.time()))
            if child.poll() is not None and (RUN/f'SAFETY_ATOMIC_REPLACEMENT_REQUEST_{arm}.json').exists() and available(assigned[arm]['devices']):
                replacements[arm]=recovery_namespace(arm,assigned[arm]['devices'])
                atomic_json(RUN/'SAFETY_REPLACEMENTS.json',replacements);continue
            if child.poll() is not None and not (RUN/f'TRAINING_COMPLETE_{arm}.json').exists():
                atomic_json(RUN/f'OWNER_EXIT_{arm}_{child.pid}.json',dict(returncode=child.returncode,unix=time.time()))
                if (RUN/f'ATOMIC_EXIT_{arm}.json').exists():continue
                if available(assigned[arm]['devices']):
                    with (RUN/f'owner_retry_{arm}_{time.time_ns()}.log').open('a') as log:
                        children[arm]=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'arm-owner','--arm',arm,'--devices',assigned[arm]['devices']],env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    assigned[arm]['pid']=children[arm].pid;atomic_json(RUN/'HANDOFF_ASSIGNMENTS.json',assigned)
        if cpu is None or cpu.poll() is not None:
            with (RUN/f'cpu_owner_retry_{time.time_ns()}.log').open('a') as log:
                cpu=subprocess.Popen([sys.executable,str(ROOT/'scripts/je03_bun_selfplay_evaluator_20261012.py')],env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(RUN/'CPU_LAUNCH.json',dict(pid=cpu.pid,cuda_visible_devices='',unix=time.time()))
        time.sleep(15)
    atomic_json(RUN/'QUEUE_FINISHED.json',dict(training_complete=all((RUN/f'TRAINING_COMPLETE_{a}.json').exists() for a in ARMS),arms=ARMS,replacements=replacements,unix=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','handoff','arm-owner','cpu-probe','preview']);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--devices',default='');args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='preview':prepare(False);atomic_json(RUN/'PARENT_CURVE_LATEST.json',dict(rows=source_evidence(),selection=choose(source_evidence())))
    elif args.command=='handoff':handoff()
    elif args.command=='cpu-probe':cpu_probe()
    else:arm_owner(args.arm,args.devices)
