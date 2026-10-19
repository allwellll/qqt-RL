"""JE05 integrated native Bun392 recipe and durable single-arm orchestration."""
import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha,lock
from scripts.je03_bun_stage1_20261011 import digest,read,verify_checkpoint,gpu_owners
from scripts.je03_bun831_20261011 import save_pickle,child_env
RUN=ROOT/'runs/je05_bun_20261013/combat_transport_pfsp8_v1'
PARENT=ROOT/'runs/je03_bun_stage2_20261012/bun392_selfplay12_v2'
STAGE1=ROOT/'runs/je03_bun_stage1_20261011/bun392_retention_v2'
ARM='bun392_combat_transport_pfsp8'
POINTS=[0,8,16,32,64,128,256,512,768,1024,1536,2048,3072,4096]
POOL=['current','stage1_2048','stage1_4096','stage2_512','stage2_1024','stage2_2048','stage2_3072','ring0','ring1','ring2','ring3']
SOURCES=['scripts/je05_bun_authority.py','scripts/je05_web_hunter.js','scripts/je05_bun_recipe.py','scripts/je05_bun_ops.py','scripts/je05_bun_worker.py','scripts/je05_bun_evaluator.py','jax_bomb/je05_bun_env.py','jax_bomb/je05_core_env.py','scripts/je05_bun_transport.py','scripts/je05_bun_scenes.py','scripts/je05_web_tasks.js','scripts/je05_bun_encoding.js','scripts/je05_bun_cpu_monitor.py','scripts/je05_bun_supervisor.py','jax_bomb/jax_train.py','jax_bomb/jax_net.py','jax_bomb/levels.py','jax_bomb/bun_safety.py','jax_bomb/bun_jax_bots.py','scripts/je03_bun_stage1_20261011.py','web/sim.js','web/assets/maps/levels.json']

def plan():
    return dict(experiment_id='JE05-BUN392-COMBAT-TRANSPORT-PFSP8-20261013-v1',arms=[ARM],devices=list(range(8)),
        architecture=dict(obs=[24,13,15],global_dim=24,embed=392,depth=4,heads=4,joint_actions=15,independent_critic=True),
        generations=4096,envs_per_device=1024,steps=128,minibatch_per_device=4096,epochs=1,updates_per_generation=64,
        fresh_ticks_per_generation=1048576,total_fresh_ticks=4294967296,checkpoints=POINTS,recovery_every=64,
        lr=7.5e-5,clip=.1,gradient_clip=.5,entropy=.005,gamma=.995,lam=.95,value_coef=.5,ema_decay=.995,optimizer_eps=1e-8,
        seed=2026101305,initial_policy='stage2 dynamic g3072 raw AND corresponding critic; fresh Adam/RNG/env',
        opponents=dict(rule=.30,g0=.20,stage1_teacher=.20,pfsp_league=.30,pool=POOL,history_interval=128,
            pfsp='per-device rollout opponent sampled from smoothed (1-win)^2 + death/self funnel deficiency; epsilon floor; counters checkpointed',
            current='raw actor updated after every generation',ring='4 snapshots rotate each128 generations; immutable six historical members preserved'),
        curriculum=dict(combat_fraction=.45,adaptive='online per-scene funnel conversion EMA; bounded .5..1.5 difficulty; combat .45; pure carry <=.20; tasks .55; CPU scores never used',
            stages=['approach','contact','legal_attack','request_bomb','placed_bomb','threat','continued_pressure','kill','settled_survival','break_route','enter_base','steal','carry','return','capture','use_item'],
            reset='FULL lesson for all scenes; independent scene label; no scripted enemy; legal topology region spawn, balanced learner seats; native sugar60/contact41px/respawn100; native item state versioned separately'),
        teacher=dict(g0_combat=.10,g0_task=.025,stage1_combat=.04,stage1_task=.01,
            scope='Actor only; move KL and conditional none/bomb KL; use_item logit has zero teacher gradient; danger rows strong'),
        reward=dict(profile='combat_transport_v1',composition='JE04 global combat/safety + bounded route/recover lesson + eligible-steal/highwater-region/capture, before GAE',
            positive_shaping_episode_cap=8.,positive_task_episode_cap=4.,transport_episode_cap=4.,capture_reward=24.,
            task='once route .5 and recover .25 only; replace steal/half/base/cross/return/capture with region ledger: conditional steal .75, .0625 per new band, thresholds/home .25 each, cap4, capture24',
            global_terms='safe settled kill +16; threat chain .2; staggered pressure .2; route-block reduction .1; settled safe escape .5; death-12,self-16,trade-12,avoidable-4,unsafe drop-.5,blocked route-.25',
            settlement='positive threat/kill/escape queued until all live bombs/flames AND native sugar traps settle; terminal lesson waits for residual, forbids new bombs while pending; death/trade cancels pending and reclaims all previous episode positives',
            legacy_zero='all v3 coefficients zero returns exactly supplied legacy reward; default v3 ignores overridden legacy magnitude'),
        protections=['NaN','crash','checkpoint corruption','duplicate owner/write collision'],no_ability_gate=True,no_cpu_gate=True,no_deadline=True,
        evaluation=dict(namespace='authority_je05_v3',hunter_games_per_tier=64,tactical_games=64,bun_task_games=16,
            protocol='same frozen Stage2 seed/spawn/source protocol; baseline g3072 freshly evaluated; CPU-only; 64/tier screening not significance'),
        input_encoding='bun24-trap-timer-v2',native_sugar=True,native_inventory=dict(slots=7,glue_gain=3,banana_gain=1,default_slot=0,death='inventory retained per authoritative web rule',movement='72px/s carry/slow; 268px/s slide; cell40px;10Hz'),transport_regions=dict(version='map806-topology-regions-v1',bands=32,brick_entry_cost=4,table_variants=['native_bricks','opened_bricks'],hot_path='cell lookup + per-origin highwater; no BFS',shape_cap=4.,capture=24.,capture_settlement='queue until all bombs/flames/sugar settle; death cancels pending; region balance retained until settlement',region_delta=.0625,milestone=.25,eligible_steal=.75,home=.25,clawback='death/drop/trade outstanding custody balance; highwater persists',curriculum='combat .45; pure transport <=.20; all use FULL lesson and responsive opponent'),
        scene_names=['combat','vulnerable_steal','carry_route_far','carry_route_mid','carry_route_near','defend_and_steal','escort_under_pressure','route_break_capture','full_bun'],
        namespaces=dict(state='state',cache='cache',judge='judge',eval='authority_je05_v3'))

def unrep(q):
    if isinstance(q,dict):return {k:unrep(v) for k,v in q.items()}
    if isinstance(q,tuple):return tuple(unrep(v) for v in q)
    if isinstance(q,list):return [unrep(v) for v in q]
    return np.asarray(q)[0].copy()

def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PREPARED.json').exists():
        assert read(RUN/'SOURCE_SEAL.json')['sources']=={f:sha(ROOT/f) for f in SOURCES}
        assert read(RUN/'SOURCE_SEAL.json')['plan_sha256']==sha(RUN/'plan.json')
        assert all(sha(RUN/'inputs'/f)==h for f,h in read(RUN/'INITIALIZATION.json')['input_hashes'].items())
        return
    p=plan();atomic_json(RUN/'plan.json',p)
    from scripts import je05_bun_scenes as scenes
    save_pickle(RUN/'inputs/regions.pkl',scenes.TABLES)
    atomic_json(RUN/'regions.json',[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in table.items()} for table in scenes.TABLES])
    atomic_json(RUN/'REGIONS.json',dict(version='map806-topology-regions-v1',map_sha256=sha(ROOT/'web/assets/maps/levels.json'),tables_sha256=sha(RUN/'inputs/regions.pkl'),names=list(scenes.NAMES),legal_region_counts=np.asarray(scenes.MEMBERS).sum(-1).tolist(),members_digest=digest(np.asarray(scenes.MEMBERS)),cage_distance=[t['cage_distance'].tolist() for t in scenes.TABLES],native_map_mirror_exact=bool(np.array_equal(scenes.TABLES[0]['regions'][0],scenes.TABLES[0]['regions'][1,:,::-1]))))
    atomic_json(RUN/'REGIONS.json',dict(version='map806-topology-regions-v1',map_sha256=sha(ROOT/'web/assets/maps/levels.json'),tables_sha256=sha(RUN/'inputs/regions.pkl'),names=list(scenes.NAMES),legal_region_counts=np.asarray(scenes.MEMBERS).sum(-1).tolist(),cage_distance=[t['cage_distance'].tolist() for t in scenes.TABLES],native_map_mirror_exact=bool(np.array_equal(scenes.TABLES[0]['regions'][0],scenes.TABLES[0]['regions'][1,:,::-1]))))
    cp=PARENT/'formal/bun392_selfplay_dynamic/gen003072';m=verify_checkpoint(cp,sha(PARENT/'plan.json'))
    assert m['policy_sha256']=='dc3919a5e7a2f415f951e23f20069d14c3c1df332acf5dfa7b1bb21314ca4f70'
    with (cp/'state.pkl').open('rb') as f:state=pickle.load(f)
    params=unrep(state['live'][0]);del state
    with (cp/'policy.pt').open('rb') as f:raw=pickle.load(f)
    # Compare leaves without importing JAX, so prepare cannot initialize CUDA.
    def equal(a,b):
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return np.array_equal(a,b)
    assert equal(params['actor'],raw)
    save_pickle(RUN/'inputs/initial.pkl',params)
    sources={'g0':dict(policy_sha256=sha(PARENT/'inputs/bun392_gen0.pkl')),'stage1_teacher':dict(policy_sha256=sha(PARENT/'inputs/stage1_actor.pkl')),'initial':dict(checkpoint=str(cp.relative_to(ROOT)),committed_sha256=sha(cp/'COMMITTED.json'),**m)}
    fixed={}
    with (PARENT/'inputs/bun392_gen0.pkl').open('rb') as f:fixed['g0']=pickle.load(f)
    with (PARENT/'inputs/stage1_actor.pkl').open('rb') as f:fixed['stage1_teacher']=pickle.load(f)
    for name,g,parent,arm in [('stage1_2048',2048,STAGE1,'bun392_retention_teacher'),('stage1_4096',4096,STAGE1,'bun392_retention_teacher'),
        *[(f'stage2_{g}',g,PARENT,'bun392_selfplay_dynamic') for g in [512,1024,2048,3072]]]:
        c=parent/'formal'/arm/f'gen{g:06d}';proof=verify_checkpoint(c,sha(parent/'plan.json'))
        with (c/'policy.pt').open('rb') as f:fixed[name]=pickle.load(f)
        sources[name]=dict(checkpoint=str(c.relative_to(ROOT)),committed_sha256=sha(c/'COMMITTED.json'),policy_sha256=proof['policy_sha256'])
    save_pickle(RUN/'inputs/frozen.pkl',fixed)
    for name in ['state','cache','judge','authority_je05_v3']:(RUN/name).mkdir(exist_ok=True)
    atomic_json(RUN/'INITIALIZATION.json',dict(sources=sources,input_hashes={f.name:sha(f) for f in (RUN/'inputs').iterdir()},fresh_optimizer=True,fresh_rng=True,fresh_env=True))
    atomic_json(RUN/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},plan_sha256=sha(RUN/'plan.json')))
    atomic_json(RUN/'PREPARED.json',dict(passed=True,unix=time.time(),cpu_only=True,not_a_start_gate=True))

_CHECKPOINT_CACHE={}

def verify(cp,force=False):
    cp=Path(cp)
    signature=(sha(cp/'COMMITTED.json'),*[(name,(cp/name).stat().st_size,(cp/name).stat().st_mtime_ns) for name in ['state.pkl','policy.pt','ema_policy.pt']])
    cache=_CHECKPOINT_CACHE.get(str(cp))
    if not force and cache and cache[0]==signature:return cache[1]
    m=verify_checkpoint(cp,sha(RUN/'plan.json'))
    assert sha(cp/'ema_policy.pt')==m['ema_policy_sha256']
    assert m['arm']==ARM
    _CHECKPOINT_CACHE[str(cp)]=(signature,m)
    return m


def resolve_checkpoint(generation=None,force=False):
    candidates=sorted((RUN/'formal'/ARM).glob('gen*/COMMITTED.json'),reverse=True)
    for marker in candidates:
        if generation is not None and not marker.parent.name.startswith(f'gen{generation:06d}'):continue
        try:
            proof=verify(marker.parent,force=force)
            if generation is None or proof['generation']==generation:return marker.parent
        except Exception as error:
            # Never modify a corrupt historical checkpoint. Recovery writes a
            # separate suffixed directory; preserve truthful technical evidence.
            key=sha(marker)
            target=RUN/'state'/f'CORRUPT_{key}.json'
            if not target.exists():atomic_json(target,dict(checkpoint=str(marker.parent.relative_to(ROOT)),error=repr(error)))
    return None

def notify(event,message):
    file=RUN/'state'/f'NOTIFIED_{event}.json'
    if file.exists():return
    atomic_json(file,dict(claimed=True,unix=time.time()))
    try:r=subprocess.run(['jdme-notify','send',message],capture_output=True,text=True,timeout=30);code=r.returncode
    except Exception:code=-1
    atomic_json(file,dict(claimed=True,returncode=code,unix=time.time()))

def launch(mode,*args,cpu=False):
    env=child_env('' if cpu else '0,1,2,3,4,5,6,7',cpu)
    cmd=[sys.executable,'scripts/je05_bun_worker.py',mode,*map(str,args)]
    if cpu:cmd=['taskset','-c','80-87',*cmd]
    with (RUN/f'{mode}_{time.time_ns()}.log').open('a') as log:
        return subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]!='Z'
    except FileNotFoundError:return False

class Adopted:
    def __init__(self,pid):self.pid=pid
    def poll(self):return None if alive(self.pid) else 0
    @property
    def returncode(self):return self.poll()
    def wait(self):
        while self.poll() is None:time.sleep(1)
        return self.returncode

def owner():
    prepare();own=lock(RUN/'owner.lock');atomic_json(RUN/'OWNER.json',dict(pid=os.getpid(),devices=list(range(8)),unix=time.time()))
    evaluator=Adopted(read(RUN/'EVAL_LAUNCH.json')['pid']) if (RUN/'EVAL_LAUNCH.json').exists() and alive(read(RUN/'EVAL_LAUNCH.json')['pid']) else None
    def ensure_eval():
        nonlocal evaluator
        if evaluator is None or evaluator.poll() is not None:
            with (RUN/f'evaluator_{time.time_ns()}.log').open('a') as log:
                evaluator=subprocess.Popen(['taskset','-c','64-79',sys.executable,'scripts/je05_bun_evaluator.py','owner'],cwd=ROOT,
                    env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(RUN/'EVAL_LAUNCH.json',dict(pid=evaluator.pid,cuda_visible_devices='',unix=time.time()))
    ensure_eval()
    (RUN/'formal'/ARM).mkdir(parents=True,exist_ok=True)
    for attempt in range(1,33):
        if (RUN/'TRAINING_COMPLETE.json').exists():break
        old=read(RUN/'TRAIN_LAUNCH.json') if (RUN/'TRAIN_LAUNCH.json').exists() else {}
        adopted=old.get('pid') and alive(old['pid'])
        assert adopted or all(not p for p in gpu_owners().values()),'GPU ownership conflict; no process is killed'
        resume=resolve_checkpoint(force=True)
        if resume and verify(resume)['generation']==4096:
            atomic_json(RUN/'TRAINING_COMPLETE.json',dict(passed=True,generation=4096,checkpoint=str(resume.relative_to(ROOT))));break
        child=Adopted(old['pid']) if adopted else launch('formal',*(['--resume',resume] if resume else []))
        atomic_json(RUN/'TRAIN_LAUNCH.json',dict(pid=child.pid,attempt=attempt,devices=list(range(8)),resume=str(resume.relative_to(ROOT)) if resume else None,unix=time.time()))
        restored=False
        while child.poll() is None:
            ensure_eval()
            statefile=RUN/'formal'/ARM/'STATE.json'
            if statefile.exists() and not (RUN/'state/FIRST_NONZERO_UPDATE.json').exists():
                row=read(statefile)
                if row['finite'] and min(row['actor_update_l2'])>0 and min(row['critic_update_l2'])>0:
                    atomic_json(RUN/'state/FIRST_NONZERO_UPDATE.json',dict(worker_pid=child.pid,**row))
                    notify('first_update',f"JE05正式PID {child.pid}，GPU0–7，预算4096代/4.29B ticks；g{row['generation']} finite非零更新，Actor L2 {row['actor_update_l2'][0]:.6f}、Critic {row['critic_update_l2'][0]:.6f}，吞吐{row['fresh_ticks_per_second']:.0f} ticks/s；恢复及CPU评估异步。")
            evidence_cp=resolve_checkpoint(1)
            if evidence_cp is not None and not restored and not (RUN/'state/RESTORE_MONITOR.json').exists():
                monitor=launch('restore-monitor','--resume',evidence_cp,cpu=True)
                atomic_json(RUN/'RESTORE_LAUNCH.json',dict(pid=monitor.pid,cuda_visible_devices='',nonblocking=True));restored=True
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
            with (RUN/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),worker_pid=child.pid,gpu=sample))+'\n')
            time.sleep(5)
        if child.returncode==0 and (RUN/'formal'/ARM/'TRAINER_EXIT.json').exists():
            d=read(RUN/'formal'/ARM/'TRAINER_EXIT.json')
            if d['generation']==4096:
                atomic_json(RUN/'TRAINING_COMPLETE.json',d);break
        atomic_json(RUN/f'TECHNICAL_FAILURE_attempt{attempt:03d}.json',dict(returncode=child.returncode,unix=time.time()))
        notify('technical_'+str(attempt),'JE05训练技术故障，保留证据并从最近完整checkpoint自动恢复。')
        time.sleep(5)
    else:raise RuntimeError('Technical recovery retry budget exhausted')
    notify('training_complete','JE05整体配方4096代训练完成，CPU权威评估继续异步收尾。')
    while not (RUN/'authority_je05_v3/EVALUATION_COMPLETE.json').exists():ensure_eval();time.sleep(10)
    evaluator.wait()
    atomic_json(RUN/'EXPERIMENT_COMPLETE.json',dict(passed=True,training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),evaluation_sha256=sha(RUN/'authority_je05_v3/EVALUATION_COMPLETE.json'),unix=time.time()))
    notify('experiment_complete','JE05整体配方训练与全部CPU权威评估完成；保留中间点，不默认终点最好。')
    atomic_json(RUN/'OWNER_FINISHED.json',dict(passed=True,unix=time.time()))

def finalize_preparation():
    # Explicit CPU development revision, before any JE05 formal/eval writer.
    # Preserve the incomplete seal/plan evidence, never touch predecessor runs.
    for marker in ['OWNER.json','EVAL_LAUNCH.json','HANDOFF_OWNER.json']:
        if (RUN/marker).exists():assert not alive(read(RUN/marker)['pid'])
    history=RUN/'validation/preparation_revisions';history.mkdir(parents=True,exist_ok=True)
    for name in ['SOURCE_SEAL.json','plan.json','PREPARED.json']:
        file=RUN/name
        if file.exists():
            target=history/(file.stem+'_'+sha(file)+'.json')
            if not target.exists():atomic_json(target,read(file))
    from scripts import je05_bun_scenes as scenes
    atomic_json(RUN/'plan.json',plan())
    atomic_json(RUN/'regions.json',[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in table.items()} for table in scenes.TABLES])
    atomic_json(RUN/'REGIONS.json',dict(version='map806-topology-regions-v1',map_sha256=sha(ROOT/'web/assets/maps/levels.json'),tables_sha256=sha(RUN/'inputs/regions.pkl'),names=list(scenes.NAMES),legal_region_counts=np.asarray(scenes.MEMBERS).sum(-1).tolist(),members_digest=digest(np.asarray(scenes.MEMBERS)),cage_distance=[t['cage_distance'].tolist() for t in scenes.TABLES],native_map_mirror_exact=bool(np.array_equal(scenes.TABLES[0]['regions'][0],scenes.TABLES[0]['regions'][1,:,::-1]))))
    atomic_json(RUN/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},plan_sha256=sha(RUN/'plan.json')))
    atomic_json(RUN/'PREPARED.json',dict(passed=True,cpu_only=True,not_a_start_gate=True,final_schema='je05-bun-native-sugar-inventory-region-v1',unix=time.time()))
    prepare()


def handoff():
    # CPU authority completion is explicitly NOT a prerequisite for GPU reuse.
    prepare();lease=lock(RUN/'handoff.lock')
    atomic_json(RUN/'HANDOFF_OWNER.json',dict(pid=os.getpid(),cuda_visible_devices='',unix=time.time()))
    previous=ROOT/'runs/je04_bun_20261012/retention3_pfsp8_v1'
    while True:
        marker=previous/'TRAINING_COMPLETE.json';checkpoint=previous/'formal/bun392_retention3_pfsp8/gen004096'
        result=dict(training_complete=marker.exists(),trainer_exited=False,gpus_free=False,not_waiting_for_cpu_eval=True,unix=time.time())
        if marker.exists():
            try:
                proof=verify_checkpoint(checkpoint,sha(previous/'plan.json'))
                assert proof['generation']==4096 and proof['fresh_ticks']==4294967296
                assert read(marker)['generation']==4096
                launch_record=read(previous/'TRAIN_LAUNCH.json')
                result['trainer_exited']=not alive(launch_record['pid'])
                result['gpus_free']=all(not v for v in gpu_owners().values())
                result['checkpoint_sha256']=sha(checkpoint/'COMMITTED.json')
            except Exception as e:result['technical_error']=repr(e)
        atomic_json(RUN/'HANDOFF_STATE.json',result)
        if result['training_complete'] and result['trainer_exited'] and result['gpus_free']:
            atomic_json(RUN/'GPU_HANDOFF_RELEASED.json',result)
            notify('handoff','JE04八卡训练结束，JE05整体运输配方自动接管；CPU评估不阻塞。')
            owner();return
        time.sleep(10)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','finalize','owner','handoff']);args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='finalize':finalize_preparation()
    elif args.command=='handoff':handoff()
    else:owner()
