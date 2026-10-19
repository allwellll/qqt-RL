"""Holistic native Bun392 continuation; isolated production smoke and durable owner."""
import argparse,json,os,pickle,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha,lock
from scripts.je03_bun_stage1_20261011 import digest,read,verify_checkpoint,gpu_owners
from scripts.je03_bun831_20261011 import save_pickle,child_env
from scripts.je05_bun_recipe import POOL,unrep,alive,Adopted
BASE=ROOT/'runs/je06_bun_20261013/holistic_long256_pfsp8_v1'
SMOKE_NAME='production_smoke_restart_v3'
RUN=BASE/SMOKE_NAME if os.environ.get('JE06_PHASE')=='smoke' else BASE
PREVIOUS=ROOT/'runs/je05_bun_20261013/combat_transport_pfsp8_v1'
PARENT_CP=PREVIOUS/'formal/bun392_combat_transport_pfsp8/gen001536'
ARM='bun392_holistic_long256_pfsp8'
POINTS=[0,8,16,32,64,128,256,512,768,1024,1536,2048,3072,4096]
SOURCES=['scripts/je06_bun_recipe.py','scripts/je06_bun_worker.py','scripts/je06_bun_authority.py','scripts/je06_bun_evaluator.py',
 'scripts/je06_bun_supervisor.py','scripts/je06_web_tasks.js','scripts/je06_bun_candidate_audit.py','scripts/je06_bun_side_diagnostics.py',
 *read(PREVIOUS/'SOURCE_SEAL.json')['sources'].keys()]

def plan():
    from scripts.je05_bun_recipe import plan as previous_plan
    p=previous_plan();p.update(experiment_id='JE06-BUN392-HOLISTIC-LONG256-PFSP8-20261013-v1',arms=[ARM],generations=2048,steps=256,
        seed=2026101306,updates_per_generation=128,fresh_ticks_per_generation=2097152,total_fresh_ticks=4294967296,
        extension_generations=4096,extension_fresh_ticks=8589934592,checkpoints=[g for g in POINTS if g<=2048],
        initial_policy='JE05 g1536 raw Actor AND corresponding Critic; fresh Adam/RNG/env',fresh_optimizer=True,fresh_rng=True,fresh_env=True,
        smoke_generations=2,attribution='holistic capability continuation; no A/B or single-factor causal claim',
        extension_rule='after first 2048 generations, continue same state to4096 if complete CPU curve shows holistic positive improvement; otherwise retain all checkpoints and finish',
        evaluation=dict(namespace='authority_je06_recovery_v2',hunter_games_per_tier=64,tactical_games=64,bun_task_games=16,protocol='frozen JE05 web native task v2 + Stage2 Hunter/Tactical; raw parent re-evaluated; screening only'),
        namespaces=dict(state='state',cache='cache',judge='judge',eval='authority_je06_recovery_v2'))
    return p

def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    if not (BASE/'PREPARED.json').exists():
        assert (BASE/'PARENT_SELECTION.json').exists(),'freeze evidence-based parent selection'
        cp=PARENT_CP;m=verify_checkpoint(cp,sha(PREVIOUS/'plan.json'))
        assert m['generation']==1536 and sha(cp/'ema_policy.pt')==m['ema_policy_sha256']
        with (cp/'state.pkl').open('rb') as f:s=pickle.load(f)
        assert digest(s['live'])==m['full_state_digest'];params=unrep(s['live']['params']);del s
        with (cp/'policy.pt').open('rb') as f:raw=pickle.load(f)
        assert digest(params['actor'])==digest(raw)
        save_pickle(BASE/'inputs/initial.pkl',params)
        for name in ['frozen.pkl','regions.pkl']:
            with (PREVIOUS/'inputs'/name).open('rb') as f:save_pickle(BASE/'inputs'/name,pickle.load(f))
        atomic_json(BASE/'regions.json',read(PREVIOUS/'regions.json'))
        atomic_json(BASE/'INITIALIZATION.json',dict(parent=str(cp.relative_to(ROOT)),policy_kind='raw',parent_committed_sha256=sha(cp/'COMMITTED.json'),
            parent_policy_sha256=m['policy_sha256'],parent_ema_sha256=m['ema_policy_sha256'],parent_state_sha256=m['state_sha256'],actor_digest=digest(params['actor']),critic_digest=digest(params['critic']),
            fresh_optimizer=True,fresh_rng=True,fresh_env=True,input_hashes={f.name:sha(f) for f in (BASE/'inputs').iterdir()}))
        atomic_json(BASE/'plan.json',plan())
        for name in ['state','cache','judge','validation','authority_je06_recovery_v2']:(BASE/name).mkdir(exist_ok=True)
        atomic_json(BASE/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},plan_sha256=sha(BASE/'plan.json')))
        atomic_json(BASE/'PREPARED.json',dict(passed=True,unix=time.time()))
    seal=read(BASE/'SOURCE_SEAL.json');assert seal['sources']=={s:sha(ROOT/s) for s in SOURCES}
    assert seal['plan_sha256']==sha(BASE/'plan.json')
    assert all(sha(BASE/'inputs'/f)==h for f,h in read(BASE/'INITIALIZATION.json')['input_hashes'].items())
    if RUN!=BASE:materialize_smoke()

def materialize_smoke():
    child=BASE/SMOKE_NAME
    child.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','state','validation']:(child/name).mkdir(exist_ok=True)
    for f in (BASE/'inputs').iterdir():
        target=child/'inputs'/f.name
        if not target.exists():os.link(f,target)
        assert sha(target)==sha(f)
    p=read(BASE/'plan.json');p.update(generations=2,checkpoints=[0,1,2],total_fresh_ticks=4194304)
    if (child/'plan.json').exists():assert read(child/'plan.json')==p
    else:atomic_json(child/'plan.json',p)
    return child

def process_matches(pid,script):
    try:return alive(pid) and script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def restart_preparation():
    """Archive pre-GPU preparation evidence; preserve failed smoke untouched."""
    for marker,script in [('SUPERVISOR_OWNER.json','je06_bun_supervisor.py'),('OWNER_LAUNCH.json','je06_bun_recipe.py'),
                          ('EVAL_LAUNCH.json','je06_bun_evaluator.py'),('SMOKE_LAUNCH.json','je06_bun_worker.py')]:
        if (BASE/marker).exists():assert not process_matches(read(BASE/marker)['pid'],script)
    assert not list((BASE/'formal').glob('*/gen*/COMMITTED.json')),'cannot revise an active formal plan'
    assert not (BASE/'PRODUCTION_SMOKE_COMPLETE.json').exists(),'failed smoke must not be passed'
    archive=BASE/'validation/pre_restart_evidence';archive.mkdir(parents=True,exist_ok=True)
    for name in ['SOURCE_SEAL.json','plan.json','PREPARED.json','SUPERVISOR_OWNER.json','OWNER_LAUNCH.json','OWNER.json','EVAL_LAUNCH.json','SMOKE_LAUNCH.json']:
        file=BASE/name
        if file.exists():
            target=archive/(file.stem+'_'+sha(file)+'.json')
            if not target.exists():atomic_json(target,read(file))
    atomic_json(BASE/'plan.json',plan())
    atomic_json(BASE/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},plan_sha256=sha(BASE/'plan.json')))
    atomic_json(BASE/'RESTART_PREPARATION.json',dict(passed=True,old_smoke_preserved='production_smoke',new_smoke=SMOKE_NAME,
        no_formal_checkpoint=True,old_processes_exited=True,unix=time.time()))
    prepare();materialize_smoke()

def verify(cp,force=False):
    m=verify_checkpoint(Path(cp),sha(RUN/'plan.json'))
    assert m['arm']==ARM and sha(Path(cp)/'ema_policy.pt')==m['ema_policy_sha256']
    return m

def resolve_checkpoint(generation=None,force=False):
    for marker in sorted((RUN/'formal'/ARM).glob('gen*/COMMITTED.json'),reverse=True):
        if generation is not None and not marker.parent.name.startswith(f'gen{generation:06d}'):continue
        try:
            m=verify(marker.parent)
            if generation is None or m['generation']==generation:return marker.parent
        except Exception as e:
            target=RUN/'state'/f'CORRUPT_{sha(marker)}.json'
            if not target.exists():atomic_json(target,dict(checkpoint=str(marker.parent.relative_to(ROOT)),error=repr(e)))
    return None

def notify(event,message):
    p=BASE/'state'/f'NOTIFIED_{event}.json'
    if p.exists():return
    atomic_json(p,dict(claimed=True,unix=time.time()))
    try:code=subprocess.run(['jdme-notify','send',message],capture_output=True,timeout=30).returncode
    except Exception:code=-1
    atomic_json(p,dict(claimed=True,returncode=code,unix=time.time()))

def launch(mode,*args,cpu=False,smoke=False):
    if smoke:materialize_smoke()
    env=child_env('' if cpu else '0,1,2,3,4,5,6,7',cpu);env['JE06_PHASE']='smoke' if smoke else 'formal'
    command=[sys.executable,'scripts/je06_bun_worker.py',mode,*map(str,args)]
    if cpu:command=['taskset','-c','80-87',*command]
    with (BASE/f'{mode}_{"smoke" if smoke else "formal"}_{time.time_ns()}.log').open('a') as log:
        return subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def gpu_sample(pid,phase):
    sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
    with (BASE/'gpu_samples.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(unix=time.time(),pid=pid,phase=phase,gpu=sample))+'\n')

def smoke():
    prepare();assert all(not p for p in gpu_owners().values()),'no GPU takeover'
    p=launch('formal',smoke=True);atomic_json(BASE/'SMOKE_LAUNCH.json',dict(pid=p.pid,devices=list(range(8))))
    while p.poll() is None:gpu_sample(p.pid,'production_smoke');time.sleep(5)
    assert p.returncode==0,'production smoke failed; preserve evidence'
    child=BASE/SMOKE_NAME
    cp=child/'formal'/ARM/'gen000002';m=verify_checkpoint(cp,sha(child/'plan.json'))
    rows=[__import__('json').loads(x) for x in (cp.parent/'events.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(r['finite'] and r['nonzero_update'] and r['optimizer_updates']==128 for r in rows)
    cpu=launch('restore-monitor','--resume',cp,cpu=True,smoke=True);assert cpu.wait()==0
    # Fresh GPU process executes the second generation from full g1 state and
    # compares it with uninterrupted g2. It never overwrites a smoke checkpoint.
    gpu=launch('replay','--resume',cp.parent/'gen000001',smoke=True)
    while gpu.poll() is None:gpu_sample(gpu.pid,'fresh_process_replay');time.sleep(5)
    assert gpu.returncode==0 and read(child/'state/FRESH_GPU_REPLAY.json')['passed']
    atomic_json(BASE/'PRODUCTION_SMOKE_COMPLETE.json',dict(passed=True,shape=[8,1024,256,4096],finite_nonzero=True,replicas_exact=m['replicas_exact'],
        smoke_namespace=SMOKE_NAME,checkpoint_sha256=sha(cp/'COMMITTED.json'),cpu_restore_sha256=sha(child/'state/RESTORE_MONITOR.json'),
        gpu_replay_sha256=sha(child/'state/FRESH_GPU_REPLAY.json'),xla_memory=[r['xla_memory'] for r in rows],
        steady_fresh_ticks_per_second=rows[-1]['fresh_ticks_per_second'],updates_per_generation=128,unix=time.time()))

def extension_positive(base,rows):
    def totals(d,key):return sum(d['hunter'][t][key] for t in ['hard','normal','easy'])
    checks=[]
    for row in rows:
        combat=totals(row,'safe_win')>=totals(base,'safe_win') and sum(row['hunter'][t]['safe_win'] for t in ['hard','normal'])>=sum(base['hunter'][t]['safe_win'] for t in ['hard','normal'])
        safety=all(totals(row,k)<=totals(base,k)+(2 if k=='death_episode' else 1) for k in ['death_episode','self_episodes','trade_episodes'])
        tactical=row['tactical']['safe_win']>=base['tactical']['safe_win']-2 and row['tactical']['self_episodes']<=base['tactical']['self_episodes']
        tasks=row['tasks'];bt=base['tasks']
        task=tasks['carry_home']['capture_episodes']>=bt['carry_home']['capture_episodes']-1 and tasks['near_steal']['steal_episodes']>=bt['near_steal']['steal_episodes']-1 and tasks['full']['self_episodes']<=bt['full']['self_episodes']
        gain=totals(row,'safe_win')>totals(base,'safe_win') or tasks['full']['capture_episodes']>bt['full']['capture_episodes'] or tasks['route_break']['capture_episodes']>bt['route_break']['capture_episodes']
        checks.append(dict(combat=combat,safety=safety,tactical=tactical,tasks=task,gain=gain))
    return bool(checks) and all(all(c.values()) for c in checks),checks

def train_block(target):
    for attempt in range(1,33):
        old=read(BASE/'TRAIN_LAUNCH.json') if (BASE/'TRAIN_LAUNCH.json').exists() else {}
        adopted=old.get('pid') and process_matches(old['pid'],'scripts/je06_bun_worker.py')
        assert adopted or all(not p for p in gpu_owners().values()),'no GPU takeover'
        cp=resolve_checkpoint(force=True)
        if cp and verify(cp)['generation']==target:return read(BASE/'formal'/ARM/'TRAINER_EXIT.json')
        child=Adopted(old['pid']) if adopted else launch('formal',*(['--resume',cp] if cp else []))
        atomic_json(BASE/'TRAIN_LAUNCH.json',dict(pid=child.pid,devices=list(range(8)),resume=str(cp.relative_to(ROOT)) if cp else None,attempt=attempt,target=target))
        while child.poll() is None:
            gpu_sample(child.pid,'formal')
            sf=BASE/'formal'/ARM/'STATE.json'
            if sf.exists() and not (BASE/'state/FIRST_NONZERO_UPDATE.json').exists():
                row=read(sf)
                if row['finite'] and row['nonzero_update']:
                    atomic_json(BASE/'state/FIRST_NONZERO_UPDATE.json',dict(worker_pid=child.pid,**row))
                    notify('first_update',f"JE06正式PID {child.pid}，GPU0–7，g{row['generation']}，finite非零Actor/Critic {row['actor_update_l2'][0]:.6f}/{row['critic_update_l2'][0]:.6f}，预算2048代/42.95亿fresh ticks，吞吐{row['fresh_ticks_per_second']:.0f}ticks/s。")
            time.sleep(5)
        final=BASE/'formal'/ARM/'TRAINER_EXIT.json'
        if child.returncode==0 and final.exists() and read(final)['generation']==target:return read(final)
        atomic_json(BASE/f'TECHNICAL_FAILURE_{target}_{attempt:03d}.json',dict(returncode=child.returncode,unix=time.time()))
        time.sleep(5)
    raise RuntimeError('technical recovery retry limit')

def owner():
    prepare();lease=lock(BASE/'owner.lock');atomic_json(BASE/'OWNER.json',dict(pid=os.getpid(),devices=list(range(8)),unix=time.time()))
    if not (BASE/'PRODUCTION_SMOKE_COMPLETE.json').exists():
        while any(gpu_owners().values()):
            atomic_json(BASE/'GPU_WAIT_STATE.json',dict(phase='WAIT_EXTERNAL_GPU_RELEASE',owners=gpu_owners(),unix=time.time(),no_takeover=True))
            time.sleep(10)
        smoke()
    if not (BASE/'BLOCK_TRAINING_COMPLETE_2048.json').exists():atomic_json(BASE/'BLOCK_TRAINING_COMPLETE_2048.json',train_block(2048))
    # This decision only governs the optional extra budget, never stops the
    # already authorized first2048 or waits at a mid-training checkpoint.
    while not (BASE/'authority_je06_recovery_v2/BLOCK_EVALUATION_COMPLETE_2048.json').exists():time.sleep(10)
    decision=BASE/'EXTENSION_DECISION.json'
    if not decision.exists():
        out=BASE/'authority_je06_recovery_v2';base=read(out/'baseline_g001536/COMPLETE.json');base['tasks']=read(out/'baseline_g001536/COMPLETE_WITH_BUN_TASKS.json')['tasks']
        rows=[]
        for g in [1536,2048]:
            folder=out/f'{ARM}_g{g:06d}';row=read(folder/'COMPLETE.json');row['tasks']=read(folder/'COMPLETE_WITH_BUN_TASKS.json')['tasks'];rows.append(row)
        positive,checks=extension_positive(base,rows)
        if positive:atomic_json(BASE/'EXTENSION_AUTHORIZED.json',dict(passed=True,target=4096,fresh_ticks=8589934592,checks=checks))
        atomic_json(decision,dict(positive=positive,checks=checks,screening_only=True,no_midrun_ability_gate=True))
    target=4096 if read(decision)['positive'] else 2048
    final=train_block(4096) if target==4096 else read(BASE/'BLOCK_TRAINING_COMPLETE_2048.json')
    atomic_json(BASE/'TRAINING_COMPLETE.json',final)
    notify('training_complete','JE06整体能力训练预算完成，CPU权威曲线异步收尾。')
    while not (BASE/'authority_je06_recovery_v2/EVALUATION_COMPLETE.json').exists():time.sleep(10)
    while not (BASE/'side_diagnostics_600_900_v1/OWNER_FINISHED.json').exists():time.sleep(10)
    atomic_json(BASE/'EXPERIMENT_COMPLETE.json',dict(passed=True,training_sha256=sha(BASE/'TRAINING_COMPLETE.json'),evaluation_sha256=sha(BASE/'authority_je06_recovery_v2/EVALUATION_COMPLETE.json'),side_sha256=sha(BASE/'side_diagnostics_600_900_v1/EVALUATION_COMPLETE.json')))
    notify('experiment_complete','JE06训练、全部权威评估和600/900tick旁路诊断完成；完整曲线保留，不默认终点最好。')
    atomic_json(BASE/'OWNER_FINISHED.json',dict(passed=True,unix=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','restart-prepare','owner']);a=ap.parse_args()
    if a.command=='restart-prepare':restart_preparation()
    elif a.command=='prepare':prepare()
    else:owner()
