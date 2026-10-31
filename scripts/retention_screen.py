"""Single-owner same-parent retention screen; never launches historical relay."""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/retention_20261019/r1'
PARENT=Path('runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032')
ARM='bun392_bridge_pfsp8'
from scripts.population_run_state import atomic_json,sha,lock


def declaration():
    active=BASE/'RESUMPTION_DECLARATION.json'
    if active.exists():return read(active)
    seeds=[2026101901,2026101902,2026101903]
    return dict(experiment_id='RETENTION-20261019-R1',parent=str(PARENT),train_seeds=seeds,
        generations=32,points=[0,8,16,24,32],auto_relay=False,synthetic_training_carry=False,
        return_multiplier=2,route_dev_seeds=list(range(2026102100,2026102132)),
        route_test_seeds=list(range(2026102900,2026102964)),
        runs=[dict(arm=arm,seed=seed,parent=str(PARENT),namespace=f'runs/retention_20261019/r1/{arm}_s{seed}') for seed in seeds for arm in ['A0','A1']])


def require_gpu_training_authorized():
    """A persistent user GPU pause overrides every experiment-local permit.

    Keep the marker and historical receipts when an explicit user resume is
    handled. Only its linked explicit authorization can supersede this hold.
    CPU evaluation does not call this guard.
    """
    marker=ROOT/'runs/retention_20261021/USER_GPU_PAUSE.json'
    if not marker.exists():return
    try:
        pause=read(marker)
        receipt=Path(pause.get('resume_receipt',''))
        if receipt.is_absolute() or '..' in receipt.parts:
            raise ValueError('resume receipt must be repository relative')
        resume=read(ROOT/receipt)
        authorized=(pause.get('status')=='SUPERSEDED_BY_EXPLICIT_USER_RESUME'
            and pause.get('automatic_training_allowed') is True
            and resume.get('decision')=='resume_gpu_experiments'
            and resume.get('authorized_by')=='explicit_user_instruction'
            and resume.get('supersedes')==str(marker.relative_to(ROOT))
            and resume.get('authorization_contract')=='.hermes/r5_authorized_resume_plan.md')
    except (OSError,ValueError,TypeError,AttributeError):
        authorized=False
    if not authorized:
        raise RuntimeError('Training blocked by user GPU pause; explicit user resume required')


def require_training_authorized(run=None):
    """A user semantic hold blocks all training launches, including retries."""
    require_gpu_training_authorized()
    marker=BASE/'SEMANTIC_AUDIT_HOLD.json'
    if marker.exists():
        decision=read(marker)
        authorized=(decision.get('automatic_training_allowed') is True
            and decision.get('audit_complete') is True
            and decision.get('status')=='AUDIT_COMPLETE_USER_AUTHORIZED_RESUME')
        if not authorized:raise RuntimeError('Training blocked by user semantic audit hold')
    if run is not None and any(run.glob('EXCLUDED*.json')):
        raise RuntimeError('Excluded training namespace cannot be resumed')


def resource_idle(rows,clients):
    return (len(rows)==8 and {r[0] for r in rows}==set(range(8)) and not clients
        and all(memory<1024 and util<5 for _,memory,util in rows))


def probe():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[tuple(map(int,line.split(','))) for line in raw.splitlines() if line.strip()]
    clients=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
    return resource_idle(rows,clients),rows,len(clients)


def wait_idle():
    stable=0
    while stable<2:
        require_training_authorized()
        idle,rows,clients=probe();stable=stable+1 if idle else 0
        atomic_json(BASE/'RESOURCE.json',dict(idle=idle,compute_clients=clients,consecutive_idle_samples=stable,gpus=rows,unix=time.time()))
        if stable<2:time.sleep(10)


def validate_update(row):
    if not row.get('finite') or not row.get('nonzero_update'):raise ValueError('invalid real update')
    for name in ['actor_update_l2','critic_update_l2']:
        values=row[name]
        if len(values)!=8 or not all(math.isfinite(v) and v>0 for v in values):raise ValueError(name)


def read(path):return json.loads(Path(path).read_text())


def env_for(run,cpu=False):
    from scripts.je03_bun831_20261011 import child_env
    env=child_env('' if cpu else '0,1,2,3,4,5,6,7',cpu)
    env.update(JE10_NAMESPACE=str(run.relative_to(ROOT)),JE09_H6_VERIFIER_SLOTS='4',
        JAX_COMPILATION_CACHE_DIR=str(run/'cache'/('cpu' if cpu else 'gpu')),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env


def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'PREPARED.json').exists():return
    d=declaration()
    if (BASE/'DECLARATION.json').exists():assert read(BASE/'DECLARATION.json')==d
    atomic_json(BASE/'DECLARATION.json',d)
    cp=ROOT/PARENT;marker=read(cp/'COMMITTED.json')
    assert marker['passed'] and sha(cp/'state.pkl')==marker['state_sha256'] and sha(cp/'policy.pt')==marker['policy_sha256']
    with (cp/'state.pkl').open('rb') as stream:saved=pickle.load(stream)
    import jax
    from scripts.je03_bun_stage1_20261011 import digest
    from scripts.je03_bun831_20261011 import save_pickle
    assert digest(saved['live'])==marker['full_state_digest']
    initial=jax.tree.map(lambda x:x[0],saved['live']['params']);pool=saved['live']['pool']
    parent_run=cp.parents[2]
    for spec in d['runs']:
        run=ROOT/spec['namespace'];run.mkdir(parents=True,exist_ok=True)
        for folder in ['inputs','state','formal','cache','validation','diagnostics']:(run/folder).mkdir(exist_ok=True)
        p=read(parent_run/'plan.json')
        p.update(experiment_id=d['experiment_id']+'-'+spec['arm']+'-'+str(spec['seed']),seed=spec['seed'],generations=32,
            checkpoints=d['points'],total_fresh_ticks=32*2097152,recovery_every=4,
            state_schema='retention-r1-'+spec['arm'],retention_core=spec['arm']=='A1',ring_generation_offset=32)
        p['campaign']['stopping']='Fixed32 complete; technical failure or user stop only. No relay or ability gates.'
        p['parent_selection']=dict(checkpoint=str(PARENT),policy_sha256=marker['policy_sha256'],optimizer='fresh on every arm',critic='same-parent matching raw')
        atomic_json(run/'plan.json',p)
        atomic_json(run/'ROUND_DECLARATION.json',dict(**spec,evaluation=dict(seed=20261019,points=d['points'],route_games=32,route_seed_start=2026102100)))
        save_pickle(run/'inputs/initial.pkl',initial);save_pickle(run/'inputs/retention_parent.pkl',initial)
        save_pickle(run/'inputs/initial_pool.pkl',pool)
        for name in ['frozen.pkl','regions.pkl','return_replay.pkl','RETURN_REPLAY.json']:shutil.copyfile(parent_run/'inputs'/name,run/'inputs'/name)
        from scripts.je10_causal_replay import load_verified
        load_verified(run/'inputs/return_replay.pkl',run/'inputs/RETURN_REPLAY.json')
        shutil.copyfile(parent_run/'regions.json',run/'regions.json')
    # One immutable code snapshot manifest for the complete matrix. No seal of
    # historical runs is modified, and no source is rewritten by the launcher.
    sources={str(p.relative_to(ROOT)):sha(p) for folder,pattern in [('scripts','*.py'),('scripts','*.js'),('jax_bomb','*.py'),('web','*.js')] for p in (ROOT/folder).glob(pattern)}
    atomic_json(BASE/'SOURCE_SEAL.json',dict(sources=sources))
    atomic_json(BASE/'PREPARED.json',dict(passed=True,parent_policy_sha256=marker['policy_sha256'],runs=6,unix=time.time()))
    status('PREPARED',training_ticks=0)


def check_seal():
    seal=BASE/'RESUMPTION_SOURCE_SEAL.json'
    if not seal.exists():seal=BASE/'SOURCE_SEAL.json'
    for path,expected in read(seal)['sources'].items():
        if sha(ROOT/path)!=expected:raise ValueError('screen source seal changed: '+path)


def configure(run):
    from scripts import je10_campaign as campaign
    campaign.BASE=run
    worker=campaign.configure_worker(run)
    if read(run/'plan.json')['retention_core']:
        from scripts.retention_core_replay import install
        install()
    return worker


def spawn_worker(run,command='worker',resume=None,cpu=False):
    if command=='worker':require_training_authorized(run)
    args=['taskset','-c','96-111' if cpu else '0-79',sys.executable,'scripts/retention_screen.py',command,'--run',str(run.relative_to(ROOT))]
    if resume:args+=['--resume',str(resume.relative_to(ROOT))]
    return subprocess.Popen(args,cwd=ROOT,env=env_for(run,cpu),stdout=(run/(command+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)


def status(state,**extra):atomic_json(BASE/'STATUS.json',dict(state=state,unix=time.time(),**extra))


def notify(message):
    print(message,flush=True)
    try:subprocess.run(['jdme-notify','send',message],timeout=20,check=False,stdout=subprocess.DEVNULL)
    except (FileNotFoundError,subprocess.TimeoutExpired):pass


def smoke(spec):
    require_training_authorized(ROOT/spec['namespace'])
    source=ROOT/spec['namespace'];run=BASE/spec.get('smoke_name','smoke_'+spec['arm'])
    if (run/'TECHNICAL_COMPLETE.json').exists():return
    if not run.exists():
        shutil.copytree(source/'inputs',run/'inputs');shutil.copyfile(source/'regions.json',run/'regions.json')
        for folder in ['state','formal','cache','diagnostics']:(run/folder).mkdir(exist_ok=True)
        plan=read(source/'plan.json');plan.update(generations=1,checkpoints=[0,1],seed=2026101999,total_fresh_ticks=2097152)
        atomic_json(run/'plan.json',plan)
    wait_idle();status('TECHNICAL_SMOKE',arm=spec['arm'],run=str(run.relative_to(ROOT)))
    proc=spawn_worker(run);assert proc.wait()==0,'technical smoke failed'
    row=read(run/'formal'/ARM/'STATE.json');validate_update(row)
    cp=run/'formal'/ARM/'gen000001'
    assert spawn_worker(run,'restore',cp,cpu=True).wait()==0,'fresh CPU restore failed'
    restored=read(run/'state/RESTORE_MONITOR.json');assert restored['fresh_process'] and restored['passed']
    atomic_json(run/'TECHNICAL_COMPLETE.json',dict(passed=True,finite=True,nonzero_actor_critic=True,fresh_process_restore=True,capability_gate=False))


def owner():
    BASE.mkdir(parents=True,exist_ok=True)
    require_training_authorized()
    unique=lock(BASE/'owner.lock');check_seal()
    # Also serialize against the legacy campaign administrative lock without
    # invoking its successor mechanism. Old paused namespace is read-only.
    d=declaration();status('WAITING_RESOURCE',training_ticks=0)
    for spec in d['runs'][:2]:
        if not (ROOT/spec['namespace']/'TRAINING_COMPLETE.json').exists():smoke(spec)
    evaluator=subprocess.Popen(['taskset','-c','80-95',sys.executable,'scripts/retention_evaluate.py'],cwd=ROOT,
        env=env_for(BASE,True),stdout=(BASE/'evaluation.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(BASE/'EVALUATOR_LAUNCH.json',dict(unique_owner=True,state='authority_je10/QUEUE.json',unix=time.time()))
    total=0
    for spec in d['runs']:
        run=ROOT/spec['namespace'];out=run/'formal'/ARM
        if (run/'TRAINING_COMPLETE.json').exists():total+=32*2097152;continue
        wait_idle();check_seal();status('TRAINING',run=spec['namespace'],arm=spec['arm'],seed=spec['seed'],training_ticks=total)
        resume=None
        for attempt in range(3):
            if (out/'LATEST_CHECKPOINT.json').exists():resume=ROOT/read(out/'LATEST_CHECKPOINT.json')['checkpoint']
            proc=spawn_worker(run,resume=resume)
            atomic_json(run/'OWNER_STATE.json',dict(launched=True,attempt=attempt,run=spec['namespace'],restore_checkpoint=str(resume.relative_to(ROOT)) if resume else None,unix=time.time()))
            announced=False
            while proc.poll() is None:
                state=out/'STATE.json'
                if state.exists() and not announced:
                    row=read(state);validate_update(row)
                    if row['generation']>=1 and (out/'gen000001'/'COMMITTED.json').exists():
                        atomic_json(run/'FORMAL_NONZERO_CONFIRMED.json',dict(passed=True,formal_nonzero_verified=True,**{k:row[k] for k in ['generation','finite','nonzero_update','actor_update_l2','critic_update_l2','fresh_ticks']}))
                        notify(f"保持筛选 {spec['arm']} seed{spec['seed']} 已确认正式g{row['generation']} finite，Actor/Critic八副本真实非零更新。")
                        announced=True
                time.sleep(10)
            if proc.returncode==0 and (run/'TRAINING_COMPLETE.json').exists():break
            atomic_json(run/f'TECHNICAL_ATTEMPT_{attempt}.json',dict(exit_code=proc.returncode,unix=time.time()))
            notify(f"保持筛选技术故障 {spec['arm']} seed{spec['seed']}，保留证据，从最近原子checkpoint恢复。")
            wait_idle()
        else:raise RuntimeError('technical retries exhausted')
        terminal=read(out/'TRAINER_EXIT.json');assert terminal['generation']==32 and not terminal['atomic_stop']
        validate_update(read(out/'STATE.json'));total+=32*2097152
        atomic_json(run/'FIXED_RUN_TERMINAL.json',dict(terminal,passed=True))
        wait_idle()
        atomic_json(run/'GPU_RELEASE_RECEIPT.json',dict(released=True,compute_clients=0,
            consecutive_idle_samples=2,resource=read(BASE/'RESOURCE.json'),unix=time.time()))
        notify(f"保持筛选固定run完成 {spec['arm']} seed{spec['seed']}：32代，67,108,864 fresh ticks；评估异步继续。")
    status('TRAINING_COMPLETE_EVALUATING',training_ticks=total)
    if evaluator.wait()!=0:raise RuntimeError('CPU evaluation failed; training artifacts retained')
    from scripts.retention_results import summarize
    result=summarize(BASE)
    atomic_json(BASE/'SCREEN_COMPLETE.json',result);status('SCREEN_COMPLETE',training_ticks=total,candidate=result['candidate'])
    notify('同父多seed保持筛选完成；候选：'+str(result['candidate'])+'。结果见 reports/retention_experiment_20261019.md。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','owner','worker','restore']);ap.add_argument('--run');ap.add_argument('--resume');args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='owner':owner()
    else:
        if args.command=='worker':require_training_authorized(ROOT/args.run)
        check_seal();run=ROOT/args.run;worker=configure(run)
        if args.command=='restore':worker.restore_monitor(ROOT/args.resume)
        else:worker.worker(ROOT/args.resume if args.resume else None)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if BASE.exists():atomic_json(BASE/'TECHNICAL_FAILURE.json',dict(type=type(error).__name__,error=str(error),unix=time.time()))
        raise
