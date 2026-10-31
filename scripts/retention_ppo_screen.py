"""Fresh paired certified-return PPO screen after the completed R2 decision."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import retention_screen as engine
from scripts.population_run_state import atomic_json,sha,lock

BASE=ROOT/'runs/retention_20261021/r3'
R2=ROOT/'runs/retention_20261020/r2'
SCRIPT='scripts/retention_ppo_screen.py'
ARM=engine.ARM


def declaration():return engine.read(BASE/'DECLARATION.json')


def checkpoint_receipt(run,generation):
    cp=run/'formal'/ARM/f'gen{generation:06d}'
    marker=engine.read(cp/'COMMITTED.json')
    assert marker['passed'] and marker['replicas_exact'] and marker['devices']==8
    assert marker['generation']==generation and marker['adam_steps']==[128*generation]*8
    assert marker['state_schema']==engine.read(run/'plan.json')['state_schema']
    assert sha(run/'plan.json')==marker['plan_sha256']
    assert sha(cp/'state.pkl')==marker['state_sha256']
    assert sha(cp/'policy.pt')==marker['policy_sha256']
    return dict(passed=True,generation=generation,atomic_sha_verified=True,
        replicas_exact=True,adam_steps=marker['adam_steps'],
        policy_sha256=marker['policy_sha256'])


def configure(run):
    worker=engine.configure(run)
    plan=engine.read(run/'plan.json')
    if plan['core_history_fraction']==1.:
        from scripts.retention_core_exposure import install
        install()
    from scripts.retention_return_ppo import install,install_worker_telemetry
    installed=install(plan['certified_return_ppo_factor'])
    from scripts import je10_ops
    assert je10_ops.ppo is installed
    assert installed._return_ppo_factor==plan['certified_return_ppo_factor']
    atomic_json(run/'state/RETURN_PPO_INSTALL.json',dict(passed=True,
        worker_call_site='scripts.je10_ops.ppo',
        certified_return_ppo_factor=installed._return_ppo_factor,
        in_memory_only=True,original_denominator=True))
    install_worker_telemetry(worker)
    return worker


def launch_worker(run,command='worker',resume=None,cpu=False):
    engine.require_training_authorized(run)
    if command=='worker' and run.name.startswith('smoke_'):
        seed_baseline_compilation(run)
    args=['taskset','-c','96-111' if cpu else '0-79',sys.executable,SCRIPT,
        command,'--run',str(run.relative_to(ROOT))]
    if resume:args+=['--resume',str(resume.relative_to(ROOT))]
    return subprocess.Popen(args,cwd=ROOT,env=engine.env_for(run,cpu),
        stdout=(run/(command+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)


def seed_baseline_compilation(run):
    """Warm only immutable executable keys; JAX recompiles changed losses."""
    if (run/'BASELINE_COMPILE_CACHE_REUSE.json').exists():return
    source=R2/'smoke_A0/cache/gpu';files=[]
    for path in sorted(source.rglob('*-cache')):
        if not path.is_file():continue
        relative=path.relative_to(source);target=run/'cache/gpu'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():continue
        shutil.copyfile(path,target)
        assert sha(path)==sha(target)
        files.append(dict(name=str(relative),sha256=sha(target)))
    atomic_json(run/'BASELINE_COMPILE_CACHE_REUSE.json',dict(files=files,
        source=str(source.relative_to(ROOT)),learner_state_copied=False,
        independent_cache_directory=True,changed_hlo_recompiled_by_jax=True))


def reuse_smoke_compilation(run,arm):
    """Copy executable caches only; every learner field starts from raw parent."""
    if (run/'SMOKE_COMPILE_CACHE_REUSE.json').exists():return
    source=BASE/('smoke_'+arm)
    assert engine.read(source/'TECHNICAL_COMPLETE.json')['passed']
    files=[]
    for path in sorted((source/'cache/gpu').rglob('*-cache')):
        if not path.is_file():continue
        relative=path.relative_to(source/'cache/gpu');target=run/'cache/gpu'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists(),'fresh compilation directory expected'
        shutil.copyfile(path,target)
        assert sha(path)==sha(target)
        files.append(dict(name=str(relative),sha256=sha(target)))
    atomic_json(run/'SMOKE_COMPILE_CACHE_REUSE.json',dict(files=files,
        source=str(source.relative_to(ROOT)),same_arm_only=True,
        independent_cache_directory=True,learner_state_copied=False))


def prepare():
    assert not BASE.exists(),'fresh R3 namespace required'
    assert (R2/'SCREEN_COMPLETE.json').exists() and (R2/'FULL_COMPARISON_20261021.json').exists()
    result=engine.read(R2/'FULL_COMPARISON_20261021.json')
    core_fraction=1. if result['candidate']=='A1_pending_confirmation' else .5
    source=R2/'A0_s2026102001'
    seeds=[2026102101,2026102102,2026102103]
    specs=[dict(arm=arm,seed=seed,parent=str(engine.PARENT),
        core_history_fraction=core_fraction,certified_return_ppo_factor=1. if arm=='A0' else .5,
        namespace=f'runs/retention_20261021/r3/{arm}_s{seed}') for seed in seeds for arm in ['A0','A1']]
    d=dict(experiment_id='RETENTION-20261021-R3',parent=str(engine.PARENT),train_seeds=seeds,
        generations=32,points=[0,8,16,24,32],return_multiplier=2,synthetic_training_carry=False,
        runs=specs,dev_rng=20261019,test_rng=20261102,
        route_dev_seeds=list(range(2026102100,2026102132)),
        route_test_seeds=list(range(2026110300,2026110364)),
        sole_factor='Certified native ordinary-return Actor clipped-surrogate row weight 1 versus 0.5; original denominator',
        shared_core_fraction=core_fraction,r2_decision_sha256=sha(R2/'FULL_COMPARISON_20261021.json'),
        experiment_report='reports/retention_ppo_experiment_20261021.md',
        evaluation_report='reports/retention_ppo_evaluation_20261021.md',
        stopping='Fixed32, technical recovery or explicit user stop; no capability admission/promotion gate',
        actual_fresh_ticks_budget=402653184,
        reward_hypotheses_installed=False,
        unchanged=['reward','GAE','bootstrap','advantage normalization','Critic','entropy','KL',
            'DAgger return x2','replay rows and selection','native custody authority','curriculum','opponents'])
    BASE.mkdir(parents=True)
    atomic_json(BASE/'DECLARATION.json',d)
    atomic_json(BASE/'SEMANTIC_AUDIT_HOLD.json',dict(status='AUDIT_COMPLETE_USER_AUTHORIZED_RESUME',
        automatic_training_allowed=True,audit_complete=True,
        reason='User authorized post-R2 evidence-driven independent iterations; no excluded namespace resume.',
        semantic_decision='Free-alive autonomous stock remains legal; teacher/PBRS current-window only; no synthetic custody.'))
    for spec in specs:
        run=ROOT/spec['namespace']
        shutil.copytree(source/'inputs',run/'inputs')
        shutil.copyfile(source/'regions.json',run/'regions.json')
        for folder in ['state','formal','cache','validation','diagnostics']:(run/folder).mkdir()
        plan=engine.read(source/'plan.json')
        plan.update(experiment_id=d['experiment_id']+'-'+spec['arm']+'-'+str(spec['seed']),
            seed=spec['seed'],retention_core=True,core_history_fraction=core_fraction,
            certified_return_ppo_factor=spec['certified_return_ppo_factor'],
            state_schema='retention-r3-'+spec['arm'])
        atomic_json(run/'plan.json',plan)
        atomic_json(run/'ROUND_DECLARATION.json',dict(**spec,evaluation=dict(
            seed=d['dev_rng'],points=d['points'],route_games=32,route_seed_start=2026102100)))
    sources={str(p.relative_to(ROOT)):sha(p) for folder,pattern in
        [('scripts','*.py'),('scripts','*.js'),('jax_bomb','*.py'),('web','*.js')]
        for p in (ROOT/folder).glob(pattern)}
    atomic_json(BASE/'SOURCE_SEAL.json',dict(sources=sources))
    atomic_json(BASE/'PREPARED.json',dict(passed=True,fresh_namespaces=True,
        common_parent_inputs_sha256={p.name:sha(p) for p in (source/'inputs').iterdir() if p.is_file()},
        runtime_install_only=True,r2_numerical_source_unchanged=True))


def evaluation_job(run,split,g):
    d=declaration();env=engine.env_for(run,True)
    env.update(JE10_ROUTE_PARALLEL='1',JE10_DIAGNOSTIC_REPORT=d['evaluation_report'],
        JE10_EVAL_SEED=str(d['dev_rng'] if split=='dev' else d['test_rng']),
        JE10_ROUTE_GAMES='32' if split=='dev' else '64',
        JE10_ROUTE_SEED_START=str(d['route_dev_seeds'][0] if split=='dev' else d['route_test_seeds'][0]),
        JE10_EVAL_NAMESPACE='authority_je10' if split=='dev' else 'authority_retention_test')
    return subprocess.run([sys.executable,SCRIPT,'evalchild','--run',str(run.relative_to(ROOT)),
        '--split',split,'--generation',str(g)],cwd=ROOT,env=env,
        stdout=(run/(split+'_eval.log')).open('a'),stderr=subprocess.STDOUT).returncode


def owner():
    unique=lock(BASE/'owner.lock');engine.require_training_authorized();engine.check_seal()
    assert (R2/'SCREEN_COMPLETE.json').exists(),'R2 fixed decision prerequisite'
    # R2 owner has no training left; evaluator takes the same global CPU lock.
    evaluator=subprocess.Popen(['taskset','-c','80-95',sys.executable,SCRIPT,'evalowner'],cwd=ROOT,
        env=engine.env_for(BASE,True),stdout=(BASE/'evaluation.log').open('a'),
        stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(BASE/'EVALUATOR_LAUNCH.json',dict(unique_owner=True,unix=time.time()))
    d=declaration()
    for spec in d['runs'][:2]:engine.smoke(spec)
    for run_index,spec in enumerate(d['runs']):
        run=ROOT/spec['namespace'];out=run/'formal'/ARM
        if (run/'TRAINING_COMPLETE.json').exists():continue
        engine.wait_idle();engine.check_seal()
        if not (out/'LATEST_CHECKPOINT.json').exists():reuse_smoke_compilation(run,spec['arm'])
        engine.status('TRAINING',run=spec['namespace'],arm=spec['arm'],seed=spec['seed'])
        for attempt in range(3):
            latest=out/'LATEST_CHECKPOINT.json'
            resume=ROOT/engine.read(latest)['checkpoint'] if latest.exists() else None
            proc=launch_worker(run,resume=resume)
            atomic_json(run/'OWNER_STATE.json',dict(attempt=attempt,restore_checkpoint=str(resume.relative_to(ROOT)) if resume else None))
            announced=False
            while proc.poll() is None:
                if (out/'STATE.json').exists() and not announced:
                    row=engine.read(out/'STATE.json');engine.validate_update(row)
                    if (out/'gen000001/COMMITTED.json').exists():
                        proof=checkpoint_receipt(run,1)
                        atomic_json(run/'FORMAL_NONZERO_CONFIRMED.json',dict(passed=True,
                            **{k:row[k] for k in ['generation','finite','nonzero_update','actor_update_l2','critic_update_l2','fresh_ticks']},
                            first_atomic_checkpoint=proof,
                            return_ppo_install=engine.read(run/'state/RETURN_PPO_INSTALL.json'),
                            certified_return_exposure=row['native_return_retention_exposure']))
                        engine.notify(f"R3 {spec['arm']} seed{spec['seed']}正式首个finite/Actor-Critic非零更新确认。")
                        announced=True
                time.sleep(10)
            if proc.returncode==0 and (run/'TRAINING_COMPLETE.json').exists():break
            atomic_json(run/f'TECHNICAL_ATTEMPT_{attempt}.json',dict(exit_code=proc.returncode))
            engine.notify('R3技术故障，保留证据并从最近原子点恢复。');engine.wait_idle()
        else:raise RuntimeError('R3 technical retries exhausted')
        terminal=engine.read(out/'TRAINER_EXIT.json')
        assert terminal['generation']==32 and not terminal['atomic_stop']
        engine.validate_update(engine.read(out/'STATE.json'))
        atomic_json(run/'FIXED_RUN_TERMINAL.json',dict(terminal,atomic_checkpoint=checkpoint_receipt(run,32)))
        engine.wait_idle()
        atomic_json(run/'GPU_RELEASE_RECEIPT.json',dict(released=True,consecutive_idle_samples=2,
            resource=engine.read(BASE/'RESOURCE.json')))
        successor=(d['runs'][run_index+1]['namespace'] if run_index+1<len(d['runs'])
            else '固定训练预算完成；唯一CPU/JS继续已声明评估')
        engine.notify(f"R3 {spec['arm']} seed{spec['seed']}固定g32，原子state/policy SHA通过，worker正常退出；"
            f"8卡连续两次释放核验，唯一owner持续、CPU/JS独立；后继{successor}。")
    engine.status('TRAINING_COMPLETE_EVALUATING')
    assert evaluator.wait()==0,'R3 CPU evaluation failed'
    from scripts import retention_exposure_screen as shared
    shared.BASE=BASE;shared.declaration=declaration
    result=shared.results();atomic_json(BASE/'SCREEN_COMPLETE.json',result)
    engine.status('SCREEN_COMPLETE');engine.notify('R3完整固定筛选完成，按完整自然/战斗/安全证据选择后继。')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','owner','worker','restore','evalowner','evalchild'])
    p.add_argument('--run');p.add_argument('--resume');p.add_argument('--split');p.add_argument('--generation',type=int)
    args=p.parse_args()
    engine.BASE=BASE;engine.declaration=declaration;engine.spawn_worker=launch_worker
    if args.command=='prepare':prepare()
    elif args.command=='owner':owner()
    elif args.command in ['evalowner','evalchild']:
        from scripts import retention_evaluate as evaluator
        if args.command=='evalowner':evaluator.child=evaluation_job;evaluator.owner()
        else:evaluator.evaluate_child(ROOT/args.run,args.split,args.generation)
    else:
        engine.require_training_authorized(ROOT/args.run);engine.check_seal()
        worker=configure(ROOT/args.run)
        if args.command=='restore':worker.restore_monitor(ROOT/args.resume)
        else:worker.worker(ROOT/args.resume if args.resume else None)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if BASE.exists():
            atomic_json(BASE/'TECHNICAL_FAILURE.json',dict(type=type(exc).__name__,error=str(exc)))
            engine.notify('R3技术停止：'+type(exc).__name__+'；保留最后原子点，需修复后恢复，未作能力判停。')
        raise
