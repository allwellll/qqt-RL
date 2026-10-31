"""Independent R2 paired exposure screen; R1 and reward candidates stay read-only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import retention_screen as engine
from scripts.population_run_state import atomic_json, sha, lock

BASE = ROOT / 'runs/retention_20261020/r2'
ARM = engine.ARM
R1 = ROOT / 'runs/retention_20261019/r1'
SCRIPT = 'scripts/retention_exposure_screen.py'


def declaration():
    return engine.read(BASE / 'DECLARATION.json')


def configure(run):
    worker = engine.configure(run)
    if engine.read(run / 'plan.json')['core_history_fraction'] == 1.:
        from scripts.retention_core_exposure import install
        install()
    return worker


def launch_worker(run, command='worker', resume=None, cpu=False):
    engine.require_training_authorized(run)
    args = ['taskset', '-c', '96-111' if cpu else '0-79', sys.executable,
            SCRIPT, command, '--run', str(run.relative_to(ROOT))]
    if resume:
        args += ['--resume', str(resume.relative_to(ROOT))]
    return subprocess.Popen(args, cwd=ROOT, env=engine.env_for(run, cpu),
        stdout=(run / (command + '.log')).open('a'), stderr=subprocess.STDOUT,
        start_new_session=True)


def prepare():
    assert not (BASE / 'DECLARATION.json').exists(), 'fresh R2 namespace required'
    seeds = [2026102001, 2026102002, 2026102003]
    specs = [dict(arm=arm, seed=seed, parent=str(engine.PARENT),
                  core_history_fraction=.5 if arm == 'A0' else 1.,
                  namespace=f'runs/retention_20261020/r2/{arm}_s{seed}')
             for seed in seeds for arm in ['A0', 'A1']]
    d = dict(experiment_id='RETENTION-20261020-R2', parent=str(engine.PARENT),
             train_seeds=seeds, generations=32, points=[0,8,16,24,32],
             return_multiplier=2, runs=specs, synthetic_training_carry=False,
             route_dev_seeds=list(range(2026102100,2026102132)),
             route_test_seeds=list(range(2026110100,2026110164)),
             dev_rng=20261019, test_rng=20261031,
             experiment_report='reports/retention_exposure_experiment_20261020.md',
             evaluation_report='reports/retention_exposure_evaluation_20261020.md',
             prerequisite='R1 fixed screen complete, evidence frozen; never an ability threshold',
             sole_factor='half versus all existing history ordinary-return slots from identical core',
             stopping='fixed32; technical fault recovery or explicit user stop; no ability gate')
    BASE.mkdir(parents=True)
    atomic_json(BASE / 'DECLARATION.json', d)
    source = R1 / 'A0_s2026101901'
    for spec in specs:
        run = ROOT / spec['namespace']
        shutil.copytree(source / 'inputs', run / 'inputs')
        shutil.copyfile(source / 'regions.json', run / 'regions.json')
        for folder in ['state','formal','cache','validation','diagnostics']:
            (run / folder).mkdir()
        plan = engine.read(source / 'plan.json')
        plan.update(experiment_id=d['experiment_id']+'-'+spec['arm']+'-'+str(spec['seed']),
                    seed=spec['seed'], retention_core=True,
                    core_history_fraction=spec['core_history_fraction'],
                    state_schema='retention-r2-'+spec['arm'])
        atomic_json(run / 'plan.json', plan)
        atomic_json(run / 'ROUND_DECLARATION.json', dict(**spec, evaluation=dict(
            seed=d['dev_rng'], points=d['points'], route_games=32, route_seed_start=2026102100)))
    # These administrative preparations do not alter or install any R1 source.
    sources = {str(p.relative_to(ROOT)): sha(p)
               for folder, pattern in [('scripts','*.py'),('scripts','*.js'),
                                       ('jax_bomb','*.py'),('web','*.js')]
               for p in (ROOT / folder).glob(pattern)}
    atomic_json(BASE / 'SOURCE_SEAL.json', dict(sources=sources))
    atomic_json(BASE / 'PREPARED.json', dict(passed=True, fresh_namespaces=True,
        common_parent_inputs_sha256={p.name: sha(p) for p in (source/'inputs').iterdir() if p.is_file()},
        numerical_installation_in_R1=False))


def evaluate_child(run, split, g):
    from scripts import retention_evaluate as evaluator
    evaluator.evaluate_child(run, split, g)


def evaluation_job(run, split, g):
    d = declaration()
    env = engine.env_for(run, True)
    env.update(JE10_ROUTE_PARALLEL='1', JE10_DIAGNOSTIC_REPORT=d['evaluation_report'],
        JE10_EVAL_SEED=str(d['dev_rng'] if split=='dev' else d['test_rng']),
        JE10_ROUTE_GAMES='32' if split=='dev' else '64',
        JE10_ROUTE_SEED_START=str(d['route_dev_seeds'][0] if split=='dev' else d['route_test_seeds'][0]),
        JE10_EVAL_NAMESPACE='authority_je10' if split=='dev' else 'authority_retention_test')
    return subprocess.run([sys.executable, SCRIPT, 'evalchild', '--run', str(run.relative_to(ROOT)),
        '--split', split, '--generation', str(g)], cwd=ROOT, env=env,
        stdout=(run/(split+'_eval.log')).open('a'), stderr=subprocess.STDOUT).returncode


def evaluation_owner():
    from scripts import retention_evaluate as evaluator
    evaluator.child = evaluation_job
    evaluator.owner()


def results():
    import numpy as np
    from scripts.retention_results import metrics, paired_interval
    d = declaration(); records=[]; pairs={}
    for spec in d['runs']:
        run=ROOT/spec['namespace']
        summaries=[engine.read(run/'authority_je10'/f'g{g:06d}'/'SUMMARY.json') for g in d['points']]
        points=[metrics(x) for x in summaries]
        tail={k:float(np.mean([x[k] for x in points[-3:]])) for k in points[0]}
        curve=[x['natural'] for x in points]
        auc=sum((b-a)*(curve[i]+curve[i+1])/2 for i,(a,b) in enumerate(zip(d['points'][:-1],d['points'][1:])))/32
        holdout=metrics(engine.read(run/'authority_retention_test/g000032/SUMMARY.json'))
        record=dict(**spec,curve=curve,tail=tail,worst=min(curve[-3:]),auc=auc,holdout=holdout)
        records.append(record);pairs[spec['arm'],spec['seed']]=record
    differences=[pairs['A1',s]['tail']['natural']-pairs['A0',s]['tail']['natural'] for s in d['train_seeds']]
    result=dict(passed=True,experiment_id=d['experiment_id'],runs=records,
        paired_tail_delta=differences,seed_bootstrap95=paired_interval(differences),
        actual_fresh_ticks=6*32*2097152,
        limitation='Three independent training seeds: exploratory only; positive training-state fit does not imply natural retention.')
    atomic_json(BASE/'RESULTS.json',result)
    lines=['\n## 固定筛选终态\n',f'实际训练量{result["actual_fresh_ticks"]:,} fresh ticks；配对自然尾差{differences}；seed-bootstrap95%{result["seed_bootstrap95"]}。',
        '|臂|seed|自然g0/8/16/24/32|尾均值|最差|AUC|test|','|---|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append('|'+r['arm']+'|'+str(r['seed'])+'|'+ '/'.join(f'{x:.3f}' for x in r['curve'])+
            f'|{r["tail"]["natural"]:.3f}|{r["worst"]:.3f}|{r["auc"]:.3f}|{r["holdout"]["natural"]:.3f}|')
    report=ROOT/d['experiment_report']
    with report.open('a') as stream:stream.write('\n'.join(lines)+'\n')
    return result


def preceding_training_released():
    active=R1/'RESUMPTION_DECLARATION.json'
    if not active.exists():return False
    specs=engine.read(active)['runs']
    for spec in specs:
        run=ROOT/spec['namespace']
        if not (run/'TRAINING_COMPLETE.json').exists():return False
        # The first control finished before release receipts were introduced;
        # its release was independently verified before the resumption smoke.
        if spec is not specs[0] and not (run/'GPU_RELEASE_RECEIPT.json').exists():return False
    return engine.read(R1/'STATUS.json')['state'] in ['TRAINING_COMPLETE_EVALUATING','SCREEN_COMPLETE']


def owner():
    # Only the six fixed trainer terminals and resource release define this
    # handoff. R1 holdout can continue on CPU; no performance threshold waits.
    while not preceding_training_released():
        engine.status('WAITING_R1_FIXED_TRAINING_RELEASE',controls_gpu=False)
        time.sleep(10)
    unique=lock(BASE/'owner.lock')
    engine.check_seal();engine.require_training_authorized()
    evaluator=subprocess.Popen(['taskset','-c','80-95',sys.executable,SCRIPT,'evalowner'],cwd=ROOT,
        env=engine.env_for(BASE,True),stdout=(BASE/'evaluation.log').open('a'),
        stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(BASE/'EVALUATOR_LAUNCH.json',dict(unique_owner=True,unix=time.time()))
    d=declaration()
    for spec in d['runs'][:2]:
        engine.smoke(spec)
    for spec in d['runs']:
        run=ROOT/spec['namespace'];out=run/'formal'/ARM
        if (run/'TRAINING_COMPLETE.json').exists():continue
        engine.wait_idle();engine.check_seal()
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
                        atomic_json(run/'FORMAL_NONZERO_CONFIRMED.json',dict(passed=True,
                            **{k:row[k] for k in ['generation','finite','nonzero_update','actor_update_l2','critic_update_l2','fresh_ticks']}))
                        engine.notify(f"R2 {spec['arm']} seed{spec['seed']}正式非零更新已确认。")
                        announced=True
                time.sleep(10)
            if proc.returncode==0 and (run/'TRAINING_COMPLETE.json').exists():break
            atomic_json(run/f'TECHNICAL_ATTEMPT_{attempt}.json',dict(exit_code=proc.returncode))
            engine.notify('R2技术故障，保留证据并从最近原子点恢复。');engine.wait_idle()
        else:raise RuntimeError('R2 technical retries exhausted')
        terminal=engine.read(out/'TRAINER_EXIT.json')
        assert terminal['generation']==32 and not terminal['atomic_stop']
        engine.validate_update(engine.read(out/'STATE.json'))
        atomic_json(run/'FIXED_RUN_TERMINAL.json',terminal)
        engine.wait_idle()
        atomic_json(run/'GPU_RELEASE_RECEIPT.json',dict(released=True,consecutive_idle_samples=2,
            resource=engine.read(BASE/'RESOURCE.json')))
        engine.notify(f"R2固定run终态 {spec['arm']} seed{spec['seed']} 32代，接续队列继续。")
    engine.status('TRAINING_COMPLETE_EVALUATING')
    assert evaluator.wait()==0,'R2 evaluation failed'
    result=results();atomic_json(BASE/'SCREEN_COMPLETE.json',result)
    engine.status('SCREEN_COMPLETE')
    engine.notify('R2固定筛选完成，完整证据已保存，需按自然/战斗/安全尾曲线选择后继。')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','owner','worker','restore','evalowner','evalchild'])
    p.add_argument('--run');p.add_argument('--resume');p.add_argument('--split');p.add_argument('--generation',type=int)
    args=p.parse_args()
    engine.BASE=BASE;engine.declaration=declaration;engine.spawn_worker=launch_worker
    if args.command=='prepare':prepare()
    elif args.command=='owner':owner()
    elif args.command=='evalowner':evaluation_owner()
    elif args.command=='evalchild':evaluate_child(ROOT/args.run,args.split,args.generation)
    else:
        engine.require_training_authorized(ROOT/args.run);engine.check_seal()
        worker=configure(ROOT/args.run)
        if args.command=='restore':worker.restore_monitor(ROOT/args.resume)
        else:worker.worker(ROOT/args.resume if args.resume else None)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if BASE.exists():atomic_json(BASE/'TECHNICAL_FAILURE.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
