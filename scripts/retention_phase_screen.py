"""R5 stage conditioning; serial after complete R4, no import side effects."""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import retention_screen as engine
from scripts.population_run_state import atomic_json,sha,lock

BASE = ROOT/'runs/retention_20261021/r5'
PRIOR = ROOT/'runs/retention_20261021/r4'
SCRIPT = 'scripts/retention_phase_screen.py'
ARM = engine.ARM


def declaration():
    return engine.read(BASE/'DECLARATION.json')


def build_probe(source):
    import numpy as np
    target=BASE/'FIXED_PHASE_PROBE.json'
    subprocess.run(['node','scripts/retention_phase_probe_states.js',
                    str(source/'authority_je10/g000000'),str(target)],cwd=ROOT,check=True)
    data=engine.read(target);rows=data['rows']
    np.savez(BASE/'FIXED_PHASE_PROBE.npz',
        obs=np.asarray([r['obs'] for r in rows],np.float32).reshape(-1,24,13,15),
        gv=np.asarray([r['gv'] for r in rows],np.float32),
        task=np.asarray([r['task'] for r in rows]),
        artificial_carry=np.asarray([r['artificial_carry'] for r in rows],bool))
    atomic_json(BASE/'PHASE_PROBE_MANIFEST.json',dict(passed=True,states=len(rows),
        source=str((source/'authority_je10/g000000').relative_to(ROOT)),sources=data['sources'],
        sample_sha256=sha(BASE/'FIXED_PHASE_PROBE.npz'),diagnostic_only=True,controls_training=False,
        selection='Every20 ticks, all eight tasks, original fixed first seed; no favourable-seed replacement',
        artificial_carry_labelled_separately=True))


def prepare():
    engine.require_gpu_training_authorized()
    assert not BASE.exists(),'fresh R5 namespace required'
    prior=engine.read(PRIOR/'DECLARATION.json')
    assert (PRIOR/'SCREEN_COMPLETE.json').exists(),'complete R4 required'
    assert (PRIOR/'EVALUATION_COMPLETE.json').exists(),'complete R4 holdout required'
    evaluation=engine.read(PRIOR/'EVALUATION_COMPLETE.json')
    assert evaluation['passed'] and len(evaluation['completed'])==30
    for spec in prior['runs']:
        run=ROOT/spec['namespace']
        terminal=engine.read(run/'FIXED_RUN_TERMINAL.json')
        assert terminal['generation']==32 and not terminal['atomic_stop']
        assert engine.read(run/'GPU_RELEASE_RECEIPT.json')['released']
        for g in [0,8,16,24,32]:
            assert (run/'authority_je10'/f'g{g:06d}'/'COMPLETE.json').exists()
        for g in [0,32]:
            assert (run/'authority_retention_test'/f'g{g:06d}'/'COMPLETE.json').exists()
    conclusion=PRIOR/'FULL_COMPARISON_20261021.json'
    decision=engine.read(conclusion)
    assert decision['candidate'] in ['A1_pending_confirmation','A0_or_no_improving_candidate']
    priority=decision['candidate']=='A1_pending_confirmation'
    # Both arms reuse the same frozen raw parent, never a selected R4 peak.
    source=ROOT/prior['runs'][0]['namespace']
    template=engine.read(source/'plan.json')
    parent=template['parent_selection']['checkpoint']
    seeds=[2026102107,2026102108,2026102109]
    specs=[dict(arm=arm,seed=seed,parent=parent,phase_conditioning=arm=='A1',
                namespace=str((BASE/f'{arm}_s{seed}').relative_to(ROOT)))
           for seed in seeds for arm in ['A0','A1']]
    d=dict(experiment_id='RETENTION-20261021-R5',parent=parent,train_seeds=seeds,
        generations=32,points=[0,8,16,24,32],runs=specs,dev_rng=20261019,test_rng=20261106,
        route_dev_seeds=list(range(2026102100,2026102132)),
        route_test_seeds=list(range(2026110700,2026110764)),
        sole_factor='Current-visible exclusive five-phase one-hot in Actor/Critic state token',
        shared_core_error_priority=priority,shared_core_fraction=.5,
        shared_certified_return_ppo_factor=1.,return_multiplier=2,
        phase_names=['COMBAT_CONTROL','STOCK_WINDOW','DROP_RECOVERY','CARRY_RETURN','CAGE_DELIVERY'],
        phase_priority='delivery > enemy cargo > visible enemy-origin drop > current stock window > combat',
        phase_encoding='phase5-visible-v1',phase_history=False,continuous_fields=False,
        delivery_boundary='self splat mass >= .5 on own cage plus four-neighbor halo',
        initialization='append five zero state_w rows, preserve all old leaves; fresh optimizer/env/RNG/FIFO/core',
        raw_observation_dimensions=[24,24],conditioned_state_dimension=29,
        synthetic_training_carry=False,reward_hypotheses_installed=False,
        old_compilation_cache_reused=False,actual_fresh_ticks_budget=402653184,
        prior_conclusion_sha256=sha(conclusion),
        experiment_report='reports/retention_phase_conditioning_experiment_20261021.md',
        evaluation_report='reports/retention_phase_conditioning_evaluation_20261021.md',
        stopping='Fixed32; technical recovery or explicit user stop only, no capability gate',
        unchanged=['reward','PBRS','curriculum','opponents','core sampling shared per complete R4 conclusion',
                   'replay rows and strata','PPO','DAgger return x2','safety','causal custody','budget'])
    BASE.mkdir(parents=True)
    atomic_json(BASE/'DECLARATION.json',d)
    atomic_json(BASE/'SEMANTIC_AUDIT_HOLD.json',dict(status='AUDIT_COMPLETE_USER_AUTHORIZED_RESUME',
        automatic_training_allowed=True,audit_complete=True,
        reason='User authorized serial R5 after complete R4; no excluded namespace.',
        semantic_decision='Free-alive autonomous stock legal; teacher/PBRS current-window only; no synthetic custody.'))
    from scripts.retention_phase_conditioning import migrate
    raw_hashes={p.name:sha(p) for p in (source/'inputs').iterdir() if p.is_file()}
    for spec in specs:
        run=ROOT/spec['namespace']
        shutil.copytree(source/'inputs',run/'inputs')
        shutil.copyfile(source/'regions.json',run/'regions.json')
        for folder in ['state','formal','cache','validation','diagnostics']:(run/folder).mkdir()
        if spec['phase_conditioning']:
            for name in ['initial.pkl','retention_parent.pkl','initial_pool.pkl','frozen.pkl']:
                path=run/'inputs'/name
                with path.open('rb') as stream:params=pickle.load(stream)
                with path.open('wb') as stream:pickle.dump(migrate(params),stream,protocol=pickle.HIGHEST_PROTOCOL)
        plan=dict(template)
        plan.update(experiment_id=d['experiment_id']+'-'+spec['arm']+'-'+str(spec['seed']),
                    seed=spec['seed'],phase_conditioning=spec['phase_conditioning'],
                    core_error_priority=priority,core_history_fraction=.5,certified_return_ppo_factor=1.,
                    state_schema='retention-r5-'+spec['arm'])
        atomic_json(run/'plan.json',plan)
        atomic_json(run/'ROUND_DECLARATION.json',dict(**spec,evaluation=dict(
            seed=d['dev_rng'],points=d['points'],route_games=32,route_seed_start=2026102100)))
        atomic_json(run/'INPUT_MIGRATION.json',dict(passed=True,phase_conditioning=spec['phase_conditioning'],
            raw_parent_inputs_sha256=raw_hashes,
            actual_inputs_sha256={p.name:sha(p) for p in (run/'inputs').iterdir() if p.is_file()},
            only_state_projection_zero_rows_added=spec['phase_conditioning'],
            no_optimizer_or_env_loaded=True))
    sources={str(p.relative_to(ROOT)):sha(p) for folder,pattern in
        [('scripts','*.py'),('scripts','*.js'),('jax_bomb','*.py'),('web','*.js')]
        for p in (ROOT/folder).glob(pattern)}
    atomic_json(BASE/'SOURCE_SEAL.json',dict(sources=sources))
    build_probe(source)
    atomic_json(BASE/'PREPARED.json',dict(passed=True,fresh_namespaces=True,
        prior_complete=True,prior_conclusion_sha256=sha(conclusion),raw_parent_inputs_sha256=raw_hashes))


def environment(run,cpu=False):
    env=engine.env_for(run,cpu)
    # An isolated process add-on; shared production JS source stays unchanged.
    env['NODE_OPTIONS']=(env.get('NODE_OPTIONS','')+' --require ./web/retention_phase_conditioning.js').strip()
    return env


def launch_worker(run,command='worker',resume=None,cpu=False):
    engine.require_training_authorized(run)
    args=['taskset','-c','96-111' if cpu else '0-79',sys.executable,SCRIPT,command,
          '--run',str(run.relative_to(ROOT))]
    if resume:args+=['--resume',str(resume.relative_to(ROOT))]
    return subprocess.Popen(args,cwd=ROOT,env=environment(run,cpu),
        stdout=(run/(command+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)


def evaluation_job(run,split,g):
    d=declaration();env=environment(run,True)
    env.update(JE10_ROUTE_PARALLEL='1',JE10_DIAGNOSTIC_REPORT=d['evaluation_report'],
        JE10_EVAL_SEED=str(d['dev_rng'] if split=='dev' else d['test_rng']),
        JE10_ROUTE_GAMES='32' if split=='dev' else '64',
        JE10_ROUTE_SEED_START=str(d['route_dev_seeds'][0] if split=='dev' else d['route_test_seeds'][0]),
        JE10_EVAL_NAMESPACE='authority_je10' if split=='dev' else 'authority_retention_test')
    return subprocess.run([sys.executable,SCRIPT,'evalchild','--run',str(run.relative_to(ROOT)),
        '--split',split,'--generation',str(g)],cwd=ROOT,env=env,
        stdout=(run/(split+'_eval.log')).open('a'),stderr=subprocess.STDOUT).returncode


def configure(run):
    from scripts import retention_phase_conditioning as phase
    phase.install()
    from scripts import retention_core_priority_screen as shared
    worker=shared.configure(run)
    plan=engine.read(run/'plan.json')
    atomic_json(run/'state/PHASE_INSTALL.json',dict(passed=True,
        phase_conditioning=plan['phase_conditioning'],raw_inputs=[24,24],
        actual_forward='scripts.retention_phase_conditioning.forward',
        actor_and_critic=True,teacher_context_used=False,only_current_visible_inputs=True,
        old_networks_exact_bypass=True,in_memory_only=True))
    return worker


def prepare_evaluation_runtime():
    """Set only this CPU process's callbacks, including its tactical child."""
    from scripts import retention_phase_conditioning as phase
    phase.install()
    from scripts import je08_bun_authority as authority
    from scripts.retention_phase_export import authority_export
    authority.export=authority_export
    original=subprocess.Popen
    def launch(args,*a,**kw):
        if len(args)>2 and args[1:3]==['scripts/je10_evaluate.py','tactical']:
            args=[args[0],SCRIPT,'tactical','--policy',args[3],'--target',args[4]]
        return original(args,*a,**kw)
    subprocess.Popen=launch


def owner():
    unique=lock(BASE/'owner.lock');engine.require_training_authorized();engine.check_seal()
    evaluator=subprocess.Popen(['taskset','-c','80-95',sys.executable,SCRIPT,'evalowner'],cwd=ROOT,
        env=environment(BASE,True),stdout=(BASE/'evaluation.log').open('a'),
        stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(BASE/'EVALUATOR_LAUNCH.json',dict(unique_owner=True,unix=time.time()))
    d=declaration()
    for spec in d['runs'][:2]:
        engine.smoke(spec)
        engine.notify('R5 '+spec['arm']+'技术smoke通过：finite、Actor/Critic非零、原子保存/新CPU进程恢复；无能力门禁。')
    from scripts.retention_core_priority_screen import checkpoint_receipt
    for index,spec in enumerate(d['runs']):
        run=ROOT/spec['namespace'];out=run/'formal'/ARM
        if (run/'TRAINING_COMPLETE.json').exists():continue
        engine.wait_idle();engine.check_seal()
        engine.status('TRAINING',run=spec['namespace'],arm=spec['arm'],seed=spec['seed'])
        for attempt in range(3):
            latest=out/'LATEST_CHECKPOINT.json'
            resume=ROOT/engine.read(latest)['checkpoint'] if latest.exists() else None
            proc=launch_worker(run,resume=resume)
            if resume is None and not (run/'FRESH_G0_AUDIT.json').exists():
                subprocess.Popen(['taskset','-c','112-127',sys.executable,
                    'scripts/retention_phase_fresh_audit.py',str(run.relative_to(ROOT))],cwd=ROOT,
                    env=environment(run,True),stdout=(run/'fresh_audit.log').open('a'),
                    stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(run/'OWNER_STATE.json',dict(attempt=attempt,
                restore_checkpoint=str(resume.relative_to(ROOT)) if resume else None))
            announced=False
            while proc.poll() is None:
                if (out/'STATE.json').exists() and not announced:
                    row=engine.read(out/'STATE.json');engine.validate_update(row)
                    if (out/'gen000001/COMMITTED.json').exists():
                        proof=checkpoint_receipt(run,1)
                        atomic_json(run/'FORMAL_NONZERO_CONFIRMED.json',dict(passed=True,
                            **{k:row[k] for k in ['generation','finite','nonzero_update','actor_update_l2','critic_update_l2','fresh_ticks']},
                            first_atomic_checkpoint=proof,phase_install=engine.read(run/'state/PHASE_INSTALL.json')))
                        engine.notify(f"R5 {spec['arm']} seed{spec['seed']}正式首代finite/Actor-Critic非零，原子SHA通过；8卡worker/唯一owner运行，CPU独立。")
                        announced=True
                time.sleep(10)
            if proc.returncode==0 and (run/'TRAINING_COMPLETE.json').exists():break
            atomic_json(run/f'TECHNICAL_ATTEMPT_{attempt}.json',dict(exit_code=proc.returncode))
            engine.notify('R5技术故障，保留证据并恢复最近原子点。');engine.wait_idle()
        else:raise RuntimeError('R5 technical retries exhausted')
        terminal=engine.read(out/'TRAINER_EXIT.json')
        assert terminal['generation']==32 and not terminal['atomic_stop']
        engine.validate_update(engine.read(out/'STATE.json'))
        atomic_json(run/'FIXED_RUN_TERMINAL.json',dict(terminal,atomic_checkpoint=checkpoint_receipt(run,32)))
        engine.wait_idle()
        atomic_json(run/'GPU_RELEASE_RECEIPT.json',dict(released=True,consecutive_idle_samples=2,
            resource=engine.read(BASE/'RESOURCE.json')))
        successor=d['runs'][index+1]['namespace'] if index+1<len(d['runs']) else '固定训练完成，CPU/JS继续全部固定评估'
        engine.notify(f"R5 {spec['arm']} seed{spec['seed']}正常g32终态，原子state/policy SHA通过，worker退出；8卡连续两次释放，唯一owner/CPU独立；后继{successor}。")
    engine.status('TRAINING_COMPLETE_EVALUATING')
    assert evaluator.wait()==0,'R5 CPU evaluation failed'
    from scripts import retention_exposure_screen as shared
    shared.BASE=BASE;shared.declaration=declaration
    result=shared.results();atomic_json(BASE/'SCREEN_COMPLETE.json',result)
    engine.status('SCREEN_COMPLETE');engine.notify('R5六run/全部固定dev与新保留集完成；按冻结全曲线证据作结论。')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','owner','worker','restore','evalowner','evalchild','tactical'])
    parser.add_argument('--run');parser.add_argument('--resume');parser.add_argument('--split')
    parser.add_argument('--generation',type=int);parser.add_argument('--policy');parser.add_argument('--target')
    args=parser.parse_args()
    engine.BASE=BASE;engine.declaration=declaration;engine.spawn_worker=launch_worker
    if args.command=='prepare':prepare()
    elif args.command=='owner':owner()
    elif args.command=='tactical':
        prepare_evaluation_runtime()
        from scripts.je10_evaluate import tactical
        tactical(args.policy,args.target)
    elif args.command in ['evalowner','evalchild']:
        from scripts import retention_evaluate as evaluator
        if args.command=='evalowner':evaluator.child=evaluation_job;evaluator.owner()
        else:
            prepare_evaluation_runtime()
            evaluator.evaluate_child(ROOT/args.run,args.split,args.generation)
            run=ROOT/args.run
            if args.split=='test':
                receipt=run/'authority_retention_test/g000000/SHARED_PARENT_RECEIPT.json'
                if receipt.exists():
                    content=engine.read(receipt)
                    conditioned=engine.read(run/'plan.json')['phase_conditioning']
                    content.update(same_parent_params=not conditioned,
                        functional_zero_extension_equivalence=conditioned,
                        independent_samples=False,
                        explanation='One raw parent evaluation. A1 adds zero projection rows; '
                            'same function at initialization, not identical parameter shape.')
                    atomic_json(receipt,content)
            cp=run/'formal'/ARM/f'gen{args.generation:06d}'/'state.pkl'
            target=run/os.environ['JE10_EVAL_NAMESPACE']/f'g{args.generation:06d}'/'PHASE_COUNTERFACTUAL.json'
            if not target.exists():
                from scripts.retention_phase_probe import audit
                audit(cp,BASE/'FIXED_PHASE_PROBE.npz',target)
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
            engine.notify('R5技术停止：'+type(exc).__name__+'，保留最近原子点；未作能力判停。')
        raise
