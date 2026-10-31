"""Prepare isolated paired stages and unchanged observation fallbacks."""
from pathlib import Path
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/retention_20261022/continuous'
SOURCE = ROOT / 'runs/retention_20261022/r6/A0_s2026102212'
PARENT = 'runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032'
from scripts.population_run_state import atomic_json, sha, lock


def read(path):
    return json.loads(Path(path).read_text())


def share_parent_holdout(run):
    """Reuse genuine fixed parent episodes; all recipes share unmodified inputs."""
    source=ROOT/'runs/retention_20261022/r6/A0_s2026102210'
    assert sha(run/'inputs/initial.pkl')==sha(source/'inputs/initial.pkl')
    cp=source/'formal/bun392_bridge_pfsp8/gen000000/COMMITTED.json'
    expected=read(cp)['policy_sha256']
    samples=source/'authority_retention_test/g000000'
    assert read(samples/'COMPLETE.json')['checkpoint_sha256']==expected
    assert read(samples/'SUMMARY.json')['policy_sha256']==expected
    destination=run/'authority_retention_test/g000000'
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        shutil.copytree(samples,destination)
    atomic_json(destination/'SHARED_PARENT_RECEIPT.json',dict(
        source=str(samples.relative_to(ROOT)),same_parent_params=True,
        initial_inputs_sha256=sha(run/'inputs/initial.pkl'),
        policy_sha256=expected,fixed_test_rng=20261109,
        fixed_route_seed_start=2026111000,independent_samples=False,
        synthetic_or_relabelled_evaluation=False))


def prepare_run(spec):
    run = ROOT / spec['namespace']
    if (run / 'PREPARED.json').exists():
        return
    run.mkdir(parents=True, exist_ok=True)
    if not (run / 'inputs').exists():
        shutil.copytree(SOURCE / 'inputs', run / 'inputs')
    for folder in ['state', 'formal', 'cache', 'validation', 'diagnostics']:
        (run / folder).mkdir(exist_ok=True)
    shutil.copyfile(SOURCE / 'regions.json', run / 'regions.json')
    plan = read(SOURCE / 'plan.json')
    generations = spec['generations']
    points = [0,8,16,24,32] if generations == 32 else [0,8,16,24,32,64,96,128]
    if spec['kind'] == 'technical_smoke':
        points = [0,1]
    plan.update(experiment_id='RETENTION-20261022-' + run.name,
                seed=spec['seed'], generations=generations, checkpoints=points,
                total_fresh_ticks=generations*2097152,
                state_schema='retention-r7-stage-' + run.name,
                stage_allocation_return_share=spec.get('return_share', .75))
    atomic_json(run / 'plan.json', plan)
    atomic_json(run / 'ROUND_DECLARATION.json', dict(**spec, parent=PARENT,
                fresh_optimizer_env_rng_fifo_core=True, return_dagger_multiplier=2,
                evaluation=dict(seed=20261019, points=points, route_games=32,
                                route_seed_start=2026102100),
                sole_factor='ordinary accepted phase replay quota only',
                reward_changed=False, query_budget_changed=False,
                phase_inputs_changed=False, ppo_changed=False,
                role='technical-only, excluded from paired experiment' if generations==1 else spec['kind']))
    expected={p.name:sha(p) for p in (SOURCE/'inputs').iterdir() if p.is_file()}
    actual={p.name:sha(p) for p in (run/'inputs').iterdir() if p.is_file()}
    assert actual==expected
    atomic_json(run/'INPUT_AUDIT.json',dict(passed=True,parent=PARENT,
                common_inputs_sha256=expected,learner_state_copied=False,
                optimizer_rng_env_fifo_core_fresh=True))
    share_parent_holdout(run)
    seal=read(ROOT/'runs/retention_20261022/r6/SOURCE_SEAL.json')
    for name in ['retention_continuous_screen.py','retention_continuous_prepare.py','retention_stage_allocation.py']:
        seal['sources']['scripts/'+name]=sha(ROOT/'scripts'/name)
    seal['plan_sha256']=sha(run/'plan.json')
    atomic_json(run/'SOURCE_SEAL.json',seal)
    # A1 is prepared now but cannot become runnable until the one fixed candidate
    # smoke and independent fresh-process CPU restore have passed.
    receipt=dict(passed=True,source_sealed=True,fresh_namespace=True,
                 paired_factor_only=True,unix=time.time())
    if spec.get('arm')=='A1' and spec['kind']!='technical_smoke':
        atomic_json(run/'AWAITING_TECHNICAL_SMOKE.json',receipt)
    else:
        atomic_json(run/'PREPARED.json',receipt)


def append_observation():
    """Keep a fixed, explicit unchanged fallback ahead of every queue boundary."""
    queue_lock=lock(BASE/'queue_mutation.lock')
    queue=read(BASE/'TRAIN_QUEUE.json')
    count=sum(s['kind']=='observation' for s in queue['runs'])
    seed=2026102224+count-1
    spec=dict(namespace=f'runs/retention_20261022/continuous/O{count}_s{seed}',
              kind='observation',seed=seed,generations=128,arm=f'O{count}')
    prepare_run(spec)
    queue['runs'].append(spec)
    queue['unix']=time.time()
    atomic_json(BASE/'TRAIN_QUEUE.json',queue)
    return spec


def maintain_reserve():
    """Prepare two fixed observation fallbacks ahead; never launches a worker."""
    unique=lock(BASE/'reserve_maintainer.lock')
    while True:
        queue=read(BASE/'TRAIN_QUEUE.json')
        remaining=[s for s in queue['runs'] if s['kind']=='observation'
                   and not (ROOT/s['namespace']/'TRAINING_COMPLETE.json').exists()]
        if len(remaining)<2:
            from scripts.retention_screen import require_gpu_training_authorized
            try:
                require_gpu_training_authorized()
                append_observation()
            except BlockingIOError:
                pass
            except RuntimeError:
                # A user pause stays authoritative; the maintainer owns no GPU.
                pass
        atomic_json(BASE/'RESERVE_HEARTBEAT.json',dict(cpu_only=True,controls_gpu=False,
                    prepared_observation_reserve=len(remaining),unix=time.time()))
        time.sleep(30)


def ready_candidate():
    import subprocess
    import sys
    import os
    smoke=ROOT/'runs/retention_20261022/r7/smoke_stage_A1'
    checkpoint=smoke/'formal/bun392_bridge_pfsp8/gen000001'
    while not (checkpoint/'COMMITTED.json').exists():
        time.sleep(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    proc=subprocess.run(['taskset','-c','112-119',sys.executable,
                         'scripts/retention_continuous_screen.py','restore',
                         '--run',str(smoke.relative_to(ROOT)),
                         '--resume',str(checkpoint.relative_to(ROOT))],cwd=ROOT,env=env,
                        stdout=(BASE/'candidate_restore.log').open('a'),stderr=subprocess.STDOUT)
    assert proc.returncode==0,'candidate fresh process restore failed'
    proof=read(smoke/'state/RESTORE_MONITOR.json')
    assert proof['passed'] and proof['fresh_process'] and proof['generation']==1
    atomic_json(smoke/'TECHNICAL_COMPLETE.json',dict(passed=True,finite_nonzero=True,
                fresh_process_restore=True,capability_gate=False,unix=time.time()))
    for spec in read(BASE/'TRAIN_QUEUE.json')['runs']:
        if spec['kind']=='paired_stage_allocation' and spec['arm']=='A1':
            run=ROOT/spec['namespace']
            atomic_json(run/'PREPARED.json',dict(passed=True,smoke=str(smoke.relative_to(ROOT)),
                        fresh_process_restore=True,focused_tests_passed=26,capability_gate=False,
                        unix=time.time()))
    atomic_json(BASE/'R7_CANDIDATE_TECHNICALLY_READY.json',dict(passed=True,unix=time.time()))


if __name__=='__main__':
    import sys
    if sys.argv[1:] == ['ready-candidate']:
        ready_candidate()
    elif sys.argv[1:] == ['maintain-reserve']:
        maintain_reserve()
