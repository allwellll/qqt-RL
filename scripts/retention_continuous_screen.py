"""Durable fixed-budget training relay; CPU evaluation never owns GPU succession."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import retention_custody_dagger_screen as recipe
from scripts.population_run_state import atomic_json, sha, lock

BASE = ROOT / 'runs/retention_20261022/continuous'
SCRIPT = 'scripts/retention_continuous_screen.py'
ARM = recipe.ARM


def read(path):
    return json.loads(Path(path).read_text())


def specs():
    return read(BASE / 'TRAIN_QUEUE.json')['runs']


def configure_engine():
    recipe._configure_engine()
    recipe.engine.BASE = BASE


def pending(items):
    return [s for s in items if not (ROOT / s['namespace'] / 'TRAINING_COMPLETE.json').exists()]


def notify(message):
    atomic_json(BASE / 'notifications' / f'event_{time.time_ns()}.json',
                dict(message=message, unix=time.time()))
    print(message, flush=True)
    try:
        subprocess.run(['jdme-notify', 'send', message], timeout=20,
                       stdout=subprocess.DEVNULL, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def guard(run=None):
    recipe.engine.require_training_authorized(run)


def verify_seal(run):
    seal = read(run / 'SOURCE_SEAL.json')
    for name, expected in seal['sources'].items():
        assert sha(ROOT / name) == expected, 'sealed numerical source changed: ' + name
    assert sha(run / 'plan.json') == seal['plan_sha256']


def configure_worker(run):
    worker = recipe.configure(run)
    plan = read(run / 'plan.json')
    if plan.get('stage_allocation_return_share') == .5:
        from scripts.retention_stage_allocation import install
        atomic_json(run / 'state/STAGE_ALLOCATION_INSTALL.json', install(.5))
    return worker


def spawn(run, command='worker', resume=None, cpu=False):
    guard(run)
    args = ['taskset', '-c', '96-111' if cpu else '0-79', sys.executable,
            SCRIPT, command, '--run', str(run.relative_to(ROOT))]
    if resume:
        args += ['--resume', str(resume.relative_to(ROOT))]
    return subprocess.Popen(args, cwd=ROOT, env=recipe.engine.env_for(run, cpu),
                            stdout=(run / f'{command}.log').open('a'),
                            stderr=subprocess.STDOUT, start_new_session=True)


def owner():
    unique = lock(BASE / 'owner.lock')
    guard()
    evaluator = subprocess.Popen(
        ['taskset', '-c', '80-95', sys.executable, SCRIPT, 'evalowner'], cwd=ROOT,
        env=recipe.engine.env_for(BASE, True), stdin=subprocess.DEVNULL,
        stdout=(BASE / 'evaluation.log').open('a'), stderr=subprocess.STDOUT,
        start_new_session=True)
    while True:
        guard()
        items = pending(specs())
        if not items:
            # The queue is replenished with a preregistered unchanged observation,
            # never wait for an evaluation score or report to claim GPU work.
            from scripts.retention_continuous_prepare import append_observation
            append_observation()
            continue
        spec = items[0]
        run = ROOT / spec['namespace']
        if not (run / 'PREPARED.json').exists():
            # Only technical readiness can select the declared observation fallback.
            fallback = next((s for s in items if s['kind'] == 'observation'
                             and (ROOT / s['namespace'] / 'PREPARED.json').exists()), None)
            if fallback is None:
                raise RuntimeError('No technically prepared training namespace; evaluation is not a gate')
            spec = fallback
            run = ROOT / spec['namespace']
        verify_seal(run)
        recipe.engine.wait_idle()
        out = run / 'formal' / ARM
        atomic_json(BASE / 'STATUS.json', dict(state='TRAINING', run=spec['namespace'],
                    fixed_generations=spec['generations'], unix=time.time()))
        for attempt in range(3):
            guard(run)
            latest = out / 'LATEST_CHECKPOINT.json'
            resume = ROOT / read(latest)['checkpoint'] if latest.exists() else None
            proc = spawn(run, resume=resume)
            atomic_json(run / 'OWNER_STATE.json', dict(attempt=attempt,
                        restore_checkpoint=str(resume.relative_to(ROOT)) if resume else None,
                        owner_active=True, unix=time.time()))
            announced = False
            while proc.poll() is None:
                atomic_json(BASE / 'OWNER_HEARTBEAT.json', dict(run=spec['namespace'],
                            state='TRAINING', attempt=attempt, unix=time.time()))
                cp = out / 'gen000001/COMMITTED.json'
                state = out / 'STATE.json'
                if cp.exists() and state.exists() and not announced:
                    row = read(state)
                    recipe.engine.validate_update(row)
                    proof = recipe._checkpoint_receipt(run, 1)
                    atomic_json(run / 'FORMAL_NONZERO_CONFIRMED.json', dict(
                        passed=True, generation=row['generation'], atomic_checkpoint=proof,
                        **{k: row[k] for k in ['finite', 'nonzero_update', 'actor_update_l2',
                                              'critic_update_l2', 'fresh_ticks']},
                        unix=time.time()))
                    notify(f"正式首更新 {spec['namespace']} g{row['generation']} finite/Actor-Critic非零，8卡固定{spec['generations']}代；CPU评估独立，终态立即接续队列。")
                    announced = True
                time.sleep(10)
            if proc.returncode == 0 and (run / 'TRAINING_COMPLETE.json').exists():
                break
            guard(run)
            notify(f"技术恢复 {spec['namespace']}：保留证据，从最新原子点恢复。")
            recipe.engine.wait_idle()
        else:
            raise RuntimeError('technical retry budget exhausted')
        terminal = read(out / 'TRAINER_EXIT.json')
        assert terminal['generation'] == spec['generations'] and not terminal['atomic_stop']
        recipe.engine.validate_update(read(out / 'STATE.json'))
        proof = recipe._checkpoint_receipt(run, spec['generations'])
        atomic_json(run / 'FIXED_RUN_TERMINAL.json', dict(**terminal, atomic_checkpoint=proof))
        recipe.engine.wait_idle()
        atomic_json(run / 'GPU_RELEASE_RECEIPT.json', dict(released=True,
                    consecutive_idle_samples=2, resource=read(BASE / 'RESOURCE.json')))
        notify(f"固定run终态 {spec['namespace']} g{terminal['generation']}原子SHA通过、worker退出/8卡释放；立即接续预声明队列，不等CPU评估。")
        atomic_json(BASE / 'CONTINUATION.json', dict(state='NEXT_PREDECLARED_RUN',
                    completed_run=spec['namespace'], next_runs=pending(specs()),
                    evaluation_blocks_training=False, unix=time.time()))


def evaluate_child(run, split, generation):
    plan = read(run / 'plan.json')
    recipe.engine.declaration = lambda: dict(points=plan['checkpoints'], runs=specs())
    from scripts import retention_evaluate as evaluator
    evaluator.evaluate_child(run, split, generation)


def evalowner():
    unique = lock(BASE / 'evaluator.lock')
    import fcntl
    authority = (ROOT / 'runs/je10_bun_20261018/evaluation.lock').open('a')
    fcntl.flock(authority.fileno(), fcntl.LOCK_EX)
    while True:
        jobs = []
        for spec in specs():
            run = ROOT / spec['namespace']
            if not (run / 'plan.json').exists():
                continue
            for g in read(run / 'plan.json')['checkpoints']:
                cp = run / 'formal' / ARM / f'gen{g:06d}/COMMITTED.json'
                if cp.exists() and not (run / f'authority_je10/g{g:06d}/COMPLETE.json').exists():
                    jobs.append((run, 'dev', g))
        if not jobs:
            for spec in specs():
                run = ROOT / spec['namespace']
                g = spec['generations']
                if (run / 'TRAINING_COMPLETE.json').exists() and not (run / f'authority_retention_test/g{g:06d}/COMPLETE.json').exists():
                    jobs.append((run, 'test', g))
        if not jobs:
            atomic_json(BASE / 'EVAL_QUEUE.json', dict(running=None, controls_gpu=False,
                        waiting_for_committed_checkpoints=True, unix=time.time()))
            time.sleep(10)
            continue
        run, split, g = jobs[0]
        env = recipe.engine.env_for(run, True)
        env.update(JE10_ROUTE_PARALLEL='1', JE10_EVAL_SEED='20261019' if split == 'dev' else '20261109',
                   JE10_ROUTE_GAMES='32' if split == 'dev' else '64',
                   JE10_ROUTE_SEED_START='2026102100' if split == 'dev' else '2026111000',
                   JE10_DIAGNOSTIC_REPORT='reports/retention_continuous_evaluation_20261022.md',
                   JE10_EVAL_NAMESPACE='authority_je10' if split == 'dev' else 'authority_retention_test')
        atomic_json(BASE / 'EVAL_QUEUE.json', dict(running=[str(run.relative_to(ROOT)), split, g], controls_gpu=False, unix=time.time()))
        code = subprocess.run([sys.executable, SCRIPT, 'evalchild', '--run', str(run.relative_to(ROOT)),
                               '--split', split, '--generation', str(g)], cwd=ROOT, env=env,
                              stdout=(run / f'{split}_eval.log').open('a'), stderr=subprocess.STDOUT).returncode
        if code:
            atomic_json(BASE / 'EVAL_FAILURE.json', dict(run=str(run.relative_to(ROOT)),
                        generation=g, split=split, controls_gpu=False, exit_code=code, unix=time.time()))
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['owner', 'worker', 'restore', 'evalowner', 'evalchild'])
    parser.add_argument('--run')
    parser.add_argument('--resume')
    parser.add_argument('--split')
    parser.add_argument('--generation', type=int)
    args = parser.parse_args()
    configure_engine()
    if args.command == 'owner':
        owner()
    elif args.command == 'evalowner':
        evalowner()
    elif args.command == 'evalchild':
        evaluate_child(ROOT / args.run, args.split, args.generation)
    else:
        run = ROOT / args.run
        guard(run)
        verify_seal(run)
        worker = configure_worker(run)
        if args.command == 'restore':
            worker.restore_monitor(ROOT / args.resume)
        else:
            worker.worker(ROOT / args.resume if args.resume else None)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(BASE / f'TECHNICAL_FAILURE_{time.time_ns()}.json',
                    dict(error=type(error).__name__, message=str(error), unix=time.time()))
        notify(f'连续训练技术阻塞：{type(error).__name__}；证据已保存，非能力判停。')
        raise
