"""Versioned CPU authority recovery; never mutates the training source seal."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import pickle
import signal
import shutil
import subprocess
import sys
import time
import traceback


def cpu_env(source=None):
    env = dict(os.environ if source is None else source)
    env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
               JAXBOMB_RULE='bun', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


os.environ.update(cpu_env())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import je03_bun_selfplay_20261012 as stage
from scripts import je03_bun831_evaluator_20261011 as wrapper
from scripts.population_run_state import atomic_json, sha, lock
from scripts.bun_checkpoint_state import cpu_command
from scripts.je03_bun_stage1_20261011 import read, verify_checkpoint

RUN = stage.RUN
OUT = RUN / 'authority_stage2_recovery_v2'
PRIORITY = [0, 512, 1024, 1536, 2048, 8, 16, 32, 64, 128, 256, 768, 3072, 4096]
TASKS = wrapper.TASKS


def manifest():
    return wrapper.authority.SPAWNS


def validate_complete(out, policy_hash):
    main = read(out / 'COMPLETE.json')
    full = read(out / 'COMPLETE_WITH_BUN_TASKS.json')
    assert main['policy_sha256'] == full['policy_sha256'] == policy_hash
    assert full['authority_sha256'] == sha(out / 'COMPLETE.json')
    for name, digest in {**main['shards'], **full['additional_shards']}.items():
        assert sha(out / name) == digest, name
    return main, full


def setup():
    stage.verify_seal()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(namespace=OUT.name, paired=True, stage2_enabled=True,
        hunter_games_per_tier_per_arm=64, tactical_games_per_arm=64,
        hunter_primary=300, hunter_residual=64, tactical_primary=300,
        native_bun_tasks=TASKS, native_task_games=16, native_task_seed_base=2026101131,
        native_task_ticks=300, native_web_safety_seed=392010731,
        spawns_sha256=sha(manifest()), seed=read(manifest())['seed'],
        checkpoints=stage.POINTS, priority=PRIORITY, cuda_visible_devices='',
        safety='credited kill, zero actor deaths, actor alive after residual settlement',
        task_scope='native Bun idle-opponent diagnostic, not webpage Hunter',
        sources={f: sha(ROOT / f) for f in wrapper.authority.FILES + [
            'scripts/je03_bun831_evaluator_20261011.py',
            'scripts/eval_web_native_bun_probe.js', str(Path(__file__).relative_to(ROOT))]})
    if (OUT / 'PROTOCOL.json').exists():
        assert read(OUT / 'PROTOCOL.json') == protocol
    else:
        atomic_json(OUT / 'PROTOCOL.json', protocol)
    if (OUT / 'spawns.json').exists():
        assert read(OUT / 'spawns.json') == read(manifest())
    else:
        atomic_json(OUT / 'spawns.json', read(manifest()))
    wrapper.authority.OUT = OUT
    wrapper.authority.RUN = RUN


def tasks(policy, target):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax_bomb import bun_env as env, jax_train as train
    assert jax.default_backend() == 'cpu' and train.IS_BUN
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    env.prepare()
    with Path(policy).open('rb') as stream:
        params = jax.tree.map(jnp.asarray, pickle.load(stream))
    results = {}; begin = time.time()
    for ordinal, lesson in enumerate(TASKS):
        env.configure_training(lesson + '=1', 1, reward_profile='legacy')
        key = jax.random.PRNGKey(2026101131 + ordinal)
        states = env.init_batch(key, 16); active = jnp.ones(16, jnp.bool_)

        @jax.jit
        def step(states, key, active):
            key, kact, kstep = jax.random.split(key, 3)
            masks = train.both_masks(states)
            actions, _, _ = train.sample_actions(params, 'transformer', train.both_perspectives(states)[:16],
                tuple(x[:16] for x in masks), kact, state=train.both_states(states)[:16])
            joint = jnp.stack([actions, jnp.tile(jnp.array([[4, 0]], jnp.int32), (16, 1))], 1)
            next_states, done, info = jax.vmap(lambda s, a, k: env.step(
                s, a, k, auto_reset=False, return_info=True))(states, joint, jax.random.split(kstep, 16))
            next_states = jax.tree.map(lambda n, o: jnp.where(
                active.reshape((-1,) + (1,) * (n.ndim - 1)), n, o), next_states, states)
            values = jnp.stack([info[k][:, 0].astype(jnp.int32) for k in [
                'capture', 'steal', 'return_bun', 'death', 'own_bomb_defeat']], -1)
            extra = jnp.stack([(actions[:, 1] == 2).astype(jnp.int32),
                jnp.broadcast_to(info['mutual_death'], (16,)).astype(jnp.int32),
                info['surviving_causal_kill'][:, 0].astype(jnp.int32)], -1)
            return next_states, key, active & ~done, jnp.concatenate([values, extra], -1) * active[:, None]

        totals = np.zeros((16, 8), np.int32)
        for _ in range(300):
            states, key, active, values = step(states, key, active)
            totals += np.asarray(values)
        safe = (totals[:, 7] > 0) & (totals[:, 3] == 0) & np.asarray(states.core.alive)[:, 0]
        results[lesson] = dict(games=16, capture_episodes=int((totals[:, 0] > 0).sum()),
            steal_episodes=int((totals[:, 1] > 0).sum()), return_episodes=int((totals[:, 2] > 0).sum()),
            death_episodes=int((totals[:, 3] > 0).sum()), self_episodes=int((totals[:, 4] > 0).sum()),
            item_use=int(totals[:, 5].sum()), trade_episodes=int((totals[:, 6] > 0).sum()),
            safe_win=int(safe.sum()), episodes=totals.tolist())
    atomic_json(target, dict(passed=True, policy_sha256=sha(policy), seed_base=2026101131,
        primary_ticks=300, native_bun=True, cuda_visible_devices='', results=results,
        opponent='frozen idle; diagnostic only; no web residual, not Hunter',
        columns=['capture', 'steal', 'return', 'death', 'self', 'item_use', 'trade', 'causal_kill'],
        wall_seconds=time.time() - begin))


def checkpoint(arm, generation):
    setup()
    cp = RUN / 'formal' / arm / f'gen{generation:06d}'
    verify_checkpoint(cp, sha(RUN / 'plan.json'))
    policy = cp / 'policy.pt'; label = f'{arm}_g{generation:06d}'; out = OUT / label
    if (out / 'COMPLETE_WITH_BUN_TASKS.json').exists():
        validate_complete(out, sha(policy)); return
    start = time.time()
    wrapper.authority.evaluate(label, policy, dict(generation=generation,
        committed_sha256=sha(cp / 'COMMITTED.json')))
    target = out / 'bun_tasks.json'
    if not target.exists():
        with (out / 'bun_tasks.log').open('a') as log:
            subprocess.run(cpu_command([sys.executable, str(Path(__file__).resolve()),
                'tasks', '--policy', str(policy), '--out', str(target)],
                'BUN_TASK_CPUS'), env=cpu_env(), stdout=log, stderr=subprocess.STDOUT, check=True)

    def probe(difficulty):
        path = out / f'native_probe_{difficulty}.json'
        if not path.exists():
            subprocess.run(['node', str(ROOT / 'scripts/eval_web_native_bun_probe.js'), '--model',
                str(out / 'model/model.json'), '--difficulty', difficulty, '--seed', '392010731',
                '--checkpoint-sha', sha(policy), '--out', str(path)], env=cpu_env(), check=True)
        data = read(path)
        assert data['checkpoint_sha256'] == sha(policy) and data['real_web_hunter']
        assert data['native_items'] and data['native_trap'] and len(data['episodes']) == 2
        assert all(not e['hazards_remaining'] for e in data['episodes'])
        return path.name, sha(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        shards = dict(pool.map(probe, ['hard', 'normal', 'easy']))
    task = read(target)
    assert task['policy_sha256'] == sha(policy) and task['cuda_visible_devices'] == ''
    shards['bun_tasks.json'] = sha(target)
    atomic_json(out / 'COMPLETE_WITH_BUN_TASKS.json', dict(passed=True, policy_sha256=sha(policy),
        authority_sha256=sha(out / 'COMPLETE.json'), additional_shards=shards,
        tasks=task['results'], stage2_enabled=True, wall_seconds=time.time() - start))
    validate_complete(out, sha(policy))


def publish_pair(generation):
    mains = []; fulls = []; hashes = {}; proofs = {}
    for arm in stage.ARMS:
        cp = RUN / 'formal' / arm / f'gen{generation:06d}'
        verify_checkpoint(cp, sha(RUN / 'plan.json'))
        out = OUT / f'{arm}_g{generation:06d}'
        main, full = validate_complete(out, sha(cp / 'policy.pt'))
        mains.append(main); fulls.append(full)
        hashes[arm] = sha(out / 'COMPLETE_WITH_BUN_TASKS.json')
        proofs[arm] = sha(cp / 'COMMITTED.json')
    delta = {d: {k: mains[1]['hunter'][d][k] - mains[0]['hunter'][d][k]
                    for k in mains[0]['hunter'][d]} for d in ('hard', 'normal', 'easy')}
    atomic_json(OUT / f'PAIR_g{generation:06d}.json', dict(passed=True, generation=generation,
        arms=stage.ARMS, complete_hashes=hashes, checkpoint_proofs=proofs,
        protocol_sha256=sha(OUT / 'PROTOCOL.json'), hunter_delta=delta,
        metrics={a: dict(hunter=m['hunter'], tactical=m['tactical'], tasks=f['tasks'])
                 for a, m, f in zip(stage.ARMS, mains, fulls)},
        wall_seconds=sum(f['wall_seconds'] for f in fulls), unix=time.time()))


def owner():
    # Also hold the failed legacy owner's lock: handoff retries cannot become writers.
    legacy = RUN / 'authority_stage2_v1'; legacy.mkdir(parents=True, exist_ok=True)
    legacy_lock = lock(legacy / 'owner.lock')
    OUT.mkdir(parents=True, exist_ok=True); own_lock = lock(OUT / 'owner.lock')
    setup()
    atomic_json(OUT / 'OWNER.json', dict(pid=os.getpid(), cuda_visible_devices='',
        affinity=sorted(os.sched_getaffinity(0)), legacy_lock_held=True, unix=time.time()))
    active = None

    def stop(signum, frame):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM); active.wait()
        atomic_json(OUT / 'OWNER_STOPPED.json', dict(signal=signum, unix=time.time()))
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    while True:
        for g in PRIORITY:
            cps = [RUN / 'formal' / a / f'gen{g:06d}' for a in stage.ARMS]
            if not all((c / 'COMMITTED.json').exists() for c in cps):
                continue
            for arm, cp in zip(stage.ARMS, cps):
                out = OUT / f'{arm}_g{g:06d}'
                if (out / 'COMPLETE_WITH_BUN_TASKS.json').exists():
                    validate_complete(out, sha(cp / 'policy.pt')); continue
                atomic_json(OUT / 'STATE.json', dict(pid=os.getpid(), phase='EVALUATING',
                    arm=arm, generation=g, priority=PRIORITY, unix=time.time()))
                with (OUT / f'{arm}_g{g:06d}.log').open('a') as log:
                    active = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'checkpoint',
                        '--arm', arm, '--generation', str(g)], env=cpu_env(), stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    code = active.wait()
                if code:
                    # Clean only this evaluation's group before any retry.
                    try: os.killpg(active.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    atomic_json(OUT / f'TECHNICAL_RETRY_{arm}_{g}_{time.time_ns()}.json',
                        dict(returncode=code, log=str(OUT / f'{arm}_g{g:06d}.log'), unix=time.time()))
                    active = None; time.sleep(10); break
                active = None
            else:
                publish_pair(g)
                notify = OUT / f'NOTIFICATION_g{g:06d}.json'
                if not notify.exists():
                    # Record before sending so restarts cannot send duplicate notifications.
                    atomic_json(notify, dict(claimed=True, unix=time.time()))
                    executable = shutil.which('jdme-notify')
                    if executable:
                        try:
                            result = subprocess.run([executable, 'send', f'Bun Stage2 g{g} 两臂 CPU 权威评估完成；结果 {OUT}/PAIR_g{g:06d}.json'],
                                capture_output=True, text=True, timeout=15)
                            receipt = dict(returncode=result.returncode,
                                stdout=result.stdout, stderr=result.stderr)
                        except subprocess.TimeoutExpired:
                            receipt = dict(error='notification timeout; evaluation continues')
                    else:
                        receipt = dict(skipped='jdme-notify unavailable; evaluation continues')
                    atomic_json(notify, dict(claimed=True, **receipt, unix=time.time()))
                continue
            break
        pairs = [g for g in stage.POINTS if (OUT / f'PAIR_g{g:06d}.json').exists()]
        atomic_json(OUT / 'QUEUE.json', dict(completed=pairs,
            pending=[g for g in PRIORITY if g not in pairs], unix=time.time()))
        if len(pairs) == len(stage.POINTS) and all((RUN / f'TRAINING_COMPLETE_{a}.json').exists() for a in stage.ARMS):
            atomic_json(OUT / 'EVALUATION_COMPLETE.json', dict(passed=True, technical_only=True,
                pair_hashes={str(g): sha(OUT / f'PAIR_g{g:06d}.json') for g in pairs},
                training_hashes={a: sha(RUN / f'TRAINING_COMPLETE_{a}.json') for a in stage.ARMS}, unix=time.time()))
            atomic_json(OUT / 'OWNER_FINISHED.json', dict(pid=os.getpid(), passed=True,
                evaluation_sha256=sha(OUT / 'EVALUATION_COMPLETE.json'), unix=time.time()))
            return
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['owner', 'checkpoint', 'tasks'])
    parser.add_argument('--arm', choices=stage.ARMS); parser.add_argument('--generation', type=int)
    parser.add_argument('--policy'); parser.add_argument('--out')
    args = parser.parse_args()
    if args.command == 'tasks': tasks(args.policy, Path(args.out))
    elif args.command == 'checkpoint': checkpoint(args.arm, args.generation)
    else: owner()
