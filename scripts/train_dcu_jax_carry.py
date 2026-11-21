"""Adapt duel Actors on the carry map, then train SP against frozen defenders."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import time

from qqt_rl.training.carry_league import opponent_counts

ROOT = Path(__file__).resolve().parents[1]
U930_SHA = '1b1079d213a971ed7013739868d3b9848484ecf48ade25418142e16fb3a06d27'
U930 = ROOT / 'data/carry-training-20261115/u930_actor.pt'
SP_SHA = '39b0dea7327221891bdd4cfba93ab917c897aa86fd78d83401241ad945a982cf'
DUEL_RUN = ROOT / 'runs/duel-pool-8card-e2048-mb2048-20261121'
DUEL_SEEDS = {
    'u244': '7c4a4bb0fb1cb1ef3777d819033593d6be99bb36fce71122501ca543c8383b59',
    'u220': '5cdb62df5445480d573cebbede756d27c3d673d087541afdcdf51b8ade475aae',
    'u153': 'c22aefaece05b7073ed2fec90c72e08178d9982dcca45e5e45d641cf935c55fe',
    'u930': U930_SHA,
}
SCHEMA = 'qqt.carry-actor-critic/v1'
SOURCES = ('scripts/train_dcu_jax_carry.py', 'qqt_rl/training/jax_carry_env.py',
           'qqt_rl/training/jax_carry_collect.py', 'qqt_rl/training/jax_carry_fast.py',
           'qqt_rl/training/carry_league.py',
           'qqt_rl/training/kill_ppo.py',
           'qqt_rl/training/checkpoints.py', 'qqt_rl/training/dcu.py',
           'scripts/portable_training.py', 'scripts/r4_kill_arena.py',
           'qqt_rl/training/jax_kill_env.py', 'scripts/r4_k4_observation.py',
           'scripts/r3_items_observation.py', 'scripts/retention_phase_conditioning.py',
           'scripts/je05_bun_transport.py', 'scripts/je10_causal_init.py',
           'jax_bomb/jax_net.py', 'jax_bomb/levels.py',
           'jax_bomb/je07_bun_env_r1_items.py', 'jax_bomb/je05_core_env_r1_items.py',
           'jax_bomb/je07_native_items.py', 'web/assets/maps/levels.json',
           'jax_bomb/maps/kill_maps.json')
LEGACY_SOURCES = {
    'scripts/train_dcu_jax_carry.py': 'd92bd199cca7a50c8a5ef658097a0a89d4dc961ea50928f981fabd2562b3854b',
    'qqt_rl/training/jax_carry_collect.py': 'a891bef9d3d283bb40a46d07f19a986486b906908125e6bc7b4b5310fc1bf6b3',
    'qqt_rl/training/jax_carry_fast.py': '67fae084167f2cb71cd42c49b5edd3c8c653dcbb5a8c6506af149b3e408d412f',
}
TRUSTED_TRAIN_SOURCE_SHAS = {
    # Checkpoints created before the joint alternating trainer was added.
    '9e63c3946920f645b72f131a83124488aacec359476a237224fb3f17970c4748',
    # Checkpoints from the first joint-training launch before metric-shape fix.
    'aa9c7e4585601e998739bfdbff043d1748d87dbbfd67291fd0a39ab0a15823e7',
}


def seed_path(a):
    if a.interceptor_actor:
        return a.interceptor_actor
    if a.interceptor_start == 'u930':
        return a.u930
    return DUEL_RUN / f'u{int(a.interceptor_start[1:]):06d}' / 'actor.pt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('plan', 'train'))
    p.add_argument('--stage', choices=('interceptor', 'courier'), default='courier')
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--envs', type=int, default=2048, help='per device, balanced seats')
    p.add_argument('--collector', choices=('baseline', 'optimized', 'fused'), default='fused')
    p.add_argument('--value-chunk', type=int, default=512, help='deferred Critic inference batch')
    p.add_argument('--policy-chunk', type=int, default=512, help='learner inference batch; match PPO geometry')
    p.add_argument('--compiler-profile', choices=('safe', 'throughput'), default='throughput')
    p.add_argument('--ticks', type=int, default=64)
    p.add_argument('--minibatch', type=int, default=512, help='per device')
    p.add_argument('--interceptor-fraction', type=float, default=.75)
    p.add_argument('--u930', type=Path, default=U930, help='trusted original u930 actor.pt')
    p.add_argument('--interceptor-start', choices=DUEL_SEEDS, help='stage A duel Actor (default: u930)')
    p.add_argument('--interceptor-actor', type=Path, help='alternate path for the pinned stage A Actor')
    p.add_argument('--interceptor', type=Path, help='chosen stage-A checkpoint directory; omitted uses original u930')
    p.add_argument('--courier-opponent', type=Path,
                   help='stage A: frozen courier checkpoint used while updating an interceptor')
    p.add_argument('--defender', type=Path, action='append', default=[], metavar='CHECKPOINT',
                   help='stage B: repeat to mix trained stage-A checkpoints from distinct duel seeds')
    p.add_argument('--checkpoint', type=Path, help='Actor/Critic warm start from the SAME stage')
    p.add_argument('--critic-warmup', type=int, default=10)
    p.add_argument('--max-updates', type=int, default=100000)
    p.add_argument('--hours', type=float)
    p.add_argument('--deadline-utc', help='absolute UTC ISO deadline INCLUDING compilation')
    p.add_argument('--seed', type=int, default=2026111503)
    p.add_argument('--out', type=Path)
    p.add_argument('--acknowledge-training', action='store_true')
    a = p.parse_args(argv)
    if a.hours is None:
        a.hours = .75 if a.stage == 'interceptor' else 4.
    try:
        a.device_ids = [int(x) for x in a.devices.split(',')]
    except ValueError:
        p.error('invalid device indices')
    if not a.device_ids or min(a.device_ids) < 0 or len(set(a.device_ids)) != len(a.device_ids):
        p.error('devices must be distinct nonnegative indices')
    if a.envs < 2 or a.envs % 2 or a.ticks < 1 or a.minibatch < 1 or a.envs * a.ticks % a.minibatch:
        p.error('even envs and divisible envs*ticks/minibatch required')
    if a.value_chunk < 1 or a.envs*a.ticks % min(a.envs*a.ticks, a.value_chunk):
        p.error('effective Critic batch must divide envs*ticks')
    if a.policy_chunk < 1 or a.envs % min(a.envs, a.policy_chunk):
        p.error('effective policy chunk must divide envs')
    if not 0 < a.hours <= 5 or a.max_updates < 1 or a.critic_warmup < 0:
        p.error('invalid wall/update budget')
    if not 0 <= a.interceptor_fraction <= 1:
        p.error('invalid interceptor fraction')
    if not math.isfinite(a.hours) or not 0 <= a.seed <= 0xffffffff:
        p.error('invalid hours/seed')
    if a.stage == 'interceptor' and (a.interceptor or a.defender):
        p.error('--interceptor and --defender apply only to stage B (courier)')
    if a.stage == 'courier' and a.courier_opponent:
        p.error('--courier-opponent applies only to stage A (interceptor)')
    if a.stage == 'courier' and (a.interceptor_start or a.interceptor_actor):
        p.error('--interceptor-start and --interceptor-actor apply only to stage A (interceptor)')
    if a.interceptor and a.defender:
        p.error('Choose either --interceptor or repeat --defender')
    if len(a.defender) > len(DUEL_SEEDS) or len({p.resolve() for p in a.defender}) != len(a.defender):
        p.error('Pass distinct defender directories (at most one per duel seed)')
    a.interceptor_start = a.interceptor_start or 'u930'
    try:
        counts = opponent_counts(a.envs, a.stage, a.interceptor_fraction, max(1, len(a.defender)))
        if a.defender and not all(counts[:-1]):
            raise ValueError('Each selected defender needs at least one pair of environments')
    except ValueError as error:
        p.error(str(error))
    if a.deadline_utc:
        try:
            d = datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
            if d.utcoffset() is None or d.utcoffset().total_seconds() != 0:
                raise ValueError()
            a.deadline_timestamp = d.timestamp()
        except ValueError:
            p.error('deadline must be an ISO UTC timestamp')
    if a.mode == 'train' and (not a.acknowledge_training or not a.out):
        p.error('train requires --acknowledge-training and a new --out')
    if a.out:
        a.out = a.out.resolve()
        if not a.out.is_relative_to(ROOT) or a.out == ROOT or a.out.exists():
            p.error('output must be a new directory inside this project')
    return a


def require_launch(a):
    """Keep the persistent stop and require explicit bounded-round authorization."""
    if not a.acknowledge_training:
        raise RuntimeError('Explicit new-round acknowledgement required')
    from scripts.portable_training import require_training_authorization
    require_training_authorization()


def plan(a):
    # Standard-library only: importing this entry point and planning cannot use DCUs.
    pool = opponent_pool(a)
    return dict(schema='qqt.dcu-jax-carry/v1', stage=a.stage, devices=a.device_ids,
                envs_per_device=a.envs, global_environments=a.envs * len(a.device_ids),
                ticks=a.ticks, global_samples=a.envs * a.ticks * len(a.device_ids),
                global_minibatch=a.minibatch * len(a.device_ids),
                initial_actor=a.interceptor_start if a.stage == 'interceptor' else 'SP_web_u23302',
                initial_actor_path=str(seed_path(a)) if a.stage == 'interceptor' else None,
                initial_actor_sha256=DUEL_SEEDS[a.interceptor_start] if a.stage == 'interceptor' else SP_SHA,
                opponent_fractions={row['id']: row['envs_per_device']/a.envs for row in pool},
                opponent_pool=pool, critic_warmup_updates=a.critic_warmup,
                opponents_frozen=True, online_selfplay=False,
                alternating_segment=True,
                interceptor=(str(a.interceptor) if a.interceptor else
                             'league' if a.defender else 'original_u930'),
                interceptor_transport_allowed=False, normal_opponent_transport_allowed=True,
                simulator='full map806, native Bun custody and respawns',
                rollout_gae_on_device=True, host_callbacks=False,
                collector=a.collector, value_chunk=min(a.envs*a.ticks, a.value_chunk),
                policy_chunk=min(a.envs, a.policy_chunk),
                compiler_profile=a.compiler_profile,
                collector_execution=('device lax.scan, deferred Critic' if a.collector=='fused'
                    else 'separate device executables'),
                per_tick_host_tensor_reads=False, python_tick_dispatch=a.collector!='fused',
                hours=a.hours, deadline_utc=a.deadline_utc,
                source_sha256={f: sha(ROOT/f) for f in SOURCES},
                optimizer_saved=False, precise_resume=False, automatic_successor=False,
                checkpoint_policy=dict(every_updates=5, seconds=120, keep=32,
                                       budget_gib=2, min_free_gib=8))


def load_verified(path, expected, template):
    """Hash before unpickling our own trusted weights; enforce ABI and finite data."""
    import jax
    import jax.numpy as j
    import numpy as np
    if sha(path) != expected:
        raise ValueError('Model SHA mismatch: ' + str(path))
    value = pickle.loads(Path(path).read_bytes())
    if jax.tree.structure(value) != jax.tree.structure(template):
        raise ValueError('Model tree mismatch')
    for x, y in zip(jax.tree.leaves(value), jax.tree.leaves(template)):
        if x.shape != y.shape or x.dtype != y.dtype or not np.isfinite(x).all():
            raise ValueError('Model ABI/finite mismatch')
    return jax.tree.map(j.asarray, value)


def configure_compiler(profile):
    """Process-local opt-in; preserve caller XLA_FLAGS and all installed software."""
    if profile == 'throughput':
        os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0 '
            '--xla_disable_hlo_passes=constant_folding,multi_output_fusion '
            '--xla_backend_optimization_level=2 --xla_llvm_disable_expensive_passes=true')


def read_checkpoint(path, stage):
    meta = json.loads((path/'CHECKPOINT.json').read_text())
    if meta.get('schema') != SCHEMA or meta.get('stage') != stage:
        raise ValueError('Checkpoint must come from carry trainer stage: ' + stage)
    if meta.get('u930_sha256') != U930_SHA or meta.get('SP_web_sha256') != SP_SHA:
        raise ValueError('Checkpoint ancestry mismatch')
    sources = meta['source_sha256']
    legacy = 'qqt_rl/training/carry_league.py' not in sources
    required = set(SOURCES) - ({'qqt_rl/training/carry_league.py'} if legacy else set())
    if set(sources) != required:
        raise ValueError('Checkpoint source list mismatch')
    for f, expected in sources.items():
        actual = LEGACY_SOURCES[f] if legacy and f in LEGACY_SOURCES else sha(ROOT/f)
        if f == 'scripts/train_dcu_jax_carry.py' and expected in TRUSTED_TRAIN_SOURCE_SHAS:
            continue
        if actual != expected:
            raise ValueError('Checkpoint source mismatch: ' + f)
    return meta


def opponent_pool(a):
    if a.stage == 'interceptor':
        if a.courier_opponent:
            meta = read_checkpoint(a.courier_opponent, 'courier')
            return [dict(id='courier:u' + f'{int(meta["update"]):06d}',
                         envs_per_device=a.envs, can_transport=True,
                         checkpoint=str(a.courier_opponent), seed='courier',
                         actor_sha256=meta['actor_sha256'])]
        return [dict(id='SP_web', envs_per_device=a.envs, can_transport=True,
                     actor_sha256=SP_SHA, checkpoint=None)]
    counts = opponent_counts(a.envs, a.stage, a.interceptor_fraction, max(1, len(a.defender)))
    pool = []
    for position, path in enumerate(a.defender or [a.interceptor]):
        row = dict(id='interceptor', envs_per_device=counts[position], can_transport=False)
        if path:
            meta = read_checkpoint(path, 'interceptor')
            seed = meta.get('initial_actor', 'u930')
            digest = meta['actor_sha256']
            if not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('Invalid defender Actor SHA')
            if a.defender:
                if (seed not in DUEL_SEEDS or meta.get('initial_actor_sha256') != DUEL_SEEDS[seed]
                        or meta.get('update', 0) <= meta.get('critic_warmup_updates', 10)
                        or digest == DUEL_SEEDS[seed]):
                    raise ValueError(f'Defender must have trained its Actor on the carry map: {path}')
                row['id'] = f'interceptor:{seed}'
            row.update(checkpoint=str(path), seed=seed, actor_sha256=digest)
        else:
            row.update(checkpoint=None, seed='u930', actor_sha256=U930_SHA)
        pool.append(row)
    if len({row['seed'] for row in pool}) != len(pool) or len({row['actor_sha256'] for row in pool}) != len(pool):
        raise ValueError('Defender pool must use distinct warm-start seeds and Actor weights')
    return pool + [dict(id='SP_web', envs_per_device=counts[-1], can_transport=True,
                        actor_sha256=SP_SHA, checkpoint=None)]


def run(a, started):
    import numpy as np
    import jax
    import jax.numpy as j
    from scripts import portable_training as portable
    from scripts.r4_k4_observation import actor_critic_forward
    from qqt_rl.training.jax_carry_collect import build as collector, consistency_check
    from qqt_rl.training.jax_carry_env import METRICS
    from qqt_rl.training.kill_ppo import build
    from qqt_rl.training.checkpoints import CheckpointStore, durable_json

    devices = jax.local_devices()
    cards = len(devices)
    if cards != len(a.device_ids) or any(d.platform != 'gpu' for d in devices):
        raise RuntimeError('Expected explicit ROCm DCUs; no CPU fallback')
    config = plan(a)
    config['XLA_FLAGS'] = os.environ.get('XLA_FLAGS')
    with jax.default_device(jax.devices('cpu')[0]):
        portable.verify_bundle()
        protocol = portable.read(portable.BUNDLE/'protocols/SP.json')
        if protocol['models']['SP_web']['sha256'] != SP_SHA:
            raise ValueError('SP_web identity changed')
        sp = portable.load_actor(protocol, 'SP_web')
        pool = config['opponent_pool']
        opponents = [load_verified(Path(row['checkpoint'])/'actor.pt', row['actor_sha256'], sp)
                     if row.get('checkpoint') else sp for row in pool]
        seed_actor = (load_verified(seed_path(a), config['initial_actor_sha256'], sp)
                      if a.stage == 'interceptor' else sp)
        params = portable.fresh_parameters(seed_actor, a.seed)
        if a.checkpoint:
            meta = read_checkpoint(a.checkpoint, a.stage)
            if a.stage == 'interceptor' and meta.get('initial_actor_sha256', U930_SHA) != config['initial_actor_sha256']:
                raise ValueError('Resume requires the original --interceptor-start seed')
            for b in ('actor', 'critic'):
                params[b] = load_verified(a.checkpoint/(b+'.pt'), meta[b+'_sha256'], params[b])
        config.update(u930_sha256=U930_SHA, SP_web_sha256=SP_SHA,
                      frozen_interceptor_actor_sha256=pool[0]['actor_sha256'] if len(pool) == 2 else None,
                      warm_start=str(a.checkpoint) if a.checkpoint else None,
                      fresh_adam=True, fresh_critic=not bool(a.checkpoint), jax=jax.__version__)
        init_opt, _, update = build(actor_critic_forward, devices, a.minibatch)
        collector_defenders = 1 if a.stage == 'interceptor' else len(pool) - 1
        if a.collector == 'baseline':
            init_world, collect = collector(devices, a.envs, a.ticks, stage=a.stage,
                                            interceptor_fraction=a.interceptor_fraction, defenders=collector_defenders)
        else:
            from qqt_rl.training.jax_carry_fast import build as fast_collector
            init_world, collect = fast_collector(devices, a.envs, a.ticks, stage=a.stage,
                interceptor_fraction=a.interceptor_fraction, fused=a.collector=='fused',
                value_chunk=a.value_chunk, policy_chunk=a.policy_chunk, defenders=collector_defenders)
        adam_host = init_opt(params)
        bank_host = jax.tree.map(lambda *models: j.stack(models), *opponents)
    rep = lambda x: jax.device_put_replicated(x, devices)
    p, adam, reference, bank = rep(params), rep(adam_host), rep(params['actor']), rep(bank_host)
    keys = jax.random.split(jax.random.PRNGKey(a.seed), cards)
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    durable_json(out/'PLAN.json', config)
    store = CheckpointStore(out, keep=32, budget_gib=2, min_free_gib=8, schema=SCHEMA)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    deadline = started + a.hours*3600
    if a.deadline_utc:
        deadline = min(deadline, time.monotonic() + a.deadline_timestamp - time.time())
    finite = jax.pmap(lambda q: j.all(j.stack([j.all(j.isfinite(x)) for x in jax.tree.leaves(q)])), devices=devices)
    summary = jax.pmap(lambda q: j.stack([j.stack([j.sum(x), j.sum(x*x)])
                                        for x in jax.tree.leaves(q)]), devices=devices)
    check_rollout = consistency_check(devices, a.minibatch)
    last_saved, last_save, completed, steps, healthy = None, 0., 0, 0, True

    def checkpoint(index):
        nonlocal last_saved, last_save
        if last_saved == index or not healthy:
            return
        sums = np.asarray(summary(p))
        if not np.isfinite(sums).all() or not np.array_equal(sums, np.broadcast_to(sums[0], sums.shape)):
            raise RuntimeError('Model replicas differ/nonfinite')
        host = jax.tree.map(lambda x: np.asarray(x[0]), p)
        meta = dict(schema=SCHEMA, stage=a.stage, source_sha256=config['source_sha256'],
                    u930_sha256=U930_SHA, SP_web_sha256=SP_SHA,
                    initial_actor=config['initial_actor'], initial_actor_sha256=config['initial_actor_sha256'],
                    critic_warmup_updates=a.critic_warmup, opponent_pool=pool,
                    frozen_interceptor_actor_sha256=config['frozen_interceptor_actor_sha256'],
                    collector=a.collector, compiler_profile=a.compiler_profile,
                    XLA_FLAGS=config['XLA_FLAGS'], envs_per_device=a.envs,
                    minibatch=a.minibatch, value_chunk=a.value_chunk,
                    policy_chunk=a.policy_chunk,
                    global_environment_steps=steps, cards=cards, rule='native-map806-full-role-carry',
                    mode_row=0, optimizer_saved=False, precise_resume=False)
        receipt = store.publish(index, host, meta)
        last_saved, last_save = index, time.monotonic()
        print(json.dumps(dict(event='checkpoint', update=index, **receipt)), flush=True)

    def stopped():
        require_launch(a)
        return stop[0] or time.monotonic() >= deadline or (out/'USER_STOP').exists()

    opponent_ids = [row['id'] for row in pool]
    # The collector reports one metric row per allocation group. Courier
    # stages have one row per frozen opponent; interceptor stages also include
    # the learner group, so derive the shape from the first rollout instead of
    # assuming it matches the opponent pool length.
    cumulative = None
    try:
        checkpoint(0)
        print(json.dumps(dict(event='initializing_full_bun_worlds', **config)), flush=True)
        world = init_world(keys)
        jax.block_until_ready(world)
        for index in range(1, a.max_updates+1):
            if stopped():
                break
            begin = time.monotonic()
            world, keys, batch, metrics, health = collect(p, world, bank, keys)
            jax.block_until_ready((batch, metrics, health))
            collected = time.monotonic()
            if np.asarray(health['item_overflow']).max():
                raise RuntimeError('Native item queue overflow')
            if np.asarray(health['nonfinite']).sum():
                raise RuntimeError('Nonfinite rollout logits/value/logp/reward; refusing PPO update')
            if index == 1:
                errors = np.asarray(check_rollout(p, batch))
                durable_json(out/'INFERENCE_CHECK.json', dict(
                    max_abs_logp_error=float(errors[:, 0].max()),
                    max_abs_value_error=float(errors[:, 1].max()),
                    passed=bool(np.isfinite(errors).all() and
                                (errors[:, 0] <= .03).all() and (errors[:, 1] <= .1).all())))
                if not np.isfinite(errors).all() or (errors[:, 0] > .03).any() or (errors[:, 1] > .1).any():
                    raise RuntimeError('Rollout/separate inference mismatch; refusing PPO update')
            if stopped():
                break
            keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)
            healthy = False
            (p, adam), loss = update(p, adam, reference, batch, keys,
                                     np.full(cards, index <= a.critic_warmup))
            jax.block_until_ready(loss)
            if not np.asarray(finite((p, adam))).all() or not np.isfinite(np.asarray(loss)).all():
                raise RuntimeError('Nonfinite PPO state; retaining last durable checkpoint')
            healthy = True
            elapsed = time.monotonic() - begin
            completed, steps = index, steps + cards*a.envs*a.ticks
            metric_delta = np.asarray(metrics).sum(0)
            if cumulative is None:
                cumulative = np.zeros_like(metric_delta, dtype=np.float64)
            cumulative += metric_delta
            row = dict(update=index, stage=a.stage, environment_steps=steps,
                       seconds=elapsed, environment_steps_per_second=cards*a.envs*a.ticks/elapsed,
                       collect_seconds=collected-begin, ppo_seconds=time.monotonic()-collected,
                       elapsed_seconds=time.monotonic()-started,
                       critic_only=index <= a.critic_warmup,
                       metric_names=METRICS, opponent_ids=opponent_ids,
                       metric_axes=['opponent:' + ','.join(opponent_ids), 'learner,opponent', 'metric'],
                       metrics=metric_delta.tolist(), cumulative_metrics=cumulative.tolist(),
                       loss=np.asarray(loss).mean((0, 1)).tolist())
            with (out/'updates.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(event='update', **row)), flush=True)
            if index == 1 or index == a.critic_warmup or index % 5 == 0 or time.monotonic()-last_save >= 120:
                checkpoint(index)
        checkpoint(completed)
        durable_json(out/'RESULT.json', dict(status='UPDATES_COMPLETED' if completed else 'NO_UPDATE',
                     stage=a.stage, updates=completed, environment_steps=steps,
                     elapsed_seconds=time.monotonic()-started, optimizer_saved=False, automatic_successor=False))
    finally:
        try:
            checkpoint(completed)
        finally:
            durable_json(out/'STOPPED.json', dict(updates=completed, environment_steps=steps,
                         healthy=healthy, optimizer_saved=False, automatic_successor=False,
                         stopped_at=datetime.now(timezone.utc).isoformat()))


def main(argv=None):
    started = time.monotonic()
    a = arguments(argv)
    os.chdir(ROOT)
    if a.mode == 'plan':
        print(json.dumps(plan(a), indent=2))
        return
    # Both gates run before configuring or importing JAX. Neither is rewritten.
    require_launch(a)
    from qqt_rl.training.dcu import configure
    configure_compiler(a.compiler_profile)
    configure(a.device_ids[0])
    visible = ','.join(map(str, a.device_ids))
    os.environ.update(HIP_VISIBLE_DEVICES=visible, CUDA_VISIBLE_DEVICES=visible,
                      JAX_PLATFORMS='rocm,cpu', JAX_ENABLE_COMPILATION_CACHE='false',
                      JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='false')
    run(a, started)


if __name__ == '__main__':
    main()
