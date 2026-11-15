"""Explicit, bounded single-DCU carry benchmark; never resumes/publishes training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--envs', type=int, default=128)
    p.add_argument('--ticks', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--stage', choices=('courier', 'interceptor'), default='courier')
    p.add_argument('--collector', choices=('baseline', 'optimized', 'fused'), default='baseline')
    p.add_argument('--ppo', action='store_true', help='time disposable full Actor/Critic PPO updates')
    p.add_argument('--pipeline-repeats', type=int, default=0,
                   help='actual consecutive rollout/update cycles; discard all resulting weights')
    p.add_argument('--validate-reset', action='store_true', help='exercise native timeout and fused reset')
    p.add_argument('--diagnose-mismatch', action='store_true',
                   help='report rejected inference and continue timing, never a training admission')
    p.add_argument('--minibatch', type=int, default=512)
    p.add_argument('--value-chunk', type=int, default=512)
    p.add_argument('--policy-chunk', type=int, default=512)
    p.add_argument('--compiler-profile', choices=('safe', 'throughput'), default='safe')
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--acknowledge-benchmark', action='store_true')
    a = p.parse_args(argv)
    if not a.acknowledge_benchmark:
        p.error('Explicit --acknowledge-benchmark required before selecting a DCU')
    if a.device < 0 or a.envs < 8 or a.envs > 4096 or a.envs % 8:
        p.error('Use one nonnegative device and 8..4096 envs divisible by8')
    if not 1 <= a.ticks <= 128 or not 1 <= a.repeats <= 20 or not 1 <= a.warmups <= 4:
        p.error('Invalid bounded timing geometry')
    if not 0 <= a.pipeline_repeats <= 5 or (a.pipeline_repeats and not a.ppo):
        p.error('Use --ppo and 0..5 pipeline cycles')
    if a.minibatch < 1 or a.envs*a.ticks % a.minibatch:
        p.error('minibatch must divide envs*ticks')
    if a.value_chunk < 1 or a.envs*a.ticks % min(a.envs*a.ticks, a.value_chunk):
        p.error('effective Critic chunk must divide envs*ticks')
    if a.policy_chunk < 1 or a.envs % min(a.envs, a.policy_chunk):
        p.error('effective policy chunk must divide envs')
    a.report = a.report.resolve()
    if not a.report.is_relative_to(ROOT) or a.report.exists():
        p.error('Use a new report file inside the project')
    return a


def main(argv=None):
    a = arguments(argv)
    os.chdir(ROOT)
    # The user's explicit benchmark command authorizes diagnostics only. The
    # persistent stop policy and all historical training jobs remain untouched.
    from qqt_rl.training.dcu import configure
    from scripts.train_dcu_jax_carry import configure_compiler
    configure_compiler(a.compiler_profile)
    configure(a.device)
    os.environ.update(JAX_PLATFORMS='rocm,cpu', JAX_ENABLE_COMPILATION_CACHE='false',
                      JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='false')
    import jax
    import jax.numpy as j
    import numpy as np
    from scripts import portable_training as portable
    from scripts.train_dcu_jax_carry import load_verified, U930, U930_SHA, SOURCES
    from scripts.r4_k4_observation import actor_critic_forward, forward
    from qqt_rl.training.jax_carry_collect import build, consistency_check
    from qqt_rl.training.kill_ppo import build as ppo_build, masked_logits
    from qqt_rl.training.checkpoints import durable_json

    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise RuntimeError('Expected exactly one explicitly selected DCU; no CPU fallback')
    print(json.dumps(dict(event='initialize', device=str(devices[0]),
                         device_kind=devices[0].device_kind, envs=a.envs,
                         ticks=a.ticks, stage=a.stage, collector=a.collector)), flush=True)
    started = time.monotonic()
    with jax.default_device(jax.devices('cpu')[0]):
        portable.verify_bundle()
        protocol = portable.read(portable.BUNDLE/'protocols/SP.json')
        sp = portable.load_actor(protocol, 'SP_web')
        u930 = load_verified(U930, U930_SHA, sp)
        params = portable.fresh_parameters(u930 if a.stage == 'interceptor' else sp, 2026111503)
        bank = jax.tree.map(lambda b, s: j.stack([b, s]), u930, sp)
        if a.collector == 'baseline':
            init, collect = build(devices, a.envs, a.ticks, stage=a.stage)
        else:
            from qqt_rl.training.jax_carry_fast import build as fast_build
            init, collect = fast_build(devices, a.envs, a.ticks, stage=a.stage,
                                       fused=a.collector == 'fused', value_chunk=a.value_chunk,
                                       policy_chunk=a.policy_chunk)
        if a.ppo:
            opt_init, _, update = ppo_build(actor_critic_forward, devices, a.minibatch)
            adam = opt_init(params)
    rep = lambda x: jax.device_put_replicated(x, devices)
    p, bank, reference = rep(params), rep(bank), rep(params['actor'])
    if a.ppo:
        adam = rep(adam)
    keys = jax.random.split(jax.random.PRNGKey(2026111503), 1)
    print(json.dumps(dict(event='models_prepared', seconds=time.monotonic()-started)), flush=True)
    world = init(keys)
    jax.block_until_ready(world)
    init_seconds = time.monotonic()-started
    print(json.dumps(dict(event='worlds_ready', initial_seconds=init_seconds)), flush=True)
    check = consistency_check(devices, a.minibatch)
    @jax.pmap
    def diagnose_actor(actor, b):
        ix = j.arange(b['logp'].size).reshape(-1, a.minibatch)
        def body(_, indices):
            mv, ab, _, _ = forward(actor, b['obs'][indices], b['state'][indices])
            lm, la = masked_logits(mv, ab, b['mm'][indices], b['bm'][indices])
            rows = j.arange(a.minibatch)
            acts = b['action'][indices]
            return None, lm[rows, acts[:, 0]] + la[rows, acts[:, 1]]
        return jax.lax.scan(body, None, ix)[1].reshape(-1)
    warmup_seconds, durations, errors = [], [], None
    diagnostic = None
    batch = None
    for index in range(a.warmups+a.repeats):
        print(json.dumps(dict(event='collect_start', index=index)), flush=True)
        begin = time.monotonic()
        world, keys, batch, metrics, health = collect(p, world, bank, keys)
        jax.block_until_ready((world, batch, metrics, health))
        seconds = time.monotonic()-begin
        if any(np.asarray(v).any() for v in health.values()):
            raise RuntimeError('Unhealthy rollout: ' + str(jax.device_get(health)))
        if index == 0:
            print(json.dumps(dict(event='first_rollout_ready', seconds=seconds,
                                  environment_steps_per_second=a.envs*a.ticks/seconds)), flush=True)
            checked = time.monotonic()
            errors = np.asarray(check(p, batch))
            if (not np.isfinite(errors).all() or errors[:, 0].max() > .03 or errors[:, 1].max() > .1) and not a.diagnose_mismatch:
                raise RuntimeError('Inference mismatch: ' + str(errors))
            print(json.dumps(dict(event='inference_checked', max_abs_errors=errors.tolist(),
                                  check_seconds=time.monotonic()-checked)), flush=True)
            if a.diagnose_mismatch:
                new_logp = np.asarray(diagnose_actor(p['actor'], batch))[0]
                old_logp = np.asarray(batch['logp'])[0]
                delta = np.abs(new_logp-old_logp)
                diagnostic = dict(actor_only_logp_delta_quantiles=np.quantile(delta, [.5,.9,.99,1]).tolist(),
                                  fraction_over_003=float((delta>.03).mean()))
                print(json.dumps(dict(event='actor_diagnostic', **diagnostic)), flush=True)
                selected = np.argsort(delta)[-16:]
                evidence = a.report.with_suffix('.npz')
                evidence.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(evidence, obs=np.asarray(batch['obs'])[0, selected],
                    state=np.asarray(batch['state'])[0, selected],
                    mm=np.asarray(batch['mm'])[0, selected], bm=np.asarray(batch['bm'])[0, selected],
                    action=np.asarray(batch['action'])[0, selected], old_logp=old_logp[selected],
                    actor_recomputed_logp=new_logp[selected])
                diagnostic['evidence'] = str(evidence.relative_to(ROOT))
        if index < a.warmups:
            warmup_seconds.append(seconds)
        else:
            durations.append(seconds)
        print(json.dumps(dict(event='rollout', index=index, warmup=index<a.warmups,
                              seconds=seconds, environment_steps_per_second=a.envs*a.ticks/seconds,
                              episode_ticks=np.asarray(world.state.t).max().item())), flush=True)
    update_durations = []
    if a.ppo:
        for index in range(3):
            begin = time.monotonic()
            (next_p, next_adam), loss = update(p, adam, reference, batch, keys, np.array([False]))
            jax.block_until_ready((next_p, next_adam, loss))
            seconds = time.monotonic()-begin
            # Discard all updates; reference inputs remain unchanged for timing.
            if not np.isfinite(np.asarray(loss)).all():
                raise RuntimeError('Nonfinite diagnostic PPO loss')
            if index:
                update_durations.append(seconds)
            print(json.dumps(dict(event='ppo', index=index, compile=index==0, seconds=seconds)), flush=True)
    pipeline_durations, pipeline_errors = [], []
    if a.pipeline_repeats:
        pipeline_p, pipeline_adam = p, adam
        @jax.pmap
        def finite(q):
            return j.all(j.stack([j.all(j.isfinite(x)) for x in jax.tree.leaves(q)]))
        jax.block_until_ready(finite((pipeline_p, pipeline_adam)))
        for index in range(a.pipeline_repeats):
            begin = time.monotonic()
            rollout_p = pipeline_p
            world, keys, batch, metrics, health = collect(pipeline_p, world, bank, keys)
            jax.block_until_ready(health)
            if any(np.asarray(v).any() for v in health.values()):
                raise RuntimeError('Unhealthy consecutive rollout')
            (pipeline_p, pipeline_adam), loss = update(pipeline_p, pipeline_adam, reference,
                                                      batch, keys, np.array([False]))
            jax.block_until_ready(loss)
            if not np.isfinite(np.asarray(loss)).all() or not np.asarray(finite((pipeline_p, pipeline_adam))).all():
                raise RuntimeError('Nonfinite consecutive diagnostic update')
            seconds = time.monotonic()-begin
            pipeline_durations.append(seconds)
            print(json.dumps(dict(event='pipeline', index=index, seconds=seconds,
                environment_steps_per_second=a.envs*a.ticks/seconds)), flush=True)
            # Validation is outside the timed interval; use the parameters
            # that actually sampled these actions, before this PPO update.
            cycle_errors = np.asarray(check(rollout_p, batch))
            pipeline_errors.append(cycle_errors.tolist())
            if not np.isfinite(cycle_errors).all() or cycle_errors[:, 0].max() > .03 or cycle_errors[:, 1].max() > .1:
                raise RuntimeError('Consecutive inference mismatch: ' + str(cycle_errors))
    reset_result = None
    if a.validate_reset:
        @jax.pmap
        def at_timeout(w):
            return w._replace(state=w.state._replace(core=w.state.core._replace(t=j.full_like(w.state.t, 2399))))
        reset_world = at_timeout(init(keys))
        reset_world, _, _, reset_metrics, reset_health = collect(p, reset_world, bank, keys)
        jax.block_until_ready(reset_world)
        games = float(np.asarray(reset_metrics)[..., 8].sum()/2)
        reset_result = dict(games=games, max_ticks=int(np.asarray(reset_world.state.t).max()))
        if games < a.envs or reset_result['max_ticks'] >= a.ticks or any(np.asarray(v).any() for v in reset_health.values()):
            raise RuntimeError('Timeout reset validation failed: ' + str(reset_result))
        print(json.dumps(dict(event='reset_verified', **reset_result)), flush=True)
    median = float(np.median(durations))
    row = dict(schema='qqt.single-dcu-carry-benchmark/v1',
               recorded_at=datetime.now(timezone.utc).isoformat(),
               stage=a.stage, collector=a.collector, device_kind=devices[0].device_kind,
               jax=jax.__version__, devices=1, envs=a.envs, ticks=a.ticks,
               minibatch=a.minibatch, samples=a.envs*a.ticks, initial_seconds=init_seconds,
               value_chunk=min(a.envs*a.ticks, a.value_chunk),
               policy_chunk=min(a.envs, a.policy_chunk),
               compiler_profile=a.compiler_profile,
               warmup_seconds=warmup_seconds, rollout_seconds=durations,
               median_rollout_seconds=median, rollout_environment_steps_per_second=a.envs*a.ticks/median,
               ppo_seconds=update_durations,
               pipeline_seconds=pipeline_durations,
               pipeline_inference_errors=pipeline_errors,
               measured_pipeline_environment_steps_per_second=(a.envs*a.ticks/float(np.median(pipeline_durations)))
                    if pipeline_durations else None,
               reset_validation=reset_result,
               timed_pipeline_environment_steps_per_second=(a.envs*a.ticks/(median+float(np.median(update_durations))))
                    if update_durations else None,
               inference_errors=errors.tolist(), memory_stats=devices[0].memory_stats(),
               diagnostic=diagnostic,
               inference_admitted=bool(np.isfinite(errors).all() and errors[:, 0].max()<=.03 and errors[:, 1].max()<=.1),
               optimizer_updates_discarded=3+a.pipeline_repeats if a.ppo else 0, checkpoint_writes=0,
               automatic_successor=False, XLA_FLAGS=os.environ.get('XLA_FLAGS'),
               source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES})
    if a.collector != 'baseline':
        f = 'qqt_rl/training/jax_carry_fast.py'
        row['source_sha256'][f] = hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    row['training_environment_steps_per_second'] = (
        row['timed_pipeline_environment_steps_per_second'] if row['inference_admitted'] else None)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    durable_json(a.report, row)
    print(json.dumps(dict(event='result', **row)), flush=True)


if __name__ == '__main__':
    main()
