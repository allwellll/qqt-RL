"""Bounded DCU migration check using the verified DUEL/SP release assets.

Default: real model forward only. --update: one small rollout and PPO update;
discard all parameters afterwards, without publishing models or resuming queues.
Run each line in a fresh process because the sealed runtime installs modules.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import time

from scripts import portable_training as portable
from qqt_rl.training.dcu import configure


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--line', choices=('DUEL', 'SP'), required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--ticks', type=int, default=2)
    parser.add_argument('--minibatch', type=int, default=8)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error('--device must be nonnegative')
    if not (4 <= args.envs <= 32 and args.envs % 4 == 0):
        parser.error('--envs must be a multiple of 4 between 4 and 32')
    if not (1 <= args.ticks <= 8):
        parser.error('--ticks must be between 1 and 8 for this diagnostic')
    if args.minibatch < 1 or args.envs * args.ticks % args.minibatch:
        parser.error('envs * ticks must be divisible by a positive minibatch')
    if args.report and args.report.exists():
        parser.error('report already exists; choose a new path')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.report:
        args.report = args.report.resolve()
    os.chdir(portable.ROOT)
    # Select ROCm explicitly: a missing plugin must fail, never fall back to CPU.
    # HIP selection refers to physical devices; JAX then sees one logical device.
    configure(args.device)
    portable.verify_bundle()
    protocol = portable.read(portable.BUNDLE / 'protocols' / (args.line + '.json'))
    import jax
    import jax.numpy as j
    import jaxlib
    import numpy as np
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != 'gpu' or 'cuda' in devices[0].device_kind.lower():
        raise RuntimeError(f'Expected exactly one ROCm DCU, got {devices}')
    print(json.dumps(dict(stage='initializing', line=args.line,
        device=str(devices[0]), device_kind=devices[0].device_kind)), flush=True)
    started = time.monotonic()
    _, mixed, bs, ba, spawns, params, states, aux, keys, bank = portable.initialize(protocol, args.envs)
    from scripts.r4_k4_observation import actor_critic_forward
    backend = mixed.bun_backend()
    # Compile observation construction with inference. Eager DCU execution of
    # its many small danger-map operations otherwise compiles hundreds of kernels.
    def infer(p, st, seats, ledger):
        obs, global_state, _, _ = backend.inputs(st, seats, ledger)
        return obs, global_state, actor_critic_forward(p, obs, global_state)
    obs, global_state, outputs = jax.jit(infer)(params, states,
        aux['combat_mix']['learner_seat'], aux['combat_mix']['bomb_threat']['r3'])
    jax.block_until_ready(outputs)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(outputs)):
        raise RuntimeError('Nonfinite model forward')
    result = dict(status='PASS_DCU_FORWARD', line=args.line, ruleset=protocol['ruleset'],
        jax=jax.__version__, jaxlib=jaxlib.__version__, device_kind=devices[0].device_kind,
        envs=args.envs, obs_shape=list(obs.shape), global_shape=list(global_state.shape),
        forward_seconds=time.monotonic()-started, rollout_calls=0, optimizer_updates=0,
        checkpoint_writes=0, automatic_successor=False)
    print(json.dumps(result), flush=True)
    if args.update:
        import optax
        optimizer = optax.adam(protocol['spec']['lr'], eps=1e-8)
        adam = {branch: optimizer.init(params[branch]) for branch in ('actor', 'critic')}
        name = 'scripts.cx_duel_updater_v1' if args.line == 'DUEL' else 'scripts.cc_sil_updater_v1'
        update = importlib.import_module(name).updater(protocol['spec']['sil_coef'])(args.minibatch, optimizer)
        collect = jax.jit(mixed.collector(args.ticks, args.envs, protocol['seed'], spawns))
        print(json.dumps(dict(stage='collecting', ticks=args.ticks)), flush=True)
        started = time.monotonic()
        (_, _, _), batch, exposure = collect(params, states, aux, keys, bs, ba, bank, j.asarray(0, j.int32))
        jax.block_until_ready((batch, exposure))
        result['rollout_seconds'] = time.monotonic()-started
        print(json.dumps(dict(stage='updating', rollout_seconds=result['rollout_seconds'])), flush=True)
        started = time.monotonic()
        (next_params, next_adam), loss = update(params, adam, batch,
            jax.random.PRNGKey(protocol['seed'] ^ 0x1EA6))
        jax.block_until_ready((next_params, next_adam, loss))
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((next_params, next_adam, loss))):
            raise RuntimeError('Nonfinite PPO/Adam result')
        changes = {branch: float(j.sqrt(sum(j.sum((a-b)**2) for a, b in
            zip(jax.tree.leaves(next_params[branch]), jax.tree.leaves(params[branch])))))
            for branch in ('actor', 'critic')}
        if not all(value > 0 for value in changes.values()):
            raise RuntimeError(f'Expected actor and critic updates: {changes}')
        result.update(status='PASS_DCU_UPDATE', ticks=args.ticks, minibatch=args.minibatch,
            rollout_calls=1, optimizer_updates=args.envs*args.ticks//args.minibatch,
            update_seconds=time.monotonic()-started, parameter_delta_l2=changes,
            loss=np.asarray(loss).mean(0).tolist())
    result['memory_stats'] = devices[0].memory_stats()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
