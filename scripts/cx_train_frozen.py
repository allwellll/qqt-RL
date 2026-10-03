#!/usr/bin/env python3
"""Matched one-seat PPO: current versus frozen it4000 opponent.

Reuses existing JAX rollout/GAE/PPO without altering evaluator dependencies.
Both arms use P0-only learning and the same optimizer update budget.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('JAXBOMB_RULE', 'bun')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_bomb import jax_train as jt
from qqt_rl.training.io import atomic_write_json, sha256_file
from qqt_rl.training.jax_cache import configure_persistent_cache

CONFIG = {'num_envs': 512, 'num_steps': 64, 'minibatch': 512, 'epochs': 1,
          'arch': 'transformer', 'embed': 192, 'depth': 4, 'lr': 3e-4,
          'gamma': .995, 'lam': .95, 'clip_eps': .2, 'vf_coef': .5, 'ent_coef': .01}
SPAWNS = 'native=0.3,near=0.1,mid=0.15,far=0.1,below=0.15,upper_left=0.1,upper_right=0.1'
REWARDS = {'danger_escape_reward': .75, 'avoidable_danger_death_penalty': 4.,
           'tactical_bomb_placement_reward': .6, 'tactical_bomb_resolution_reward': 1.,
           'base_bomb_reward': .06, 'forced_kill_reward': 2., 'enemy_threat_reward': .4}


def opponent_parameters(current, reference, mode):
    if mode not in ('current', 'frozen'):
        raise ValueError(f'unknown opponent mode: {mode}')
    # 对手动作不参与梯度；current每轮rollout开始取学习者当前权重。
    chosen = reference if mode == 'frozen' else current
    return jax.tree.map(jax.lax.stop_gradient, chosen)


def learner_batch(batch, n):
    # 原rollout按P0/P1拼接；只保留P0，冻结对手样本不进入PPO。
    return tuple(x[:, :n] for x in batch[:-1]) + (
        tuple(mask[:, :n] for mask in batch[-1]),)


def build_one_iter(opt, reference, mode, config=CONFIG):
    weights = jnp.asarray(jt.parse_bun_opponent_weights('old=1'))
    n = config['num_envs']

    @jax.jit
    def one_iter(params, opt_state, states, key):
        opponent = opponent_parameters(params, reference, mode)
        states, batch, _, _ = jt.collect_rollout(
            params, config['arch'], states, key, config['num_steps'],
            flee_bot_ratio=0., bun_opponent_weights=weights,
            bun_opponent_weak_params=opponent, bun_opponent_old_params=opponent,
            bun_opponent_recent_params=opponent)
        obs, gv, acts, lps, vals, rew, done, masks = learner_batch(batch, n)
        fobs, fgv = jt.both_perspectives(states), jt.both_states(states)
        fm = jt.both_masks(states)
        _, _, tail = jt.sample_actions(params, config['arch'], fobs[:n],
            (fm[0][:n], fm[1][:n]), jax.random.split(key)[0], state=fgv[:n])
        nxt = jnp.concatenate([vals[1:], tail[None]], axis=0)
        adv = jt.compute_gae(rew, vals, nxt, done, config['gamma'], config['lam'])
        params, opt_state, loss = jt.ppo_update(
            params, opt, opt_state, config['arch'],
            (obs, gv, acts, lps, adv, adv + vals, masks), key,
            config['minibatch'], config['clip_eps'], config['vf_coef'],
            config['ent_coef'], config['epochs'], return_loss=True)
        return params, opt_state, states, jax.random.split(key)[0], loss
    return one_iter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', required=True, choices=['current', 'frozen'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--iters', type=int, required=True)
    ap.add_argument('--save-every', type=int, required=True)
    ap.add_argument('--load', required=True)
    ap.add_argument('--save', required=True)
    args = ap.parse_args()
    if args.iters < 1 or args.save_every < 1:
        raise ValueError('iters and save-every must be positive')
    run_dir = Path(args.save).resolve().parent.parent
    configure_persistent_cache(run_dir=run_dir)
    devices = jt.setup_platform()
    print(f'devices: {jt.device_summary(devices)}', flush=True)
    jt.prepare_environment()
    jt.configure_training('danger_arena=1', 1, reward_profile='danger_arena', **REWARDS)
    jt.configure_start_state_curriculum(None, '')
    print(f'spawn buckets={jt.configure_spawn_buckets(SPAWNS)}', flush=True)
    key = jax.random.PRNGKey(args.seed)
    states = jt.init_batch(key, CONFIG['num_envs'])
    key, _ = jax.random.split(key)
    reference = jt.load_params(args.load)
    params = reference
    opt = optax.adam(CONFIG['lr'])
    opt_state = opt.init(params)
    one_iter = build_one_iter(opt, reference, args.mode)
    meta = {**CONFIG, 'seed': args.seed, 'mode': args.mode, 'source_root': str(ROOT),
            'interpreter': sys.executable, 'init_sha256': sha256_file(args.load),
            'reward': REWARDS, 'spawns': SPAWNS, 'learner_seat': 0,
            'opponent_gradient': False, 'optimizer_steps_per_iteration': 32,
            'warmup_updates_committed': False, 'iterations': args.iters}
    atomic_write_json(run_dir / 'trainer_config.json', meta)
    print(f'config={meta}', flush=True)
    t0 = time.time()
    warm = (params, opt_state, states, key)
    for _ in range(2):
        out = one_iter(*warm)
        jax.block_until_ready(out)
        warm = out[:4]
        if not np.isfinite(float(out[4])):
            raise FloatingPointError('nonfinite warmup loss')
    del warm, out
    print(f'warmup done ({time.time()-t0:.1f}s)', flush=True)

    def save(path, updates):
        jt.save_params(params, str(path))
        atomic_write_json(Path(path).with_suffix('.json'), {**meta, 'updates_completed': updates})

    started = time.time()
    for iteration in range(args.iters):
        tick = time.time()
        params, opt_state, states, key, loss = one_iter(params, opt_state, states, key)
        jax.block_until_ready(params)
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at iteration {iteration}')
        elapsed = time.time() - tick
        sps = CONFIG['num_envs'] * CONFIG['num_steps'] / elapsed
        print(f'[iter {iteration}] {elapsed:.2f}s  sps={sps:,.0f} loss={loss:.6f}', flush=True)
        if iteration and iteration % args.save_every == 0:
            base = Path(args.save)
            save(base.with_name(f'{base.stem}_it{iteration}.pt'), iteration + 1)
    save(Path(args.save), args.iters)
    print(f'FINAL training_seconds={time.time()-started:.3f} updates={args.iters}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
