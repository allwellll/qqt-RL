#!/usr/bin/env python3
"""Vary only the danger-arena reward for an existing surviving-kill event."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_ema as ema
from jax_bomb import bun_env
import jax.numpy as jnp
from qqt_rl.training.io import atomic_write_json, sha256_file


def make_reward(original, amount):
    if amount not in (12., 24.):
        raise ValueError('surviving-kill reward must be 12 or 24')
    if amount == 12.:
        return original

    def reward(*args, **kwargs):
        value = original(*args, **kwargs)
        info = kwargs['rule_info']
        eligible = (info['surviving_kill'].astype(jnp.bool_)
                    & ~info['mutual_death'][:, None]
                    & (info['lesson'] == bun_env.LESSON_DANGER_ARENA)[:, None])
        return value + (amount - 12.) * eligible.astype(jnp.float32)

    return reward


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--surviving-kill-reward', type=float, choices=[12., 24.], required=True)
    args, rest = parser.parse_known_args()
    save = Path(rest[rest.index('--save') + 1]).resolve()
    # Inner EMA writers compare checkpoint directories to this absolute path.
    rest[rest.index('--save') + 1] = str(save)
    training = ema.reward.training
    original_reward = training.jt.reward_from_events
    original_writer, original_argv = training.atomic_write_json, sys.argv
    meta = {'surviving_kill_reward': args.surviving_kill_reward,
            'baseline_surviving_kill_reward': 12.,
            'event': 'existing rule_info.surviving_kill; same-tick survivor and causal/physical credit',
            'lesson': 'danger_arena', 'mutual_death_bonus': 0.,
            'unchanged_terms': ['death', 'self_kill', 'avoidable_death', 'mutual',
                                'escape', 'hit', 'positive_tactical_shaping'],
            'reward_shaping_scale': .6, 'parameter_scope': 'full', 'ema_decay': .95,
            'source_sha256': sha256_file(__file__),
            'reward_source_sha256': sha256_file(ROOT / 'jax_bomb/bun_env.py')}
    atomic_write_json(save.parent.parent / 'kill_reward_config.json', meta)

    def writer(path, data):
        path = Path(path).resolve()
        if path.name == 'trainer_config.json' or path.parent == save.parent:
            data = dict(data, surviving_kill_reward=args.surviving_kill_reward,
                        baseline_surviving_kill_reward=12., kill_reward_mutual_bonus=0.)
        return original_writer(path, data)

    training.jt.reward_from_events = make_reward(original_reward, args.surviving_kill_reward)
    training.atomic_write_json = writer
    sys.argv = [sys.argv[0], '--export-mode', 'ema95', *rest]
    print(f'kill_reward_config={meta}', flush=True)
    try:
        return ema.main()
    finally:
        training.jt.reward_from_events = original_reward
        training.atomic_write_json, sys.argv = original_writer, original_argv


if __name__ == '__main__':
    sys.exit(main())
