#!/usr/bin/env python3
"""Project only critic loss targets onto the existing HL-Gauss support."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_kill_reward as base_reward
import jax.numpy as jnp
from qqt_rl.training.io import atomic_write_json, sha256_file

training = base_reward.ema.reward.training
opponents = base_reward.ema.reward.clear_opponents.opponents
bun_env = base_reward.bun_env


def make_value_loss(original, projection):
    if projection not in ('current', 'projected'):
        raise ValueError('value projection must be current or projected')
    if projection == 'current':
        return original

    def loss(logits, targets, v_min=training.jt.V_MIN, v_max=training.jt.V_MAX,
             num_bins=training.jt.NUM_VALUE_BINS, sigma=1.5):
        return original(logits, jnp.clip(targets, v_min, v_max),
                        v_min=v_min, v_max=v_max, num_bins=num_bins, sigma=sigma)
    return loss


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--value-projection', choices=['current', 'projected'], required=True)
    args, rest = parser.parse_known_args()
    for flag, expected in [('--hunter-fraction', .5), ('--reward-shaping-scale', .6)]:
        if rest.count(flag) != 1 or float(rest[rest.index(flag) + 1]) != expected:
            raise ValueError(f'{flag} must remain {expected}')
    if any(f in rest for f in ('--surviving-kill-reward', '--export-mode', '--mode')):
        raise ValueError('reward 24, EMA .95 and current reference are fixed')
    save = Path(rest[rest.index('--save') + 1]).resolve()
    rest[rest.index('--save') + 1] = str(save)
    meta = {'value_projection': args.value_projection,
            'value_support': [training.jt.V_MIN, training.jt.V_MAX],
            'value_bins': training.jt.NUM_VALUE_BINS, 'sigma': 1.5,
            'scope': 'critic HL-Gauss loss targets only',
            'gae_and_actor_advantages_changed': False, 'value_scalar_decode_changed': False,
            'surviving_kill_reward': 24., 'ema_decay': .95, 'reward_shaping_scale': .6,
            'hunter_fraction': .5, 'remaining_opponent': 'current stochastic self-play',
            'source_sha256': sha256_file(__file__),
            'loss_source_sha256': sha256_file(ROOT / 'jax_bomb/jax_train.py'),
            'value_support_source_sha256': sha256_file(ROOT / 'jax_bomb/jax_net.py'),
            'reward_source_sha256': sha256_file(ROOT / 'scripts/cx_train_kill_reward.py')}
    atomic_write_json(save.parent.parent / 'value_projection_config.json', meta)
    old_loss, old_writer = training.jt.hl_gauss_value_loss, training.atomic_write_json
    old_reward, old_bot = training.jt.reward_from_events, training.jt.bun_opponent_actions
    old_fresh, old_argv = bun_env._fresh, sys.argv

    def writer(path, data):
        path = Path(path).resolve()
        if path.name == 'trainer_config.json' or path.parent == save.parent:
            data = dict(data, value_projection=args.value_projection,
                        critic_target_support=meta['value_support'],
                        value_scalar_decode_changed=False, gae_and_actor_advantages_changed=False)
        return old_writer(path, data)

    training.jt.hl_gauss_value_loss = make_value_loss(old_loss, args.value_projection)
    training.atomic_write_json = writer
    sys.argv = [sys.argv[0], '--surviving-kill-reward', '24', *rest]
    print(f'value_projection_config={meta}', flush=True)
    try:
        return base_reward.main()
    finally:
        training.jt.hl_gauss_value_loss, training.atomic_write_json = old_loss, old_writer
        training.jt.reward_from_events, training.jt.bun_opponent_actions = old_reward, old_bot
        bun_env._fresh, sys.argv = old_fresh, old_argv


if __name__ == '__main__':
    sys.exit(main())
