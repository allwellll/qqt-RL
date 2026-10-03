#!/usr/bin/env python3
"""Matched PPO with a fixed fraction of device-side hunter_hard opponents.

The JAX bot is a training opponent only, not a substitute for the JS evaluator.
Only P1 actions change; the existing P0 rollout and PPO remain intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_frozen as training
import jax
import jax.numpy as jnp
from jax_bomb import bun_jax_bots
from qqt_rl.training.io import atomic_write_json, sha256_file


def make_opponent_actions(original, fraction):
    if fraction not in (0.0, 0.5):
        raise ValueError('hunter fraction must be 0 or 0.5')
    if fraction == 0:
        return original

    def actions(states, obs, gv, masks, arch, key, kinds,
                weak_params, old_params, recent_params):
        sampled = original(states, obs, gv, masks, arch, key, kinds,
                           weak_params, old_params, recent_params)
        n = states.pos.shape[0]
        count = int(n * fraction)
        if count == 0:
            return sampled
        sub = jax.tree.map(lambda x: x[:count], states)
        # Both perspective tensors contain all P0 rows, then all P1 rows.
        mm, bm = masks[0][n:n + count], masks[1][n:n + count]
        players = jnp.ones((count,), jnp.int32)
        tiers = jnp.full((count,), bun_jax_bots.TIER_NAMES.index('hunter_hard'), jnp.int32)
        bot = bun_jax_bots.rule_bot_actions(
            sub, players, mm, bm, jax.random.fold_in(key, 701), tiers)
        return sampled.at[:count].set(bot)
    return actions


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hunter-fraction', type=float, choices=[0.0, 0.5], required=True)
    args, rest = parser.parse_known_args()
    save = Path(rest[rest.index('--save') + 1]).resolve()
    meta = {'hunter_fraction': args.hunter_fraction,
            'training_opponent': 'jax_bomb.bun_jax_bots:hunter_hard',
            'formal_evaluation_opponent': 'web/bun_hunter_bot.js',
            'opponent_seat': 1, 'learner_seat': 0,
            'assignment': 'first fraction of environments, fixed throughout run',
            'remaining_opponent': 'current stochastic self-play',
            'source_sha256': sha256_file(__file__),
            'bot_sha256': sha256_file(ROOT / 'jax_bomb/bun_jax_bots.py')}
    atomic_write_json(save.parent.parent / 'opponents_config.json', meta)
    jt = training.jt
    jt.bun_opponent_actions = make_opponent_actions(jt.bun_opponent_actions, args.hunter_fraction)
    sys.argv = [sys.argv[0], '--mode', 'current', *rest]
    print(f'opponents_config={meta}', flush=True)
    return training.main()


if __name__ == '__main__':
    sys.exit(main())
