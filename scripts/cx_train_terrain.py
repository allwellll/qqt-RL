#!/usr/bin/env python3
"""Matched self-play PPO with native or evaluator-aligned cleared bricks.

The hook changes fresh/reset states only, not in-episode environment physics.
Spawn sampling still uses the unchanged native map and bucket distribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_frozen as training
from jax_bomb import bun_env
from jax_bomb.bun_frozen_opponents import clear_destructible_bricks
from qqt_rl.training.io import atomic_write_json, sha256_file


def make_fresh(original, terrain):
    if terrain == 'native':
        return original
    if terrain != 'clear':
        raise ValueError('terrain must be native or clear')
    def fresh(key):
        return clear_destructible_bricks(original(key))
    return fresh


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--terrain', choices=['native', 'clear'], required=True)
    args, rest = parser.parse_known_args()
    save = Path(rest[rest.index('--save') + 1]).resolve()
    meta = {'terrain': args.terrain, 'changes_at_reset_only': True,
            'cleared_fields': ['core.brick', 'core.brick_linger'] if args.terrain == 'clear' else [],
            'spawn_sampling_unchanged': True, 'reward_impulse_on_clear': False,
            'source_sha256': sha256_file(__file__),
            'clear_function_source_sha256': sha256_file(ROOT / 'jax_bomb/bun_frozen_opponents.py')}
    atomic_write_json(save.parent.parent / 'terrain_config.json', meta)
    bun_env._fresh = make_fresh(bun_env._fresh, args.terrain)
    sys.argv = [sys.argv[0], '--mode', 'current', *rest]
    print(f'terrain_config={meta}', flush=True)
    return training.main()


if __name__ == '__main__':
    sys.exit(main())
