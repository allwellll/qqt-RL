#!/usr/bin/env python3
"""Projected-value PPO; change only JAX hunter positioning in half of P1 slots."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
from scripts import cx_train_value_projection as projected
from scripts import cx_trap_route_bot as trap
from qqt_rl.training.io import atomic_write_json, sha256_file

opponents = projected.opponents
training = projected.training
HARD_FACTORY = opponents.make_opponent_actions


def make_actions(original, fraction, mode):
    if mode not in ('current', 'trap16') or fraction != .5:
        raise ValueError('trap routes require current/trap16 and half-bot slots')
    if mode == 'current':
        return HARD_FACTORY(original, fraction)

    def actions(states, obs, gv, masks, arch, key, kinds,
                weak_params, old_params, recent_params):
        sampled = original(states, obs, gv, masks, arch, key, kinds,
                           weak_params, old_params, recent_params)
        n = states.pos.shape[0]
        count = n // 2
        if not count:
            return sampled
        sub = jax.tree.map(lambda x: x[:count], states)
        bot = trap.rule_bot_actions(sub, jnp.ones(count, jnp.int32),
            masks[0][n:n + count], masks[1][n:n + count], jax.random.fold_in(key, 701),
            jnp.full(count, opponents.bun_jax_bots.TIER_NAMES.index('hunter_hard'), jnp.int32))
        return sampled.at[:count].set(bot)
    return actions


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trap-route-mode', choices=['current', 'trap16'], required=True)
    args, rest = parser.parse_known_args()
    for flag, expected in [('--hunter-fraction', .5), ('--reward-shaping-scale', .6)]:
        if rest.count(flag) != 1 or float(rest[rest.index(flag) + 1]) != expected:
            raise ValueError(f'{flag} must remain {expected}')
    if any(flag in rest for flag in ('--value-projection', '--surviving-kill-reward', '--export-mode', '--mode')):
        raise ValueError('projected critic, reward 24, EMA .95 and current P1 are fixed')
    save = Path(rest[rest.index('--save') + 1]).resolve()
    rest[rest.index('--save') + 1] = str(save)
    meta = {'mode': args.trap_route_mode, 'bot_fraction': .5, 'trap_candidates': 16, 'trap_radius': 3,
            'selection': 'nearest open cells, Manhattan then row-major',
            'goal_cost': '0 or 1+round(4*after/max(before,1)) using JAX reachable safe cells',
            'prediction': 'static JAX deadline approximation; not JS time-expanded parity',
            'training_opponent': 'scripts.cx_trap_route_bot:trap_bot_action' if args.trap_route_mode == 'trap16'
                 else 'jax_bomb.bun_jax_bots:bot_action',
            'formal_evaluation_bot': 'web/bun_hunter_bot.js unchanged',
            'value_projection': 'projected', 'surviving_kill_reward': 24., 'ema_decay': .95,
            'source_sha256': sha256_file(__file__), 'bot_sha256': sha256_file(ROOT / 'scripts/cx_trap_route_bot.py')}
    atomic_write_json(save.parent.parent / 'trap_routes_config.json', meta)
    old_factory, old_writer, old_opponent_writer, old_argv = (
        opponents.make_opponent_actions, training.atomic_write_json, opponents.atomic_write_json, sys.argv)

    def writer(path, data):
        path = Path(path).resolve()
        if path.name == 'trainer_config.json' or path.parent == save.parent:
            data = dict(data, trap_route_mode=args.trap_route_mode, trap_candidates=16)
        return old_writer(path, data)

    def opponent_writer(path, data):
        if Path(path).name == 'opponents_config.json':
            data = dict(data, trap_route_mode=args.trap_route_mode,
                        training_opponent=meta['training_opponent'], trap_routes_source_sha256=meta['bot_sha256'])
        return old_opponent_writer(path, data)

    opponents.make_opponent_actions = lambda original, fraction: make_actions(original, fraction, args.trap_route_mode)
    opponents.atomic_write_json, training.atomic_write_json = opponent_writer, writer
    sys.argv = [sys.argv[0], '--value-projection', 'projected', *rest]
    print(f'trap_routes_config={meta}', flush=True)
    try:
        return projected.main()
    finally:
        opponents.make_opponent_actions, training.atomic_write_json = old_factory, old_writer
        opponents.atomic_write_json, sys.argv = old_opponent_writer, old_argv


if __name__ == '__main__':
    sys.exit(main())
