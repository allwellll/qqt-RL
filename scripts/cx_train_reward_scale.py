#!/usr/bin/env python3
"""Scale positive tactical-bomb shaping on cleared terrain with a fixed hunter opponent."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_clear_opponents as clear_opponents
from scripts import cx_train_frozen as training
from jax_bomb import bun_env
from qqt_rl.training.io import atomic_write_json, sha256_file


def make_reward(original, scale):
    if scale not in (1.0, 0.6, 0.3, 0.05):
        raise ValueError('shaping scale must be 1, 0.6, 0.3, or 0.05')
    if scale == 1.0:
        return original

    def reward(*args, **kwargs):
        full = original(*args, **kwargs)
        info = kwargs['rule_info']
        c = training.REWARDS
        shaping = bun_env.tactical_bomb_shaping_from_events(
            info, c['tactical_bomb_placement_reward'],
            c['tactical_bomb_resolution_reward'], c['base_bomb_reward'],
            c['forced_kill_reward'], c['enemy_threat_reward'])
        return full - (1.0 - scale) * shaping

    return reward


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reward-shaping-scale', type=float, choices=[1.0, 0.6, 0.3, 0.05], required=True)
    args, rest = parser.parse_known_args()
    save = Path(rest[rest.index('--save') + 1]).resolve()
    meta = {
        'reward_shaping_scale': args.reward_shaping_scale,
        'scaled_terms': ['base_bomb', 'placement', 'resolution', 'forced', 'threat'],
        'unchanged_terms': ['surviving_kill', 'death', 'self_kill', 'avoidable_death', 'escape', 'hit', 'mutual'],
        'terrain': 'clear', 'source_sha256': sha256_file(__file__),
        'reward_source_sha256': sha256_file(ROOT / 'jax_bomb/bun_env.py'),
    }
    atomic_write_json(save.parent.parent / 'reward_scale_config.json', meta)
    training.jt.reward_from_events = make_reward(training.jt.reward_from_events,
                                                  args.reward_shaping_scale)
    sys.argv = [sys.argv[0], *rest]
    print(f'reward_scale_config={meta}', flush=True)
    return clear_opponents.main()


if __name__ == '__main__':
    sys.exit(main())
