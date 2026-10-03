#!/usr/bin/env python3
"""Export raw or exponential moving average weights from otherwise unchanged PPO."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_reward_scale as reward
import jax
from qqt_rl.training.io import atomic_write_json, sha256_file


class ExportEMA:
    """Observe committed iteration outputs; warmup trajectories never enter EMA."""

    def __init__(self, decay=.95, warmup_calls=2):
        if not 0 < decay < 1 or warmup_calls < 0:
            raise ValueError('invalid EMA decay or warmup count')
        self.decay = decay
        self.warmup_calls = warmup_calls
        self.calls = 0
        self.updates = 0
        self.average = None
        self.blend = jax.jit(lambda old, new: jax.tree.map(
            lambda x, y: decay * x + (1. - decay) * y, old, new))

    def wrap(self, compiled):
        def step(params, opt_state, states, key):
            result = compiled(params, opt_state, states, key)
            if self.calls >= self.warmup_calls:
                if self.average is None:
                    self.average = params
                self.average = self.blend(self.average, result[0])
                self.updates += 1
                jax.block_until_ready(self.average)
            self.calls += 1
            # Raw parameters, optimizer, rollouts and keys are fed back unchanged.
            return result
        return step


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--export-mode', choices=['raw', 'ema95'], required=True)
    args, rest = parser.parse_known_args()
    save = Path(rest[rest.index('--save') + 1]).resolve()
    out = save.parent.parent
    training = reward.training
    observer = ExportEMA() if args.export_mode == 'ema95' else None
    original_builder = training.build_one_iter
    original_save = training.jt.save_params
    original_writer = training.atomic_write_json
    original_argv = sys.argv
    raw_checkpoints = {}
    meta = {'export_mode': args.export_mode, 'decay_per_committed_iteration': .95 if observer else None,
            'initial_average': 'historical it4000', 'warmup_calls_excluded': 2,
            'feedback_to_policy_or_optimizer': False,
            'average_scope': 'all parameter leaves after each complete PPO iteration (32 optimizer steps)',
            'raw_checkpoint_directory': 'raw_ckpt' if observer else 'ckpt',
            'source_sha256': sha256_file(__file__),
            'training_source_sha256': sha256_file(ROOT / 'scripts/cx_train_frozen.py')}
    atomic_write_json(out / 'ema_config.json', meta)

    def builder(*a, **k):
        return observer.wrap(original_builder(*a, **k))

    def save_params(params, path):
        if observer.average is None or observer.updates < 1:
            raise RuntimeError('cannot export EMA before a committed update')
        raw = out / 'raw_ckpt' / Path(path).name
        raw.parent.mkdir(parents=True, exist_ok=True)
        original_save(params, str(raw))
        raw_checkpoints[raw.name] = {'sha256': sha256_file(raw), 'updates_completed': observer.updates}
        atomic_write_json(out / 'raw_checkpoints.json', raw_checkpoints)
        original_save(observer.average, path)

    def writer(path, data):
        path = Path(path)
        if path.name == 'trainer_config.json' or path.parent == save.parent:
            data = dict(data, export_mode=args.export_mode,
                        ema_decay=.95 if observer else None, ema_feedback=False)
            if observer and 'updates_completed' in data:
                if data['updates_completed'] != observer.updates:
                    raise ValueError('EMA update count differs from checkpoint sidecar')
                data['raw_checkpoint_sha256'] = raw_checkpoints[path.with_suffix('.pt').name]['sha256']
        return original_writer(path, data)

    if observer:
        training.build_one_iter = builder
        training.jt.save_params = save_params
    training.atomic_write_json = writer
    sys.argv = [sys.argv[0], *rest]
    print(f'ema_config={meta}', flush=True)
    try:
        return reward.main()
    finally:
        training.build_one_iter = original_builder
        training.jt.save_params = original_save
        training.atomic_write_json = original_writer
        sys.argv = original_argv


if __name__ == '__main__':
    sys.exit(main())
