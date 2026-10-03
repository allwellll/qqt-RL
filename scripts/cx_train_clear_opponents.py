#!/usr/bin/env python3
"""Compare opponent mixtures on common evaluator-aligned cleared terrain."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cx_train_opponents as opponents
from scripts.cx_train_terrain import make_fresh, bun_env
from qqt_rl.training.io import atomic_write_json, sha256_file


def main():
    save = Path(sys.argv[sys.argv.index('--save') + 1]).resolve()
    meta = {'terrain': 'clear', 'changes_at_reset_only': True,
            'cleared_fields': ['core.brick', 'core.brick_linger'],
            'spawn_sampling_unchanged': True, 'reward_impulse_on_clear': False,
            'source_sha256': sha256_file(__file__),
            'terrain_hook_sha256': sha256_file(ROOT / 'scripts/cx_train_terrain.py')}
    atomic_write_json(save.parent.parent / 'terrain_config.json', meta)
    bun_env._fresh = make_fresh(bun_env._fresh, 'clear')
    print(f'terrain_config={meta}', flush=True)
    return opponents.main()


if __name__ == '__main__':
    sys.exit(main())
