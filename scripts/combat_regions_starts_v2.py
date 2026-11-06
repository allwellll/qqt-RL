"""Frozen reachable starts outside cage and push-box cells, without rotation."""
import json
from pathlib import Path
import numpy as np
from scripts.multi_region_combat_v1 import REGIONS, DISTANCES, legal_pairs
from scripts.bomb_threat_a3sr_spawn import base_mask, validate_cell

B = Path('runs/combat_regions_independent_20261104')


def manifest(level, seed, games_per_stratum=16):
    if games_per_stratum % 4:
        raise ValueError('role and displacement balance requires groups of four')
    wall = np.asarray(level['wall'], bool).reshape(13, 15)
    brick = np.asarray(level['brick'], bool).reshape(13, 15)
    push = np.asarray(level.get('pushable', [False] * 195), bool).reshape(13, 15)
    bases = base_mask(level)
    speed = .3 * float(level['initial_stats']['speed'])
    blast = int(level['initial_stats']['blast'])
    rng = np.random.default_rng(seed)
    rows, missing = [], []
    for region in REGIONS:
        for distance in DISTANCES:
            pairs = legal_pairs(wall | push, brick, region, distance)
            def valid(cell):
                return validate_cell(wall, brick, bases, push, divmod(cell, 15), blast, speed, 30)['valid']
            eligible = {cell: valid(cell) for pair in pairs for cell in pair}
            pairs = [(a, b) for a, b in pairs if eligible[a] and eligible[b]]
            if not pairs:
                missing.append(dict(region=region, distance=distance, reason='no native legal safe pair outside cages/push boxes'))
                continue
            for group in range(games_per_stratum // 4):
                vertical = group % 2 == 0
                choices = [(a, b) for a, b in pairs if (abs(a // 15 - b // 15) > abs(a % 15 - b % 15)) == vertical]
                a, b = (choices or pairs)[int(rng.integers(len(choices or pairs)))]
                for flip, seat in ((False, 0), (False, 1), (True, 0), (True, 1)):
                    pair = (b, a) if flip else (a, b)
                    own, enemy = pair[seat], pair[1 - seat]
                    dy, dx = enemy // 15 - own // 15, enemy % 15 - own % 15
                    heading = ('south' if dy > 0 else 'north') if abs(dy) > abs(dx) else ('east' if dx > 0 else 'west')
                    rows.append(dict(seed=seed + len(rows), region=region, distance=distance,
                        learner_seat=seat, cells=[list(divmod(c, 15)) for c in pair],
                        relative_heading=heading, initiative='simultaneous native step',
                        outside_cages=True, native_map=True, no_preplaced_bombs=True))
    return dict(schema='combat-regions-starts-v2', seed=seed, rows=rows, missing=missing,
                geometry_augmentation=False, reward_changes=False)


def main():
    level = next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x.get('qqt_id') == 806)
    for split, seed in (('train', 2026544100), ('dev', 2026545100), ('holdout', 2026546100)):
        value = manifest(level, seed)
        (B / (split + '_starts.json')).write_text(json.dumps(value, indent=2) + '\n')
        print(split, len(value['rows']), value['missing'])


if __name__ == '__main__':
    main()
