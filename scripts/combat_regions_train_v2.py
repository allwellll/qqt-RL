"""Independent complete-state G worker; inherited PPO and reward unchanged."""
from pathlib import Path


def main():
    source = Path('scripts/combat_specialist_g_league_train.py').read_text()
    replacements = {
        "BASE = Path('runs/combat_specialist_league_20261101')": "BASE = Path('runs/combat_regions_independent_20261104/training')",
        'combat_specialist_g_mixed as mixed': 'combat_regions_mixed_v2 as mixed',
        'combat_specialist_snapshots as snapshots': 'combat_regions_snapshots_v2 as snapshots',
        "    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])":
            "    aux = mixed.migrate_aux(aux)\n    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])",
        "    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path":
            "    for path, expected in protocol['source_sha256'].items(): assert sha(path) == expected, path\n    course=protocol['curriculum']; assert sha(course['train_manifest']) == course['train_manifest_sha256']",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    scope = {'__name__': 'independent_regions_worker', '__file__': __file__}
    exec(compile(source, __file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
