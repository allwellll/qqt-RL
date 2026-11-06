"""Versioned worker adapter; original live G worker/source remains unchanged."""
from pathlib import Path


def main():
    source=Path('scripts/combat_specialist_g_league_train.py').read_text()
    source=source.replace("BASE = Path('runs/combat_specialist_league_20261101')",
        "BASE = Path('runs/multi_region_combat_20261104/training')")
    source=source.replace('combat_specialist_g_mixed as mixed','multi_region_g_mixed_v1 as mixed')
    source=source.replace("    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])",
        "    aux = mixed.migrate_aux(aux)\n    env_steps = int(payload['environment_steps']); opt_steps = int(payload['optimizer_steps'])")
    scope={'__name__':'multi_region_g_private_worker','__file__':__file__}
    exec(compile(source,__file__,'exec'),scope)
    scope['main']()


if __name__=='__main__':
    main()
