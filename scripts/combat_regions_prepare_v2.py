"""Prepare a reviewable isolated protocol from a main-owner committed boundary."""
import argparse
import json
import os
import shutil
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

B = Path('runs/combat_regions_independent_20261104')
SOURCE = Path(os.environ.get('QQT_SOURCE_ROOT', '.'))


def sources():
    imported = json.loads((B / 'SOURCE_INPUTS.json').read_text())['sources']
    paths = set(imported) | {'scripts/combat_regions_train_v2.py', 'scripts/combat_regions_mixed_v2.py',
        'scripts/combat_regions_snapshots_v2.py', 'scripts/combat_regions_owner_v2.py'}
    return {p: sha(p) for p in sorted(paths)}


def prepare(checkpoint):
    cp = Path(checkpoint)
    if cp.is_absolute() or '..' in cp.parts:
        raise ValueError('source checkpoint must be input-repository relative')
    source = SOURCE / cp
    receipt = json.loads((source / 'COMMITTED.json').read_text())
    parent = B / 'parent' / f"u{receipt['update']:06d}"
    parent.mkdir(parents=True, exist_ok=True)
    for name, key in (('state.pkl', 'state_sha256'), ('policy.pt', 'policy_sha256'),
                      ('policy_bun_mode0.pt', 'bun_mode_zero_policy_sha256')):
        assert sha(source / name) == receipt[key]
        dest = parent / name
        if not dest.exists():
            temp = dest.with_suffix(dest.suffix + '.tmp')
            shutil.copyfile(source / name, temp)
            temp.replace(dest)
        assert sha(dest) == receipt[key]
    atomic_json(parent / 'SOURCE_COMMITTED.json', receipt)
    local = dict(receipt, checkpoint=str(parent))
    atomic_json(parent / 'COMMITTED.json', local)
    p = json.loads((B / 'inputs/source_protocol.json').read_text())
    old_arm = next(iter(p['arms']))
    run = 'G_REGIONS_V2_c000'
    p.update(namespace=str(B / 'training/formal'), namespace_id='COMBAT-REGIONS-INDEPENDENT-V2',
        arms={run: p['arms'][old_arm]}, source_sha256=sources(), continuation_source=local,
        chunk_start_update=receipt['update'], campaign_start_update=receipt['update'],
        chunk_updates=1280, seed=2026548100, environment_steps_per_run=1280 * 2048 * 128,
        curriculum=dict(natural_recipe=.70, regional=.30, clean_natural=.525, genuine_bank=.175,
            train_manifest=str(B / 'train_starts.json'), train_manifest_sha256=sha(B / 'train_starts.json'),
            source_episodes='all source states/ledgers/RNG retained; course applied at actual resets only',
            no_cage_starts=True, native_regional_map=True, center_coverage='unavailable',
            schema='move5/ability3, unchanged G mode token; passive a3sr keys only',
            reward_changes=False, added_telemetry=['mr_distance', 'mr_heading', 'ca_last_tick', 'ca_eligible_age']),
        evaluation=dict(fixed_updates=[80, 320, 640, 960, 1280], full_holdout=[640, 1280],
            modes=['greedy', 'categorical'], seats=[0, 1], weakest_region_required=True,
            preserved_risk_lines=[3, 5], light_safety50='preserved on GPU5 independent risk line; incompatible G ability3 schema is not silently migrated'))
    path = B / 'training/protocols' / (run + '.json')
    atomic_json(path, p)
    atomic_json(B / 'training/CURRENT.json', dict(protocol=str(path), run=run))
    atomic_json(B / 'PREPARED.json', dict(status='PREPARED_NOT_LAUNCHED', source_checkpoint=str(cp),
        parent=str(parent), parent_update=receipt['update'], source_state_sha256=receipt['state_sha256'],
        protocol=str(path), protocol_sha256=sha(path), target_end_update=receipt['update'] + 1280,
        updates=1280, all_source_environment_and_optimizer_steps_retained=True))
    print((B / 'PREPARED.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    prepare(args.checkpoint)
