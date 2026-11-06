"""Freeze the CC GPU4-7 protocol and seed its league snapshots (CPU only, run once)."""
import json, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

ROOT = Path('runs/cc_train_20261105'); BASE = ROOT / 'training'
OLD = Path('runs/combat_regions_independent_20261104/training')
SOURCE = OLD / 'formal/G_REGIONS_V2_c000/u018262'
ARMS = {'N0_native': 0., 'S05_sil': .5, 'S10_sil': 1., 'S20_sil': 2.}
UPDATES = 1280


def main():
    path = BASE / 'protocols/CC_REGIONS_V1.json'
    assert not path.exists(), 'protocol already frozen'
    old = json.loads((OLD / 'protocols/G_REGIONS_V2_c000.json').read_text())
    base_arm = old['arms']['G_REGIONS_V2_c000']
    cm = json.loads((SOURCE / 'COMMITTED.json').read_text()); assert sha(SOURCE / 'state.pkl') == cm['state_sha256']
    for name in ['scripts/cc_train_v1.py', 'scripts/cc_sil_updater_v1.py', 'scripts/cc_snapshots_v1.py', 'scripts/model_cuda_overlay.py']:
        old['source_sha256'][name] = sha(name)
    start = int(cm['update'])
    protocol = dict(old, namespace=str(BASE / 'formal'), namespace_id='CC-REGIONS-GPU4567-V1',
        arms={arm: dict(base_arm, sil_coef=coef) for arm, coef in ARMS.items()},
        chunk_start_update=start, campaign_start_update=start, chunk_updates=UPDATES,
        continuation_source=dict(cm, metrics={'source': 'G_frozen fixed-eval best of 8 (runs/cc_train_20261105/eval_v1)'}),
        compatible_checkpoint_protocol_sha256=[],
        cc=dict(owner='Claude Code GPU4-7', created_unix=time.time(),
            reason='G_REGIONS_V2 u018262 is best on fixed 88-game eval; all corner/resume-protocol descendants regress within 8 updates',
            recipe='inherited G_REGIONS_V2 league PPO unchanged; arms differ only in placement self-imitation coefficient',
            cuda='model_cuda_overlay (7 cond guards) + --xla_disable_hlo_passes=multi_output_fusion',
            evaluation='runs/cc_train_20261105 fixed eval v1 (88 seat-0 games) at every 80-update save; trainer save every 40',
            successor='after COMPLETE: same arm continues with a new chunk; failure: one resume retry from last COMMITTED, then mark FAILED'))
    atomic_json(path, protocol)
    import os
    old_g = json.loads((OLD / 'league_snapshots/G.json').read_text())
    for arm in ARMS:
        root = BASE / 'league' / arm; snaps = root / 'league_snapshots'; snaps.mkdir(parents=True, exist_ok=True)
        os.environ['CC_SNAPSHOT_ROOT'] = str(root)
        import importlib, scripts.cc_snapshots_v1 as S; S = importlib.reload(S)
        # G: current = source, ring = source lineage points up to 18262, fixed = original fixed history.
        ring = [S.entry(str(OLD / f'formal/G_REGIONS_V2_c000/u{u:06d}')) for u in (18262, 18222, 18182, 18142)]
        atomic_json(snaps / 'G.json', dict(schema=S.SCHEMA, family='G', current=ring[0], ring=ring, fixed=old_g['fixed'],
                                           ring_fallback=old_g.get('ring_fallback')))
        atomic_json(snaps / 'C.json', json.loads((OLD / 'league_snapshots/C.json').read_text()))
        for f in ('G', 'C'):
            S.load(f)
    print(path, sha(path))


if __name__ == '__main__':
    main()
