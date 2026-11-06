"""Freeze CC_SAFE_V1 (survival-filtered SIL arm) from CC_REGIONS_V1 and seed its league snapshots."""
import json, os, time, importlib, shutil
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training')


def main(arm='S10_safe', coef=1.):
    path = BASE / 'protocols/CC_SAFE_V1.json'; assert not path.exists()
    p = json.loads((BASE / 'protocols/CC_REGIONS_V1.json').read_text())
    for name in ['scripts/cc_train_v2.py', 'scripts/cc_sil_updater_v2.py', 'scripts/cc_supervise_v2.py']:
        p['source_sha256'][name] = sha(name)
    base_arm = dict(p['arms']['N0_native'], sil_coef=coef, sil_filter='learner live for next 40 ticks of the same rollout')
    p = dict(p, namespace_id='CC-SAFE-GPU7-V1', arms={arm: base_arm},
             cc=dict(p['cc'], created_unix=time.time(), recipe='as CC_REGIONS_V1; SIL weight kept only for placements the learner survives 40 ticks',
                     reason='S20_sil (unfiltered, 2.0) failed u018342: categorical 10W/32L 27 self-deaths; S10 self-deaths 9/21 vs G_frozen 1/13'))
    atomic_json(path, p)
    src = BASE / 'league/N0_native/league_snapshots'; dst = BASE / 'league' / arm / 'league_snapshots'; dst.mkdir(parents=True, exist_ok=True)
    # N0_native's seed G.json has been republished by its run; rebuild the original seed instead of copying.
    os.environ['CC_SNAPSHOT_ROOT'] = str(BASE / 'league' / arm)
    import scripts.cc_snapshots_v1 as S; S = importlib.reload(S)
    old = Path('runs/combat_regions_independent_20261104/training')
    ring = [S.entry(str(old / f'formal/G_REGIONS_V2_c000/u{u:06d}')) for u in (18262, 18222, 18182, 18142)]
    old_g = json.loads((old / 'league_snapshots/G.json').read_text())
    atomic_json(dst / 'G.json', dict(schema=S.SCHEMA, family='G', current=ring[0], ring=ring, fixed=old_g['fixed'], ring_fallback=old_g.get('ring_fallback')))
    atomic_json(dst / 'C.json', json.loads((old / 'league_snapshots/C.json').read_text()))
    S.load('G'); S.load('C'); print(path, sha(path))


if __name__ == '__main__':
    main()
