"""Freeze CC_TRAP_V1 (survival+untrapped-filtered SIL arm) from CC_REGIONS_V1 and seed its league snapshots."""
import json, os, time, importlib, shutil
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training')


def main(arm='S10_trap', coef=1.):
    path = BASE / 'protocols/CC_TRAP_V1.json'; assert not path.exists()
    p = json.loads((BASE / 'protocols/CC_REGIONS_V1.json').read_text())
    for name in ['scripts/cc_train_v3.py', 'scripts/cc_sil_updater_v3.py', 'scripts/cc_supervise_v3.py', 'scripts/cc_sil_updater_v2.py']:
        p['source_sha256'][name] = sha(name)
    base_arm = dict(p['arms']['N0_native'], sil_coef=coef, sil_filter='learner live and untrapped (move mask not stay-only or bomb legal) for next 40 ticks of the same rollout')
    p = dict(p, namespace_id='CC-TRAP-GPU7-V1', arms={arm: base_arm},
             cc=dict(p['cc'], created_unix=time.time(), recipe='as CC_REGIONS_V1; SIL weight kept only for placements the learner survives untrapped 40 ticks',
                     reason='S10_safe u018342 greedy self-deaths 14 (S10 9, G_frozen 1): native trap kills ~60 ticks after trapping, outside the 40-tick live window; filter now also excludes trapped ticks'))
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
