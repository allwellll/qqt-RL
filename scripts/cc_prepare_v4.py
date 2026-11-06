"""Freeze a CC_ANCHOR_V1 protocol (SIL + KL anchor to G_frozen actor) and seed its league snapshots.
usage: python -m scripts.cc_prepare_v4 ARM SIL_COEF KL_COEF TRAP_FILTER(0/1) [FAMILY]"""
import importlib, json, os, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training'); OLD = Path('runs/combat_regions_independent_20261104/training')
ANCHOR = OLD / 'formal/G_REGIONS_V2_c000/u018262/policy.pt'


def main(arm, sil, kl, trap, family='CC_ANCHOR'):
    path = BASE / f'protocols/{family}_V1.json'
    p = json.loads(path.read_text()) if path.exists() else json.loads((BASE / 'protocols/CC_REGIONS_V1.json').read_text())
    assert arm not in p['arms'] or path.exists() is False
    for name in ['scripts/cc_train_v4.py', 'scripts/cc_sil_updater_v4.py', 'scripts/cc_sil_updater_v2.py', 'scripts/cc_supervise_v3.py', 'scripts/cc_prepare_v4.py']:
        p['source_sha256'][name] = sha(name)
    spec = dict(json.loads((BASE / 'protocols/CC_REGIONS_V1.json').read_text())['arms']['N0_native'], sil_coef=sil, kl_coef=kl, trap_filter=bool(trap),
                anchor_policy=str(ANCHOR), anchor_policy_sha256=sha(ANCHOR))
    arms = dict(p['arms']) if path.exists() else {}
    arms[arm] = spec
    p = dict(p, namespace_id=f'{family}-V1', arms=arms,
             cc=dict(p['cc'], created_unix=time.time(), recipe='CC_REGIONS_V1 + KL(G_frozen actor || pi) on live rows; optional v3 trap-filtered SIL',
                     reason='fixed eval: original G_REGIONS_V2 lineage also degrades after u18262 (u018342 greedy 13W/5 kills, u019022 9W/1 kill); every CC arm drifts toward self-deaths; N0 hit the 2-point win+delivery decline stop rule'))
    atomic_json(path, p)
    root = BASE / 'league' / arm; dst = root / 'league_snapshots'; dst.mkdir(parents=True, exist_ok=True)
    os.environ['CC_SNAPSHOT_ROOT'] = str(root)
    import scripts.cc_snapshots_v1 as S; S = importlib.reload(S)
    ring = [S.entry(str(OLD / f'formal/G_REGIONS_V2_c000/u{u:06d}')) for u in (18262, 18222, 18182, 18142)]
    old_g = json.loads((OLD / 'league_snapshots/G.json').read_text())
    atomic_json(dst / 'G.json', dict(schema=S.SCHEMA, family='G', current=ring[0], ring=ring, fixed=old_g['fixed'], ring_fallback=old_g.get('ring_fallback')))
    atomic_json(dst / 'C.json', json.loads((OLD / 'league_snapshots/C.json').read_text()))
    S.load('G'); S.load('C'); print(path, sha(path), list(arms))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2]), int(a[3]), *(a[4:5]))
