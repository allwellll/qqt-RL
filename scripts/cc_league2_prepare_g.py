"""Freeze a CC_LEAGUE_V2 protocol for an arm that starts from G_frozen u018262 (CPU only, once per arm).

usage: python -m scripts.cc_league2_prepare_g NEW_ARM SIL_COEF
  recipe = CC_REGIONS_V1 base arm (G_REGIONS_V2 league PPO) + cc_sil_updater_v1(SIL_COEF) + CC_LEAGUE_V2 opponents.
  G manifest = the original G_frozen seed (current/ring u018262..u018142, inherited fixed history), as CC_REGIONS_V1.
  Separate file so cc_league2_prepare.py (sha-pinned by running protocols) stays unchanged.
"""
import importlib, json, os, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.cc_league2_prepare import BASE, SCRIPTS, WEIGHTS, LEAGUE2

OLD = Path('runs/combat_regions_independent_20261104/training')
SOURCE = OLD / 'formal/G_REGIONS_V2_c000/u018262'


def main(arm, coef):
    family = f'CC_LEAGUE2_{arm}'; path = BASE / f'protocols/{family}_V1.json'
    assert not path.exists(), path
    src_path = BASE / 'protocols/CC_REGIONS_V1.json'; src = json.loads(src_path.read_text())
    cm = json.loads((SOURCE / 'COMMITTED.json').read_text())
    assert sha(SOURCE / 'state.pkl') == cm['state_sha256'] == src['continuation_source']['state_sha256']
    for name in SCRIPTS + ['scripts/cc_league2_prepare_g.py']:
        src['source_sha256'][name] = sha(name)
    base = src['arms']['N0_native']
    spec = dict(base, sil_coef=float(coef), opponent_weights=WEIGHTS,
                league_opponent_mix=dict(rule_bot=.15, G_snapshots=dict(current=.25, ring=.20, fixed_history=.05),
                    C_snapshots=dict(peer_latest=.20, peer_ring=.10, G_frozen=.05), boundary='every completed 40-update checkpoint'),
                league2=dict(LEAGUE2, sil_updater='cc_sil_updater_v1', weights_since_update=int(cm['update']), weight_history=[
                    dict(update=int(cm['update']), weights=WEIGHTS, previous=base['opponent_weights'], reason='CC_LEAGUE_V2 start')],
                    source=dict(arm='G_frozen', checkpoint=str(SOURCE), protocol=str(src_path), protocol_sha256=sha(src_path))))
    start = int(cm['update'])
    protocol = dict(src, namespace=str(BASE / 'formal'), namespace_id=f'{family}-V1', arms={arm: spec},
        chunk_start_update=start, campaign_start_update=start, chunk_updates=1280, league2_chunk_updates=1280,
        continuation_source=dict(cm, metrics={'source': 'G_frozen u018262 (user orchestration: strong eval, S10 weakest)'}),
        compatible_checkpoint_protocol_sha256=[],
        cc=dict(src['cc'], owner='cc-league-v2', created_unix=time.time(),
            recipe=f'G_frozen u018262 + cc_sil_updater_v1 (sil_coef {coef}) + CC_LEAGUE_V2 opponents',
            reason='user: start from G_frozen; strong eval S10_sil vs G_frozen greedy 18W/19L, categorical 15W/28L'))
    atomic_json(path, protocol)
    root = BASE / 'league' / arm; dst = root / 'league_snapshots'; dst.mkdir(parents=True, exist_ok=True)
    os.environ['CC_SNAPSHOT_ROOT'] = str(root); os.environ['CC_LEAGUE2_ARM'] = arm
    os.environ['CC_LEAGUE2_PEERS'] = json.dumps(LEAGUE2['peers'])
    import scripts.cc_league2_lib as L; L = importlib.reload(L)
    ring = [L.entry(str(OLD / f'formal/G_REGIONS_V2_c000/u{u:06d}')) for u in (18262, 18222, 18182, 18142)]
    old_g = json.loads((OLD / 'league_snapshots/G.json').read_text())
    atomic_json(dst / 'G.json', dict(schema=L.SCHEMA, family='G', current=ring[0], ring=ring, fixed=old_g['fixed'],
                                     ring_fallback=old_g.get('ring_fallback')))
    L.load('G'); c = L.load('C', start)
    print(path, sha(path), 'start', start, 'peer', c['peer_arm'], 'live', c['live_peers'], 'fallback', c['fallback'])


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))
