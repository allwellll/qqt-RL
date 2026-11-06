"""Freeze a CC_LEAGUE_V2 protocol for one arm and seed its league snapshots (CPU only, once per arm).

usage: python -m scripts.cc_league2_prepare NEW_ARM SOURCE_ARM SIL_COEF [SOURCE_UPDATE]
  SOURCE_ARM: CC arm whose COMMITTED save is the start point (latest unless SOURCE_UPDATE is given);
              its recipe (SIL updater/filter) is inherited, SIL_COEF overrides the coefficient.
  writes runs/cc_train_20261105/training/protocols/CC_LEAGUE2_<NEW_ARM>_V1.json  (family CC_LEAGUE2_<NEW_ARM>)
         runs/cc_train_20261105/training/league/<NEW_ARM>/league_snapshots/{G,C}.json
"""
import importlib, json, os, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training')
SCRIPTS = ['scripts/cc_league2_train.py', 'scripts/cc_league2_lib.py', 'scripts/cc_league2_supervise.py',
           'scripts/cc_league2_prepare.py', 'scripts/cc_sil_updater_v1.py', 'scripts/cc_sil_updater_v3.py',
           'scripts/model_cuda_overlay.py', 'scripts/combat_regions_mixed_v2.py']
# rule hunter hard | Gcurrent Gring Gfixed | peer-latest peer-ring G_frozen
WEIGHTS = [0., .05, .10, .25, .20, .05, .20, .10, .05]
LEAGUE2 = dict(
    version='CC_LEAGUE_V2',
    slots=['rule', 'hunter', 'hard', 'Gcurrent', 'Gring(own last 4 saves)', 'Gfixed(inherited G fixed history)',
           'Ccurrent=peer latest COMMITTED', 'Cring=peer previous saves', 'Cfixed=G_frozen u018262'],
    peers=dict(max_age_hours=2, exclude=[], rotation='live peers sorted by name, one per 40-update generation',
               fallback='G_frozen u018262 in all C slots, recorded in C_history.jsonl'),
    progression=dict(window=80, threshold=.55, min_games=400, pools=['Gcurrent', 'Gring', 'Ccurrent', 'Cring'],
                     action='bot weights (rule+hunter+hard) halved, freed mass to peer-latest; below .02 total -> 0'))


def main(arm, source_arm, coef, source_update=None):
    family = f'CC_LEAGUE2_{arm}'; path = BASE / f'protocols/{family}_V1.json'
    assert not path.exists(), path
    src_out = BASE / 'formal' / source_arm
    sup = json.loads((src_out / 'SUPERVISOR.json').read_text())
    if 'protocol' not in sup:
        sup = json.loads(next(l for l in reversed((src_out / 'supervisor.jsonl').read_text().splitlines()) if '"protocol"' in l))
    src_protocol = json.loads(Path(sup['protocol']).read_text())
    rows = sorted(src_out.glob('u*/COMMITTED.json'))
    cp = src_out / f'u{int(source_update):06d}' if source_update else rows[-1].parent
    cm = json.loads((cp / 'COMMITTED.json').read_text())
    assert sha(cp / 'state.pkl') == cm['state_sha256'] and sha(cp / 'policy.pt') == cm['policy_sha256']
    src_spec = src_protocol['arms'][source_arm]
    assert 'kl_coef' not in src_spec, 'KL-anchored source not supported by cc_league2_train'
    sil = 'cc_sil_updater_v3' if src_spec.get('sil_filter') else 'cc_sil_updater_v1'
    for name in SCRIPTS:
        src_protocol['source_sha256'][name] = sha(name)
    spec = dict(src_spec, sil_coef=float(coef), opponent_weights=WEIGHTS,
                league_opponent_mix=dict(rule_bot=.15, G_snapshots=dict(current=.25, ring=.20, fixed_history=.05),
                    C_snapshots=dict(peer_latest=.20, peer_ring=.10, G_frozen=.05), boundary='every completed 40-update checkpoint'),
                league2=dict(LEAGUE2, sil_updater=sil, weights_since_update=int(cm['update']), weight_history=[
                    dict(update=int(cm['update']), weights=WEIGHTS, previous=src_spec['opponent_weights'], reason='CC_LEAGUE_V2 start')],
                    source=dict(arm=source_arm, protocol=sup['protocol'], protocol_sha256=sha(sup['protocol']))))
    start = int(cm['update'])
    protocol = dict(src_protocol, namespace=str(BASE / 'formal'), namespace_id=f'{family}-V1', arms={arm: spec},
        chunk_start_update=start, chunk_updates=1280, league2_chunk_updates=1280,
        continuation_source=dict(cm, metrics={'source': f'{source_arm} latest COMMITTED at CC_LEAGUE_V2 switch'}),
        compatible_checkpoint_protocol_sha256=[],
        cc=dict(src_protocol['cc'], owner='cc-league-v2', created_unix=time.time(),
            recipe=f'{source_arm} recipe ({sil}, sil_coef {coef}) + CC_LEAGUE_V2 opponents (peer arms, fewer rule bots, progressive)',
            reason='CC_REGIONS_V1 league: 40% rule bots (learner wins >99%), C family frozen at old u4952 (~92%); '
                   'only 25% opponents improved with training'))
    atomic_json(path, protocol)
    root = BASE / 'league' / arm; dst = root / 'league_snapshots'; dst.mkdir(parents=True, exist_ok=True)
    os.environ['CC_SNAPSHOT_ROOT'] = str(root); os.environ['CC_LEAGUE2_ARM'] = arm
    os.environ['CC_LEAGUE2_PEERS'] = json.dumps(LEAGUE2['peers'])
    import scripts.cc_league2_lib as L; L = importlib.reload(L)
    old_g = json.loads((BASE / 'league' / source_arm / 'league_snapshots/G.json').read_text())
    ring = [L.entry(str(c.parent)) for c in rows if int(c.parent.name[1:]) <= start][-4:][::-1]
    atomic_json(dst / 'G.json', dict(schema=L.SCHEMA, family='G', current=L.entry(str(cp)), ring=ring, fixed=old_g['fixed'],
                                     ring_fallback=old_g.get('ring_fallback')))
    L.load('G'); c = L.load('C', start)
    print(path, sha(path), 'start', start, 'peer', c['peer_arm'], 'live', c['live_peers'], 'fallback', c['fallback'])


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]), *(a[3:4]))
