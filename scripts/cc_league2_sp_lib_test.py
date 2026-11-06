"""Unit tests for cc_league2_sp_lib (CPU, no JAX). usage: python -m scripts.cc_league2_sp_lib_test
Writes runs/cc_train_20261105/training/league/SP_LIB_TESTS.json."""
import json, os, shutil, tempfile, time
import numpy as np
from pathlib import Path
from scripts import cc_league2_sp_lib as L
from scripts.population_run_state import atomic_json

OUT = {}
def check(name, cond, detail):
    OUT[name] = dict(passed=bool(cond), detail=str(detail)); assert cond, (name, detail)

item = lambda u: dict(update=u, policy_sha256=f'sha{u}', policy=f'p{u}', checkpoint=f'c{u}')
pool = [item(u) for u in range(18262, 19900, 40)]          # 41 entries
ring = L.ring_from_pool(pool)
check('ring_max_8', len(ring) <= 8, len(ring))
check('ring_has_last_4', [x['update'] for x in ring[-4:]] == [x['update'] for x in pool[-4:]], [x['update'] for x in ring])
hist = [x['update'] for x in ring[:-4]]
check('ring_history_one_per_200', len({u // 200 for u in hist}) == len(hist) and len(hist) == 4, hist)
check('ring_small_pool', [x['update'] for x in L.ring_from_pool(pool[:3])] == [x['update'] for x in pool[:3]], 'pool<4 -> all')
rates = {'sha19782': [90., 100.], 'sha19822': [50., 100.], 'sha19862': [10., 100.]}
r3 = [item(19782), item(19822), item(19862), item(19742)]
w, p = L.pfsp_weights(r3, rates)
check('pfsp_weights_formula', np.allclose(w * 0 + np.maximum((1 - p) ** 2, .05) / np.maximum((1 - p) ** 2, .05).sum(), w), (w, p))
check('pfsp_harder_drawn_more', w[2] > w[1] > w[0], w)
check('pfsp_floor', np.isclose(w[0] / w[3], .05 / .25), 'p=.9 -> .05 floor; unseen -> p .5 -> .25')
check('pfsp_unseen_p_half', p[3] == .5, p)
check('gate_empty_admits', L.gate([], item(1), {})[0], 'first')
check('gate_few_games_admits', L.gate([item(1)], item(2), {'sha1': [0., 10.]})[0], 'games<40')
check('gate_blocks_below_half', not L.gate([item(1)], item(2), {'sha1': [19., 40.]})[0], '19/40')
check('gate_admits_at_half', L.gate([item(1)], item(2), {'sha1': [20., 40.]})[0], '20/40')
row = lambda gen, ring, w, l, d: dict(league_generation=gen, league_manifests={str(gen): dict(G=dict(ring=ring))},
    exposure=dict(bun=dict(opponent_win=[0, 0, 0, 0, w, 0, 0, 0, 0], opponent_loss=[0, 0, 0, 0, l, 0, 0, 0, 0], opponent_draw=[0, 0, 0, 0, d, 0, 0, 0, 0])))
rr = L.record_rates([row(0, [item(5)], 3, 1, 1), row(1, [item(5)], 2, 2, 0), row(0, [item(5), item(6)], 9, 9, 9)])
check('record_rates_single_ring_only', rr == {'sha5': [5., 9.]}, rr)
# Exercise the real hash-verified manifest loader with isolated serialized policies.
# Production checkpoints are external inputs, not unit-test fixtures.
from scripts.cc_league2_lib import entry
import pickle
from scripts.population_run_state import sha
scratch = Path('runs/cc_train_20261105/training/league')
scratch.mkdir(parents=True, exist_ok=True)
tmp = Path(os.path.relpath(tempfile.mkdtemp(dir=scratch))); (tmp / 'league_snapshots').mkdir()
src = []
for u in (19302, 19262, 19222, 19182):
    cp = tmp / f'u{u:06d}'; cp.mkdir()
    (cp / 'policy.pt').write_bytes(pickle.dumps({'fixture_update': u}))
    (cp / 'state.pkl').write_bytes(pickle.dumps({'fixture_state': u}))
    atomic_json(cp / 'COMMITTED.json', dict(update=u, policy_sha256=sha(cp / 'policy.pt'), state_sha256=sha(cp / 'state.pkl')))
    src.append(entry(str(cp)))

try:
    atomic_json(tmp / 'league_snapshots/G.json', dict(schema=L.SCHEMA, family='G', current=src[0], ring=src, fixed=[src[-1]]))
    os.environ.update(CC_SNAPSHOT_ROOT=str(tmp), CC_LEAGUE2_ARM='L2_S05g', CC_LEAGUE2_SEED='2026679302')
    L.base._snap['BASE'] = tmp  # base snapshot functions read their own BASE global
    m1 = L.load('G', 19342); m2 = L.load('G', 19342)
    check('load_single_ring_entry', len(m1['ring']) == 1 and m1['ring'][0]['update'] in (19182, 19222, 19262, 19302), m1['sp'])
    check('load_deterministic', m1['sp'] == m2['sp'], (m1['sp'], m2['sp']))
    pool0 = json.loads((tmp / 'league_snapshots/G_sp_pool.json').read_text())['pool']
    check('pool_seeded_oldest_first', [x['update'] for x in pool0] == [19182, 19222, 19262, 19302], [x['update'] for x in pool0])
    hist = [json.loads(l) for l in (tmp / 'league_snapshots/G_history.jsonl').read_text().splitlines()]
    check('history_logged', any(h['event'] == 'pfsp' for h in hist), len(hist))
finally:
    shutil.rmtree(tmp)
atomic_json('runs/cc_train_20261105/training/league/SP_LIB_TESTS.json', dict(status='PASS', tests=OUT, unix=time.time()))
print('PASS', len(OUT), 'tests')
