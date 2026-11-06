"""CC_LEAGUE_V2 high self-play (L2_SP) league helpers: cc_league2_lib plus a gated, history-extended, PFSP G ring.

Only G changes (C peers, remap, progression are cc_league2_lib unchanged):
  * entry gate: a new save enters the self-play pool only if the learner's win rate in the Gring slot (kind 4)
    against the last pool entry, over the last 40 updates, is >= GATE (.5); fewer than MIN_GATE_GAMES -> admit.
  * ring: last 4 pool entries + 1 historical entry per 200 updates (oldest kept), at most 8.
  * PFSP: at each 40-update generation one ring entry is drawn with rng(seed ^ index), weight
    max((1 - p)^2, .05), p = learner win rate vs that entry in the Gring slot (no games -> p = .5).
    The returned manifest's ring holds only that entry, so the worker's uniform ring pick selects it.
Every gate decision and PFSP draw is appended to league/<arm>/league_snapshots/G_history.jsonl.
Win attribution: a row's Gring-slot games are credited to its generation's single ring entry (rows whose
generation ring has >1 entry, e.g. inherited from the fork source, are skipped).
"""
import json, os, time
from pathlib import Path
import numpy as np
from scripts import cc_league2_lib as base
from scripts.cc_league2_lib import *  # noqa: F401,F403  (remap_kinds, progression, window_rates, actor, SCHEMA, ...)
from scripts.population_run_state import atomic_json

GATE, MIN_GATE_GAMES, RECENT, HISTORY_EVERY, RING_MAX, FLOOR = .5, 40, 4, 200, 8, .05
GRING = 4


def _root():
    return Path(os.environ['CC_SNAPSHOT_ROOT']) / 'league_snapshots'


def _note(**kw):
    with (_root() / 'G_history.jsonl').open('a') as f:
        f.write(json.dumps(dict(unix=time.time(), **kw)) + '\n')


def record_rates(rows):
    """{policy_sha256: [wins, games]} for the Gring slot, from update rows."""
    out = {}
    for r in rows:
        ring = r.get('league_manifests', {}).get(str(r.get('league_generation')), {}).get('G', {}).get('ring', [])
        if len(ring) != 1:
            continue
        e = r['exposure']['bun']; key = ring[0]['policy_sha256']
        w = e['opponent_win'][GRING]; g = w + e['opponent_loss'][GRING] + e['opponent_draw'][GRING]
        acc = out.setdefault(key, [0., 0.]); acc[0] += w; acc[1] += g
    return out


def _rows(out_dir, last=None):
    f = Path(out_dir) / 'updates.jsonl'
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    return rows[-last:] if last else rows


def ring_from_pool(pool):
    """Last RECENT entries + one per HISTORY_EVERY updates from older ones (oldest first kept), at most RING_MAX."""
    recent = pool[-RECENT:]; older = pool[:-RECENT] if len(pool) > RECENT else []
    history, bucket = [], None
    for item in older:
        b = int(item['update']) // HISTORY_EVERY
        if b != bucket:
            history.append(item); bucket = b
    history = history[-(RING_MAX - len(recent)):] if len(recent) < RING_MAX else []
    return history + recent


def pfsp_weights(ring, rates):
    p = np.array([rates.get(item['policy_sha256'], [0., 0.])[0] / rates[item['policy_sha256']][1]
                  if rates.get(item['policy_sha256'], [0., 0.])[1] > 0 else .5 for item in ring])
    w = np.maximum((1. - p) ** 2, FLOOR)
    return w / w.sum(), p


def gate(pool, candidate, rates):
    if not pool:
        return True, None, 0
    w, g = rates.get(pool[-1]['policy_sha256'], [0., 0.])
    if g < MIN_GATE_GAMES:
        return True, None, int(g)
    return (w / g) >= GATE, w / g, int(g)


def _pool_path():
    return _root() / 'G_sp_pool.json'


def _pool():
    f = _pool_path()
    if f.exists():
        return json.loads(f.read_text())['pool']
    seed = json.loads((_root() / 'G.json').read_text())['ring'][::-1]  # fork seed: source ring, oldest first
    atomic_json(f, dict(schema='cc-league2-sp-pool-v1', pool=seed))
    _note(event='pool_seed', updates=[int(x['update']) for x in seed])
    return seed


def publish(family, checkpoint, fixed_checkpoint=None):
    manifest = base.publish(family, checkpoint, fixed_checkpoint)
    if family != 'G':
        return manifest
    pool = _pool(); item = manifest['current']
    if pool and pool[-1]['policy_sha256'] == item['policy_sha256']:
        return manifest
    rates = record_rates(_rows(base.FORMAL / os.environ['CC_LEAGUE2_ARM'], 40))
    ok, rate, games = gate(pool, item, rates)
    if ok:
        pool = pool + [item]; atomic_json(_pool_path(), dict(schema='cc-league2-sp-pool-v1', pool=pool))
    _note(event='gate', update=int(item['update']), admitted=ok, win_rate_vs_last=rate, games=games,
          last=int(pool[-1]['update']) if pool else None, pool_size=len(pool))
    return manifest


def load(family, index=None):
    if family != 'G':
        return base.load(family, index)
    manifest = base._load('G')
    pool = _pool(); ring = ring_from_pool(pool)
    rates = record_rates(_rows(base.FORMAL / os.environ['CC_LEAGUE2_ARM']))
    w, p = pfsp_weights(ring, rates)
    rng = np.random.default_rng(int(os.environ['CC_LEAGUE2_SEED']) ^ int(index))
    k = int(rng.choice(len(ring), p=w)); pick = ring[k]
    for item in [pick]:
        assert base.sha(item['policy']) == item['policy_sha256']
    _note(event='pfsp', index=int(index), ring=[int(x['update']) for x in ring], weights=[round(float(x), 4) for x in w],
          win_rate=[round(float(x), 4) for x in p], picked=int(pick['update']))
    return dict(manifest, ring=[pick], sp=dict(ring=[int(x['update']) for x in ring], weights=w.round(4).tolist(), picked=int(pick['update'])))
