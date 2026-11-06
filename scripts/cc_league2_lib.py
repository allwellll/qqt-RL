"""CC_LEAGUE_V2 helpers: peer opponents, kind remap, difficulty progression (CPU-side, no JAX at import).

Slot layout is unchanged from the 9-slot G league (combat_specialist_g_mixed.NAMES); only the meaning
of the C family and the weights change:
  0 rule  1 hunter  2 hard  3 Gcurrent  4 Gring(own last 4 saves)  5 Gfixed(inherited G fixed history)
  6 Ccurrent = latest COMMITTED save of a live peer arm (rotated by 40-update generation)
  7 Cring    = one of that peer's previous saves (up to 4)
  8 Cfixed   = G_frozen u018262
Peer checkpoints are read only (sha checked against COMMITTED.json). No live peer -> G_frozen, recorded.
"""
import json, os, time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training')
FORMAL = BASE / 'formal'
G_FROZEN = 'runs/combat_regions_independent_20261104/training/formal/G_REGIONS_V2_c000/u018262'
BOTS = (0, 1, 2)
STRONG = (3, 4, 6, 7)
PEER_LATEST = 6

# Snapshot module with the per-arm root (same as cc_snapshots_v1), then C is replaced by peers.
_source = Path('scripts/combat_specialist_snapshots.py').read_text()
_old = "BASE = Path('runs/combat_specialist_league_20261101')"
assert _source.count(_old) == 1
_snap = dict(os=os, __name__='cc_league2_snapshots')
exec(compile(_source.replace(_old, "BASE = Path(os.environ.get('CC_SNAPSHOT_ROOT', 'runs/cc_train_20261105/training/league/_unset'))"),
             'cc_league2_snapshots', 'exec'), _snap)
SCHEMA, entry, publish, actor, _load = _snap['SCHEMA'], _snap['entry'], _snap['publish'], _snap['actor'], _snap['load']


def committed(arm_dir):
    rows = []
    for f in Path(arm_dir).glob('u*/COMMITTED.json'):
        try:
            rows.append((int(json.loads(f.read_text())['update']), f.parent))
        except (ValueError, KeyError, OSError):
            continue
    return sorted(rows)


def live_peers(self_arm, policy):
    """Arms in training: SUPERVISOR.json RUNNING and a COMMITTED save newer than policy['max_age_hours']."""
    now = time.time(); out = []
    for d in sorted(FORMAL.iterdir()):
        if not d.is_dir() or d.name == self_arm or d.name in policy.get('exclude', []):
            continue
        try:
            if json.loads((d / 'SUPERVISOR.json').read_text()).get('status') != 'RUNNING':
                continue
        except (OSError, ValueError):
            continue
        rows = committed(d)
        if not rows:
            continue
        u, cp = rows[-1]
        if now - (cp / 'COMMITTED.json').stat().st_mtime > float(policy['max_age_hours']) * 3600:
            continue
        out.append((d.name, rows))
    return out


def peer_manifest(self_arm, policy, index):
    peers = live_peers(self_arm, policy)
    fixed = entry(G_FROZEN)
    if not peers:
        current = fixed; ring = [fixed]; arm = None; fallback = 'no live peer: G_frozen'
    else:
        arm, rows = peers[(index // 40) % len(peers)]
        current = entry(str(rows[-1][1]))
        ring = [entry(str(cp)) for _, cp in rows[-5:-1][::-1]] or [current]
        fallback = None
    return dict(schema=SCHEMA, family='C', current=current, ring=ring, fixed=[fixed],
                peer_arm=arm, live_peers=[name for name, _ in peers], fallback=fallback, index=index,
                ring_fallback='peer has fewer than five saves; ring = available previous saves or latest')


def load(family, index=None):
    """Worker snapshots.load replacement. G: own per-arm manifest. C: peer manifest built now and recorded."""
    if family == 'G':
        return _load('G')
    arm = os.environ['CC_LEAGUE2_ARM']; policy = json.loads(os.environ['CC_LEAGUE2_PEERS'])
    manifest = peer_manifest(arm, policy, int(index))
    root = Path(os.environ['CC_SNAPSHOT_ROOT']) / 'league_snapshots'
    atomic_json(root / 'C.json', manifest)
    with (root / 'C_history.jsonl').open('a') as f:
        f.write(json.dumps(dict(unix=time.time(), index=manifest['index'], peer_arm=manifest['peer_arm'],
            current=manifest['current']['checkpoint'], ring=[r['checkpoint'] for r in manifest['ring']],
            live_peers=manifest['live_peers'], fallback=manifest['fallback'])) + '\n')
    value = _load('C'); assert value['current']['policy_sha256'] == manifest['current']['policy_sha256']
    return manifest


def remap_kinds(aux, count, seed, weights, allocation):
    """Align every physical env's stored opponent kind with the static allocation for the new weights.
    In-flight episodes keep their state; their opponent switches to the new slot from the next tick."""
    import jax.numpy as j
    kinds = np.asarray(allocation(count, seed, weights))
    old = np.asarray(aux['combat_mix']['kind'])
    mix = dict(aux['combat_mix'], kind=j.asarray(kinds.astype(old.dtype)))
    share = (np.bincount(kinds, minlength=9) / count).round(5).tolist()
    return dict(aux, combat_mix=mix), dict(changed_envs=int((old != kinds).sum()), env_share=share)


def window_rates(out, start, end):
    """Learner win rate per pool over updates (start, end] from updates.jsonl."""
    W = np.zeros(9); L = np.zeros(9); D = np.zeros(9); T = np.zeros(9); n = 0
    with (Path(out) / 'updates.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if not start < int(r['new_update']) <= end:
                continue
            e = r['exposure']['bun']; n += 1
            W += e['opponent_win']; L += e['opponent_loss']; D += e['opponent_draw']; T += e['pool_ticks']
    games = W + L + D
    strong = [i for i in STRONG]
    return dict(rows=n, strong_games=int(games[strong].sum()),
                strong_win_rate=float(W[strong].sum() / max(1., games[strong].sum())),
                pool_win_rate=(W / np.maximum(games, 1)).round(4).tolist(), pool_games=games.astype(int).tolist(),
                tick_share=(T / max(1., T.sum())).round(4).tolist())


def harder(weights, floor=.02):
    w = np.asarray(weights, float).copy()
    bots = w[list(BOTS)].sum()
    if bots <= 0:
        return None
    moved = bots if bots / 2 < floor else bots / 2
    w[list(BOTS)] *= (bots - moved) / bots
    w[PEER_LATEST] += moved
    return [round(float(x), 6) for x in w / w.sum()]


def progression(out, spec, index):
    """Called by the worker after an 80-update atomic save. True -> worker exits for a reweighted protocol."""
    rule = spec['league2']['progression']
    begin = max(index - int(rule['window']), int(spec['league2']['weights_since_update']))
    stats = window_rates(out, begin, index)
    new = harder(spec['opponent_weights'])
    ready = stats['rows'] >= .75 * int(rule['window']) and stats['strong_games'] >= int(rule['min_games'])
    fire = bool(ready and new is not None and stats['strong_win_rate'] > float(rule['threshold']))
    row = dict(unix=time.time(), update=index, window_start=begin, ready=ready, fired=fire,
               weights=spec['opponent_weights'], new_weights=new if fire else None, **stats)
    with (Path(out) / 'PROGRESSION.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    if fire:
        atomic_json(Path(out) / 'PROGRESSION_REQUEST.json', row)
    return fire
