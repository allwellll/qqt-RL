"""Fixed paired evaluation: 44 seat-0 scenarios x {greedy, categorical} vs frozen parent18102.

usage: model_evaluation.py <config> with args [label, checkpoint_or_model_json, out_root]
Games are frozen in PROTOCOL.json on first use; every later model reuses the same seeds.
Adds movement oscillation (adjacent opposite moves) from the verified review.json actions.
"""
import concurrent.futures, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.model_launch import manifest, command

PROTOCOL = Path('runs/cc_train_20261105/eval_protocol_v1/PROTOCOL.json')
OPPONENT = 'runs/corner_counterplay_20261104/exports/parent18102/model.json'
SHARD = 4
CPU_ENV = {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu', 'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


def protocol():
    if PROTOCOL.exists():
        return json.loads(PROTOCOL.read_text())
    prior = json.loads(Path('runs/corner_counterplay_20261104/counter_train_v1_20261105/evaluation/PROTOCOL.json').read_text())['pair_games']
    # every seat-0 scenario of the prior pair protocol; new seeds make repeated spawns distinct games
    scenarios = [g for g in prior if g['seat'] == 0 and g['mode'] == 'greedy']
    games = []
    for i, g in enumerate(scenarios):
        seed = 2026611100 + i
        for mode in ('greedy', 'categorical'):
            games.append(dict(g, seed=seed, action_seed=seed, seat=0, mode=mode, opponent_mode='greedy',
                              evaluation_split='cc_fixed_eval_v1'))
    value = dict(status='FROZEN_BEFORE_ANY_GAME', opponent=dict(file=OPPONENT, sha256=sha(OPPONENT)),
                 scenarios=len(scenarios), games=games, seat='learner seat 0 (P1) only', unix=time.time())
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True); atomic_json(PROTOCOL, value)
    return value


def oscillation(review, seat):
    acts = review['actions']; col = 2 * seat
    moves = [a[col] for a in acts]; abil = [a[col + 1] for a in acts]
    opp = sum(1 for x, y in zip(moves, moves[1:]) if OPPOSITE.get(x) == y)
    return dict(ticks=len(moves), opposite_adjacent=opp, stay=moves.count(4), bomb_decisions=abil.count(1))


def run_shard(label, model_json, out, index, games):
    folder = out / f'shard_{index:02d}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE.json').exists():
        return folder
    cfg = folder / 'CONFIG.json'
    atomic_json(cfg, dict(candidate=label, target='parent18102', model_json=str(model_json), opponent_model_json=OPPONENT,
                          games=games, output=str(folder)))
    pool = list(range(0, 48)) + list(range(96, 144))  # NUMA 0/1; GPU4-7 training is pinned to NUMA 2/3
    cpu = pool[(index * 7 + sum(map(ord, label))) % len(pool)]
    with (folder / 'worker.log').open('w') as f:
        r = subprocess.run(['taskset', '-c', str(cpu)] + command('evaluation', manifest(dict(script='scripts/resume_gpu67_eval_v1.js', args=[str(cfg)], env=CPU_ENV)), node=True),
                           stdout=f, stderr=subprocess.STDOUT, timeout=3600, env=dict(os.environ, **CPU_ENV))
    assert r.returncode == 0, f'{folder} exit {r.returncode}'
    return folder


def summarize(rows):
    from scripts.corner_counterplay_stage_report_v1 import measures
    out = {}
    for mode in ('greedy', 'categorical'):
        rs = [r for r in rows if r['mode'] == mode]
        ms = [measures(r) for r in rs]
        t = {k: sum(m[k] for m in ms if m[k] is not None) for k in ms[0] if k != 'eligible_gap_median'}
        t['losses'] = sum(int(r['winner'] == 1 - r['seat']) for r in rs)
        t['draws'] = len(rs) - t['wins'] - t['losses']
        for k in ('ticks', 'opposite_adjacent', 'stay', 'bomb_decisions'):
            t[k] = sum(r['oscillation'][k] for r in rs)
        t['opposite_rate'] = t['opposite_adjacent'] / max(t['ticks'], 1)
        t['placed_per_1k_ticks'] = 1000 * t['placed'] / max(t['ticks'], 1)
        t['zero_placement_games'] = sum(1 for m in ms if m['placed'] == 0)
        strata = defaultdict(lambda: defaultdict(int))
        for r, m in zip(rs, ms):
            s = strata[r.get('brick_stage', 'native_complete')]
            for k in ('games', 'wins', 'deaths', 'self_deaths', 'causal_kills', 'placed', 'deliveries', 'item_pickups'):
                s[k] += m[k]
            s['opposite_adjacent'] += r['oscillation']['opposite_adjacent']; s['ticks'] += r['oscillation']['ticks']
        t['by_brick_stage'] = {k: dict(v) for k, v in strata.items()}
        out[mode] = t
    return out


def evaluate(label, source, out_root, workers=11):
    p = protocol(); assert sha(p['opponent']['file']) == p['opponent']['sha256']
    out = Path(out_root) / label; out.mkdir(parents=True, exist_ok=True)
    if (out / 'SUMMARY.json').exists():
        return json.loads((out / 'SUMMARY.json').read_text())
    source = Path(source)
    if source.is_dir():
        export = out / 'export'
        if not (export / 'model.json').exists():
            with (out / 'export.log').open('w') as f:
                r = subprocess.run(command('evaluation', manifest(dict(module='scripts.r4_k4_export', args=[str(source), str(export)], env=CPU_ENV))),
                                   stdout=f, stderr=subprocess.STDOUT, timeout=900, env=dict(os.environ, **CPU_ENV))
            assert r.returncode == 0, 'export failed'
        model_json = export / 'model.json'
        state = sha(source / 'state.pkl') if (source / 'state.pkl').exists() else None
    else:
        model_json, state = source, None
    games = p['games']; shards = [games[i:i + SHARD] for i in range(0, len(games), SHARD)]
    begun = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        folders = list(pool.map(lambda a: run_shard(label, model_json, out, *a), enumerate(shards)))
    rows = []
    for folder in folders:
        for r in json.loads((folder / 'COMPLETE.json').read_text())['results']:
            rid = f"match-{r['seed']}-P{r['seat']}-{r['mode']}-{r['opponent_mode']}"
            r['oscillation'] = oscillation(json.loads((folder / rid / 'review.json').read_text()), r['seat'])
            rows.append(r)
    assert len(rows) == len(games)
    from scripts.corner_counterplay_stage_report_v1 import measures
    per_game = [dict(seed=r['seed'], mode=r['mode'], brick_stage=r.get('brick_stage'), region=r['start_region'], winner=r['winner'], seat=r['seat'],
                     **measures(r), **r['oscillation']) for r in rows]
    value = dict(status='COMPLETE_FIXED_EVAL_V1', label=label, source=str(source), state_sha256=state, model_json=str(model_json),
                 model_json_sha256=sha(model_json), protocol_sha256=sha(PROTOCOL), games=len(rows), by_mode=summarize(rows),
                 per_game=per_game, seconds=time.time() - begun, unix=time.time())
    atomic_json(out / 'SUMMARY.json', value)
    return value


def main():
    args = sys.argv[1:]
    # pairs of label source, final arg is out_root
    out_root = args[-1]; pairs = list(zip(args[:-1:2], args[1:-1:2]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        for v in pool.map(lambda a: evaluate(a[0], a[1], out_root, workers=int(os.environ.get('CC_EVAL_WORKERS', max(4, 66 // len(pairs))))), pairs):
            print('EVALUATED', v['label'], json.dumps({m: {k: x for k, x in t.items() if k != 'by_brick_stage'} for m, t in v['by_mode'].items()}), flush=True)


if __name__ == '__main__':
    main()
