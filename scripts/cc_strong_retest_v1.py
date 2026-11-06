"""Same-checkpoint retest on fresh seeds (replaces "two consecutive save points pass").

Protocol = cc_strong_eval_v1 (same 44 scenarios: map, spawns, native respawn, region, brick stage; both seats;
greedy + categorical learner; opponent G_frozen argmax), only the seeds are replaced by a fresh, never-used range:
scenario i gets seeds BASE+2i and BASE+2i+1 (seed == action_seed), so every seat x mode has 88 games and a point has
352.  The protocol, the decision rule and the reference are frozen in retest_v1/PROTOCOL.json before any game.
usage (model_evaluation.py <config>): args = ['freeze'] | ['run', spec, ...] | ['report']
"""
import concurrent.futures, glob, json, math, os, re, subprocess, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.model_launch import manifest, command
from scripts.cc_fixed_eval_v1 import CPU_ENV, oscillation
from scripts import cc_strong_eval_v1 as E

OUT = E.OUT / 'retest_v1'
PROTOCOL = OUT / 'PROTOCOL.json'
BASE = 2026612000
SPECS = ['G_frozen', 'L2_S05g@u019302', 'L2_S05b@u019782', 'L2_S05g@u019622']
RULE = ('PASS iff, on the retest sample only: score(greedy) > .500 + SE(greedy) and score(categorical) > .386 + SE(categorical), '
        'score = (W + D/2)/n over both seats (n = 176 per mode), SE = sd(per-game score)/sqrt(n); and net wins (W-L) and '
        'deliveries, each summed over both seats and both modes, >= those of G_frozen self-play on the same retest seeds. '
        'Reference .500/.386 = G_frozen two-seat self-play of cc_strong_eval_v1 (greedy self-play is .500 by symmetry). '
        'Seed-clustered SE is reported as a supplement only; merged (original + retest) scores and SE are reported, not used for the verdict.')


def used_seeds():
    seeds = set()
    for f in glob.glob('runs/**/PROTOCOL*.json', recursive=True) + glob.glob('runs/**/protocol*.json', recursive=True):
        if Path(f).resolve() == PROTOCOL.resolve():
            continue
        try:
            v = json.loads(Path(f).read_text())
        except Exception:
            continue
        stack = [v]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, y in x.items():
                    if k in ('seed', 'action_seed') and isinstance(y, int):
                        seeds.add(y)
                    stack.append(y)
            elif isinstance(x, list):
                stack += x
    for d in glob.glob('runs/cc_train_20261105/**/match-*', recursive=True):
        m = re.search(r'match-(\d+)-', d)
        if m:
            seeds.add(int(m.group(1)))
    return seeds


def freeze():
    if PROTOCOL.exists():
        p = json.loads(PROTOCOL.read_text())
        assert sha(E.PROTOCOL) == p['source_protocols']['seat0'] and sha(E.PROTOCOL_P1) == p['source_protocols']['seat1']
        return p
    src = {0: json.loads(E.PROTOCOL.read_text())['games'], 1: json.loads(E.PROTOCOL_P1.read_text())['games']}
    scen = {}
    for seat, games in src.items():
        rows = [g for g in games if g['mode'] == 'greedy']
        assert len(rows) == 44
        scen[seat] = rows
    new = {BASE + k for k in range(88)}; clash = new & used_seeds()
    assert not clash, f'seed range already used: {sorted(clash)[:5]}'
    games = {}
    for seat in (0, 1):
        out = []
        for i, g in enumerate(scen[seat]):
            for r in (0, 1):
                seed = BASE + 2 * i + r
                for mode in ('greedy', 'categorical'):
                    out.append(dict(g, seed=seed, action_seed=seed, seat=seat, mode=mode, opponent_mode='greedy',
                                    evaluation_split='cc_strong_retest_v1', scenario_index=i, replica=r))
        games[str(seat)] = out
    value = dict(status='FROZEN_BEFORE_ANY_GAME', created=time.strftime('%Y-%m-%dT%H:%M%z'),
                 source_protocols=dict(seat0=sha(E.PROTOCOL), seat1=sha(E.PROTOCOL_P1)),
                 g_frozen=dict(file=E.G_FROZEN, sha256=sha(E.G_FROZEN), source='G_REGIONS_V2_c000/u018262'),
                 seed_range=[BASE, BASE + 87], seeds_checked_unused=True, points=SPECS, decision_rule=RULE,
                 reference=dict(greedy=.500, categorical=.386), games_per_point=352, games=games)
    atomic_json(PROTOCOL, value)
    return value


def pair_dir(m, seat):
    return OUT / 'pairs' / f'{m["label"]}__vs__G_frozen__P{seat}'


def run_shard(m, g, seat, index, games):
    folder = pair_dir(m, seat) / f'shard_{index:02d}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE.json').exists():
        return folder
    cfg = folder / 'CONFIG.json'
    atomic_json(cfg, dict(candidate=m['label'], target='G_frozen', model_json=m['model_json'], opponent_model_json=g['model_json'],
                          games=games, output=str(folder)))
    with E.Slot() as cpu, (folder / 'worker.log').open('w') as f:
        r = subprocess.run(['taskset', '-c', str(cpu)] + command('evaluation', manifest(dict(script='scripts/resume_gpu67_eval_v1.js', args=[str(cfg)], env=CPU_ENV)), node=True),
                           stdout=f, stderr=subprocess.STDOUT, timeout=3600, env=dict(os.environ, **CPU_ENV))
    assert r.returncode == 0, f'{folder} exit {r.returncode}'
    return folder


def finalize(m, g, seat, games):
    from scripts.corner_counterplay_stage_report_v1 import measures
    d = pair_dir(m, seat); rows = []
    for folder in sorted(d.glob('shard_*')):
        for r in json.loads((folder / 'COMPLETE.json').read_text())['results']:
            rid = f"match-{r['seed']}-P{r['seat']}-{r['mode']}-{r['opponent_mode']}"; review = folder / rid / 'review.json'
            assert r['review_verified'] and sha(review) == r['review_sha256'], review
            rv = json.loads(review.read_text())
            assert rv['meta']['model_json_sha256'] == m['model_json_sha256'] and rv['meta']['opponent_model_json_sha256'] == g['model_json_sha256']
            r['oscillation'] = oscillation(rv, r['seat']); r['opponent'] = E.opponent_deaths(r); rows.append(r)
    assert sorted((r['seed'], r['mode']) for r in rows) == sorted((x['seed'], x['mode']) for x in games), f'{d} incomplete'
    per_game = [dict(seed=r['seed'], seat=r['seat'], mode=r['mode'], winner=r['winner'], **measures(r), **r['oscillation'], **r['opponent']) for r in rows]
    by_mode = E.summarize(rows)
    v = dict(status='COMPLETE_STRONG_RETEST_V1', learner=m, seat=seat, protocol_sha256=sha(PROTOCOL), games=len(rows), by_mode=by_mode, per_game=per_game, unix=time.time())
    atomic_json(d / 'SUMMARY.json', v)
    return v


def run(specs):
    p = freeze(); g = E.ensure_export(E.resolve('G_frozen'))
    models = [E.ensure_export(E.resolve(s)) for s in specs]
    jobs = []
    for m in models:
        for seat in (0, 1):
            games = p['games'][str(seat)]
            if not (pair_dir(m, seat) / 'SUMMARY.json').exists():
                jobs += [(m, g, seat, i, games[k:k + E.SHARD]) for i, k in enumerate(range(0, len(games), E.SHARD))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(E.cpus())) as pool:
        list(pool.map(lambda j: run_shard(*j), jobs))
    for m in models:
        for seat in (0, 1):
            if not (pair_dir(m, seat) / 'SUMMARY.json').exists():
                finalize(m, g, seat, p['games'][str(seat)])
    print('RETEST_RUN_DONE', len(jobs), 'shards', flush=True)


def stats(rows):
    """rows: per-game dicts of one mode (both seats). iid SE and seed-clustered SE of the score."""
    sc = [1. if r['winner'] == r['seat'] else 0. if r['winner'] == 1 - r['seat'] else .5 for r in rows]
    n = len(sc); mean = sum(sc) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in sc) / (n - 1)); se = sd / math.sqrt(n)
    cl = {}
    for r, x in zip(rows, sc):
        cl.setdefault(r['seed'], []).append(x - mean)
    G = len(cl); se_cl = math.sqrt(G / (G - 1) * sum(sum(v) ** 2 for v in cl.values())) / n
    W = sum(1 for r in rows if r['winner'] == r['seat']); L = sum(1 for r in rows if r['winner'] == 1 - r['seat'])
    return dict(n=n, wins=W, losses=L, draws=n - W - L, score=mean, se=se, se_seed_cluster=se_cl, net=W - L,
                deliveries=sum(r['deliveries'] for r in rows), self_deaths=sum(r['self_deaths'] for r in rows),
                causal_kills=sum(r['causal_kills'] for r in rows), deaths=sum(r['deaths'] for r in rows))


def rows_of(paths):
    out = []
    for f in paths:
        v = json.loads(Path(f).read_text()); seat = v.get('seat', 0)
        out += [dict(r, seat=r.get('seat', seat)) for r in v['per_game']]
    return out


def report():
    p = freeze(); res = {}
    for spec in p['points']:
        label = E.resolve(spec)['label']
        new = rows_of([pair_dir(dict(label=label), s) / 'SUMMARY.json' for s in (0, 1)])
        old_files = [E.OUT / 'pairs' / f'{label}__vs__G_frozen' / 'SUMMARY.json', E.OUT / 'pairs' / f'{label}__vs__G_frozen__P1' / 'SUMMARY.json']
        old = rows_of([f for f in old_files if f.exists()]) if all(f.exists() for f in old_files) else []
        res[label] = {md: dict(retest=stats([r for r in new if r['mode'] == md]),
                               original=stats([r for r in old if r['mode'] == md]) if old else None,
                               merged=stats([r for r in old + new if r['mode'] == md]) if old else None) for md in ('greedy', 'categorical')}
    ref = res['G_frozen']; tot = lambda x, k: x['greedy']['retest'][k] + x['categorical']['retest'][k]
    verdict = {}
    for label, x in res.items():
        if label == 'G_frozen':
            continue
        g, c = x['greedy']['retest'], x['categorical']['retest']
        checks = dict(greedy=g['score'] > .500 + g['se'], categorical=c['score'] > .386 + c['se'],
                      net=tot(x, 'net') >= tot(ref, 'net'), deliveries=tot(x, 'deliveries') >= tot(ref, 'deliveries'))
        verdict[label] = dict(PASS=all(checks.values()), checks=checks, net=tot(x, 'net'), deliveries=tot(x, 'deliveries'),
                              thresholds=dict(greedy=.500 + g['se'], categorical=.386 + c['se']))
    value = dict(status='STRONG_RETEST_V1', protocol_sha256=sha(PROTOCOL), rule=p['decision_rule'], results=res, verdict=verdict,
                 g_frozen_selfplay_retest=dict(net=tot(ref, 'net'), deliveries=tot(ref, 'deliveries')), unix=time.time())
    atomic_json(OUT / 'RETEST.json', value)
    L = [f'# 同点新 seed 复测 retest_v1（{time.strftime("%Y-%m-%d %H:%M")}）', '',
         f'协议 `retest_v1/PROTOCOL.json`（开跑前冻结）：cc_strong_eval_v1 的 44 个开局，seed 换成 {p["seed_range"][0]}–{p["seed_range"][1]}（每开局 2 个）；双座，对手 G_frozen argmax；每座每模式 88 局，每点 352 局。', '',
         '| 点 | 模式 | 复测 胜-负-平 | 复测得分 ± SE（seed 聚类 SE） | 门槛 | 原评估得分 ± SE（n） | 合并得分 ± SE（n） | 复测净胜 | 复测交付 | 复测自死 |', '|' + '---|' * 10]
    for label, x in res.items():
        for md in ('greedy', 'categorical'):
            r, o, m = x[md]['retest'], x[md]['original'], x[md]['merged']
            th = '—' if label == 'G_frozen' else f"{verdict[label]['thresholds'][md]:.3f}"
            L.append(f"| {label} | {md} | {r['wins']}-{r['losses']}-{r['draws']} | {r['score']:.3f} ± {r['se']:.3f}（{r['se_seed_cluster']:.3f}） | {th} | "
                     + (f"{o['score']:.3f} ± {o['se']:.3f}（{o['n']}）" if o else '—') + ' | ' + (f"{m['score']:.3f} ± {m['se']:.3f}（{m['n']}）" if m else '—')
                     + f" | {r['net']} | {r['deliveries']} | {r['self_deaths']} |")
    L += ['', '## 判定（只用复测样本）', '', '| 点 | greedy > .500+SE | categorical > .386+SE | 净胜 ≥ G_frozen | 交付 ≥ G_frozen | 结论 |', '|---|---|---|---|---|---|']
    for label, v in verdict.items():
        c = v['checks']; ok = lambda b: '是' if b else '否'
        L.append(f"| {label} | {ok(c['greedy'])} | {ok(c['categorical'])} | {ok(c['net'])}（{v['net']} vs {value['g_frozen_selfplay_retest']['net']}） | "
                 f"{ok(c['deliveries'])}（{v['deliveries']} vs {value['g_frozen_selfplay_retest']['deliveries']}） | **{'通过' if v['PASS'] else '不通过'}** |")
    (OUT / 'RETEST.md').write_text('\n'.join(L) + '\n')
    return value


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    {'freeze': lambda: print(json.dumps({k: v for k, v in freeze().items() if k != 'games'}, ensure_ascii=False)),
     'run': lambda: run(sys.argv[2:] or SPECS), 'report': lambda: print(json.dumps(report()['verdict'], ensure_ascii=False))}[sys.argv[1]]()
