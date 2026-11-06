"""Chain / fountain metrics for cc_strong_eval_v1 (offline, CPU only, read-only on replays and checkpoints).

Every review.json of a strong-eval pair is replayed in the frozen production simulator (web/sim-bun-native-v2.js,
hash checks + winner verified) by scripts/cc_strong_metrics_fountain_v1.js; per-game metrics and an event timeline
are written to eval_strong_v1/fountain_metrics_v1/pairs/<pair>/GAMES.jsonl, aggregates to SUMMARY.json.
usage (model_evaluation.py <config>): args = ['baseline'] | ['pairs', <pair dir>...] | ['point', <label>...]
cc_strong_eval_v1.point() calls point_summary() so every new strong-eval point carries these metrics.
"""
import concurrent.futures, json, math, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json
from scripts.model_launch import manifest, command
from scripts.cc_fixed_eval_v1 import CPU_ENV
from scripts import cc_strong_eval_v1 as E
import subprocess, os

VERSION = 'cc-strong-metrics-fountain-v1'
OUT = E.OUT / 'fountain_metrics_v1'
WORKER = 'scripts/cc_strong_metrics_fountain_v1.js'
CHUNK, WORKERS = 11, 12
DEFINITIONS = """# 连锁/喷泉指标口径（cc_strong_metrics_fountain_v1）

计算方式：把强评估已有的每局 review.json 放进冻结的生产模拟器 web/sim-bun-native-v2.js（SHA f6e346f3…）逐 tick 重放，
每个 hash 检查点和胜者都必须一致，否则该局记为错误、不进统计。只在 `Sim._resolveExplosions` 外面包一层只读记录，
不改规则、RNG 和动作。tick 记为该步执行后的 `s.t`（与 review.checks 同一时钟）。每局对两名玩家分别计算；
报表的"本方"指被评估模型（learner 座位），两座合并（seat 0 + seat 1），argmax（greedy）和采样（categorical）分列。

## 批次（与 r3_items_reward.batch_update / r3_items_metrics 相同）
某方放泡时，若距离该方上一次放泡 ≥ 8 tick，批次号 +1，否则沿用当前批次。泡的批次在放下时记下，炸掉后删除。
同一 tick 里一起炸掉的本方泡，以其中最小的批次号作为这次爆炸的批次（sourcebatch，同 r3_items_reward）。

## 1. chain_events（跨批次连锁引爆次数）
在一次爆炸结算里，先找出本方的自然到时泡（引信为 0 的根），然后沿"源泡的十字火覆盖到另一颗引信仍 > 0 的泡"这条边，只经过本方的泡，
求出整个连通的引爆集合（同 r3_items_metrics 的 chains）。如果其中至少有一条边把另一批次的本方泡提前引爆（源批次 ≠ 目标批次），
就计 1 次 chain_event。同一 tick 同一方最多计 1 次；被对手的泡引爆、或只在同批次内部互相引爆的不算。
本 tick 刚放下、又在同一 tick 被引爆的泡没有批次（与 r3_items_reward 一样，batch_update 在这一步之后才执行），不计入批次。
chain_links = 本次被提前引爆的、与源批次不同的批次数（对应 cross_batch_count 的 k），逐次相加。
校验：每局 chain_events 和 chain_len_max 都与评估 worker 写进 RESULT.json 的 r3.chains（batch_count ≥ 2 的记录）比对，结果列为 r3_chain_check。

## 2. chain_len_max（单次最长连锁批次数）
一次连锁里涉及的不同批次数（包括根所在批次），取全局最大值。2 = 旧泡引爆了 1 个新批次。

## 3. chain_near_enemy（连锁的火覆盖对手 3 格内的次数）
chain_event 里，整个引爆集合（根加被引爆的泡）的十字火中，只要有一格与对手所在格（玩家中心所在格）的曼哈顿距离 ≤ 3，
且对手在这次结算前还活着，就计 1 次。另报 chain_hit_enemy_cell：火覆盖了对手所在格本身的次数。

## 4. 有效爆炸批次、fountain_streak、fountain_ticks
一次本方爆炸（本方拥有、并由本方引起的泡；causeMask 含本方）满足下列条件时算有效：
- 本方在结算前活着且没被困，对手在结算前活着；
- 本方火覆盖的格子（去掉同一 tick 对手火也覆盖的格子）中满足任一项：
  a) 对手 3 格内（曼哈顿距离）的"新格"，即这一格只落在本方一颗活泡的十字火上（同 r3_items_reward.novel_explosion_cover：
     不是几颗本方泡重复覆盖的格子），
  b) 一块新炸的砖（结算前 brickLinger = 0，不算残骸重复覆盖），
  c) 对手所在格；
- 批次号与本方上一次有效爆炸不同（同批次分几次炸只计一次，同 r3_items_reward 的 sourcebatch != last_explosion_batch）。
把本方所有有效爆炸的 tick 排成序列，相邻两次的间隔 ≤ 24 tick 就算同一串。
fountain_streak = 最长串里的有效批次数（没有有效爆炸为 0，单独一次为 1）。
fountain_ticks = 长度 ≥ 2 的串从第一次到最后一次有效爆炸所跨的 tick 数之和（end − start + 1）。
另报 effective_batches（有效批次总数）、fountain_streaks_ge2/ge3（长度 ≥ 2/≥ 3 的串数）。
严格版 strict_*：要求 a) 成立，并且同时覆盖砖、宝箱或对手所在格（与 r3_items_reward 当前训练用的
explosion & (novel & near) 的组合一致）。只作参考。
本指标只作测量：死亡或受控后不清零已形成的串，也不扣分，这和奖励不同。

## 5. 放在本方活泡射线上的放泡占比
本方放泡时（引信尚未递减的新泡），检查这一格是否落在本方另一颗活泡（引信 > 0，或本 tick 正要爆炸）的十字火上；
射线的阻挡和模拟器一致：墙挡且不覆盖，砖和其他泡被覆盖后挡住。
on_own_ray_rate = Σon_own_ray / Σplaced；另报 conditional = Σon_own_ray / Σplaced_with_own_live（放泡时已有本方活泡的放泡），
other_batch = 射线来自不同批次的活泡的放泡占比（真正的"连泡在旧泡射线上"；同一批次的连放多数是边走边并排放），
triggered_early = 这些射线上的泡最后确实被提前引爆的占比。只作观测，不进奖励。

## 统计
每局指标报均值 ± 标准误（sd/√n，n = 局数）；占比为合并比值 Σx/Σy，标准误按局做 delta 法。
per_1k_ticks 是按对局时长归一后的合并比率（赢得早的局 tick 少）。对手侧（G_frozen）的同口径数据一并列出作对照。
"""

SOURCE_GROUPS = [
    ('G_frozen self (orig)', ['pairs/G_frozen__vs__G_frozen', 'pairs/G_frozen__vs__G_frozen__P1']),
    ('G_frozen self (retest_v1)', ['retest_v1/pairs/G_frozen__vs__G_frozen__P0', 'retest_v1/pairs/G_frozen__vs__G_frozen__P1']),
    ('L2_S05g u019302 (orig)', ['pairs/L2_S05g_u019302__vs__G_frozen', 'pairs/L2_S05g_u019302__vs__G_frozen__P1']),
    ('L2_S05g u019302 (retest_v1)', ['retest_v1/pairs/L2_S05g_u019302__vs__G_frozen__P0', 'retest_v1/pairs/L2_S05g_u019302__vs__G_frozen__P1']),
    ('L2_ATK u019382', ['pairs/L2_ATK_u019382__vs__G_frozen', 'pairs/L2_ATK_u019382__vs__G_frozen__P1']),
    ('L2_ATK u019462', ['pairs/L2_ATK_u019462__vs__G_frozen', 'pairs/L2_ATK_u019462__vs__G_frozen__P1']),
    ('L2_ATK u019542', ['pairs/L2_ATK_u019542__vs__G_frozen', 'pairs/L2_ATK_u019542__vs__G_frozen__P1']),
]
COMBINED = [('G_frozen self (orig+retest)', ['G_frozen self (orig)', 'G_frozen self (retest_v1)']),
            ('L2_S05g u019302 (orig+retest)', ['L2_S05g u019302 (orig)', 'L2_S05g u019302 (retest_v1)'])]
COUNTS = ('placed', 'placed_with_own_live', 'on_own_ray', 'on_own_ray_other_batch', 'on_ray_triggered_early', 'explosion_ticks',
          'chain_events', 'chain_links', 'chain_len_max', 'chain_near_enemy', 'chain_hit_enemy_cell', 'effective_batches',
          'fountain_streak', 'fountain_ticks', 'fountain_streaks_ge2', 'fountain_streaks_ge3',
          'strict_effective_batches', 'strict_fountain_streak', 'strict_fountain_ticks')
RATIOS = dict(on_own_ray_rate=('on_own_ray', 'placed'), on_own_ray_conditional=('on_own_ray', 'placed_with_own_live'),
              on_own_ray_other_batch_rate=('on_own_ray_other_batch', 'placed'), on_ray_triggered_early=('on_ray_triggered_early', 'on_own_ray'),
              chain_near_share=('chain_near_enemy', 'chain_events'))
PER_1K = ('chain_events', 'chain_near_enemy', 'effective_batches', 'fountain_ticks', 'placed')


def key_of(pair):
    rel = Path(pair).resolve().relative_to(E.OUT.resolve()).as_posix()
    return rel.replace('retest_v1/pairs/', 'retest_v1__').replace('pairs/', '')


def reviews(pair):
    return sorted(str(p) for p in Path(pair).glob('shard_*/match-*/review.json'))


def run_chunk(folder, index, files):
    out = folder / f'chunk_{index:02d}.jsonl'
    if out.exists() and sum(1 for _ in out.open()) == len(files):
        return out
    cfg = folder / f'chunk_{index:02d}.json'; atomic_json(cfg, dict(files=files, output=str(out)))
    with E.Slot() as cpu, (folder / f'chunk_{index:02d}.log').open('w') as f:
        r = subprocess.run(['taskset', '-c', str(cpu)] + command('evaluation', manifest(dict(script=WORKER, args=[str(cfg)], env=CPU_ENV)), node=True),
                           stdout=f, stderr=subprocess.STDOUT, timeout=3600, env=dict(os.environ, **CPU_ENV))
    assert r.returncode == 0, f'{folder} chunk {index} exit {r.returncode}'
    return out


def compute(pairs):
    """pairs: strong-eval pair directories (each must have SUMMARY.json). Returns {pair: pair SUMMARY}."""
    jobs, todo = [], []
    for pair in pairs:
        pair = Path(pair); assert (pair / 'SUMMARY.json').exists(), f'{pair} not finalized'
        folder = OUT / 'pairs' / key_of(pair); files = reviews(pair); s = folder / 'SUMMARY.json'
        if s.exists() and json.loads(s.read_text())['games'] == len(files):
            continue
        folder.mkdir(parents=True, exist_ok=True); todo.append((pair, folder, files))
        jobs += [(folder, i, files[k:k + CHUNK]) for i, k in enumerate(range(0, len(files), CHUNK))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(WORKERS, len(E.cpus()))) as pool:
        list(pool.map(lambda j: run_chunk(*j), jobs))
    for pair, folder, files in todo:
        rows = [json.loads(l) for f in sorted(folder.glob('chunk_*.jsonl')) for l in f.open() if l.strip()]
        errors = [r for r in rows if 'error' in r]; rows = [r for r in rows if 'error' not in r]
        with (folder / 'GAMES.jsonl.tmp').open('w') as f:
            for r in rows: f.write(json.dumps(r) + '\n')
        (folder / 'GAMES.jsonl.tmp').rename(folder / 'GAMES.jsonl')
        summary = json.loads((pair / 'SUMMARY.json').read_text())
        atomic_json(folder / 'SUMMARY.json', dict(version=VERSION, pair=str(pair), learner=summary['learner']['label'],
            opponent=summary.get('opponent', {}).get('label', 'G_frozen'), seat=summary.get('seat', 0), games=len(files), verified=len(rows),
            errors=[dict(file=r['file'], error=r['error'][:300]) for r in errors],
            r3_chain_check=dict(ok=sum(r['r3_chain_check'] is True for r in rows), bad=sum(r['r3_chain_check'] is False for r in rows),
                                missing=sum(r['r3_chain_check'] is None for r in rows)),
            by_mode={m: dict(learner=aggregate([r for r in rows if r['mode'] == m], 'learner'),
                             opponent=aggregate([r for r in rows if r['mode'] == m], 'opponent')) for m in ('greedy', 'categorical')},
            unix=time.time()))
    return {str(p): json.loads((OUT / 'pairs' / key_of(p) / 'SUMMARY.json').read_text()) for p in pairs}


def rows_of(pairs):
    return [json.loads(l) for p in pairs for l in (OUT / 'pairs' / key_of(p) / 'GAMES.jsonl').open() if l.strip()]


def mean_se(xs):
    n = len(xs)
    if not n: return dict(mean=None, se=None)
    m = sum(xs) / n; sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.
    return dict(mean=m, se=sd / math.sqrt(n))


def ratio(xs, ys):
    X, Y, n = sum(xs), sum(ys), len(xs)
    if not Y: return dict(value=None, se=None, num=X, den=Y)
    r = X / Y; se = math.sqrt(n / (n - 1) * sum((x - r * y) ** 2 for x, y in zip(xs, ys))) / Y if n > 1 else 0.
    return dict(value=r, se=se, num=X, den=Y)


def aggregate(rows, side='learner'):
    ps = [r['players'][r['seat'] if side == 'learner' else 1 - r['seat']] for r in rows]
    if not ps: return dict(games=0)
    ticks = [r['ticks'] for r in rows]
    out = dict(games=len(ps), ticks_mean=sum(ticks) / len(ticks),
               per_game={k: mean_se([p[k] for p in ps]) for k in COUNTS},
               ratios={k: ratio([p[a] for p in ps], [p[b] for p in ps]) for k, (a, b) in RATIOS.items()},
               per_1k_ticks={k: 1000 * sum(p[k] for p in ps) / sum(ticks) for k in PER_1K},
               games_with_chain=sum(p['chain_events'] > 0 for p in ps) / len(ps),
               games_with_chain_near=sum(p['chain_near_enemy'] > 0 for p in ps) / len(ps),
               games_streak_ge3=sum(p['fountain_streak'] >= 3 for p in ps) / len(ps),
               chain_len_max_overall=max(p['chain_len_max'] for p in ps), fountain_streak_overall=max(p['fountain_streak'] for p in ps))
    return out


def summarize_pairs(pairs):
    rows = rows_of(pairs)
    return {m: dict(learner=aggregate([r for r in rows if r['mode'] == m], 'learner'),
                    opponent=aggregate([r for r in rows if r['mode'] == m], 'opponent')) for m in ('greedy', 'categorical')}


def point_summary(pairs):
    """Hook for cc_strong_eval_v1.point(): pairs = pair dirs of one point (any opponents/seats)."""
    pairs = [Path(p) for p in pairs]; compute(pairs)
    vsG = [p for p in pairs if '__vs__G_frozen' in p.name]
    return dict(version=VERSION, definitions=str(OUT / 'DEFINITIONS.md'), vs_G_frozen=compact_all(summarize_pairs(vsG)) if vsG else None,
                pairs=[key_of(p) for p in pairs])


def compact(a):
    if not a.get('games'): return a
    g, r = a['per_game'], a['ratios']
    return dict(games=a['games'], chain_events=g['chain_events']['mean'], chain_events_se=g['chain_events']['se'],
                chain_len_max=g['chain_len_max']['mean'], chain_len_max_overall=a['chain_len_max_overall'],
                chain_near_enemy=g['chain_near_enemy']['mean'], chain_near_enemy_se=g['chain_near_enemy']['se'],
                effective_batches=g['effective_batches']['mean'], fountain_streak=g['fountain_streak']['mean'],
                fountain_streak_se=g['fountain_streak']['se'], fountain_streak_overall=a['fountain_streak_overall'],
                fountain_ticks=g['fountain_ticks']['mean'], fountain_ticks_se=g['fountain_ticks']['se'],
                games_streak_ge3=a['games_streak_ge3'], on_own_ray_rate=r['on_own_ray_rate']['value'],
                on_own_ray_rate_se=r['on_own_ray_rate']['se'], on_own_ray_other_batch_rate=r['on_own_ray_other_batch_rate']['value'],
                on_own_ray_conditional=r['on_own_ray_conditional']['value'], placed=g['placed']['mean'],
                strict_fountain_streak=g['strict_fountain_streak']['mean'], per_1k_ticks=a['per_1k_ticks'])


def compact_all(s):
    return {m: dict(learner=compact(v['learner']), opponent=compact(v['opponent'])) for m, v in s.items()}
def fmt(v, se=None, d=2):
    if v is None: return '—'
    return f'{v:.{d}f}' + (f'±{se:.{d}f}' if se is not None else '')


HEAD = ('| 对象 | 局数 | chain_events | chain_len_max 均值/最大 | chain_near_enemy | 有效批次 | fountain_streak 均值/最大 | '
        'fountain_ticks | streak≥3 局占比 | 射线上放泡占比 | 其中异批次 | 每局放泡 |\n|---|---|---|---|---|---|---|---|---|---|---|---|')


def line(name, a):
    c = compact(a)
    if not c.get('games'): return f'| {name} | 0 |' + ' |' * 11
    g = a['per_game']; r = a['ratios']
    return (f"| {name} | {c['games']} | {fmt(c['chain_events'], c['chain_events_se'])} | {fmt(c['chain_len_max'])} / {c['chain_len_max_overall']} | "
            f"{fmt(c['chain_near_enemy'], c['chain_near_enemy_se'])} | {fmt(c['effective_batches'], g['effective_batches']['se'], 1)} | "
            f"{fmt(c['fountain_streak'], c['fountain_streak_se'])} / {c['fountain_streak_overall']} | {fmt(c['fountain_ticks'], c['fountain_ticks_se'], 1)} | "
            f"{c['games_streak_ge3']:.3f} | {fmt(c['on_own_ray_rate'], c['on_own_ray_rate_se'], 3)} | {fmt(c['on_own_ray_other_batch_rate'], r['on_own_ray_other_batch_rate']['se'], 3)} | "
            f"{fmt(c['placed'], g['placed']['se'], 1)} |")


def baseline():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'DEFINITIONS.md').write_text(DEFINITIONS)
    groups = {name: [E.OUT / p for p in ps] for name, ps in SOURCE_GROUPS}
    compute([p for ps in groups.values() for p in ps])
    for name, parts in COMBINED: groups[name] = [p for x in parts for p in groups[x]]
    order = [n for n, _ in SOURCE_GROUPS[:2]] + [COMBINED[0][0]] + [n for n, _ in SOURCE_GROUPS[2:4]] + [COMBINED[1][0]] + [n for n, _ in SOURCE_GROUPS[4:]]
    value = dict(version=VERSION, definitions=str(OUT / 'DEFINITIONS.md'), groups={n: [key_of(p) for p in groups[n]] for n in order},
                 checks={key_of(p): {k: v for k, v in json.loads((OUT / 'pairs' / key_of(p) / 'SUMMARY.json').read_text()).items()
                                     if k in ('games', 'verified', 'errors', 'r3_chain_check')} for n in order for p in groups[n]},
                 results={n: summarize_pairs(groups[n]) for n in order}, unix=time.time())
    atomic_json(OUT / 'BASELINE.json', value)
    md = ['# 连锁/喷泉指标基线（fountain_metrics_v1）', '',
          f"口径见 DEFINITIONS.md。每行两座合并，均值±标准误（每局）。所有对局都在生产模拟器重放并通过 hash 校验。", '']
    for side, title in (('learner', '被评估模型（本方）'), ('opponent', '对手 G_frozen（同一批对局的另一方，作对照）')):
        for m, mt in (('greedy', 'argmax'), ('categorical', '采样')):
            md += [f'## {title}，{mt}', '', HEAD] + [line(n, value['results'][n][m][side]) for n in order] + ['']
    md += ['## 每 1000 tick 合并比率（本方）', '', '| 对象 | 模式 | chain_events | chain_near_enemy | 有效批次 | fountain_ticks | 放泡 |', '|---|---|---|---|---|---|---|']
    for n in order:
        for m in ('greedy', 'categorical'):
            a = value['results'][n][m]['learner']
            if a.get('games'): md.append(f"| {n} | {m} | " + ' | '.join(f"{a['per_1k_ticks'][k]:.2f}" for k in PER_1K) + ' |')
    bad = {k: v for k, v in value['checks'].items() if v['errors'] or v['r3_chain_check']['bad']}
    md += ['', f"校验：{sum(v['verified'] for v in value['checks'].values())}/{sum(v['games'] for v in value['checks'].values())} 局重放通过；"
           f"与 RESULT.json r3.chains 比对一致 {sum(v['r3_chain_check']['ok'] for v in value['checks'].values())} 局，"
           f"不一致 {sum(v['r3_chain_check']['bad'] for v in value['checks'].values())} 局。" + (f' 异常：{json.dumps(bad, ensure_ascii=False)[:2000]}' if bad else '')]
    (OUT / 'BASELINE.md').write_text('\n'.join(md) + '\n')
    print('FOUNTAIN_BASELINE', OUT / 'BASELINE.md', flush=True)


def main():
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == 'baseline': baseline()
    elif mode == 'pairs': print(json.dumps({k: v['by_mode'] for k, v in compute(rest).items()})[:4000])
    elif mode == 'point':  # backfill: attach metrics to existing point json(s) of the given labels
        for label in rest:
            for f in sorted((E.OUT / 'points').glob(f'{label}*.json')):
                v = json.loads(f.read_text()); seats = v.get('seats', [0])
                pairs = [E.OUT / 'pairs' / (f'{label}__vs__{o}' + ('__P1' if s else '')) for s in seats for o in v['opponents']]
                v['fountain'] = point_summary([p for p in pairs if (p / 'SUMMARY.json').exists()]); atomic_json(f, v)
                print('FOUNTAIN_POINT', f, json.dumps(v['fountain']['vs_G_frozen'])[:600], flush=True)
    else: raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
