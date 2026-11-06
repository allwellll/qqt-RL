"""Strong-opponent paired evaluation: cc_fixed_eval_v1 scenarios with the opponent replaced by strong policies.

Same 44 seat-0 scenarios x {greedy, categorical} and seeds as eval_protocol_v1; learner is P0, opponent always argmax.
Opponents: G_frozen (u018262 export) and the latest COMMITTED save point of every other active arm.
Each pair reuses scripts/resume_gpu67_eval_v1.js (fresh-simulator review.json replay check), CPU only.

usage (via model_evaluation.py <config>, config args):
  point  <learner_spec> [--seats 0,1] [opponent_spec ...]   one learner vs G_frozen + active arms (default) or given opponents
  points [--seats 0,1] [--opponents a,b] <learner_spec> ...   several learners in one CPU queue, then report
  matrix <spec> <spec> ...                            all ordered pairs, then MATRIX.json/.md
  report                                              rebuild MATRIX.json/.md from finished pairs
seat 1 (learner P2) uses PROTOCOL_P1.json: the same 88 games with seat=1, frozen on first use; pair dirs end with __P1.
spec: G_frozen | <arm> (latest COMMITTED) | <arm>@u018662 | <checkpoint dir> | <model.json>
env: CC_STRONG_CPUS (default 36-47,132-143; NUMA 0/1 only), CC_STRONG_ACTIVE_HOURS (default 6)
"""
import concurrent.futures, fcntl, json, os, subprocess, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.model_launch import manifest, command
from scripts.cc_fixed_eval_v1 import PROTOCOL as FIXED_PROTOCOL, CPU_ENV, oscillation, summarize

ROOT = Path('runs/cc_train_20261105')
OUT = ROOT / 'eval_strong_v1'
PROTOCOL = OUT / 'PROTOCOL.json'
PROTOCOL_P1 = OUT / 'PROTOCOL_P1.json'
FORMAL = ROOT / 'training' / 'formal'
G_FROZEN = 'runs/corner_counterplay_20261104/exports/G_frozen/model.json'
SHARD = 4
NUMA01 = set(range(0, 48)) | set(range(96, 144))
SLOTS = OUT / 'cpu_slots'


def cpus():
    spec = os.environ.get('CC_STRONG_CPUS', '36-47,132-143'); out = []
    for part in spec.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    assert out and set(out) <= NUMA01, 'strong eval must stay on NUMA 0/1 (training owns NUMA 2/3)'
    return out


def protocol(seat=0):
    if seat == 1:
        base = protocol(0)
        if PROTOCOL_P1.exists():
            p = json.loads(PROTOCOL_P1.read_text()); assert p['source_protocol_sha256'] == sha(PROTOCOL); return p
        games = [dict(g, seat=1, evaluation_split='cc_strong_eval_v1_P1') for g in base['games']]
        value = dict(status='FROZEN_BEFORE_ANY_GAME', source_protocol=str(PROTOCOL), source_protocol_sha256=sha(PROTOCOL),
                     g_frozen=base['g_frozen'], games=games, seat='learner seat 1 (P2); same seeds/spawns as seat 0',
                     opponent_mode='greedy (argmax) always', unix=time.time())
        atomic_json(PROTOCOL_P1, value)
        return value
    if PROTOCOL.exists():
        p = json.loads(PROTOCOL.read_text())
        assert sha(FIXED_PROTOCOL) == p['source_protocol_sha256'] and sha(G_FROZEN) == p['g_frozen']['sha256']
        return p
    fixed = json.loads(FIXED_PROTOCOL.read_text())
    games = [dict(g, opponent_mode='greedy', evaluation_split='cc_strong_eval_v1') for g in fixed['games']]
    assert len(games) == 88 and all(g['seat'] == 0 for g in games)
    value = dict(status='FROZEN_BEFORE_ANY_GAME', source_protocol=str(FIXED_PROTOCOL), source_protocol_sha256=sha(FIXED_PROTOCOL),
                 g_frozen=dict(file=G_FROZEN, sha256=sha(G_FROZEN), source='G_REGIONS_V2_c000/u018262'),
                 games=games, seat='learner seat 0 (P1) only', opponent_mode='greedy (argmax) always', unix=time.time())
    atomic_json(PROTOCOL, value)
    return value


def committed_points(arm):
    return sorted(int(f.parent.name[1:]) for f in (FORMAL / arm).glob('u*/COMMITTED.json'))


def active_arms():
    hours = float(os.environ.get('CC_STRONG_ACTIVE_HOURS', 6)); arms = []
    for d in sorted(p for p in FORMAL.iterdir() if p.is_dir()):
        points = committed_points(d.name)
        if not points or (d / 'USER_STOP.json').exists() or (d / 'FAILED.json').exists():
            continue
        if time.time() - (d / f'u{points[-1]:06d}' / 'COMMITTED.json').stat().st_mtime <= hours * 3600:
            arms.append(d.name)
    return arms


def resolve(spec):
    """-> dict(label, arm, update, checkpoint, model_json); checkpoints are only read."""
    if spec == 'G_frozen':
        return dict(label='G_frozen', arm='G_frozen', update=18262, checkpoint=None, model_json=G_FROZEN)
    arm, _, at = spec.partition('@')
    if (FORMAL / arm).is_dir() and '/' not in arm:
        u = int(at.lstrip('u')) if at else committed_points(arm)[-1]
        cp = FORMAL / arm / f'u{u:06d}'
        assert (cp / 'COMMITTED.json').exists(), f'{cp} not committed'
        return dict(label=f'{arm}_u{u:06d}', arm=arm, update=u, checkpoint=str(cp), model_json=None)
    p = Path(spec)
    if p.is_dir():
        assert (p / 'COMMITTED.json').exists(), f'{p} not committed'
        return dict(label=f'{p.parent.name}_{p.name}', arm=p.parent.name, update=int(p.name.lstrip('u')), checkpoint=str(p), model_json=None)
    assert p.name == 'model.json' and p.exists(), spec
    return dict(label=p.parent.name, arm=p.parent.name, update=None, checkpoint=None, model_json=str(p))


def ensure_export(m):
    """Reuse an exact K4 export (eval_v1 or ours) of the committed policy, else export under OUT/exports."""
    if m['model_json']:
        m['model_json_sha256'] = sha(m['model_json']); return m
    cp = Path(m['checkpoint']); committed = json.loads((cp / 'COMMITTED.json').read_text())
    for d in (ROOT / 'eval_v1' / m['label'] / 'export', OUT / 'exports' / m['label']):
        idf = d / 'K4_MODE0_IDENTITY.json'
        if idf.exists() and (d / 'model.json').exists():
            ident = json.loads(idf.read_text())
            if (ident['source_policy_sha256'] == committed['policy_sha256'] and Path(ident['source_checkpoint']).resolve() == cp.resolve()
                    and sha(d / 'model.json') == ident['model_sha256']):
                m.update(model_json=str(d / 'model.json'), model_json_sha256=ident['model_sha256'], export=str(d))
                return m
    dest = OUT / 'exports' / m['label']; dest.mkdir(parents=True, exist_ok=True)
    with (dest.parent / f'{m["label"]}.lock').open('a+') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (dest / 'K4_MODE0_IDENTITY.json').exists():
            with (dest.parent / f'{m["label"]}.log').open('w') as f:
                cfg = manifest(dict(module='scripts.r4_k4_export', args=[str(cp), str(dest)], env=CPU_ENV))
                r = subprocess.run(['taskset', '-c', ','.join(map(str, cpus()[:4]))] + command('evaluation', cfg),
                                   stdout=f, stderr=subprocess.STDOUT, timeout=1800, env=dict(os.environ, **CPU_ENV))
            assert r.returncode == 0, f'export {m["label"]} failed'
    return ensure_export(m)


class Slot:
    """One evaluation process per CPU in the pool, enforced across all strong-eval processes by flock."""
    def __enter__(self):
        SLOTS.mkdir(parents=True, exist_ok=True)
        while True:
            for c in cpus():
                f = (SLOTS / f'cpu{c:03d}.lock').open('a+')
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    f.close(); continue
                self.f, self.cpu = f, c
                return c
            time.sleep(5)

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def pair_dir(learner, opponent, seat=0):
    return OUT / 'pairs' / (f'{learner["label"]}__vs__{opponent["label"]}' + ('__P1' if seat else ''))


def run_shard(learner, opponent, seat, index, games):
    folder = pair_dir(learner, opponent, seat) / f'shard_{index:02d}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE.json').exists():
        return folder
    cfg = folder / 'CONFIG.json'
    atomic_json(cfg, dict(candidate=learner['label'], target=opponent['label'], model_json=learner['model_json'],
                          opponent_model_json=opponent['model_json'], games=games, output=str(folder)))
    with Slot() as cpu, (folder / 'worker.log').open('w') as f:
        r = subprocess.run(['taskset', '-c', str(cpu)] + command('evaluation', manifest(dict(script='scripts/resume_gpu67_eval_v1.js', args=[str(cfg)], env=CPU_ENV)), node=True),
                           stdout=f, stderr=subprocess.STDOUT, timeout=3600, env=dict(os.environ, **CPU_ENV))
    assert r.returncode == 0, f'{folder} exit {r.returncode}'
    return folder


def opponent_deaths(r):
    q = 1 - r['seat']; ds = [d for d in r['source_metrics']['deaths'] if d['victim'] == q]
    return dict(opp_deaths=len(ds), opp_self_deaths=sum(1 for d in ds if (d.get('initial_control') or {}).get('causal_kind') == 'unique_self'))


def finalize(learner, opponent, games, seat=0):
    d = pair_dir(learner, opponent, seat); rows = []
    for folder in sorted(d.glob('shard_*')):
        for r in json.loads((folder / 'COMPLETE.json').read_text())['results']:
            rid = f"match-{r['seed']}-P{r['seat']}-{r['mode']}-{r['opponent_mode']}"
            review = folder / rid / 'review.json'
            assert r['review_verified'] and sha(review) == r['review_sha256'], f'{review} not verified'
            rv = json.loads(review.read_text())
            assert rv['meta']['model_json_sha256'] == learner['model_json_sha256'] and rv['meta']['opponent_model_json_sha256'] == opponent['model_json_sha256']
            r['oscillation'] = oscillation(rv, r['seat']); r['opponent'] = opponent_deaths(r)
            rows.append(r)
    assert sorted((r['seed'], r['mode'], r['seat']) for r in rows) == sorted((g['seed'], g['mode'], g['seat']) for g in games), f'{d} incomplete'
    from scripts.corner_counterplay_stage_report_v1 import measures
    by_mode = summarize(rows)
    for mode, t in by_mode.items():
        for k in ('opp_deaths', 'opp_self_deaths'):
            t[k] = sum(r['opponent'][k] for r in rows if r['mode'] == mode)
    per_game = [dict(seed=r['seed'], mode=r['mode'], brick_stage=r.get('brick_stage'), region=r['start_region'], winner=r['winner'], seat=r['seat'],
                     **measures(r), **r['oscillation'], **r['opponent']) for r in rows]
    value = dict(status='COMPLETE_STRONG_EVAL_V1', learner=learner, opponent=opponent, seat=seat, protocol_sha256=sha(PROTOCOL_P1 if seat else PROTOCOL),
                 games=len(rows), by_mode=by_mode, per_game=per_game, unix=time.time())
    atomic_json(d / 'SUMMARY.json', value)
    return value


def run_pairs(pairs):
    """pairs: (learner, opponent) or (learner, opponent, seat); returns {(learner_label, opponent_label, seat): SUMMARY}."""
    pairs = [tuple(pr) + (0,) * (3 - len(pr)) for pr in pairs]
    games = {seat: protocol(seat)['games'] for seat in {pr[2] for pr in pairs}}
    models = {}
    for m in [x for pr in pairs for x in pr[:2]]:
        models.setdefault(m['label'], m)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(ensure_export, models.values()))
    todo, done = [], {}
    for a, b, seat in pairs:
        a, b = models[a['label']], models[b['label']]; s = pair_dir(a, b, seat) / 'SUMMARY.json'
        if s.exists():
            v = json.loads(s.read_text())
            assert v['learner']['model_json_sha256'] == a['model_json_sha256'] and v['opponent']['model_json_sha256'] == b['model_json_sha256'], f'{s} label reused for another model'
            done[(a['label'], b['label'], seat)] = v; continue
        todo.append((a, b, seat))
    jobs = [(a, b, seat, i, games[seat][k:k + SHARD]) for a, b, seat in todo for i, k in enumerate(range(0, len(games[seat]), SHARD))]
    begun = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus())) as pool:
        list(pool.map(lambda j: run_shard(*j), jobs))
    for a, b, seat in todo:
        v = finalize(a, b, games[seat], seat); v['seconds_batch'] = time.time() - begun; atomic_json(pair_dir(a, b, seat) / 'SUMMARY.json', v)
        done[(a['label'], b['label'], seat)] = v
    return done


KEYS = ('games', 'wins', 'losses', 'draws', 'deaths', 'self_deaths', 'causal_kills', 'placed', 'zero_placement_games', 'opposite_adjacent',
        'ticks', 'acquisitions', 'deliveries', 'carry_failures', 'item_pickups', 'opp_deaths', 'opp_self_deaths')


def aggregate(summaries, mode):
    t = {k: sum(s['by_mode'][mode][k] for s in summaries) for k in KEYS}
    t['opposite_rate'] = t['opposite_adjacent'] / max(t['ticks'], 1)
    t['score'] = (t['wins'] + .5 * t['draws']) / max(t['games'], 1)
    return t


def selection_key(summaries):
    """Pre-declared ranking: net wins (both modes), then deliveries, causal kills, fewer self-deaths."""
    g, c = aggregate(summaries, 'greedy'), aggregate(summaries, 'categorical')
    return [g['wins'] + c['wins'] - g['losses'] - c['losses'], g['deliveries'] + c['deliveries'],
            g['causal_kills'] + c['causal_kills'], -(g['self_deaths'] + c['self_deaths'])]


def compact(t):
    return {k: t[k] for k in ('wins', 'losses', 'draws', 'score', 'deaths', 'self_deaths', 'causal_kills', 'placed', 'zero_placement_games',
                              'opposite_rate', 'acquisitions', 'deliveries', 'carry_failures', 'item_pickups', 'opp_self_deaths') if k in t}


def load_summaries(labels=None, seat=0):
    """{(learner, opponent): SUMMARY} for one seat; seat=None -> {(learner, opponent, seat): SUMMARY}."""
    out = {}
    for f in sorted((OUT / 'pairs').glob('*/SUMMARY.json')):
        v = json.loads(f.read_text()); a, b, k = v['learner']['label'], v['opponent']['label'], v.get('seat', 0)
        if (labels is None or (a in labels and b in labels)) and (seat is None or k == seat):
            v['by_mode'] = {m: dict(t, score=(t['wins'] + .5 * t['draws']) / t['games']) for m, t in v['by_mode'].items()}
            out[(a, b) if seat is not None else (a, b, k)] = v
    return out


def two_seat_lines(value, lines, fmt, cols):
    """Learner in P0 and in P1 against G_frozen (argmax), both modes; 88 games per mode."""
    A = load_summaries(seat=None)
    learners = sorted({a for a, b, k in A if b == 'G_frozen' and (a, b, 1 - k) in A}, key=lambda x: (x != 'G_frozen', x))
    if not learners:
        return
    value['two_seat_vs_G_frozen'] = {}
    lines += ['', '## 双座对 G_frozen（learner 分别坐 P0、P1，对手 G_frozen argmax；每模式 88 局）', '',
              '| learner | 模式 | P0 胜-负-平 | P1 胜-负-平 | 合计得分 | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 5)]
    for a in learners:
        vs = [A[(a, 'G_frozen', 0)], A[(a, 'G_frozen', 1)]]; value['two_seat_vs_G_frozen'][a] = {}
        for mode in ('greedy', 'categorical'):
            t = aggregate(vs, mode); s0, s1 = (v['by_mode'][mode] for v in vs)
            value['two_seat_vs_G_frozen'][a][mode] = dict(compact(t), P0=compact(s0), P1=compact(s1))
            lines.append(f"| {a} | {mode} | {s0['wins']}-{s0['losses']}-{s0['draws']} | {s1['wins']}-{s1['losses']}-{s1['draws']} | {t['score']:.3f} | " + fmt(t) + ' |')
        value['two_seat_vs_G_frozen'][a]['selection_key'] = selection_key(vs)
    lines += ['', 'G_frozen 行是 G_frozen 对自身 argmax 的参照：greedy 行接近确定性自对局，categorical 行表示"同一模型采样打 argmax"的基准得分，'
              '各臂 categorical 得分应与它比较，而不是与 0.5 比较。']


def report(labels=None, name='MATRIX'):
    """Seat-0 cross matrix over `labels` (default: the labels of the existing MATRIX.json), plus the two-seat G_frozen table."""
    if labels is None and (OUT / f'{name}.json').exists():
        labels = json.loads((OUT / f'{name}.json').read_text())['labels']
    S = {k: v for k, v in load_summaries(labels).items() if k[0] != k[1]}  # self-reference lives in the two-seat table
    order = labels or sorted({x for k in S for x in k}, key=lambda x: (x != 'G_frozen', x))
    value = dict(status='STRONG_EVAL_V1_MATRIX', protocol_sha256=sha(PROTOCOL), labels=order, pairs={}, learners={}, unix=time.time())
    lines = [f'# 强对手评估矩阵 cc_strong_eval_v1（{time.strftime("%Y-%m-%d %H:%M")}）', '',
             'learner 在 P0（行），对手 argmax（列）；每格 胜-负-平（得分=(胜+0.5平)/44）。44 开局与 cc_fixed_eval_v1 相同。', '']
    for mode in ('greedy', 'categorical'):
        lines += [f'## 胜率矩阵：learner {mode}', '', '| learner \\ 对手 | ' + ' | '.join(order) + ' | 合计 |', '|' + '---|' * (len(order) + 2)]
        for a in order:
            cells, mine = [], []
            for b in order:
                v = S.get((a, b))
                if v:
                    t = v['by_mode'][mode]; mine.append(v); cells.append(f"{t['wins']}-{t['losses']}-{t['draws']} ({t['score']:.2f})")
                else:
                    cells.append('—')
            agg = aggregate(mine, mode) if mine else None
            lines.append(f'| {a} | ' + ' | '.join(cells) + (f" | {agg['wins']}-{agg['losses']}-{agg['draws']} ({agg['score']:.2f}) |" if agg else ' | — |'))
        lines.append('')
    # learner is always P0; merge A(P0) vs B and B(P0) vs A so the seat advantage cancels
    value['seat_balanced'] = {}
    for mode in ('greedy', 'categorical'):
        lines += [f'## 双座合并对战得分：{mode}（A 作 P0 对 B + B 作 P0 对 A，A 视角，88 局；>0.5 表示行方更强）', '',
                  '| A \\ B | ' + ' | '.join(order) + ' | 平均 |', '|' + '---|' * (len(order) + 2)]
        for a in order:
            cells, xs = [], []
            for b in order:
                u, w = S.get((a, b)), S.get((b, a))
                if u and w:
                    x = (u['by_mode'][mode]['score'] + 1 - w['by_mode'][mode]['score']) / 2; xs.append(x); cells.append(f'{x:.2f}')
                    value['seat_balanced'].setdefault(mode, {})[f'{a}__vs__{b}'] = x
                else:
                    cells.append('—')
            lines.append(f'| {a} | ' + ' | '.join(cells) + (f' | {sum(xs) / len(xs):.3f} |' if xs else ' | — |'))
        lines.append('')
    cols = ['胜', '负', '平', '死亡', '自死', '因果杀', '放泡', '零放泡局', '反向率', '取包', '交付', '携包失败', '道具', '对手自死']
    keys = ['wins', 'losses', 'draws', 'deaths', 'self_deaths', 'causal_kills', 'placed', 'zero_placement_games', 'opposite_rate',
            'acquisitions', 'deliveries', 'carry_failures', 'item_pickups', 'opp_self_deaths']
    fmt = lambda t: ' | '.join(f'{t[k]:.3f}' if k == 'opposite_rate' else str(t[k]) for k in keys)
    lines += ['## 分项表（每对 44 局）', '', '| learner | 对手 | 模式 | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 3)]
    for a in order:
        for b in order:
            if (a, b) in S:
                v = S[(a, b)]; value['pairs'][f'{a}__vs__{b}'] = {m: compact(t) for m, t in v['by_mode'].items()}
                for mode in ('greedy', 'categorical'):
                    lines.append(f'| {a} | {b} | {mode} | ' + fmt(v['by_mode'][mode]) + ' |')
    lines += ['', '## 每个 learner 合计（对全部强对手 / 只对 G_frozen）', '', '| learner | 范围 | 模式 | ' + ' | '.join(cols) + ' | 得分 |', '|' + '---|' * (len(cols) + 4)]
    for a in order:
        mine = [v for (x, b), v in S.items() if x == a]
        if not mine:
            continue
        scopes = dict(all=mine, G_frozen=[v for v in mine if v['opponent']['label'] == 'G_frozen'],
                      peers=[v for v in mine if v['opponent']['label'] != 'G_frozen'])
        if len(scopes['peers']) == len(mine):
            scopes.pop('peers')
        value['learners'][a] = dict(opponents=[v['opponent']['label'] for v in mine], selection_key=selection_key(mine),
                                    **{s: {m: compact(aggregate(vs, m)) for m in ('greedy', 'categorical')} for s, vs in scopes.items() if vs})
        for s, vs in scopes.items():
            for mode in ('greedy', 'categorical'):
                if vs:
                    t = aggregate(vs, mode); lines.append(f'| {a} | {s} | {mode} | ' + fmt(t) + f" | {t['score']:.3f} |")
    lines += ['', '口径：死亡/自死/因果杀/取包/交付/携包失败/道具同 cc_fixed_eval_v1（learner 侧）；对手自死=对手死亡中 causal_kind=unique_self；'
              '反向率=相邻 tick 反向移动/总 tick（取自已校验的 review.json）。所有 review.json 均由新模拟器重放校验 checkpoint hash 与胜者。']
    two_seat_lines(value, lines, fmt, cols)
    atomic_json(OUT / f'{name}.json', value); (OUT / f'{name}.md').write_text('\n'.join(lines) + '\n')
    return value


def point(learner_spec, opponent_specs, seats=(0,), run=True):
    learner = ensure_export(resolve(learner_spec))
    explicit = bool(opponent_specs)
    if not explicit:
        opponent_specs = ['G_frozen'] + [a for a in active_arms() if a != learner['arm']]
    opps = [ensure_export(resolve(s)) for s in opponent_specs]
    if not explicit:  # an explicitly requested opponent (e.g. G_frozen self reference) is always played
        opps = [o for o in opps if o['model_json_sha256'] != learner['model_json_sha256']]
    pairs = [(learner, o, seat) for seat in seats for o in opps]
    if not run:
        return pairs
    done = run_pairs(pairs); mine = [done[(learner['label'], o['label'], seat)] for _, o, seat in pairs]
    for v in mine:
        v['by_mode'] = {m: dict(t, score=(t['wins'] + .5 * t['draws']) / t['games']) for m, t in v['by_mode'].items()}
    fixed = ROOT / 'eval_v1' / learner['label'] / 'SUMMARY.json'
    vsG = [v for v in mine if v['opponent']['label'] == 'G_frozen']
    value = dict(status='STRONG_EVAL_V1_POINT', learner=learner, opponents=[o['label'] for o in opps], seats=list(seats),
                 vs={v['opponent']['label'] + ('__P1' if v.get('seat') else ''): {m: compact(t) for m, t in v['by_mode'].items()} for v in mine},
                 all={m: compact(aggregate(mine, m)) for m in ('greedy', 'categorical')},
                 G_frozen={m: compact(aggregate(vsG, m)) for m in ('greedy', 'categorical')} if vsG else None,
                 selection_key=selection_key(mine), selection_key_vs_G_frozen=selection_key(vsG) if vsG else None,
                 regression_vs_parent18102=({m: compact(dict(t, score=(t['wins'] + .5 * t['draws']) / t['games']))
                                             for m, t in json.loads(fixed.read_text())['by_mode'].items()} if fixed.exists() else None),
                 unix=time.time())
    try:  # chain / fountain metrics (offline replay, CPU); a metrics failure never blocks the point
        from scripts.cc_strong_metrics_fountain_v1 import point_summary
        value['fountain'] = point_summary([pair_dir(learner, o, seat) for _, o, seat in pairs])
    except Exception as e:
        value['fountain'] = dict(error=repr(e)[:500])
    tag = '' if list(seats) == [0] else '__seats' + ''.join(map(str, seats))
    atomic_json(OUT / 'points' / f'{learner["label"]}{tag}.json', value)
    print('STRONG_EVAL', learner['label'] + tag, json.dumps(dict(selection_key=value['selection_key'], all=value['all'])), flush=True)
    return value


def options(args):
    """--seats 0,1 and --opponents a,b,... ; returns (seats, opponents, rest)."""
    seats, opps, rest = (0,), [], []
    it = iter(args)
    for x in it:
        if x == '--seats':
            seats = tuple(int(c) for c in next(it).split(','))
        elif x == '--opponents':
            opps = next(it).split(',')
        else:
            rest.append(x)
    assert set(seats) <= {0, 1}
    return seats, opps, rest


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True); protocol()
    if mode == 'point':
        seats, opps, rest = options(args)
        point(rest[0], opps + rest[1:], seats)
    elif mode == 'points':
        # several learners, one shared CPU queue in learner order, then one point file each
        seats, opps, rest = options(args)
        run_pairs([pr for spec in rest for pr in point(spec, opps, seats, run=False)])
        for spec in rest:
            point(spec, opps, seats)
        report()
    elif mode == 'matrix':
        name = 'MATRIX'
        if args and args[0] == '--name':
            name, args = args[1], args[2:]   # a named matrix keeps MATRIX.md (round 1) intact
        ms = [resolve(s) for s in args]
        run_pairs([(a, b) for a in ms for b in ms if a['label'] != b['label']])
        print(json.dumps(report([m['label'] for m in ms], name=name)['learners'], indent=1), flush=True)
    elif mode == 'report':
        report(args or None)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
