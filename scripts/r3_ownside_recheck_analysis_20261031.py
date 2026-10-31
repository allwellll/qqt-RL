"""Read-only comparison/paired intervals for the frozen 16:20 CPU evaluation."""
import json
import time
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json, sha
from scripts.r3_ownside_recheck_1620_20261031 import OUT, REPORT, read


def paired_interval(rows, seat, rng, repeats=20000):
    values = np.array([r['score_difference'] for r in rows if r['seat'] == seat])
    assert len(values) == 128
    sums = rng.choice(values, size=(repeats, len(values)), replace=True).sum(axis=1)
    return dict(delta_win_minus_loss=int(values.sum()),
                paired_seed_bootstrap_95=np.quantile(sums, [.025, .975]).tolist(),
                improved=int((values > 0).sum()), worse=int((values < 0).sum()),
                equal=int((values == 0).sum()), games=128), sums


def main():
    complete = read(OUT/'COMPLETE.json')
    assert complete['status'] == 'COMPLETE' and sha(OUT/'RESULTS.json') == complete['results_sha256']
    r = read(OUT/'RESULTS.json'); p = read(OUT/'HOLDOUT_PREFIX_COMPARISON_PROTOCOL.json')
    baseline = r['holdout_matched_prefix']; comparisons = {'baseline': baseline}; sources = {}
    for label, ref in p['references'].items():
        source = Path(ref['path'])
        assert sha(source) == ref['sha256']
        d = read(source); assert d['status'] == 'COMPLETE'
        assert all(sha(Path(f)) == h for f, h in d['files'].items())
        raw = {m: [] for m in ('greedy', 'categorical')}
        for f in d['files']:
            if not f.endswith('.outcome.json'):
                continue
            z = read(f); raw[z['mode']].extend(z['episodes'])
        for mode, rows in raw.items():
            assert len(rows) == 128
            assert {(e['seed'], e['seat']) for e in rows} == {(2026317200+i, i%2) for i in range(128)}
            assert all(d['metrics']['holdout'][mode][s]['games'] == 64 for s in ('P0', 'P1'))
        comparisons[label] = d['metrics']['holdout']; sources[label] = ref
    rng = np.random.default_rng(2026311620); intervals = {}
    for label, cs in r['paired_per_game'].items():
        intervals[label] = {}; sums = np.zeros(20000)
        for condition, rows in cs.items():
            intervals[label][condition] = {}
            for seat in (0, 1):
                interval, draws = paired_interval(rows, seat, rng)
                intervals[label][condition][f'P{seat}'] = interval
                if seat == 1:
                    sums += draws
        intervals[label]['P1_sum_three_tiers'] = dict(
            delta_win_minus_loss=sum(z['P1']['delta_win_minus_loss'] for z in intervals[label].values()),
            stratified_paired_seed_bootstrap_95=np.quantile(sums, [.025, .975]).tolist())
    result = dict(status='COMPLETE_ANALYSIS_ONLY', original_results_sha256=sha(OUT/'RESULTS.json'),
        common_prefix_holdout=comparisons, common_prefix_sources=sources,
        paired_intervals=intervals, bootstrap_repeats=20000, bootstrap_seed=2026311620,
        limitations='conditional on fixed repeated own-side spawn geometries; not independent-map inference; no multiple-comparison correction; intervals do not alter frozen selector',
        no_publish=True, unix=time.time())
    atomic_json(OUT/'COMPARISON_ANALYSIS.json', result)
    lines = ['', '## 相同holdout前缀对照', '',
        '下表严格使用seed2026317200起前128局，每模式每席64局。候选沿用已完成数据，u3040扩展后128局不混入这张等N对照。', '',
        '|模型|模式|席位|局数|胜/平/负|胜负差|击杀/局|', '|---|---|---|---:|---|---:|---:|']
    for label, modes in comparisons.items():
        for mode, seats in modes.items():
            for seat, v in seats.items():
                lines.append(f"|{label}|{mode}|{seat}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['win_minus_loss']:+d}|{v['causal_kills']/v['games']:.4f}|")
    lines += ['', '## fresh逐局配对不确定性', '',
        '|候选|档位|席位|胜负差变化|配对seed bootstrap95%区间|改善/相同/变差局|', '|---|---|---|---:|---|---|']
    for label, cs in intervals.items():
        for condition, seats in cs.items():
            if condition == 'P1_sum_three_tiers':
                continue
            for seat, v in seats.items():
                lines.append(f"|{label}|{condition}|{seat}|{v['delta_win_minus_loss']:+d}|{v['paired_seed_bootstrap_95']}|{v['improved']}/{v['equal']}/{v['worse']}|")
        v = cs['P1_sum_three_tiers']
        lines.append(f"|{label}|三档合计|P1|{v['delta_win_minus_loss']:+d}|{v['stratified_paired_seed_bootstrap_95']}|按档分层重采样|")
    lines += ['', '每次20000次逐seed配对重采样，三档合计按档分层。区间只条件于固定出生几何与对手，不证明跨地图泛化；未作多候选校正，不以区间事后改变选型门槛。所有未记录的因果收益仍不可得，不推断意图。',
        '', '附加复算：`.venv/bin/python -m scripts.r3_ownside_recheck_analysis_20261031`。']
    report = REPORT.read_text(); marker = '\n## 相同holdout前缀对照\n'
    if marker in report:
        report = report.split(marker)[0]
    REPORT.write_text(report+'\n'.join(lines)+'\n')
    complete.update(report_sha256=sha(REPORT), comparison_analysis_sha256=sha(OUT/'COMPARISON_ANALYSIS.json'))
    atomic_json(OUT/'COMPLETE.json', complete)
    print(json.dumps(dict(selection=r['selection'], paired_intervals=intervals), ensure_ascii=False))


if __name__ == '__main__':
    main()
