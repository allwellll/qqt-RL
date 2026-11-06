"""Strict 704-game artifact audit and paired, stratified holdout analysis. CPU only."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scripts.corner_counterplay_stage_report_v1 import measures, paired
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/eval_strong_v1/cx_holdout_20261106_v1')
REPORT = Path('reports/cx_training_followup_20261106.md')


def key(r):
    return r['seed'], r['seat'], r['mode']


def utility(r):
    return .5 if r['winner'] is None else float(r['winner'] == r['seat'])


def total(rows):
    metrics = [measures(r) for r in rows]
    wins = sum(r['winner'] == r['seat'] for r in rows)
    draws = sum(r['winner'] is None for r in rows)
    return dict(games=len(rows), wins=wins, losses=len(rows)-wins-draws, draws=draws,
                score=(wins+.5*draws)/len(rows),
                **{name: sum(m[name] for m in metrics) for name in ('causal_controls', 'causal_kills', 'self_deaths',
                    'deaths', 'acquisitions', 'deliveries', 'carry_failures', 'item_pickups', 'item_releases')})


def paired_score(reference, candidate):
    ref = {key(r): r for r in reference}
    clusters = defaultdict(list)
    assert set(ref) == {key(r) for r in candidate}
    for r in candidate:
        clusters[r['seed']].append(utility(r)-utility(ref[key(r)]))
    blocks = list(clusters.values())
    delta = np.asarray([sum(b) for b in blocks]); sizes = np.asarray([len(b) for b in blocks])
    ids = np.random.default_rng(2026616088).integers(0, len(blocks), (10000, len(blocks)))
    boot = delta[ids].sum(1)/sizes[ids].sum(1)
    return dict(mean=float(delta.sum()/sizes.sum()), lower95=float(np.quantile(boot,.025)),
                upper95=float(np.quantile(boot,.975)), paired_games=len(candidate), seed_clusters=len(blocks),
                bootstrap_replicates=10000)


def main():
    p = json.loads((BASE/'PROTOCOL.json').read_text())
    frozen_sha = sha(BASE/'PROTOCOL.json')
    summary = json.loads((BASE/'SUMMARY.json').read_text())
    assert summary['status'] == 'COMPLETE' and summary['protocol_sha256'] == frozen_sha
    labels = ['G_frozen', p['source_checkpoint']['label']]
    expected = {(g['seed'], g['seat'], g['mode']): g for games in p['games'].values() for g in games}
    assert len(expected) == 352
    models = {}; verified_reviews = 0; hash_checks = 0; model_hashes = {}
    for label in labels:
        rows = []
        for seat in (0, 1):
            folder = BASE/'pairs'/f'{label}__vs__G_frozen__P{seat}'
            s = json.loads((folder/'SUMMARY.json').read_text())
            assert s['protocol_sha256'] == frozen_sha and s['games'] == 176
            model = s['learner']; assert model['label'] == label
            assert sha(model['model_json']) == model['model_json_sha256']
            model_hashes.setdefault(label, model['model_json_sha256'])
            assert model_hashes[label] == model['model_json_sha256']
            summaries = {key(r): r for r in s['per_game']}
            for complete in sorted(folder.glob('shard_*/COMPLETE.json')):
                complete_rows = json.loads(complete.read_text())['results']
                assert len(complete_rows) == 4
                for r in complete_rows:
                    k = key(r); game = expected[k]
                    name = f"match-{r['seed']}-P{seat}-{r['mode']}-{r['opponent_mode']}"
                    d = complete.parent/name
                    actual = json.loads((d/'RESULT.json').read_text()); assert actual == r
                    assert r['status'] == 'PASS' and r['complete'] and r['review_verified']
                    assert r['candidate'] == label and r['target'] == 'G_frozen' and r['seat'] == seat
                    assert r['opponent_mode'] == 'greedy'
                    assert r['model_json_sha256'] == model_hashes[label]
                    assert r['opponent_model_json_sha256'] == p['g_frozen']['sha256']
                    review = json.loads((d/'review.json').read_text())
                    assert sha(d/'review.json') == r['review_sha256']
                    assert review['meta']['game'] == game
                    assert review['summary']['done'] and review['summary']['winner'] == r['winner']
                    assert review['summary']['ticks'] == r['ticks'] == len(review['actions'])
                    assert review['meta']['model_json_sha256'] == r['model_json_sha256']
                    assert review['meta']['opponent_model_json_sha256'] == p['g_frozen']['sha256']
                    assert sha('web/sim-bun-native-v2.js') == review['meta']['simulator_sha256']
                    for metric, value in measures(r).items():
                        assert summaries[k][metric] == value, (label, k, metric)
                    r = dict(r, start_region=game['region'], brick_stage=game['brick_stage'])
                    rows.append(r); verified_reviews += 1; hash_checks += len(review['checks'])
        assert len(rows) == 352 and len({key(r) for r in rows}) == 352
        assert {key(r) for r in rows} == set(expected)
        models[label] = rows
    assert verified_reviews == 704
    output = dict(status='PASS_COMPLETE_704_ARTIFACT_AUDIT', unix=time.time(), protocol_sha256=frozen_sha,
                  verified_reviews=verified_reviews, upstream_fresh_simulator_hash_checkpoints=hash_checks,
                  audit_note='Independently verified file SHA/schema/game/model/summary consistency; upstream evaluator performed fresh-simulator replay of every game.',
                  model_json_sha256=model_hashes, models={}, paired={}, strata={}, promotion='NOT_AUTOMATIC')
    for label, rows in models.items():
        output['models'][label] = {mode: total([r for r in rows if r['mode'] == mode]) for mode in ('greedy', 'categorical')}
        groups = defaultdict(list)
        for r in rows:
            groups[f"{r['mode']}/seat{r['seat']}/{r['start_region']}/{r['brick_stage']}"].append(r)
        output['strata'][label] = {group: total(rs) for group, rs in sorted(groups.items())}
    candidate = labels[1]
    for mode in ('greedy', 'categorical'):
        reference = [r for r in models['G_frozen'] if r['mode'] == mode]
        mine = [r for r in models[candidate] if r['mode'] == mode]
        output['paired'][mode] = dict(score=paired_score(reference, mine),
                                      metrics=paired(reference, mine))
    atomic_json(BASE/'ACCEPTANCE_PAIRED_STRATA.json', output)
    lines = ['\n## 704局完整验收与同批配对结论', '',
             f'冻结协议SHA：{frozen_sha}。704局RESULT/COMPLETE/review文件、模型身份、场景/seed/席位集合、摘要计数逐项通过；没有缺局或重复。原评估器每局做fresh模拟器重放，本轮另独立核验文件和schema一致性，不冒称再次重放。', '',
             '| 模型 | 模式 | 胜/负/平 | score | 真实控制/因果杀 | 交付 | 自死 | 携包失败 | 道具拾取 |',
             '|---|---|---|---|---|---|---|---|---|']
    for label, modes in output['models'].items():
        for mode, r in modes.items():
            lines.append(f"| {label} | {mode} | {r['wins']}/{r['losses']}/{r['draws']} | {r['score']:.4f} | {r['causal_controls']}/{r['causal_kills']} | {r['deliveries']} | {r['self_deaths']} | {r['carry_failures']} | {r['item_pickups']} |")
    lines += ['', '配对区间按环境seed聚类，避免把两座当独立样本；同一checkpoint和同批G_frozen参考，不使用旧.386阈值。', '']
    for mode, r in output['paired'].items():
        d = r['score']
        lines.append(f"{mode}：{candidate}−G_frozen配对score差={d['mean']:+.4f}，95% bootstrap区间[{d['lower95']:+.4f},{d['upper95']:+.4f}]，{d['paired_games']}局/{d['seed_clusters']}个seed簇。")
        for name in ('deliveries', 'causal_controls', 'causal_kills', 'self_deaths', 'carry_failures', 'item_pickups'):
            metric = r['metrics']['intervals'].get(name)
            if metric:
                lines.append(f"- {name}每局差={metric['mean']:+.4f}，95%区间[{metric['lower95']:+.4f},{metric['upper95']:+.4f}]。")
        lines.append('')
    lines += ['### 位置、砖阶段与席位分层', '',
              '以下各层仅描述本次固定44场景，不声称未知地图泛化；小层不单独用于选模。', '',
              '| 模型 | mode/席位/位置/砖阶段 | 局数 | 胜/负/平 | 控制/因果杀 | 交付 | 自死 | 携包失败 |',
              '|---|---|---|---|---|---|---|---|']
    for label, groups in output['strata'].items():
        for group, r in groups.items():
            lines.append(f"| {label} | {group} | {r['games']} | {r['wins']}/{r['losses']}/{r['draws']} | {r['causal_controls']}/{r['causal_kills']} | {r['deliveries']} | {r['self_deaths']} | {r['carry_failures']} |")
    document = '\n'.join(lines)+'\n'
    (BASE/'ACCEPTANCE_PAIRED_STRATA.md').write_text(document)
    with REPORT.open('a') as report:
        report.write(document)
    print(json.dumps(dict(status=output['status'], models=output['models'],
                         paired_score={mode:r['score'] for mode,r in output['paired'].items()})), flush=True)


if __name__ == '__main__':
    main()
