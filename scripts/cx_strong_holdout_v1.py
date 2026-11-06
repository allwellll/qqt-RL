"""One frozen-checkpoint holdout plus same-batch G_frozen reference; CPU slots only."""
import json
import time
from pathlib import Path

from scripts import cc_strong_retest_v1 as R
from scripts import cc_strong_eval_v1 as E
from scripts.population_run_state import atomic_json, sha

OUT = E.OUT / 'cx_holdout_20261106_v1'
SPECS = ['G_frozen', 'L2_S05h@u020742']
BASE = 2026616000
RULE = ('Report both modes/seats separately and paired outcomes against G_frozen self-play on identical new seeds. '
        'Selection key: net wins, deliveries, causal kills, negative self-deaths. '
        'No automatic promotion, no old categorical .386 threshold, no bubble-count selection.')


def check_seeds(new):
    # Inspect the declared campaign protocols directly. Never recursively walk checkpoint/replay trees.
    campaign = E.ROOT
    dirs = [campaign/'training/protocols', campaign/'eval_protocol_v1', E.OUT]
    dirs += [d for d in E.OUT.iterdir() if d.is_dir() and d != OUT]
    paths = sorted({p for d in dirs for p in d.glob('*.json') if 'protocol' in p.name.lower()})
    used = set()
    for path in paths:
        stack = [json.loads(path.read_text())]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('seed', 'action_seed') and isinstance(item, int):
                        used.add(item)
                    if key == 'seed_range' and isinstance(item, list) and len(item) == 2:
                        assert not any(item[0] <= seed <= item[1] for seed in new), str(path)
                    stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
    assert not new & used, 'holdout range conflicts with a declared campaign protocol'
    return [str(p) for p in paths]


def freeze():
    f = OUT / 'PROTOCOL.json'
    if f.exists():
        p = json.loads(f.read_text())
        assert p['source_protocols'] == dict(seat0=sha(E.PROTOCOL), seat1=sha(E.PROTOCOL_P1))
        assert p['decision_rule'] == RULE and p['points'] == SPECS
        return p
    new = set(range(BASE, BASE+88))
    inventory = check_seeds(new)
    games = {}
    for seat, source in ((0, E.PROTOCOL), (1, E.PROTOCOL_P1)):
        scenarios = [g for g in json.loads(source.read_text())['games'] if g['mode'] == 'greedy']
        assert len(scenarios) == 44
        games[str(seat)] = [dict(g, seed=BASE+2*i+r, action_seed=BASE+2*i+r,
                                mode=mode, seat=seat, scenario_index=i, replica=r,
                                evaluation_split='cx_holdout_20261106_v1')
                           for i, g in enumerate(scenarios) for r in (0, 1) for mode in ('greedy', 'categorical')]
    p = dict(status='FROZEN_BEFORE_ANY_GAME', source_protocols=dict(seat0=sha(E.PROTOCOL), seat1=sha(E.PROTOCOL_P1)),
             g_frozen=dict(file=E.G_FROZEN, sha256=sha(E.G_FROZEN)), seed_range=[BASE, BASE+87],
             seeds_checked_unused_in_declared_campaign_protocols=True, checked_protocols=inventory,
             points=SPECS, decision_rule=RULE, games_per_point=352, games=games,
             source_checkpoint=E.resolve(SPECS[1]), created_unix=time.time())
    atomic_json(f, p)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R.OUT = OUT; R.PROTOCOL = OUT / 'PROTOCOL.json'; R.BASE = BASE; R.SPECS = SPECS
    R.freeze = freeze
    atomic_json(OUT / 'STATUS.json', dict(status='FREEZING_CHECKING_CAMPAIGN_PROTOCOLS', unix=time.time(), cpu_only=True))
    p = freeze()
    atomic_json(OUT / 'STATUS.json', dict(status='RUNNING', unix=time.time(), protocol_sha256=sha(R.PROTOCOL),
                                         specs=SPECS, cpu_only=True, owner='CX', decision_rule=RULE))
    R.run(SPECS)
    result = dict(status='COMPLETE', protocol_sha256=sha(R.PROTOCOL), decision_rule=RULE, models={})
    for spec in SPECS:
        model = E.resolve(spec)
        rows = R.rows_of([R.pair_dir(model, s) / 'SUMMARY.json' for s in (0, 1)])
        result['models'][model['label']] = {mode: R.stats([r for r in rows if r['mode'] == mode])
                                           for mode in ('greedy', 'categorical')}
    atomic_json(OUT / 'SUMMARY.json', result)
    atomic_json(OUT / 'STATUS.json', dict(status='COMPLETE', unix=time.time(), summary=str(OUT/'SUMMARY.json')))
    lines = ['\n## h u020742 独立新seed复测（20261106，自动完成记录）\n',
             '同批G_frozen参考；完整砖/部分砖/位置/双席位/两种模式，352局/模型；不使用旧.386阈值、不自动晋升。\n']
    for label, modes in result['models'].items():
        for mode, r in modes.items():
            lines.append(f"- {label} {mode}: {r['wins']}胜/{r['losses']}负/{r['draws']}平；交付{r['deliveries']}，因果杀{r['causal_kills']}，自死{r['self_deaths']}；score={r['score']:.4f}±{r['se']:.4f}。\n")
    with Path('reports/cx_takeover_training_20261106.md').open('a') as report:
        report.write('\n'.join(lines))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
