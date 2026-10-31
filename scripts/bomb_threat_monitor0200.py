"""Passive, bounded completion receipt for the authorized 02:00 monitoring round."""
import json
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
TARGETS = ('bt_slot03_c000025_u004800', 'bt_slot07_c000024_u000480')
NEIGHBORS = ('bt_slot03_c000025_u004760', 'bt_slot03_c000025_u004800',
             'bt_slot03_c000025_u004840', 'bt_slot07_c000024_u000440',
             'bt_slot07_c000024_u000480', 'bt_slot07_c000024_u000520')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def regression(points):
    """Strict thresholds; an unmeasured fourth point never becomes a verdict."""
    flagged = []
    for update in (6960, 7000, 7040, 7080):
        m = points.get(update, {})
        if m.get('games') != 64 or any(m.get(k) is None for k in ('causal_kills', 'zero_bomb_games')):
            return dict(status='INCOMPLETE', flagged_updates=flagged, required_points=4)
        if m['causal_kills'] < 10 or m['zero_bomb_games'] > 7:
            flagged.append(update)
    return dict(status='REGRESSION' if len(flagged) >= 3 else 'THRESHOLD_NOT_MET',
                flagged_updates=flagged, required_points=4,
                definition='at least3of4: causal_kills<10 OR opportunity-conditioned zero_bomb_games>7')


def measurement(name):
    from scripts.bomb_threat_gate_report import recompute_natural, reward_monitor
    from scripts.bomb_threat_gates_v2 import evaluate
    out = BASE / 'eval' / name
    marker_path = out / 'NATURAL_COMPLETE.json'
    if not marker_path.exists():
        return dict(status='PENDING', request=name)
    marker, natural = recompute_natural(out)
    split = marker['split']
    parent_marker, parent = recompute_natural(BASE / 'eval' / ('baseline_parent' if split == 'holdout' else 'baseline_parent_dev'))
    match_fields = ('rule_version', 'split', 'seed', 'spawn_sha256', 'simulator_sha256')
    assert all(marker.get(k) == parent_marker.get(k) for k in match_fields), name
    refs = {}
    for ref in ('reference_transport1536', 'reference_selfplay3072'):
        ref_marker, ref_values = recompute_natural(BASE / 'eval' / ref)
        if all(marker.get(k) == ref_marker.get(k) for k in match_fields):
            refs[ref] = ref_values
    cp = ROOT / marker['checkpoint']
    commit = read(cp / 'COMMITTED.json')
    assert commit['policy_sha256'] == marker['policy_sha256']
    monitor = reward_monitor(cp, int(cp.name[1:]))
    gates = evaluate(natural, parent, monitor, split, refs)
    return dict(status='COMPLETE', request=name, split=split, metrics=natural,
                gate_A=gates['relative'], gate_B=gates['absolute'],
                native_complete=(out / 'native_hunter256_v1/NATIVE_HUNTER_COMPLETE.json').exists(),
                policy_sha256=marker['policy_sha256'], natural_sha256=sha(marker_path),
                raw_hashes_verified=True, reward_monitor=monitor)


def continuity_audit():
    runs = ('bt_slot06_c000024', 'bt_slot06_c000025')
    rows = {}
    for run in runs:
        rows[run] = [json.loads(line) for line in (BASE / 'formal' / run / 'updates.jsonl').read_text().splitlines()]
    previous, current = rows[runs[0]][-40:], rows[runs[1]][:40]
    def metrics(values):
        import statistics
        ticks = sum(sum(r['exposure']['pool_ticks']) for r in values)
        pool = [sum(r['exposure']['pool_ticks'][i] for r in values) / ticks for i in range(6)]
        live = sum(sum(r['exposure']['pool_live_ticks']) for r in values)
        live_pool = [sum(r['exposure']['pool_live_ticks'][i] for r in values) / live for i in range(6)]
        return dict(updates=[values[0]['new_update'], values[-1]['new_update']],
                    configuration=values[-1]['configuration'],
                    loss_schema='legacy: total,ppo,value,entropy,approxKL,clip,grad,legacy-extra',
                    entropy_mean=statistics.mean(r['loss'][3] for r in values),
                    approximate_KL_mean=statistics.mean(r['loss'][4] for r in values),
                    pool_tick_fraction=pool, pool_live_tick_fraction=live_pool,
                    median_cycle_seconds=statistics.median(r['cycle_seconds'] for r in values))
    source = read(BASE / 'formal' / runs[1] / 'SOURCE_RESTORE.json')
    restore = read(BASE / 'formal' / runs[1] / 'RESTORE.json')
    a, b = metrics(previous), metrics(current)
    return dict(previous_last40=a, current_first40=b,
                configuration_equal=a['configuration'] == b['configuration'],
                source_restore=source, fresh_restore=restore,
                interpretation='Configuration equality/restore parity constrain hypotheses; RNG/opponent responses, pool ring refresh and policy drift remain possible. No causal attribution or automatic rollback.')


def snapshot():
    names = list(dict.fromkeys([n + '_holdout' for n in TARGETS] + list(NEIGHBORS) +
                              [f'bt_slot06_c000025_u{u:06d}' for u in (6960, 7000, 7040, 7080)]))
    results = {name: measurement(name) for name in names}
    points = {u: results[f'bt_slot06_c000025_u{u:06d}'].get('metrics', {}).get('greedy', {})
              for u in (6960, 7000, 7040, 7080)}
    verdict = regression(points)
    complete = all(r['status'] == 'COMPLETE' for r in results.values())
    # Native auxiliary completion is tracked separately; it never supplies JS gates.
    native_complete = all(results[n + '_holdout'].get('native_complete') for n in TARGETS)
    status = 'COMPLETED' if complete and native_complete else 'WAITING_EVALUATION'
    slots = {}
    for slot, entry in read(BASE / 'SLOTS.json').items():
        progress = read(BASE / 'formal' / entry['run'] / 'PROGRESS.json')
        last = progress.get('last', progress.get('metrics', {}))
        slots[slot] = dict(run=entry['run'], update=last.get('new_update'),
                           finite=last.get('finite'), update_l2=last.get('update_l2'),
                           env_per_second=(262144 / last['cycle_seconds']) if last.get('cycle_seconds') else None)
    record = dict(status=status, authorization='.hermes/STEER_monitor_20261030_0200.md',
                  measurements=results, slot06_regression=verdict, continuity=continuity_audit(),
                  active_slots=slots, no_pbt_copy=True, no_rollback=True, no_recipe_change=True,
                  publication=False, fresh256_started=False, training_uninterrupted=True,
                  priority_file='runs/bomb_threat_v1/GATE_AMEND2_HOLDOUT_PRIORITY.json', unix=time.time())
    atomic_json(BASE / 'MONITOR_0200_ACCEPTANCE.json', record)
    write_report(record)
    return record


def write_report(record):
    lines = ['# BOMB-THREAT round13 — 20261030', '',
             f'执行状态：{record["status"]}。按 `.hermes/STEER_monitor_20261030_0200.md` 只调整评估顺序与被动观察；7路训练不停，本轮不复制PBT、不回滚、不改配方、不发布。', '',
             '指定holdout置顶；CPU生产JS按argmax/categorical各64局，GPU2原生辅助按原冻结协议每模式256局，新arm另含τ0.5，二者不能互代。正在执行的评估不中断。持久优先清单由报告刷新保留，避免自动排名覆盖人工指令。', '',
             '| checkpoint | split/mode | 胜/平/负 | 胜−负 | 因果击杀 | 死亡 | 泡/局 | 单方零泡 | 平局击杀差 | A相对 / B绝对 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for name, result in record['measurements'].items():
        if result['status'] != 'COMPLETE':
            lines.append(f'| {name} | PENDING | — | — | — | — | — | — | — | 未完成 |')
            continue
        for mode in ('greedy', 'categorical'):
            m = result['metrics'][mode]
            gates = f'{result["gate_A"]["status"]} / {result["gate_B"]["status"]}' if mode == 'greedy' else '辅助观察；门禁以argmax为主'
            lines.append(f'| {name} | {result["split"]}/{mode} | {m["win"]}/{m["draw"]}/{m["loss"]} | {m["win_minus_loss"]:+d} | {m["causal_kills"]} | {m["deaths"]} | {m["bombs_per_game"]:.2f} | {m["zero_bomb_games"]} | {m["draw_kill_difference"]:+d} | {gates} |')
    lines += ['', '门禁由hash匹配的原始outcome分片复算，A为gate-amend2 relative，B为absolute；dev不具备B所需独立holdout。邻点u4760/u4840及u440/u520完整列出，不将单个峰值称作稳定改善。', '']
    for name in TARGETS:
        r = record['measurements'][name + '_holdout']
        if r['status'] == 'COMPLETE':
            lines += [f'{name} holdout：A={json.dumps(r["gate_A"], ensure_ascii=False)}；B={json.dumps(r["gate_B"], ensure_ascii=False)}。原生辅助完成={r["native_complete"]}。']
        else:
            lines += [f'{name} holdout：尚未完整，不能宣称门禁通过。']
    v = record['slot06_regression']
    lines += ['', f'slot06观察：**{v["status"]}**，触发点{v["flagged_updates"]}。判据严格为四点中至少三点“击杀<10 或 单方零泡>7”，击杀10/零泡7不算触发；四点未齐不提前定性。只报告，由编排方决定是否干预。', '']
    audit = record['continuity']; a, b = audit['previous_last40'], audit['current_first40']
    lines += [f'chunk衔接审计：c24末40更新与c25首40更新记录配置相同={audit["configuration_equal"]}；学习率分别{a["configuration"]["lr"]}/{b["configuration"]["lr"]}、固定熵系数{a["configuration"]["entropy_coefficient"]}/{b["configuration"]["entropy_coefficient"]}；实际loss中的熵均值{a["entropy_mean"]:.6f}→{b["entropy_mean"]:.6f}，approxKL {a["approximate_KL_mean"]:.6f}→{b["approximate_KL_mean"]:.6f}。该legacy实现使用常数Adam学习率，未观察到chunk重置学习率退火证据。',
              f'实际对手pool ticks占比：{a["pool_tick_fraction"]}→{b["pool_tick_fraction"]}；live占比、source/fresh恢复原始字段完整保存在验收JSON。相同配比不等于相同ring策略/状态访问，ring更新、对手响应、随机流与策略漂移仍是待证假设；不能把时间相邻等同根因。', '',
              '本轮不复制PBT：slot03相对slot01虽胜负差更高，但放泡/单方零泡未满足不变差条件。A3SR按固定640预算继续；u40/u80短点不作终态判断。B未通过不启动fresh256；即使B通过也须另立冻结细则，本监控程序不会启动实验或晋升。', '',
              '| slot | 当前run | update | finite | env/s |', '|---|---|---:|---|---:|']
    for slot, r in sorted(record['active_slots'].items()):
        lines.append(f'| {slot} | {r["run"]} | {r["update"]} | {r["finite"]} | {r["env_per_second"] or 0:.1f} |')
    lines += ['', '验收与可复算证据：`runs/bomb_threat_v1/MONITOR_0200_ACCEPTANCE.json`；调度回归：`runs/bomb_threat_v1/logs/monitor0200_focused_GREEN.log`。报告是被动CPU任务，不阻塞GPU或原评估owner。', '']
    path = ROOT / 'reports/bomb_threat_v1_round13_20261030.md'
    tmp = path.with_suffix('.md.tmp'); tmp.write_text('\n'.join(lines)); tmp.replace(path)


def main():
    guard = lock(BASE / 'MONITOR_0200.lock')
    from scripts.iter_pbt_owner import notify
    started = time.time()
    while True:
        record = snapshot()
        if record['status'] == 'COMPLETED':
            verdicts = ', '.join(n + ':A=' + record['measurements'][n+'_holdout']['gate_A']['status'] +
                                '/B=' + record['measurements'][n+'_holdout']['gate_B']['status'] for n in TARGETS)
            notify('BOMB_THREAT_MONITOR0200_COMPLETE',
                   f'round13完成：{verdicts}；slot06 {record["slot06_regression"]["status"]}。训练不停，本轮不复制PBT/不发布。',
                   report='reports/bomb_threat_v1_round13_20261030.md')
            return
        if time.time() - started > 7200:
            atomic_json(BASE / 'MONITOR_0200_DELAY.json', dict(status='BLOCKED_EVALUATION_DELAY',
                        receipt='runs/bomb_threat_v1/MONITOR_0200_ACCEPTANCE.json', unix=time.time()))
            notify('BOMB_THREAT_MONITOR0200_DELAY', 'round13评估等待超过2小时，需核验队列；训练未被监控中断，详情见持久验收。')
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
