"""Render dated JE10 tables from the verified public snapshot, without run access."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('native_death_stock', 'native_sugar_stock', 'real_drop', 'carry_near',
         'carry_mid', 'carry_far', 'recovery_threat', 'natural_full2400')
METHODS = [
    '首次受控4/残余.25原生接线；PBRS .5；DAgger .2与完整custody（整合）',
    'reset mix改为.35/.20/.35/.10',
    '复用native一步端点纠正合法安全最大正进度集合',
    'PBRS/oracle共享连续三角插值场',
    '己方loose→拾回→回笼的PBRS目标链对齐',
    '共享prestock corridor目标，phase5/6与native-safe joint DAgger',
    'current/history内分阶段回放20 ordinary/10 navigation/2 opening',
]


def hunter(point):
    return '/'.join(str(point['hunter_js'][k]['safe_win']) for k in ('hard', 'normal', 'easy'))


def deliveries(point):
    return '/'.join(str(point['routes'][k]['counts']['delivery_episodes']) for k in TASKS)


def replace_block(text, name, content):
    start = f'<!-- JE10-{name}:BEGIN -->'
    end = f'<!-- JE10-{name}:END -->'
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + '\n\n' + content + '\n\n' + end + after


def render(root=ROOT):
    evidence = json.loads((root / 'reports/je10_evidence_20261018.json').read_text())
    manifest = json.loads((root / 'reports/je10_source_manifest_20261018.json').read_text())
    rounds = evidence['rounds']
    active = rounds[-1]
    last = active['points'][-1]
    cutoff = datetime.fromisoformat(evidence['read_completed_utc']).astimezone(timezone(timedelta(hours=8)))
    cutoff = cutoff.strftime('%Y-%m-%d %H:%M:%S UTC+8')
    status = [f'**证据读取截至 {cutoff}**。这是读取快照，不代表此后训练静止。', '',
        '| 身份 | 检查点 / fresh ticks | 固定能力与状态 |', '|---|---|---|',
        '| 网页当前发布综合最佳 | r01 attempt04 g8 raw / 16,777,216 | 既有发布规则选中的折中点；未随后续轮自动更新 |',
        '| 最近完整轮 | r06 g128 / 268,435,456 | 8个预声明固定点评估全部COMPLETE；full交付1/16、drop5/16、Tactical17/64 |',
        f"| 正在训练轮 | r07，STATE g{active['state']['generation']} / {active['state']['fresh_ticks']:,} | finite/nonzero={active['state']['finite']}/{active['state']['nonzero_update']}；最新COMMITTED g{active['latest_checkpoint']['generation']} / {active['latest_checkpoint']['fresh_ticks']:,}，与评估点分列 |",
        f"| r07 最新完整固定评估 | g{last['generation']} / {last['fresh_ticks']:,} | Hunter难/中/易{hunter(last)}（各64）；Tactical{last['tactical_python']['safe_win']}/64；八类交付{deliveries(last)}（各16） |", '',
        f"当前轮已完成固定点：{', '.join('g'+str(p['generation']) for p in active['points'])}；未完成的预声明点评估不进入结论。八路线顺序为 death/sugar/drop/near/mid/far/recovery/full。"]
    table = ['| 轮次/所报检查点 | 相对前轮的方法 | Hunter难/中/易 / 各64 | Tactical /64 | D/S/Drop/Near/Mid/Far/Recovery/Full，各16 |', '|---|---|---|---:|---|']
    for index, row in enumerate(rounds):
        point = row['points'][-1]
        name = f"r{index+1:02d} g{point['generation']}" + ('（未完轮）' if not row['terminal'] else '')
        table.append(f"| {name} | {METHODS[index]} | {hunter(point)} | {point['tactical_python']['safe_win']} | {deliveries(point)} |")
    exp = ['| 点与来源口径 | current | ring | fixed league | 分母 |', '|---|---:|---:|---:|---|']
    for label, point in [('r06 g128', rounds[5]['points'][-1]), (f"r07 g{last['generation']}", last)]:
        raw = point['exposure']
        counts = [raw[k] for k in ('current', 'ring', 'fixed_league')]
        denominator = sum(counts)
        exp.append('| '+label+' 底层league内部条件份额 | '+' | '.join(f'{v/denominator:.4%}' for v in counts)+f' | {int(denominator):,} 底层league ticks |')
        executed = raw['executed_after_overrides']
        if executed:
            denominator = executed['valid_decision_ticks']
            exp.append('| '+label+' 最终实际action source | '+' | '.join(f'{executed[k]:,} ({executed[k]/denominator:.4%})' for k in ('current','ring','fixed_league'))+f' | {denominator:,} alive-learner有效决策ticks |')
    main_path = root / 'docs/je10_control_route_training_summary.md'
    main = main_path.read_text()
    for key, content in [('STATUS', status), ('ROUNDS', table), ('EXPOSURE', exp)]:
        main = replace_block(main, key, '\n'.join(content))
    main_path.write_text(main)
    ledger_path = root / 'docs/training_experiment_ledger.md'
    ledger = ledger_path.read_text()
    if '<!-- JE10-LEDGER:BEGIN -->' in ledger:
        ledger_body = [
            '### JE10 v38 击杀信用与搬包路线（2026-10-18，持续研究）', '',
            f'证据截至 **{cutoff}**。长期解释与复现边界见[JE10训练总结](je10_control_route_training_summary.md)，全部固定点/曲线见[当日报告](../reports/je10_training_20261018.md)，哈希及分母见[冻结摘要](../reports/je10_evidence_20261018.json)。', '',
            '本条选择性整合训练账本的JE10方案与逐轮终态，保留本账本既有实验。网页当前发布仍为r01 attempt04 g8 raw；最近完整轮为r06 g128。', '',
            f"正在训练r07：STATE g{active['state']['generation']}，COMMITTED g{active['latest_checkpoint']['generation']}；最新完整固定评估g{last['generation']}。运行状态与固定能力分列，不把训练事件当泛化。", '',
            'r01整合首控4/主动残余.25、受控legacy16去重、连续PBRS .5、set-valued learner-state DAgger .2及完整custody；r02只改reset mix为.35 combat/.20 bridge/.35 full/.10其他原生路线；r03 native端点progress set；r04共享连续场；r05己方回收目标链对齐；r06统一prestock corridor与joint teacher；r07分阶段current/history回放。每轮8×1024×256、128代/268,435,456 fresh ticks，g0/4/8/16/32/64/96/128固定协议不变。', '',
            'r06终态natural full1/16、drop5/16、far7/16、recovery0/16，对比父点7/16、10/16、13/16、5/16均退化；Tactical从8/64升17/64，Hunter难/中/易1/3/11（各64）。原墙库存三层pickup8/8仍delivery0/8；不使用战斗末点或JAX训练事件覆盖路线失败。阶段回放是后续诊断假设，顺序迭代不构成同父单变量因果。', '',
            '按用户覆盖策略，JE10不设置能力/admission/promotion门禁；中间诊断不改变健康固定预算或阻塞后继，技术有效性另查。r08是同配方同预算连续后备预声明，尚未运行；具体下一因素须由完成点证据支持。完整启动/历史bootstrap和外部资产未入main，公开源码边界见总结，不能假称开箱可续训。', '',
            'v37技术无效attempt、INVALID_FIXTURE/INVALID_METRICS不纳入v38能力表。网页发布选择、最新完整轮与进行轮不得混称综合最佳。']
        ledger_path.write_text(replace_block(ledger, 'LEDGER', '\n'.join(ledger_body)))
    lines = ['# JE10 v38 冻结训练与文档集成报告（20261018）', '',
        f'证据读取截至 **{cutoff}**；本报告由冻结SUMMARY/COMPLETE重新生成，没有复制仍由writer维护的训练报告。长期解释见[训练总结](../docs/je10_control_route_training_summary.md)，逐点数据和原始产物哈希见[冻结摘要](je10_evidence_20261018.json)。', '',
        '## 固定点评估全表', '',
        '所有点均核验COMPLETE全部shard哈希及policy身份。Hunter难/中/易和Python Tactical各64局；八路线交付各16局，顺序D/S/Drop/Near/Mid/Far/Recovery/Full。self为Hunter三档、Tactical的自伤局，分别报告；路线自伤另列总事件（不混作战斗率）。全表未择优。', '',
        '| 轮次 | g | Hunter难/中/易，各64 | Tactical/64 | 八路线交付，各16 | self H难/中/易; T | 路线self事件 |', '|---|---:|---|---:|---|---|---:|']
    for index, row in enumerate(rounds):
        for p in row['points']:
            hself='/'.join(str(p['hunter_js'][t]['self_episodes']) for t in ('hard','normal','easy'))
            route_self=sum(p['routes'][t]['counts']['self_death'] for t in TASKS)
            lines.append(f"| r{index+1:02d} | {p['generation']} | {hunter(p)} | {p['tactical_python']['safe_win']} | {deliveries(p)} | {hself}; {p['tactical_python']['self_episodes']} | {route_self} |")
    lines += ['', '## 预算与终态', '', '| 轮次 | train / technical seed | STATE代数 | 最新COMMITTED | compute秒 / terminal wall秒 |', '|---|---|---:|---:|---|']
    for index, row in enumerate(rounds):
        terminal=row['terminal']
        times=f"{terminal['compute_seconds']:.3f} / {terminal['wall_seconds']:.3f}" if terminal else '运行中，不报完整轮耗时'
        lines.append(f"| r{index+1:02d} | {row['seed']} / {row['technical_seed']} | {row['state']['generation']} | {row['latest_checkpoint']['generation']} | {times} |")
    lines += ['', 'r01–r06各完整128代/268,435,456 fresh ticks；r07预算相同，STATE采样值和COMMITTED不同步时按各自标记报告。终态wall与STATE末代wall含不同收尾范围，不能替换。后继遵循前轮final raw Actor/Critic/pool，并fresh Adam/RNG/env/FIFO，非同父随机对照。', '',
        '## 已完成轮曲线', '', '以下从[冻结绘图输入](je10_curve_data_20261018.json)重生成；48个点逐项核对冻结SUMMARY，包含全部预声明固定点。图内JAX训练事件以full有效ticks归一，与JS/Python评估分列。r07未完成，不伪造完整曲线。', '']
    for i in range(1,7):
        lines += [f'![r{i:02d}完整固定曲线](je10_r{i:02d}_curve_20261018.png)', '']
    lines += ['## 源码集成与验证', '',
        '逐文件依赖由AST（含相对import）、JS require及显式runtime文件读取构成；manifest记录每条直接边、required_by和source/published哈希。表内包含已有main依赖用于闭包审查，不代表这些已有文件都被修改。四项历史RED断言改为当前数值行为验收，另将跨测试文件的scalar fixture内联；生产数值源码按稳定字节移植。', '',
        '| 文件 | 来源 | 直接依赖数 |', '|---|---|---:|']
    for f, record in manifest['files'].items():
        lines.append(f"| [{f}](../{f}) | {record['origin']} | {len(record['dependencies'])} |")
    lines += ['', '完整启动器、历史recipe/authority/bootstrap链和外部训练资产未纳入；不能由main独立启动完整JE10长训或Python全套异步owner。具体缺口和可运行的CPU/JS范围见长期总结。未移植网页脏改动、run/checkpoint、权重、缓存、锁/PID、临时产物或无关旧实验脚本。', '',
        '<!-- JE10-VALIDATION:BEGIN -->', '<!-- JE10-VALIDATION:END -->', '']
    report_path=root/'reports/je10_training_20261018.md'
    report='\n'.join(lines)
    if report_path.exists():
        old=report_path.read_text()
        a='<!-- JE10-VALIDATION:BEGIN -->';b='<!-- JE10-VALIDATION:END -->'
        if a in old and b in old:
            report=replace_block(report,'VALIDATION',old.split(a,1)[1].split(b,1)[0].strip())
    report_path.write_text(report)
    print(f"rendered {sum(len(r['points']) for r in rounds)} points at {cutoff}")


if __name__ == '__main__':
    render()
