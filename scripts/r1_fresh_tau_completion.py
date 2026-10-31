"""Resume the frozen fresh matrix and produce an auditable report."""
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts import r1_fresh_tau_0735 as E
from scripts.population_run_state import atomic_json, lock, sha


def checked_pair(a, b):
    for rows in (a, b):
        assert len({(r['seed'], r['seat']) for r in rows}) == len(rows)
    return E.paired(a, b)


def validate(j, p):
    rows = E.validate(j, p)
    doc = E.read(E.job_path(j))
    original = doc['results']['natural_full2400']['episodes']
    assert len(original) == len(rows)
    for a, b in zip(original, rows):
        assert (a['seed'], a['team'], a['ticks']) == (b['seed'], b['seat'], b['ticks'])
    assert sha(Path(p['tiers'][j['tier']]['spawn_manifest'])) == p['tiers'][j['tier']]['spawn_sha256']
    return rows


def context():
    roots = [(E.BASE, 'SLOTS.json'), (Path('runs/bun_native_v2_20261030'), 'R2_SLOTS.json'),
             (Path('runs/r3_items_route_20261030'), 'SLOTS.json')]
    members = {}
    for root, filename in roots:
        path = root / filename
        if not path.exists():
            continue
        for slot, entry in E.read(path).items():
            file = root / 'formal' / entry['run'] / 'updates.jsonl'
            if file.exists():
                rows = [json.loads(x) for x in file.read_text().splitlines()]
                if rows and time.time() - rows[-1]['unix'] < 180:
                    members.setdefault(str(entry['gpu']), []).append(entry['run'])
    return {k: sorted(v) for k, v in members.items()}


def comparison(now, references):
    return {k: v['env_per_second'] / references[k]['env_per_second']
            for k, v in now.items() if k in references and v['run'] == references[k]['run']}


def table(metrics):
    lines = ['|难度/模式/席位|局数|W/D/L|WL|因果击杀|泡/局|单方零泡|双方零泡|自伤|死亡|平局率|',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for tier, groups in metrics.items():
        for label, aggregate in groups.items():
            for seat, v in [('全部', aggregate), *aggregate['by_seat'].items()]:
                lines.append(f"|{tier}/{label}/{seat}|{v['games']}|{v['win']}/{v['draw']}/{v['loss']}|"
                             f"{v['win_minus_loss']:+d}|{v['causal_kills']}|{v['bombs_per_game']:.3f}|"
                             f"{v['zero_bomb_games']}|{v['structural_zero_bomb_games']}|{v['self_deaths']}|"
                             f"{v['deaths']}|{v['draw_fraction']:.4f}|")
    return lines


def report(result, receipt):
    p = E.read(E.OUT / 'PROTOCOL.json')
    death = E.read(E.OUT / 'SLOT07_RECENT8_DEATHS.json')
    lines = ['# R1 slot05 fresh256 与温度诊断（20261031）', '',
             f"状态：{result['status']}；{result['games']}/2304 局。训练持续；未做PBT或发布，网页模型及解码未改。", '',
             '## 冻结范围', '',
             '候选为首次消费指令时最新完整40倍数点 c45/u6640，不按分数选峰；对照按指令固定为 A3SR c28/u1280。', '']
    for name, m in p['models'].items():
        lines += [f"- {name}：`{m['checkpoint']}`；policy SHA `{m['policy_sha256']}`；导出 SHA `{m['model_sha256']}`。"]
    lines += ['', '生产 JS Hunter hard/normal/easy 各256局、每档P0/P1各128；greedy候选/对照及候选tau0.5，共2304局。带砖、2400tick、stock-window-v1、老师关闭。',
              '每档新seed分别从2026311000/2026312000/2026313000开始；复用64个出生几何，不是256张独立地图。动作与Hunter使用各自固定随机流；策略分歧后状态访问不同。tau0.5仅诊断，闸门仍只看argmax。', '',
              '## 完整结果', '', *table(result['metrics']), '',
              '单方零泡=己方0泡且对手>0；双方零泡=两方实际放泡均0；二者相加才是己方原始零泡。因果击杀沿用冻结ControlFunnel，不能把所有敌方死亡算作击杀。自伤列为ownBombDefeat死亡次数，非全部非致死伤害。', '',
              '## 配对差', '', '|难度/候选模式|局数|胜负分差总和|更好/相同/更差|击杀差|泡差|死亡差|自伤差|',
              '|---|---:|---:|---|---:|---:|---:|---:|']
    for tier, groups in result['paired_per_game'].items():
        for label, rows in groups.items():
            total = lambda key: sum(r[key] for r in rows)
            changes = [sum((r['score_difference'] > 0) if sign == 1 else (r['score_difference'] < 0) if sign == -1 else r['score_difference'] == 0 for r in rows) for sign in (1, 0, -1)]
            lines.append(f"|{tier}/{label}|{len(rows)}|{total('score_difference'):+d}|{'/'.join(map(str, changes))}|{total('causal_kill_difference'):+d}|{total('bomb_difference'):+d}|{total('death_difference'):+d}|{total('self_death_difference'):+d}|")
    lines += ['', '逐局及分席位可重算数据：`runs/bomb_threat_v1/fresh256_slot05_tau_20261031/RESULTS.json`。胜负分定义胜+1/平0/负−1。配对表是固定seed描述性比较，不宣称独立地图泛化或统计显著。', '',
              '## slot07 与资源', '',
              f"slot07最近8点死亡收据：`runs/bomb_threat_v1/fresh256_slot05_tau_20261031/SLOT07_RECENT8_DEATHS.json`。原始死亡7/11/10/11/14/12/12/16，均值11.625，未超过20告警线。收据内容：`{json.dumps(death, ensure_ascii=False)}`。", '',
              '评估仅CPU。原6路护栏观察到slot07吞吐18k→约4.7k，自动降并发；同期R2-c恢复共卡，不能归因CPU评估。保留原TRAINING_BEFORE/EXECUTION_STATUS，amendment01按实际共卡上下文重建对照；每批检查15%护栏。',
              f"最终调度与吞吐证据：`runs/bomb_threat_v1/fresh256_slot05_tau_20261031/SCHEDULER_STATUS.json`；耗时 {receipt['wall_seconds']:.1f} 秒（续接阶段），worker上限 {receipt['max_workers']}。", '',
              '## 复现与验收', '',
              '`taskset -c 112-127,144-159 .venv/bin/python -m scripts.r1_fresh_tau_completion`', '',
              '协议/输入/各分片SHA、唯一seed/席位、完整终局、原生逐局统计对齐均检查；已有ACCEPTED分片复用，不重跑。COMPLETION_ACCEPTANCE.json记录完整SHA与验收。未删除任何checkpoint。', '',
              '## 结论', '']
    totals = {label: sum(result['metrics'][t][label]['win_minus_loss'] for t in p['tiers']) for label in ('candidate_greedy', 'baseline_greedy', 'candidate_tau0.5')}
    lines += [f"三档WL合计：候选argmax {totals['candidate_greedy']:+d}，u1280对照 {totals['baseline_greedy']:+d}，候选tau0.5 {totals['candidate_tau0.5']:+d}。该结论只适用于本组固定生产JS协议；本任务不执行晋升、网页切换或PBT。"]
    E.REPORT.write_text('\n'.join(lines) + '\n')


def main():
    p = E.freeze()
    guard = lock(E.OUT / 'OWNER.lock')
    assert all(sha(Path(f)) == h for f, h in p['source_sha256'].items())
    amendment = E.OUT / 'SCHEDULING_AMENDMENT01.json'
    if not amendment.exists():
        atomic_json(amendment, dict(status='FROZEN', change='CPU scheduling and acceptance only',
            parent_protocol_sha256=sha(E.OUT / 'PROTOCOL.json'), max_workers=12,
            cpu_affinity='112-127,144-159', no_scientific_changes=True,
            guard='reduce workers on >15% same-context throughput loss; GPU cohort changes reset reference', unix=time.time()))
    start = time.time()
    workers = 12
    refs = E.training_snapshot()
    cohort = context()
    history = []
    pending = [j for j in E.jobs(p) if not Path(str(E.job_path(j)) + '.ACCEPTED.json').exists()]
    while pending:
        batch, pending = pending[:workers], pending[workers:]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(E.collect, j, p) for j in batch]):
                f.result()
        result = E.summarize_results(p)
        now, current = E.training_snapshot(), context()
        ratios = comparison(now, refs) if current == cohort else {}
        if current != cohort:
            refs, cohort = now, current
        if ratios and min(ratios.values()) < .85:
            workers = max(1, workers // 2)
        entry = dict(unix=time.time(),completed_games=result['games'],workers=workers,
                     training_ratios=ratios,training_current=now,gpu_cohort=current)
        history.append(entry)
        atomic_json(E.OUT / 'SCHEDULER_STATUS.json', dict(status='RUNNING', history=history))
        print(json.dumps(dict(games=result['games'],workers=workers,ratios=ratios)),flush=True)
    for j in E.jobs(p):
        validate(j, p)
    result = E.summarize_results(p)
    assert result['games'] == 2304 and result['status'] == 'COMPLETE'
    for tier in p['tiers']:
        for c in p['conditions']:
            rows = [r for start in range(0, 256, 8) for r in validate(dict(**c,tier=tier,start=start),p)]
            assert len({(r['seed'],r['seat']) for r in rows}) == 256
            assert [sum(r['seat'] == s for r in rows) for s in (0,1)] == [128,128]
    receipt = dict(status='COMPLETE',games=2304,shards=288,wall_seconds=time.time()-start,max_workers=12,
                   protocol_sha256=sha(E.OUT/'PROTOCOL.json'),results_sha256=sha(E.OUT/'RESULTS.json'),
                   scheduler_sha256=sha(E.OUT/'SCHEDULER_STATUS.json'),unix=time.time())
    report(result, receipt)
    receipt['report_sha256'] = sha(E.REPORT)
    atomic_json(E.OUT/'COMPLETION_ACCEPTANCE.json',receipt)
    atomic_json(E.OUT/'COMPLETE.json',receipt)
    atomic_json(E.OUT/'EXECUTION_STATUS.json',dict(receipt,training_current=E.training_snapshot(),no_publish=True))


if __name__ == '__main__':
    main()
