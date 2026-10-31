"""Close the user-defined candidate window, then compare every eligible point."""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts.bomb_threat_round15 import read
from scripts import bomb_threat_a3sr_lineage_watch as L
from scripts.best_publish_selection_20261030 import select

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'runs/bomb_threat_v1'
O = B / 'publish_best_20261030'
W = B / 'iter2/a3sr/lineage_watch_20261030'
C = W / 'candidate_slot00_u2400_1440'
E = O / 'additional_eligible'


def register():
    p = O / 'CANDIDATE_ELIGIBILITY_AMENDMENT01.json'
    if not p.exists():
        atomic_json(p, dict(status='FROZEN_BEFORE_CUTOFF',
            instruction='.hermes/TASK_best_publish_and_cleanup_20261030_1530.md',
            clarification='Eligibility uses completion of the existing same-rule three-tier fresh protocol during the original1024-game confirmation window. Eligible additional points are re-evaluated on the exact new selection seeds; old scores are not pooled.',
            candidate_population=[0, 3, 4, 7], update=2400,
            start_file=str((C / 'STARTED.json').relative_to(ROOT)),
            cutoff_file=str((C / 'COMPLETE.json').relative_to(ROOT)),
            inclusion='all three tiers256 completed and completion unix strictly after STARTED and at/before original COMPLETE; no score threshold',
            extra_budget_per_eligible_model=768, workers=4, CPU_affinity='132-135', GPU_training_changes=[],
            release_gate='FINAL_SELECTION_COMPLETE.json, not the first two-model SELECTION_COMPLETE.json',
            maximum_additional_models=3, unix=time.time()))


def main():
    register()
    guard = lock(O / 'ELIGIBILITY_OWNER.lock')
    while not (C / 'COMPLETE.json').exists():
        atomic_json(O / 'ELIGIBILITY_STATUS.json', dict(status='WAITING_FROZEN_CUTOFF', unix=time.time()))
        time.sleep(5)
    start = read(C / 'STARTED.json')['unix']
    cutoff = read(C / 'COMPLETE.json')['unix']
    p = E / 'PROTOCOL.json'
    if not p.exists():
        base = read(O / 'SELECTION_PROTOCOL.json')
        models = {}
        evidence = []
        for slot in (0, 3, 4, 7):
            point = W / 'point_completions' / f'slot{slot:02d}_u002400_fresh.json'
            d = read(point, {})
            eligible = bool(d and start < d['unix'] <= cutoff)
            evidence.append(dict(slot=slot, eligible=eligible, completion=d or None))
            if eligible and slot != 0:
                models[f'slot{slot:02d}_u2400'] = read(W / 'models' / f'slot{slot:02d}_u002400.json')
        atomic_json(p, dict(**{k: base[k] for k in ('tiers', 'mode', 'engine', 'rule_version', 'teacher')},
                    models=models, status='FROZEN', candidate_window=dict(start=start, cutoff=cutoff),
                    evidence=evidence, original_complete_sha256=sha(C / 'COMPLETE.json'),
                    amendment_sha256=sha(O / 'CANDIDATE_ELIGIBILITY_AMENDMENT01.json'),
                    budget_games=768 * len(models), unix=time.time()))
    proto = read(p)
    L.OUT = E
    jobs = [dict(kind='paired_fresh', slot=model['slot'], update=2400, tier=tier,
                 start=i, games=256, seed=t['seed'], spawn_manifest=t['spawn_manifest'], label=label)
            for label, model in proto['models'].items() for tier, t in proto['tiers'].items()
            for i in range(0, 256, 8)]
    # Wait until the easy lane finishes: resource use replaces that lane rather
    # than adding another continuous evaluation pool beside healthy training.
    while not (O / 'SELECTION_COMPLETE.json').exists():
        time.sleep(5)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        while True:
            for f in list(futures):
                if f.done():
                    f.result()
                    del futures[f]
            for j in jobs:
                if len(futures) >= 4:
                    break
                if j in futures.values() or Path(str(L.shard_path(j)) + '.ACCEPTED.json').exists():
                    continue
                futures[pool.submit(L.collect, j, proto['models'][j['label']], proto)] = j
            results = {label: {tier: L.summarize_group([j for j in jobs if j['label'] == label and j['tier'] == tier], m)
                              for tier in proto['tiers']} for label, m in proto['models'].items()}
            n = sum(t['games'] for v in results.values() for t in v.values())
            atomic_json(O / 'ELIGIBILITY_STATUS.json', dict(status='RUNNING', completed_extra_games=n,
                        budget=proto['budget_games'], active=list(futures.values()), unix=time.time()))
            if n == proto['budget_games']:
                first = read(O / 'SELECTION.json')
                results.update(first['results'])
                decision = select(results)
                models = dict(read(O / 'SELECTION_PROTOCOL.json')['models'], **proto['models'])
                atomic_json(O / 'FINAL_SELECTION.json', dict(**decision, results=results, models=models,
                            eligibility_protocol_sha256=sha(p), first_selection_sha256=sha(O / 'SELECTION.json'),
                            total_games=1536 + n, unix=time.time()))
                atomic_json(O / 'FINAL_SELECTION_COMPLETE.json', dict(status='COMPLETE', winner=decision['winner'],
                            publish=decision['publish'], selection_sha256=sha(O / 'FINAL_SELECTION.json'), unix=time.time()))
                from scripts.iter_pbt_owner import notify
                notify('FINAL_BEST_SELECTION_20261030', f"候选时间窗已关闭，所有合资格点同新seed比较{1536+n}局完成；winner={decision['winner']}，publish={decision['publish']}。继续导出/浏览器/发布门禁。")
                return
            time.sleep(5)


if __name__ == '__main__':
    main()
