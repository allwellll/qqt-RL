"""Bounded CPU-only completion of the authorized three-tier paired selection."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts.bomb_threat_round15 import read
from scripts import bomb_threat_a3sr_lineage_watch as L

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'runs/bomb_threat_v1'
O = B / 'publish_best_20261030'
C = L.OUT / 'candidate_slot00_u2400_1440'


def freeze():
    O.mkdir(parents=True, exist_ok=True)
    p = O / 'SELECTION_PROTOCOL.json'
    if p.exists():
        return read(p)
    previous = read(C / 'PROTOCOL.json')
    spawn = read(ROOT / read(L.OUT / 'PROTOCOL.json')['fresh']['tiers']['easy']['spawn_manifest'])
    spawn['seed'] = 2026195000
    atomic_json(O / 'easy_spawns.json', spawn)
    tiers = dict(previous['tiers'])
    tiers['easy'] = dict(seed=2026195000, games=256, seats={'P0': 128, 'P1': 128},
                         spawn_manifest=str((O / 'easy_spawns.json').relative_to(ROOT)),
                         spawn_sha256=sha(O / 'easy_spawns.json'))
    d = dict(status='FROZEN', instruction='.hermes/TASK_best_publish_and_cleanup_20261030_1530.md',
             instruction_sha256=sha(ROOT / '.hermes/TASK_best_publish_and_cleanup_20261030_1530.md'),
             amendment='Existing 1024-game matrix lacks easy. Add 512 paired easy games prospectively; reuse exact original hard/normal shards. Do not mix old seeds.',
             models=previous['models'], tiers=tiers, mode='greedy', engine='production_JS',
             rule_version='stock-window-v1', teacher=False, games_per_model=768,
             original_matrix=str(C.relative_to(ROOT)), original_protocol_sha256=sha(C / 'PROTOCOL.json'),
             candidate_cutoff='original 1024-game COMPLETE timestamp; only points completed on the same new-seed three-tier protocol are eligible',
             primary='P1 sum(W-L) across hard/normal/easy',
             guards={'each_tier_P1_delta_min': -5, 'sum_P0_delta_min': -10, 'self_deaths_delta_max': 0},
             tie='retain current online baseline', CPU_affinity='128-131', unix=time.time(),
             limitations=['64 frozen geometries repeated; 256 games per tier are not 256 independent maps',
                          'selection/dev comparison does not establish general superiority against human players'])
    atomic_json(p, d)
    atomic_json(O / 'TASK_CONSUMED.json', dict(status='EXECUTING', protocol_sha256=sha(p), training_untouched=True, unix=time.time()))
    return d


def select(results):
    baseline = results['baseline']
    detail = {}
    for label, tiers in results.items():
        p1 = {t: x['by_seat']['P1']['win_minus_loss'] for t, x in tiers.items()}
        p0 = sum(x['by_seat']['P0']['win_minus_loss'] for x in tiers.values())
        own = sum(x['self_deaths'] for x in tiers.values())
        guards = dict(each_P1=all(p1[t] >= baseline[t]['by_seat']['P1']['win_minus_loss'] - 5 for t in tiers),
                      P0=p0 >= sum(x['by_seat']['P0']['win_minus_loss'] for x in baseline.values()) - 10,
                      self_deaths=own <= sum(x['self_deaths'] for x in baseline.values()))
        detail[label] = dict(P1=p1, P1_sum=sum(p1.values()), P0_sum=p0, self_deaths=own, guards=guards,
                             eligible=all(guards.values()))
    winner = 'baseline'
    for label, d in detail.items():
        if d['eligible'] and d['P1_sum'] > detail[winner]['P1_sum']:
            winner = label
    return dict(status='PASS', winner=winner, publish=winner != 'baseline', comparison=detail)


def main():
    guard = lock(O / 'OWNER.lock')
    proto = freeze()
    if not (O / 'PROTOCOL.json').exists():
        atomic_json(O / 'PROTOCOL.json', proto)
    L.OUT = O
    jobs = [dict(kind='paired_fresh', slot=0 if label == 'candidate' else 5,
                 update=2400 if label == 'candidate' else 1280, tier='easy', start=i,
                 games=256, seed=2026195000, spawn_manifest=proto['tiers']['easy']['spawn_manifest'], label=label)
            for label in proto['models'] for i in range(0, 256, 8)]
    # The independent owner handles only new easy shards, never original jobs.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        while True:
            for f in list(futures):
                if f.done():
                    f.result()
                    del futures[f]
            for job in jobs:
                if len(futures) >= 4:
                    break
                if job in futures.values() or Path(str(L.shard_path(job)) + '.ACCEPTED.json').exists():
                    continue
                futures[pool.submit(L.collect, job, proto['models'][job['label']], proto)] = job
            results = {label: {'easy': L.summarize_group([j for j in jobs if j['label'] == label], model)}
                       for label, model in proto['models'].items()}
            original = read(C / 'RESULTS.json', {})
            for label in results:
                results[label].update(original.get('results', {}).get(label, {}))
            n = sum(x['games'] for v in results.values() for x in v.values())
            atomic_json(O / 'EXECUTION_STATUS.json', dict(status='RUNNING', games=n, budget=1536,
                        active=list(futures.values()), unix=time.time()))
            if n == 1536 and (C / 'COMPLETE.json').exists():
                assert read(C / 'COMPLETE.json')['results_sha256'] == sha(C / 'RESULTS.json')
                decision = select(results)
                atomic_json(O / 'SELECTION.json', dict(**decision, results=results,
                            protocol_sha256=sha(O / 'SELECTION_PROTOCOL.json'),
                            original_COMPLETE_sha256=sha(C / 'COMPLETE.json'), unix=time.time()))
                atomic_json(O / 'SELECTION_COMPLETE.json', dict(status='COMPLETE',
                            selection_sha256=sha(O / 'SELECTION.json'), **{k: decision[k] for k in ('winner', 'publish')}, unix=time.time()))
                from scripts.iter_pbt_owner import notify
                notify('BEST_SELECTION_20261030', f"三档同新seed配对1536局完成，预声明选择={decision['winner']}，publish={decision['publish']}；正式发布仍需导出/浏览器/线上验收。")
                return
            time.sleep(5)


if __name__ == '__main__':
    main()
