"""Fixed R1 checkpoints versus the verified live baseline; CPU-only shards."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.bomb_threat_round15 import read
from scripts import bomb_threat_a3sr_lineage_watch as L
from scripts.best_publish_selection_20261030 import select

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = BASE / 'r1_fresh_publish_1805_20261030'


def model(label, source):
    record = OUT / 'models' / f'{label}.json'
    if record.exists():
        m = read(record)
        assert sha(ROOT / m['model_json']) == m['model_sha256']
        return m
    cp = ROOT / source['checkpoint']
    assert sha(cp / 'policy.pt') == source['policy_sha256']
    dest = OUT / 'exports' / label
    dest.mkdir(parents=True, exist_ok=True)
    code = 'from pathlib import Path;from scripts.je08_bun_authority import export;import sys;export(Path(sys.argv[1]),Path(sys.argv[2]).resolve())'
    with (dest / 'EXPORT.log').open('ab') as log:
        subprocess.run([sys.executable, '-c', code, str((cp / 'policy.pt').relative_to(ROOT)), str(dest.relative_to(ROOT))],
            cwd=ROOT, env=L.cpu_env(), stdout=log, stderr=subprocess.STDOUT, check=True)
    exported = dest / 'model.json'
    assert read(exported)['meta']['checkpoint_sha256'] == source['policy_sha256']
    m = dict(source, model_json=str(exported.relative_to(ROOT)), model_sha256=sha(exported))
    atomic_json(record, m)
    return m


def collect(job, exported):
    path = L.shard_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = lock(Path(str(path) + '.lock'))
    accepted = Path(str(path) + '.ACCEPTED.json')
    if accepted.exists():
        for f, h in read(accepted)['files'].items():
            assert sha(ROOT / f) == h
        L.validate_shard(path, job, exported)
        return
    while shard_running(path):
        time.sleep(3)
    if not (path.exists() and Path(str(path) + '.outcome.json').exists()):
        env = L.cpu_env()
        env.update(BT_FRESH_TIER=job['tier'], JE10_ROUTE_TASK='natural_full2400',
            JE10_ROUTE_GAMES=str(job['games']), JE10_ROUTE_START=str(job['start']),
            JE10_ROUTE_END=str(job['start'] + 8), JE10_ROUTE_SEED_START=str(job['seed']))
        with path.with_suffix('.log').open('ab') as log:
            subprocess.run(['node', 'scripts/bomb_threat_r1_fresh_metrics_eval.js', 'greedy', exported['model_json'],
                str(path.relative_to(ROOT)), 'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',
                job['spawn_manifest']], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    L.validate_shard(path, job, exported)
    metrics = Path(str(path) + '.r1_metrics.json')
    extra = read(metrics)
    assert len(extra['episodes']) == 8 and all(e['done'] for e in extra['episodes'])
    assert [e['seed'] for e in extra['episodes']] == list(range(job['seed'] + job['start'], job['seed'] + job['start'] + 8))
    atomic_json(accepted, dict(job=job, model=exported, protocol_sha256=sha(OUT / 'PROTOCOL.json'),
        files={str(f.relative_to(ROOT)): sha(f) for f in (path, Path(str(path) + '.outcome.json'), metrics)}, unix=time.time()))


def shard_running(path):
    output = str(path.relative_to(ROOT)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'scripts/bomb_threat_r1_fresh_metrics_eval.js' in args and output in args and (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                return True
        except OSError:
            pass
    return False


def main():
    guard = lock(OUT / 'OWNER.lock')
    L.OUT = OUT
    proto = read(OUT / 'PROTOCOL.json')
    models = {label: model(label, s) for label, s in proto['models'].items()}
    baseline_file = ROOT / proto['baseline_results']
    assert sha(baseline_file) == proto['baseline_results_sha256']
    baseline = read(baseline_file)['results'][proto['baseline_label']]
    for tier, result in baseline.items():
        assert result['games'] == 256
        for f, h in result['files'].items():
            assert sha(ROOT / f) == h
    jobs = [dict(kind='paired_fresh', slot=m['slot'], update=m['update'], tier=tier,
        start=i, games=256, seed=t['seed'], spawn_manifest=t['spawn_manifest'], label=label)
        for label, m in models.items() for tier, t in proto['tiers'].items() for i in range(0, 256, 8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        while True:
            for future in list(futures):
                if future.done():
                    future.result()
                    del futures[future]
            for job in jobs:
                if len(futures) >= 4:
                    break
                if job in futures.values() or Path(str(L.shard_path(job)) + '.ACCEPTED.json').exists():
                    continue
                futures[pool.submit(collect, job, models[job['label']])] = job
            results = {label: {tier: L.summarize_group([j for j in jobs if j['label'] == label and j['tier'] == tier], m)
                for tier in proto['tiers']} for label, m in models.items()}
            n = sum(x['games'] for v in results.values() for x in v.values())
            atomic_json(OUT / 'EXECUTION_STATUS.json', dict(status='RUNNING', games=n, budget=2304,
                active=list(futures.values()), models=models, unix=time.time()))
            if n == 2304:
                results = dict(baseline=baseline, **results)
                decision = select(results)
                atomic_json(OUT / 'FINAL_SELECTION.json', dict(**decision, results=results,
                    models=dict(baseline=proto['baseline'], **models), protocol_sha256=sha(OUT / 'PROTOCOL.json'), unix=time.time()))
                atomic_json(OUT / 'EVALUATION_COMPLETE.json', dict(**decision, games=n,
                    reused_baseline_games=768, selection_sha256=sha(OUT / 'FINAL_SELECTION.json'), unix=time.time()))
                from scripts.iter_pbt_owner import notify
                notify('R1_FRESH_SELECTION_1805', f'R1三候选同seed三档复核2304局完成，winner={decision["winner"]}，publish={decision["publish"]}；原始分片及护栏已保存，进入发布/录像技术验收。')
                return
            time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(OUT / 'BLOCKED.json', dict(status='BLOCKED', error_type=type(error).__name__, reason='inspect relative OWNER.log and EXPORT.log; no training changes',
            training_untouched=True, unix=time.time()))
        raise
