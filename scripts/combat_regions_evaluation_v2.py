"""Frozen paired CPU evaluation, separate from the original Hunter queue."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

B = Path('runs/combat_regions_independent_20261104')
OUT = B / 'evaluation'
CORES = (140, 141, 142, 143)
OPPONENT = 'bun-interceptor-defend-cap4-bot-v6-f32-time-ties'


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def freeze():
    from scripts.bun_spawn_side import manifest
    path = OUT / 'PROTOCOL.json'
    if path.exists():
        p = read(path)
        assert all(sha(f) == h for f, h in p['source_sha256'].items())
        return p
    conditions = {}
    for split in ('dev', 'holdout'):
        starts = read(B / (split + '_starts.json'))
        homes = manifest(starts['seed'], len(starts['rows']))['spawn_cells']
        rows = []
        counts = {}
        for i, row in enumerate(starts['rows']):
            key = (row['region'], row['distance'])
            counts[key] = counts.get(key, 0) + 1
            if split == 'dev' and counts[key] > 4:
                continue
            for mode in ('greedy', 'categorical'):
                rows.append(dict(seed=row['seed'], seat=row['learner_seat'], mode=mode,
                    spawn=row['cells'], native_respawn=homes[i], region=row['region'], distance=row['distance'],
                    action_seed=row['seed'] ^ 0x515154))
        conditions[split + '_region'] = rows
    homes = manifest(2026547200, 32)
    conditions['dev_natural'] = [dict(seed=homes['seed'] + i, seat=seat, mode=mode,
        spawn=pair, native_respawn=pair, region='natural', distance=None,
        action_seed=(homes['seed'] + i) ^ 0x515154)
        for i, pair in enumerate(homes['spawn_cells']) for seat in (0, 1) for mode in ('greedy', 'categorical')]
    files = set(read(B / 'SOURCE_INPUTS.json')['sources'])
    files.update(str(p) for p in Path('scripts').glob('combat_regions_*v2.*'))
    protocol = dict(status='FROZEN', conditions=conditions, checkpoint_offsets=[80, 320, 640, 960, 1280],
        full_holdout_offsets=[640, 1280], parent_full_holdout=True, opponent=OPPONENT,
        source_sha256={f: sha(f) for f in sorted(files)}, teacher=False, max_ticks=2400,
        modes=['greedy', 'categorical'], cores=list(CORES), natural_games_are_separate=True,
        center='no legal starts; trajectory visits still measured',
        dev_holdout='disjoint environment seeds; finite native geometry may overlap',
        low_sample='dev 12 episodes per region per mode, six per seat; low counts and no contact reported explicitly',
        acceptance=dict(min_contact_games=8, weakest_region='positive paired causal-kill-per-contact-game difference with 95% bootstrap lower bound >0',
            natural_and_bottom_right='paired win-rate lower 95% bound >= -0.10 and self-death increase upper bound <=0.10',
            continuous_attack='contact repeats plus causal control/kill conversion; bubbles alone never pass',
            transport='same-custody delivery and dangerous pickup failure compared at fixed seeds; missing acquisitions are insufficient evidence'))
    atomic_json(path, protocol)
    return protocol


def sources():
    p = read(read(B / 'training/CURRENT.json')['protocol'])
    parent = p['continuation_source']['checkpoint']
    yield 'parent', parent, 0
    for request in sorted((B / 'training/eval_requests').glob('G_u*.json')):
        r = read(request); offset = r['update'] - p['campaign_start_update']
        if offset in (80, 320, 640, 960, 1280):
            yield 'u' + str(r['update']), r['checkpoint'], offset


def export(label, cp):
    dest = OUT / 'exports' / label
    if not (dest / 'K4_MODE0_IDENTITY.json').exists():
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / 'export.log').open('ab') as stream:
            subprocess.run(['taskset', '-c', '144-145', sys.executable, '-m', 'scripts.r4_k4_export', cp, str(dest)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                stdout=stream, stderr=subprocess.STDOUT, check=True)
    proof = read(dest / 'K4_MODE0_IDENTITY.json')
    assert all(sha(f) == h for f, h in proof['files'].items())
    assert proof['source_policy_sha256'] == read(Path(cp) / 'COMMITTED.json')['policy_sha256']
    return dest / 'model.json'


def jobs(protocol):
    for label, cp, offset in sources():
        conditions = ['dev_region', 'dev_natural']
        if offset in (0, 640, 1280):
            conditions.append('holdout_region')
        for condition in conditions:
            for i in range(0, len(protocol['conditions'][condition]), 2):
                yield dict(label=label, checkpoint=cp, offset=offset, condition=condition, start=i)


def folder(job):
    return OUT / job['label'] / job['condition'] / ('shard_%04d' % job['start'])


def collect(job, protocol, model, core):
    out = folder(job); out.mkdir(parents=True, exist_ok=True)
    guard = lock(out / 'worker.lock')
    games = protocol['conditions'][job['condition']][job['start']:job['start'] + 2]
    cfg = dict(model_json=str(model), checkpoint=job['checkpoint'],
        policy_sha256=read(Path(job['checkpoint']) / 'COMMITTED.json')['policy_sha256'],
        opponent=OPPONENT, games=games, output=str(out), display_name=job['label'])
    atomic_json(out / 'CONFIG.json', cfg)
    with (out / 'worker.log').open('ab') as stream:
        subprocess.run(['taskset', '-c', str(core), 'node', 'scripts/combat_regions_eval_v2.js', str(out / 'CONFIG.json')],
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    complete = read(out / 'COMPLETE.json')
    assert complete['status'] == 'PASS' and len(complete['results']) == len(games)
    files = {str(out / 'COMPLETE.json'): sha(out / 'COMPLETE.json')}
    for expected, row in zip(games, complete['results']):
        assert all(row[k] == expected[k] for k in ('seed', 'seat', 'mode'))
        assert row['complete'] and row['review_verified']
        assert row['native_respawn_cells'] == expected['native_respawn']
        assert 'continuous_attack' in row and row['source_policy_ability_actions'] == 3
        replay = out / ('five-slots-%d-%d-%s' % (row['seed'], row['seat'], row['mode'])) / 'review.json'
        assert sha(replay) == row['review_sha256']
        files[str(replay)] = row['review_sha256']
    atomic_json(out / 'ACCEPTED.json', dict(status='PASS', job=job, protocol_sha256=sha(OUT / 'PROTOCOL.json'), files=files))


def main():
    guard = lock(OUT / 'OWNER.lock')
    protocol = freeze(); active = {}; models = {}; finished = set()
    with ThreadPoolExecutor(max_workers=len(CORES)) as pool:
        while True:
            for future in list(active):
                if future.done():
                    job, core = active.pop(future)
                    future.result()
                    finished.add(str(folder(job)))
            pending = []
            all_jobs = list(jobs(protocol))
            # Evaluate comparable dev cohorts for each arrived point before
            # filling the much larger holdout queue. GPU never waits for this.
            all_jobs.sort(key=lambda job: (job['condition'] == 'holdout_region', job['start'], job['offset']))
            for job in all_jobs:
                out = folder(job)
                if str(out) in finished:
                    continue
                accepted = read(out / 'ACCEPTED.json')
                if accepted:
                    assert accepted['protocol_sha256'] == sha(OUT / 'PROTOCOL.json')
                    assert all(sha(f) == h for f, h in accepted['files'].items())
                    finished.add(str(out)); continue
                if any(str(folder(v[0])) == str(out) for v in active.values()):
                    continue
                pending.append(job)
            for job in pending:
                if len(active) >= len(CORES):
                    break
                if job['label'] not in models:
                    models[job['label']] = export(job['label'], job['checkpoint'])
                core = next(c for c in CORES if c not in [v[1] for v in active.values()])
                active[pool.submit(collect, job, protocol, models[job['label']], core)] = (job, core)
            atomic_json(OUT / 'STATUS.json', dict(status='RUNNING', accepted_shards=len(finished),
                arrived_jobs=len(all_jobs), active=[dict(job=v[0], core=v[1]) for v in active.values()], unix=time.time()))
            subprocess.run([sys.executable, '-m', 'scripts.combat_regions_report_v2'], check=True,
                stdout=subprocess.DEVNULL)
            current = read(B / 'training/CURRENT.json')
            p = read(current['protocol']); complete = Path(p['namespace']) / current['run'] / 'COMPLETE.json'
            if complete.exists() and not pending and not active:
                atomic_json(OUT / 'COMPLETE.json', dict(status='ALL_FIXED_EVALUATIONS_COMPLETE', shards=len(finished), unix=time.time()))
                return
            if (OUT / 'USER_PAUSE.json').exists():
                return
            time.sleep(10)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(OUT / 'FAILURE.json', dict(status='TECHNICAL_BLOCKED', type=type(error).__name__,
            detail=str(error).replace(str(Path.cwd()), '<WORKSPACE>'), unix=time.time()))
        raise
