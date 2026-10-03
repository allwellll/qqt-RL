#!/usr/bin/env python3
"""Supervise the unchanged four-bot protocol in an isolated evaluation namespace.

Uses completed baseline artifacts read-only after verifying their provenance.
Never reuses candidate cells, model exports, or caches from another namespace.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_four_bot_matrix as matrix
from scripts import launch_r1_ab as launch
from scripts import score_eval_screen as score
from qqt_rl.training.io import atomic_write_json, sha256_file


def checkpoint_wait_seconds(rows, now=None):
    now = time.time() if now is None else now
    return max([0.] + [61. - (now - Path(r['path']).stat().st_mtime) for r in rows])


def process_matches(pid, ticks=None):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] != 'Z' and (ticks is None or int(fields[19]) == ticks)


def validate_resume(out):
    out = Path(out)
    status = json.loads((out / 'status.json').read_text())
    if status['state'] != 'failed':
        raise ValueError('resume requires a terminal failed supervisor')
    if process_matches(status['supervisor_pid']):
        raise ValueError('previous supervisor still alive')
    for job in status['jobs']:
        if process_matches(job['pid'], job['start_ticks']):
            raise ValueError(f'previous child still alive: {job["name"]}')
    rows = json.loads((out / 'checkpoints.json').read_text())['checkpoints']
    for row in rows:
        if row['name'] == 'hist_it4000':
            if sha256_file(row['path']) != launch.INIT_SHA256:
                raise ValueError('baseline checkpoint changed')
            continue
        path = Path(row['path'])
        verify_derived_export(path.parent.parent)
        training = json.loads((path.parent.parent / 'status.json').read_text())
        check = training['checkpoints'][path.name]
        if training['state'] != 'complete' or training['rc'] != 0 or not check['finite']:
            raise ValueError(f'training not verified: {path}')
        if sha256_file(path) != check['sha256']:
            raise ValueError(f'checkpoint changed: {path}')
    return status


def verify_baseline(source, checkpoint, prior_manifest, spawns):
    source = Path(source).resolve()
    sha = sha256_file(checkpoint)
    if sha != launch.INIT_SHA256:
        raise ValueError('baseline checkpoint mismatch')
    deps = json.loads(Path(prior_manifest).read_text())['deps_sha256']
    # Only the launcher changes; it is not executed by either evaluator.
    checked = {f: h for f, h in deps.items() if f != 'scripts/launch_r1_ab.py'}
    for f, h in checked.items():
        if sha256_file(ROOT / f) != h:
            raise ValueError(f'baseline evaluator dependency changed: {f}; re-evaluate baseline')
    p = source / 'tactical_v2.json'
    if not matrix.tactical_cell_ok(p, sha, spawns['seed'], spawns['games'], 300, spawns['spawn_cells'], trace=True):
        raise ValueError('baseline tactical protocol mismatch')
    audit = {'checkpoint_sha256': sha, 'dependency_hashes': checked, 'cells': {}}
    for name in score.CELLS:
        path = source / f'{name}.json'
        data = json.loads(path.read_text())
        if name.startswith('hunter_'):
            difficulty = name.removeprefix('hunter_')
            if not (data['checkpoint_sha256'] == sha and data['seed'] == spawns['seed']
                    and data['games'] == spawns['games'] and data['max_steps'] == 300
                    and data['spawn_manifest_sha256'] == spawns['spawn_sha256']
                    and data['source_hashes'] == matrix.node_source_hashes()
                    and data['bot']['config'] == {'difficulty': difficulty}
                    and sorted(e['game'] for e in data['episodes']) == list(range(spawns['games']))):
                raise ValueError(f'baseline hunter protocol mismatch: {name}')
            for e in data['episodes']:
                if e['spawn_cells'] != spawns['spawn_cells'][e['game']]:
                    raise ValueError('baseline episode spawn mismatch')
        traces = score.load_traces(path)
        if len(traces) != spawns['games']:
            raise ValueError('missing baseline attack traces')
        audit['cells'][name] = {'source': str(path), 'sha256': sha256_file(path),
                               'stats': score.cell_stats(score.per_game(traces))}
    return audit


def verify_derived_export(run_dir):
    manifest_path = Path(run_dir) / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('artifact_kind') == 'seed_weight_export':
            from scripts.cx_seed_average import verify_export
            verify_export(run_dir)


def verified_checkpoint_rows(run_dir, manifest, status):
    verify_derived_export(run_dir)
    points = list(range(manifest['save_every'], manifest['iters'] + 1, manifest['save_every']))
    if manifest['iters'] not in points:
        points.append(manifest['iters'])
    verified = []
    for iteration in points:
        filename = f'final_it{iteration}.pt' if iteration < manifest['iters'] else 'final.pt'
        path = run_dir / 'ckpt' / filename
        check = status['checkpoints'][filename]
        sha = matrix.sha256_file(path)
        if not check['finite'] or sha != check['sha256']:
            raise ValueError(f'checkpoint differs from supervisor verification: {path}')
        verified.append((iteration, path, sha))
    rows, seen = [], set()
    for iteration, path, sha in reversed(verified):
        if sha in seen:
            continue
        seen.add(sha)
        name = f'{run_dir.name}_it{iteration}'
        rows.append({'name': name, 'path': str(path.resolve()),
                     'source': str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
                     'group': manifest['experiment'], 'label': name,
                     'run': run_dir.name, 'iteration': iteration})
    return list(reversed(rows))


def supervise(args):
    out = Path(args.out).resolve()
    now = time.time()
    previous = json.loads((out / 'resume_history.json').read_text()) if getattr(args, 'resume', False) else []
    status = {'state': 'running', 'started': previous[0]['started'] if previous else now,
              'attempt_started': now, 'attempt': len(previous) + 1,
              'supervisor_pid': os.getpid(), 'jobs': []}
    active = []
    def save():
        status['updated'] = time.time()
        atomic_write_json(out / 'status.json', status)
    def run_group(commands, phase):
        status['phase'] = phase
        active.clear()
        for name, argv, env_add in commands:
            with open(out / f'{name}.log', 'ab') as log:
                env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', **env_add}
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record = {'name': name, 'pid': proc.pid, 'start_ticks': launch.start_ticks(proc.pid),
                      'started': time.time(), 'argv': argv, 'rc': None}
            status['jobs'].append(record)
            active.append((proc, record))
        save()
        while any(p.poll() is None for p, _ in active):
            for p, record in active:
                if record['rc'] is None and p.poll() is not None:
                    record.update(rc=p.returncode, ended=time.time())
            save()
            time.sleep(10)
        for p, record in active:
            if record['rc'] is None:
                record.update(rc=p.returncode, ended=time.time())
        save()
        failed = [r['name'] for _, r in active if r['rc'] != 0]
        if failed:
            raise RuntimeError(f'evaluation failed: {failed}; see per-job logs')
    try:
        common = [sys.executable, str(ROOT / 'scripts/eval_four_bot_matrix.py')]
        rows = json.loads((out / 'checkpoints.json').read_text())['checkpoints']
        status['phase'] = 'checkpoint_stability'
        while (delay := checkpoint_wait_seconds(rows)) > 0:
            status['checkpoint_wait_seconds'] = delay
            save()
            time.sleep(min(delay, 5.))
        if not previous:
            run_group([('spawns', common + ['spawns', '--out-dir', str(out)], {})], 'spawns')
        spawns = json.loads((out / 'spawns.json').read_text())
        audit = verify_baseline(args.baseline_cells, args.init, args.baseline_manifest, spawns)
        atomic_write_json(out / 'baseline_audit.json', audit)
        (out / 'cells').mkdir(exist_ok=True)
        baseline_link = out / 'cells/hist_it4000'
        if baseline_link.exists():
            if not baseline_link.is_symlink() or baseline_link.resolve() != Path(args.baseline_cells).resolve():
                raise ValueError('baseline reference changed')
        else:
            baseline_link.symlink_to(Path(args.baseline_cells).resolve(), target_is_directory=True)
        candidates = [r for r in rows if r['name'] != 'hist_it4000']
        base = ['--out-dir', str(out), '--manifest', str(out / 'checkpoints.json'), '--attack-trace']
        jobs = []
        for gpu in range(8):
            names = [r['name'] for r in candidates[gpu::8]]
            if names:
                jobs.append((f'tactical_g{gpu}', common + ['tactical', *base, '--device', str(gpu),
                            '--host-workers', '4', '--only', ','.join(names)], {}))
        run_group(jobs, 'tactical')
        run_group([('hunters', common + ['hunters', *base, '--workers', str(args.workers),
                   '--only', ','.join(r['name'] for r in candidates)], {'JAX_PLATFORMS': 'cpu'})], 'real_js_hunters')
        run_group([('summary', common + ['summarize', *base, '--report-dir', str(out / 'report')],
                    {'JAX_PLATFORMS': 'cpu'})], 'summary')
        base_cells = score.load_cells(out / 'cells/hist_it4000')
        results = {r['name']: score.screen(base_cells, score.load_cells(out / 'cells' / r['name'])) for r in candidates}
        atomic_write_json(out / 'screen.json', results)
        ranked = sorted(results, key=lambda n: results[n]['rank_key'], reverse=True)
        selections = {}
        for r in candidates:
            run = r['run']
            if run not in selections:
                group = [x for x in ranked if next(c for c in candidates if c['name'] == x)['run'] == run]
                selections[run] = {'early_best': group[0], 'passed': [x for x in group if results[x]['pass']],
                                   'latest': max((c for c in candidates if c['run'] == run), key=lambda c: c['iteration'])['name']}
        atomic_write_json(out / 'selection.json', selections)
        status.update(state='complete', ended=time.time(), passed=[n for n in ranked if results[n]['pass']], ranking=ranked)
        save()
        (out / 'COMPLETE').write_text('rc=0' + chr(10))
        return 0
    except Exception as exc:
        status.update(state='failed', ended=time.time(), error=repr(exc))
        save()
        (out / 'FAILED').write_text(repr(exc) + chr(10))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['launch', 'resume', 'supervise'])
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--baseline-cells', required=True)
    ap.add_argument('--baseline-manifest', required=True)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--resume', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.command == 'supervise':
        return supervise(args)
    out = Path(args.out).resolve()
    if args.command == 'resume':
        old = validate_resume(out)
        history_path = out / 'resume_history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(old)
        atomic_write_json(history_path, history)
        failed = out / 'FAILED'
        if failed.exists():
            failed.rename(out / f'FAILED_attempt{len(history)}')
        with open(out / 'supervisor.log', 'ab') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise',
                                     *sys.argv[2:], '--resume'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_write_json(out / f'resume_launch_{len(history)}.json',
                          {'pid': proc.pid, 'start_ticks': launch.start_ticks(proc.pid), 'started': time.time()})
        print(json.dumps({'pid': proc.pid, 'out': str(out), 'resume_attempt': len(history) + 1}))
        return 0
    if out.exists():
        raise FileExistsError(f'refusing to reuse namespace: {out}')
    rows = [{'name': 'hist_it4000', 'path': str(Path(args.init).resolve()), 'source': '<historical>/phase2_it4000.pt',
             'group': 'baseline', 'label': 'historical it4000'}]
    for mp in sorted(Path(args.runs).glob('*/manifest.json')):
        if mp.parent.name.startswith('smoke_'):
            continue
        m = json.loads(mp.read_text())
        status = json.loads((mp.parent / 'status.json').read_text())
        if status['state'] != 'complete' or status['rc'] != 0 or not status['finite']:
            raise ValueError(f'training incomplete: {mp.parent}')
        rows.extend(verified_checkpoint_rows(mp.parent, m, status))
    if len(rows) == 1:
        raise ValueError('no completed training runs')
    out.mkdir(parents=True)
    atomic_write_json(out / 'checkpoints.json', {'checkpoints': rows})
    with open(out / 'supervisor.log', 'ab') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise', *sys.argv[2:]],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_write_json(out / 'launch.json', {'pid': proc.pid, 'start_ticks': launch.start_ticks(proc.pid), 'started': time.time()})
    print(json.dumps({'pid': proc.pid, 'out': str(out), 'checkpoints': len(rows) - 1}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
