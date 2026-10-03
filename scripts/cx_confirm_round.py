#!/usr/bin/env python3
"""Run an independently seeded, paired four-bot confirmation under a supervisor."""
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
from scripts import cx_screen_round as screen
from scripts import eval_four_bot_matrix as matrix
from scripts import launch_r1_ab as launch
from scripts import score_eval_screen as score
from qqt_rl.training.io import atomic_write_json, sha256_file

CONFIRM_SPECS = {'EXP-CX-20261003-29': (20264011, 512)}


def confirm_seed(experiment, seed, games):
    expected_seed, expected_games = CONFIRM_SPECS[experiment]
    seed = expected_seed if seed is None else seed
    if (seed, games) != (expected_seed, expected_games):
        raise ValueError(f'{experiment} confirmation requires seed {expected_seed} and {expected_games} games')
    return seed


def validated_rows(config):
    rows = []
    if sha256_file(config['baseline']) != launch.INIT_SHA256:
        raise ValueError('baseline checkpoint hash changed')
    for name, path in [('hist_it4000', config['baseline']),
                       ('screen_candidate', config['original']),
                       ('independent_retrain', config['retrain'])]:
        path = Path(path).resolve()
        sha = sha256_file(path)
        if name != 'hist_it4000':
            screen.verify_derived_export(path.parent.parent)
            status = json.loads((path.parent.parent / 'status.json').read_text())
            recorded = status['checkpoints'][path.name]
            if status['state'] != 'complete' or status['rc'] != 0 or not status['finite']:
                raise ValueError(f'training incomplete: {path}')
            if not recorded['finite'] or sha != recorded['sha256']:
                raise ValueError(f'checkpoint differs from training supervisor: {path}')
        rows.append({'name': name, 'path': str(path), 'source': str(path),
                     'group': config['experiment'], 'label': name, 'sha256': sha})
    if rows[1]['sha256'] != config['original_sha256']:
        raise ValueError('screen candidate checkpoint hash changed')
    return rows


def verify_evaluator_sources(config):
    for name, expected in config['evaluator_sha256'].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f'evaluator source changed: {name}')


def verify_cells(out, rows, config):
    spawns = json.loads((out / 'spawns.json').read_text())
    if spawns['seed'] != config['seed'] or spawns['games'] != config['games']:
        raise ValueError('confirmation spawn seed/count mismatch')
    result = {}
    for row in rows:
        name, sha = row['name'], row['sha256']
        cell_dir = out / 'cells' / name
        tactical = cell_dir / 'tactical_v2.json'
        if not matrix.tactical_cell_ok(tactical, sha, config['seed'], config['games'], 300,
                                       spawns['spawn_cells'], trace=True):
            raise ValueError(f'tactical protocol mismatch: {name}')
        result[name] = {'checkpoint_sha256': sha, 'cells': {}}
        for bot in score.CELLS:
            path = cell_dir / f'{bot}.json'
            data = json.loads(path.read_text())
            if bot != 'tactical_v2':
                difficulty = bot.removeprefix('hunter_')
                if not (data['checkpoint_sha256'] == sha and data['seed'] == config['seed']
                        and data['games'] == config['games'] and data['max_steps'] == 300
                        and data['spawn_manifest_sha256'] == spawns['spawn_sha256']
                        and data['source_hashes'] == matrix.node_source_hashes()
                        and data['bot']['config'] == {'difficulty': difficulty}
                        and sorted(e['game'] for e in data['episodes']) == list(range(config['games']))):
                    raise ValueError(f'hunter protocol mismatch: {name}/{bot}')
                for episode in data['episodes']:
                    if episode['spawn_cells'] != spawns['spawn_cells'][episode['game']]:
                        raise ValueError(f'hunter spawn mismatch: {name}/{bot}/{episode["game"]}')
            traces = score.load_traces(path)
            if len(traces) != config['games']:
                raise ValueError(f'missing attack traces: {name}/{bot}')
            result[name]['cells'][bot] = {'sha256': sha256_file(path),
                                          'stats': score.cell_stats(score.per_game(traces))}
    return result


def supervise(out):
    out = Path(out).resolve()
    config = json.loads((out / 'confirm_config.json').read_text())
    verify_evaluator_sources(config)
    rows = validated_rows(config)
    recorded = json.loads((out / 'checkpoints.json').read_text())['checkpoints']
    if rows != recorded:
        raise ValueError('checkpoint manifest changed')
    now = time.time()
    previous = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else None
    if previous and previous['state'] != 'failed':
        raise ValueError('existing confirmation supervisor is not failed')
    if previous and screen.process_matches(previous['supervisor_pid']):
        raise ValueError('previous confirmation supervisor is alive')
    if previous and any(screen.process_matches(j['pid'], j['start_ticks']) for j in previous['jobs']):
        raise ValueError('previous confirmation child is alive')
    if previous:
        (out / 'FAILED').rename(out / f'FAILED_attempt{previous["attempt"]}')
    status = {'state': 'running', 'started': previous['started'] if previous else now,
              'attempt': (previous['attempt'] + 1) if previous else 1,
              'attempt_started': now, 'supervisor_pid': os.getpid(), 'jobs': []}

    def save():
        status['updated'] = time.time()
        atomic_write_json(out / 'status.json', status)

    def run_group(phase, jobs):
        status['phase'] = phase
        active = []
        for name, argv, env_extra in jobs:
            with open(out / f'{name}.log', 'ab') as log:
                env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', **env_extra}
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            job = {'name': name, 'pid': proc.pid, 'start_ticks': launch.start_ticks(proc.pid),
                   'started': time.time(), 'argv': argv, 'rc': None}
            active.append((proc, job))
            status['jobs'].append(job)
        save()
        while any(proc.poll() is None for proc, _ in active):
            for proc, job in active:
                if job['rc'] is None and proc.poll() is not None:
                    job.update(rc=proc.returncode, ended=time.time())
            save()
            time.sleep(10)
        for proc, job in active:
            if job['rc'] is None:
                job.update(rc=proc.returncode, ended=time.time())
        save()
        if any(job['rc'] != 0 for _, job in active):
            raise RuntimeError(f'failed evaluation jobs: {[j["name"] for _, j in active if j["rc"] != 0]}')

    try:
        common = [sys.executable, str(ROOT / 'scripts/eval_four_bot_matrix.py')]
        base = ['--out-dir', str(out), '--manifest', str(out / 'checkpoints.json'),
                '--seed', str(config['seed']), '--games', str(config['games'])]
        while (delay := screen.checkpoint_wait_seconds(rows)) > 0:
            status['phase'] = 'checkpoint_stability'
            status['checkpoint_wait_seconds'] = delay
            save()
            time.sleep(min(delay, 5))
        run_group('spawns', [('spawns', common + ['spawns', *base], {'JAX_PLATFORMS': 'cpu'})])
        names = [row['name'] for row in rows]
        run_group('tactical', [(f'tactical_{name}', common + ['tactical', *base,
                  '--attack-trace', '--device', str(gpu), '--host-workers', '4', '--only', name], {})
                  for gpu, name in zip((4, 5, 6), names)])
        run_group('real_js_hunters', [('hunters', common + ['hunters', *base,
                  '--attack-trace', '--workers', str(config['workers'])], {'JAX_PLATFORMS': 'cpu'})])
        run_group('summary', [('summary', common + ['summarize', *base,
                  '--report-dir', str(out / 'report')], {'JAX_PLATFORMS': 'cpu'})])
        audit = verify_cells(out, rows, config)
        atomic_write_json(out / 'cells_audit.json', audit)
        base_cells = score.load_cells(out / 'cells/hist_it4000')
        comparisons = {name: score.confirm(base_cells, score.load_cells(out / 'cells' / name),
                                            n_boot=10000, seed=config['bootstrap_seed'])
                       for name in names[1:]}
        atomic_write_json(out / 'confirm.json', comparisons)
        status.update(state='complete', ended=time.time(), comparisons=comparisons)
        save()
        (out / 'COMPLETE').write_text('rc=0\n')
        return 0
    except Exception as exc:
        status.update(state='failed', ended=time.time(), error=repr(exc))
        save()
        (out / 'FAILED').write_text(repr(exc) + '\n')
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('launch', 'supervise'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline')
    ap.add_argument('--original')
    ap.add_argument('--retrain')
    ap.add_argument('--original-sha256')
    ap.add_argument('--experiment', choices=sorted(CONFIRM_SPECS), default='EXP-CX-20261003-29')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--games', type=int, default=512)
    ap.add_argument('--bootstrap-seed', type=int)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if args.command == 'supervise':
        return supervise(out)
    if out.exists():
        raise FileExistsError(f'refusing to reuse confirmation namespace: {out}')
    seed = confirm_seed(args.experiment, args.seed, args.games)
    config = {'experiment': args.experiment, 'baseline': str(Path(args.baseline).resolve()),
              'original': str(Path(args.original).resolve()), 'retrain': str(Path(args.retrain).resolve()),
              'original_sha256': args.original_sha256, 'seed': seed, 'games': args.games,
              'bootstrap_seed': args.bootstrap_seed if args.bootstrap_seed is not None else seed + 1,
              'workers': args.workers,
              'evaluator_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                   [ROOT / 'scripts/eval_four_bot_matrix.py', ROOT / 'scripts/eval_web_hunter.js',
                                    ROOT / 'scripts/score_eval_screen.py', ROOT / 'web/bun_hunter_bot.js',
                                    ROOT / 'web/sim.js']}}
    rows = validated_rows(config)
    out.mkdir(parents=True)
    atomic_write_json(out / 'confirm_config.json', config)
    atomic_write_json(out / 'checkpoints.json', {'checkpoints': rows})
    with open(out / 'supervisor.log', 'ab') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise',
                                 '--out', str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    atomic_write_json(out / 'launch.json', {'pid': proc.pid, 'start_ticks': launch.start_ticks(proc.pid),
                                            'started': time.time()})
    print(json.dumps({'pid': proc.pid, 'out': str(out), 'checkpoints': len(rows)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
