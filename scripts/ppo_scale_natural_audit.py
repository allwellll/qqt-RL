"""Bounded same-seed categorical/greedy natural action-chain measurement v4."""
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha

R = Path(__file__).resolve().parents[1]
B = R / 'runs/ppo_scale_20261025/r1_longrun'


def run(checkpoint):
    cp = R / checkpoint
    unique = lock(cp / 'natural_action_audit.lock')
    receipt = json.loads((cp / 'COMMITTED.json').read_text())
    assert sha(cp / 'policy.pt') == receipt['policy_sha256']
    model = cp / 'retention/model/model.json'
    assert model.exists(), 'use the unique evaluator export; do not export concurrently'
    out = cp / 'natural_action_audit_v4'
    out.mkdir(exist_ok=True)
    protocol = dict(version=4, games_per_mode=64, seed_start=2026102480,
                    modes=['categorical', 'greedy'], engine='native JS full2400',
                    teacher_applied=False, policy_sha256=sha(cp / 'policy.pt'),
                    metric_sha256=sha(R / 'scripts/ppo_scale_item_metrics_v4.js'),
                    source_sha256=sha(R / 'scripts/dagger_rl_route.js'),
                    adapter_sha256=sha(R / 'scripts/ppo_scale_natural_audit.js'),
                    fixture=False, training_eligible=False, no_holdout_tuning=True,
                    scope='Observed whole-chain distance proxy, not causal Q; CPU only')
    manifest = out / 'PROTOCOL.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == protocol
    else:
        atomic_json(manifest, protocol)

    def job(pair):
        mode, start = pair
        folder = out / mode / f'{start:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'RESULT.json'
        if not target.exists():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                       JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='1',
                       JE10_ROUTE_TASK='natural_full2400', JE10_ROUTE_GAMES='64',
                       JE10_ROUTE_START=str(start), JE10_ROUTE_END=str(start+8),
                       JE10_ROUTE_SEED_START=str(protocol['seed_start']))
            env.pop('NODE_OPTIONS', None)
            with (folder / 'run.log').open('a') as log:
                subprocess.run(['node', 'scripts/ppo_scale_natural_audit.js', mode,
                    str(model.relative_to(R)), str(target.relative_to(R)),
                    str((B / 'cpu_eval_inputs/regions.json').relative_to(R))],
                    cwd=R, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        return mode, json.loads(target.read_text())['results']['natural_full2400']['episodes']

    grouped = {m: [] for m in protocol['modes']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for mode, episodes in pool.map(job, [(m, i) for m in protocol['modes'] for i in range(0,64,8)]):
            grouped[mode].extend(episodes)
            atomic_json(out / 'PROGRESS.json', {m: len(es) for m, es in grouped.items()})
    summary = {}
    for mode, episodes in grouped.items():
        assert len(episodes) == 64
        rows = {}
        for ep in episodes:
            assert ep['fixture'] is False and ep['action_selection'] == mode
            for phase_item, metrics in ep['item_metrics'].items():
                dest = rows.setdefault(phase_item, {})
                for key, val in metrics.items():
                    dest[key] = dest.get(key, 0) + val
        summary[mode] = dict(games=64, delivery_episodes=sum(e['delivery']>0 for e in episodes),
                             death_episodes=sum(e['death']>0 for e in episodes),
                             deaths=sum(e['death'] for e in episodes), item_metrics=rows,
                             chain_events=sum(len(e['item_chain_events']) for e in episodes))
    atomic_json(out / 'COMPLETE.json', dict(passed=True, protocol=protocol, summary=summary,
                episodes=grouped, physics_rng_parity_test='tests/ppo_scale_item_metrics_v4_test.js',
                does_not_gate_training=True))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
