"""Bounded CPU replay of the predeclared first64 production games."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.population_run_state import atomic_json, lock, sha

BASE = Path('runs/r4_combat_20261031/interceptor')


def read(p):
    return json.loads(Path(p).read_text())


def run_shard(folder, protocol, start, core, diagnostic=True):
    target = folder / f'shard_{start:03d}.json'
    receipt = Path(str(target) + '.ACCEPTED.json')
    guard = lock(Path(str(target) + '.lock'))
    if receipt.exists():
        assert all(sha(Path(p)) == h for p, h in read(receipt)['files'].items())
        return
    for p, h in protocol['sources'].items():
        assert sha(Path(p)) == h, p
    assert sha(Path(protocol['model']['model_json'])) == protocol['model']['model_sha256']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', QQT_STOCK_RULE='stock-window-v1',
        BUN_V2_RULE='bun-native-v2', BUN_V2_TIER='hard', BUN_V2_OPPONENT='bun.hunter',
        R4_PRESSURE_OPPONENT='interceptor', JE10_ROUTE_TASK='natural_full2400',
        JE10_ROUTE_GAMES='256', JE10_ROUTE_START=str(start), JE10_ROUTE_END=str(start+8),
        JE10_ROUTE_SEED_START=str(protocol['seed_start']))
    if not (target.exists() and Path(str(target)+'.outcome.json').exists()):
        with target.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset', '-c', str(core), 'node', 'tests/interceptor_diagnostic_eval.js' if diagnostic else 'tests/interceptor_timing_v2_eval.js',
                'greedy', protocol['model']['model_json'], str(target),
                'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json', protocol['spawn_manifest']],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    side = Path(str(target)+'.outcome.json')
    episodes = read(side)['episodes']
    assert len(episodes) == 8 and all(e['done'] for e in episodes)
    assert [e['seed'] for e in episodes] == list(range(protocol['seed_start']+start, protocol['seed_start']+start+8))
    assert [e['seat'] for e in episodes] == [(start+i) % 2 for i in range(8)]
    if diagnostic:
        baseline = read(BASE/'evaluation/u3040/interceptor'/side.name)['episodes']
        for a, b in zip(episodes, baseline):
            assert a['r3']['joint_actions_sha256'] == b['r3']['joint_actions_sha256'], a['seed']
            for k in ('ticks', 'winner', 'stored', 'placed', 'deaths', 'self_deaths', 'r4_pressure'):
                assert a[k] == b[k], (a['seed'], k)
            assert 'interceptor_diagnostic' in a
    atomic_json(receipt, dict(status='PASS', games=8, passive_action_parity=diagnostic,
        files={str(p):sha(p) for p in (target, side)}, unix=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=str(BASE/'diagnostic_v1'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    folder = Path(args.folder); folder.mkdir(parents=True, exist_ok=True)
    guard = lock(folder/'OWNER.lock'); p = read(folder/'PROTOCOL.json')
    if args.smoke:
        implementation = read(folder/'IMPLEMENTATION.json')
        assert implementation['protocol_sha256'] == sha(folder/'PROTOCOL.json')
        p['sources'] = dict(p['sources'], **implementation['sources'])
    start_time = time.time()
    atomic_json(folder/'EXECUTION_STATUS.json', dict(status='RUNNING', games=64, cpu_only=True, unix=time.time()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_shard, folder, p, start, 136+i, not args.smoke) for i, start in enumerate(range(0,64,8))]
        for f in futures:
            f.result()
            atomic_json(folder/'EXECUTION_STATUS.json', dict(status='RUNNING', accepted_shards=len(list(folder.glob('*.ACCEPTED.json'))), unix=time.time()))
    files = {str(f):sha(f) for f in sorted(folder.glob('shard_*.json*')) if f.suffix == '.json'}
    atomic_json(folder/'COMPLETE.json',dict(status='COMPLETE',games=64,wall_seconds=time.time()-start_time,
        passive_action_parity=not args.smoke,protocol_sha256=sha(folder/'PROTOCOL.json'),files=files,unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise
