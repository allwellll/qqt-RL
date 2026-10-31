"""Bounded CPU Review audit; exact requested seeds, raw outcome parity."""
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, sha, lock

BASE = Path('runs/bomb_threat_v1')
OUT = BASE/'iter2/zero_bomb_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    guard = lock(OUT/'OWNER.lock')
    jobs = []
    for split, offsets in [('dev', [32, 36, 42, 44]), ('holdout', [26, 30, 36, 38])]:
        for label, name in [('parent', 'baseline_parent_dev' if split == 'dev' else 'baseline_parent'),
                            ('A3u720', 'bt_slot01_c000015_u000720'+('_holdout' if split == 'holdout' else ''))]:
            for offset in offsets:
                jobs.append(dict(name=name, offset=offset, key=f'{label}_{split}_{offset}'))
    jobs += [dict(name='bt_slot07_c000015_u000800', offset=i, key=f'A2u800_dev_{i}') for i in [34, 46]]
    atomic_json(OUT/'MANIFEST.json', dict(status='RUNNING', jobs=jobs, teacher=False, gpu=False,
        requested_games=18, core_budget=2, simulator_sha256=sha('web/sim-stock-window-v1.js'),
        instruction_sha256=sha('.hermes/STEER_zero_bomb_20261029_1420.md'),
        sources={str(p):sha(p) for p in map(Path, ['scripts/bomb_threat_zero_bomb_review.js',
            'scripts/bomb_threat_eval.js','scripts/expert_validation_js_eval.js',
            'scripts/dagger_rl_route.js','web/review_protocol.js','web/bun_hunter_bot.js'])}, unix=time.time()))

    def run(job):
        dest = OUT/job['key']
        if not (dest/'AUDIT.json').exists():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', QQT_STOCK_RULE='stock-window-v1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (OUT/(job['key']+'.log')).open('ab') as log:
                subprocess.run(['node', 'scripts/bomb_threat_zero_bomb_review.js',
                    str(BASE/'eval_requests'/(job['name']+'.json')), str(job['offset']), str(dest)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        r = json.loads((dest/'AUDIT.json').read_text())
        source = BASE/'eval'/job['name']/f"natural_greedy_{job['offset']//8*8:03d}.json.outcome.json"
        marker=json.loads((source.parent/'NATURAL_COMPLETE.json').read_text())
        assert marker['files'][source.name]==sha(source)
        assert r['policy_sha256']==marker['policy_sha256']
        evaluation=json.loads((dest/'evaluation.json').read_text())
        assert evaluation['model_json_sha256']==r['model_sha256']==sha(r['model'])
        old = next(x for x in json.loads(source.read_text())['episodes'] if x['seed'] == r['seed'])
        for k in ('ticks', 'done', 'winner', 'requests', 'placed', 'deaths'):
            assert r[k] == old[k], (job, k, r[k], old[k])
        proof = dict(key=job['key'], source=str(source), source_sha256=sha(source),
            audit=str(dest/'AUDIT.json'), audit_sha256=sha(dest/'AUDIT.json'),
            replay=str(dest/'review.json'), replay_sha256=sha(dest/'review.json'),
            trace_sha256=sha(dest/'trace.json'), outcome_parity=True)
        atomic_json(dest/'ACCEPTED.json', proof)
        return proof

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    atomic_json(OUT/'COMPLETE.json', dict(status='COMPLETE', games=len(results), results=results,
        all_terminal=True, exact_original_outcome_parity=True, production_review_verified=True, unix=time.time()))


if __name__ == '__main__':
    main()
