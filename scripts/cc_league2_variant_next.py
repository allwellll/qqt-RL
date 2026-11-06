"""Write the next protocol of a reward-variant arm that switches to cc_league2_supervise_variant (working import hook).
Continues from the arm's latest COMMITTED save (same arm, same seed, same chunk end); the previous protocol is added
to compatible_checkpoint_protocol_sha256. CPU only, run after the arm stopped at a save boundary.
usage: python -m scripts.cc_league2_variant_next ARM
"""
import json, sys, time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/cc_train_20261105/training')
FILES = ['scripts/cc_reward_import_hook.py', 'scripts/cc_reward_import_hook_test.py', 'scripts/cc_league2_train_variant.py',
         'scripts/cc_league2_supervise_variant.py', 'scripts/cc_league2_variant_next.py', 'scripts/model_cuda_overlay.py',
         'scripts/r3_items_reward.py', 'scripts/cc_fountain_reward_v1.py', 'scripts/cc_attack_reward_v2.py']


def main(arm):
    family = f'CC_LEAGUE2_{arm}'
    chain = sorted((BASE / 'protocols').glob(f'{family}_V*.json'), key=lambda p: int(p.stem.rsplit('_V', 1)[1]))
    prev = chain[-1]; p = json.loads(prev.read_text())
    out = Path(p['namespace']) / arm
    saved = int(json.loads(sorted(out.glob('u*/COMMITTED.json'))[-1].read_text())['update'])
    tests = BASE / 'league/REWARD_HOOK_TESTS.json'; assert json.loads(tests.read_text())['status'] == 'PASS'
    target = int(p['chunk_start_update']) + int(p['chunk_updates'])
    nxt = BASE / 'protocols' / f'{family}_V{int(prev.stem.rsplit("_V", 1)[1]) + 1}.json'
    assert not nxt.exists(), nxt
    for name in FILES:
        p['source_sha256'][name] = sha(name)
    p.update(chunk_start_update=saved, chunk_updates=target - saved if target > saved else int(p['league2_chunk_updates']),
             compatible_checkpoint_protocol_sha256=p['compatible_checkpoint_protocol_sha256'] + [sha(prev)])
    p['cc'] = dict(p['cc'], reward_hook_fix=dict(
        unix=time.time(), previous=str(prev), resumed_from=saved, tests=str(tests), tests_sha256=sha(tests),
        trainer='scripts.cc_league2_train_variant', supervisor='scripts.cc_league2_supervise_variant',
        reason='install_on_import() finder was shadowed by model_cuda_overlay finder: variant never applied in previous '
               'protocol(s) (base r3 coefficients, PRESSURE/r3_pressure == .01). Updates before this point are base reward.'))
    atomic_json(nxt, p); print(nxt, sha(nxt), 'resume', saved, 'chunk', p['chunk_updates'])


if __name__ == '__main__':
    main(sys.argv[1])
