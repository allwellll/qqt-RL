"""Real import order test for cc_reward_import_hook (CPU, one subprocess per variant): hook -> model_cuda_overlay.install()
-> import scripts.r3_items_reward must give the overlay module WITH the variant patch, without importing jax early.
Also shows the old install_on_import() was shadowed. usage: python -m scripts.cc_reward_import_hook_test
Writes runs/cc_train_20261105/training/league/REWARD_HOOK_TESTS.json."""
import json, os, subprocess, sys, time
from scripts.population_run_state import atomic_json

CODE = """
import os, sys, json
os.environ['JAX_PLATFORMS'] = 'cpu'; os.environ['CC_REWARD_RECEIPT'] = sys.argv[3]
import importlib; V = importlib.import_module(sys.argv[1])
if sys.argv[2] == 'new':
    from scripts.cc_reward_import_hook import hook; hook(V)
else:
    V.install_on_import()
early = 'jax' in sys.modules
from scripts.model_cuda_overlay import install; install()
from scripts import r3_items_reward as R
print(json.dumps(dict(early_jax=early, overlay=bool(getattr(R, 'CUDA_OVERLAY', False)), caps=[round(float(x), 3) for x in R.CAPS],
    patched=bool(getattr(R, '_cc_fountain_v1', False) or getattr(R, '_cc_attack_v2', False)), receipt=os.path.exists(sys.argv[3]))))
"""
OUT = {}
for module, caps in (('scripts.cc_fountain_reward_v1', [.36, .6, .36, .2, .2, .6, .4, .45]), ('scripts.cc_attack_reward_v2', [.36, .6, .36, .2, .2, .4, .2, .45])):
    for mode in ('new', 'old'):
        receipt = f'/tmp/cc_hook_{os.getpid()}_{module.rsplit(".", 1)[1]}_{mode}.json'
        r = subprocess.run([sys.executable, '-c', CODE, module, mode, receipt], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='.'))
        assert r.returncode == 0, r.stderr[-800:]
        d = json.loads(r.stdout.strip().splitlines()[-1])
        ok = (d['patched'] and d['caps'] == caps and d['receipt']) if mode == 'new' else not d['patched']
        ok = ok and d['overlay'] and not d['early_jax']
        OUT[f'{module}:{mode}'] = dict(passed=ok, result=d)
        assert ok, (module, mode, d)
atomic_json('runs/cc_train_20261105/training/league/REWARD_HOOK_TESTS.json', dict(status='PASS', tests=OUT, unix=time.time()))
print('PASS', json.dumps(OUT))
