"""Approximate current reward reconstruction on genuine Web trajectories.

Missing readable legacy dependencies can be loaded from the existing local
Python 3.12 bytecode. Every used module is hashed; this is explicitly NOT a
claim of training-time source/ledger equality.
"""
import builtins
import argparse
import inspect
import copy
import functools
import hashlib
import importlib.abc
import importlib.util
import importlib.machinery
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAXBOMB_RULE='bun',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
USED = {}


class ExistingDependencyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(('scripts.', 'jax_bomb.')):
            return None
        relative = Path(*fullname.split('.'))
        if (ROOT / relative.with_suffix('.py')).exists():
            return None
        cache = ROOT / relative.parent / '__pycache__' / (relative.name + '.cpython-312.pyc')
        if cache.is_file():
            USED[str(cache.relative_to(ROOT))] = hashlib.sha256(cache.read_bytes()).hexdigest()
            loader = importlib.machinery.SourcelessFileLoader(fullname, str(cache))
            return importlib.util.spec_from_loader(fullname, loader)
        return None


sys.meta_path.insert(0, ExistingDependencyFinder())
_original_open = builtins.open

def _mapped_open(name, *args, **kwargs):
    if isinstance(name, (str, Path)) and str(name).endswith('/jax_bomb/web/assets/maps/levels.json'):
        name = ROOT / 'web/assets/maps/levels.json'
    return _original_open(name, *args, **kwargs)

builtins.open = _mapped_open


@functools.lru_cache(maxsize=1)
def configure():
    import diagnostic_reward as R
    import jax
    import jax.numpy as jnp
    from scripts import je09_stage_h8_pbrs_ops as current
    R.reward = current
    R.COMPONENTS = R.COMPONENTS + ('route_shaped_reward',)
    source = inspect.getsource(R.evaluate_chunk.__wrapped__)
    source = source.replace("info = dict(info, danger_safe_resolution=resolved,",
        "info = dict(info, danger_exposure=jnp.any(danger.exposed, axis=-1), danger_safe_resolution=resolved,")
    source = source.replace("(total, terms, info['own_bomb_defeat'])",
        "(total, terms, info['own_bomb_defeat'], jnp.stack([metrics[k] for k in route_names], -1))")
    namespace = dict(R.__dict__, route_names=tuple(current.pbrs.METRICS))
    exec(compile(source, '<current-offline-reward-adapter>', 'exec'), namespace)
    return R, current, namespace['evaluate_chunk']


def run(file, verify=False, limit=None):
    import numpy as np
    import jax
    import jax.numpy as jnp
    R, current, evaluate = configure()
    doc = json.loads(file.read_text())
    trace = ROOT / 'runs/replay_review_20261017/reward-traces' / (file.stem + '.jsonl')
    frames = [json.loads(line) for line in trace.open()]
    assert len(frames) == len(doc['actions']) + 1
    aux = current.reward_state(1)
    zeros = jnp.zeros((1, 2), bool)
    carry = (aux, jnp.zeros((1, 2), jnp.int16), zeros, zeros, zeros)
    linger = np.zeros((2, 2, 13, 15), np.int8)
    scene = 0 if doc['meta'].get('arena') else {'native': 8, 'open': 0, 'carry_home': 2}[doc['meta']['scene']]
    converted = [R.convert(frames[0]['state'], linger, scene)]
    infos, causes = [], [[False, False], [False, False]]
    for i, entry in enumerate(frames[1:]):
        entry['prior_legal'] = frames[i]['legal']
        info = R.observed(frames[i]['state'], entry, causes)
        info['h_route_terminal'] = np.array([entry['state']['done']], bool)
        infos.append(info)
        linger = np.maximum(linger - 1, 0)
        for source in entry['explosions']:
            if source['owner'] in (0, 1):
                for axis, key in enumerate(('horzCovered', 'vertCovered')):
                    linger[source['owner'], axis][np.array(source[key], bool).reshape(13, 15)] = 2
        converted.append(R.convert(entry['state'], linger, scene))
    results = []
    for start in range(0, min(len(infos), limit or len(infos)), R.CHUNK):
        n = min(R.CHUNK, len(infos) - start)
        indices = list(range(start, start + n)) + [start + n - 1] * (R.CHUNK - n)
        before = jax.tree.map(lambda *x: np.stack(x), *[converted[i] for i in indices])
        after = jax.tree.map(lambda *x: np.stack(x), *[converted[i + 1] for i in indices])
        info = jax.tree.map(lambda *x: np.stack(x), *[infos[i] for i in indices])
        carry, (reward_values, values, own, route) = evaluate(carry, (before, after, info))
        reward_values, values, own, route = map(np.asarray, (reward_values, values, own, route))
        assert np.isfinite(values).all() and np.isfinite(route).all()
        np.testing.assert_allclose(values[:n].sum(-1), reward_values[:n], atol=2e-5,
                                   err_msg='current reward component sum')
        for j in range(n):
            components = [{k: float(v) for k, v in zip(R.COMPONENTS, values[j, 0, p])} for p in range(2)]
            metrics = [{k: float(v) for k, v in zip(current.pbrs.METRICS, route[j, 0, p])} for p in range(2)]
            results.append(dict(tick=start + j + 1, components=components,
                total=[sum(c.values()) for c in components], route=metrics))
        if start % 256 == 0:
            print(file.stem, start + n, 'current reward recomputed', flush=True)
    provenance = dict(USED)
    for name, module in list(sys.modules.items()):
        if name.startswith(('scripts.', 'jax_bomb.')) or name == 'diagnostic_reward':
            source = getattr(module, '__file__', None)
            if source and Path(source).is_file():
                try:
                    relative = str(Path(source).resolve().relative_to(ROOT))
                except ValueError:
                    continue
                provenance[relative] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    provenance['scripts/recompute_v36_current_reward.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = ROOT / 'runs/replay_review_20261017/reward-traces' / (file.stem + '.reward.json')
    result = dict(schema='qqt.offline-current-reward/v1', status='offline_recomputed_current_reward',
        reward_source='current je09_stage_h8_pbrs_ops.combined_reward + diagnostic_reward Web state/event adapter',
        component_names=list(R.COMPONENTS), source_hashes=provenance,
        assumptions=['Cold-start current aux at the declared episode initial state; not training-time aux.',
            'Web native event/custody and explosion-owner adapter; current CPU safety detectors.',
            'scene mapping native=8/open=0/carry_home=2/tactical=0; terminal from actual Web Sim done.',
            'danger_exposure approximated by any current selected-action exposure; bridge window=120, scale=2.',
            'Unavailable readable legacy modules loaded from existing Python 3.12 bytecode with hashes; no sealed-source equivalence claim.'],
        ticks=results)
    if verify:
        old = json.loads(out.read_text())
        assert old == result, 'independent current reward rerun mismatch'
        print(file.stem, 'CURRENT REWARD EXACT RERUN', flush=True)
    else:
        out.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    for file in args.files:
        run(file, args.verify, args.limit)


if __name__ == '__main__':
    main()
