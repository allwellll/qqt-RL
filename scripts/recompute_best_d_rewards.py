"""Current Review ledger on observed Web transitions; no Actor/Critic inference.

This is an offline reconstruction with a fresh auxiliary ledger, not the PBT
training reward. No counterfactual A1 is claimed. Training paths are read-only.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import recompute_review_metrics as M
import numpy as np
import jax
import jax.numpy as jnp

assert jax.default_backend() == 'cpu'
# Match the deferred terminal flag used by the existing Review branch adapter.
source = M.source
source = source.replace('return (aux, pending, avoidable, chain, moved),',
                        "aux['terminal'] = aux['terminal'] | info['h_route_terminal']\n        return (aux, pending, avoidable, chain, moved),")
namespace = dict(M.ns)
exec(compile(source, '<current-review-ledger>', 'exec'), namespace)
evaluate = namespace['evaluate_chunk']


def run(file, verify):
    doc = json.loads(file.read_text())
    frames = [json.loads(line) for line in file.with_suffix('.trace.jsonl').open()]
    assert len(frames) == len(doc['actions']) + 1
    aux = M.ops.reward_state(1)
    zeros = jnp.zeros((1, 2), bool)
    carry = (aux, jnp.zeros((1, 2), jnp.int16), zeros, zeros, zeros)
    linger = np.zeros((2, 2, 13, 15), np.int8)
    causes = [[False, False], [False, False]]
    scene = dict(native=8, open=0, carry_home=2)[doc['meta']['scene']]
    converted = [M.convert(frames[0]['state'], linger, scene, causes)]
    infos = []
    for i, entry in enumerate(frames[1:]):
        assert entry['actions'] == doc['actions'][i]
        entry['prior_legal'] = frames[i]['legal']
        info = M.R.observed(frames[i]['state'], entry, causes)
        info['h_route_terminal'] = np.array([entry['state']['done']], bool)
        info['trigger_damage_source'] = np.array([entry['info']['causalDamageSource']], bool)
        pop = []
        for p in range(2):
            q = 1-p
            b = entry['bun_before']
            contact = frames[i]['state']['trapped'][p] > 0 and entry['info']['died'][p]
            contact = contact and b['alive'][q] and not b['trapped'][q]
            contact = contact and all(abs(b['pos'][p*2+k]-b['pos'][q*2+k]) < 41/40 for k in range(2))
            pop.append(q if contact else -1)
        info['native_trap_popper'] = np.array([pop], np.int32)
        infos.append(info)
        linger = M.advance_linger(linger, entry)
        converted.append(M.convert(entry['state'], linger, scene, causes))
    sparse, totals = [], np.zeros(2)
    chunk = 16
    for start in range(0, len(infos), chunk):
        n = min(chunk, len(infos)-start)
        indices = list(range(start, start+n)) + [start+n-1]*(chunk-n)
        stack = lambda values: jax.tree.map(lambda *x: np.stack(x), *values)
        before = stack([converted[i] for i in indices])
        after = stack([converted[i+1] for i in indices])
        info = stack([infos[i] for i in indices])
        carry, (total, parts, route) = evaluate(carry, (before, after, info))
        total, parts = map(np.asarray, (total, parts))
        assert np.isfinite(parts).all() and np.isfinite(total).all()
        np.testing.assert_allclose(parts[:n].sum(-1), total[:n], atol=2e-5)
        for j in range(n):
            row = [{k: float(v) for k, v in zip(M.COMPONENTS, parts[j, 0, p]) if v != 0}
                   for p in range(2)]
            if any(row):
                sparse.append([start+j+1, *row])
            totals += parts[j, 0].sum(-1)
    if verify:
        assert sparse == doc['rewards'], 'independent reward rerun differs'
        np.testing.assert_array_equal(totals, doc['summary']['reward'])
    else:
        used = {}
        root = Path.cwd().resolve()
        for name, module in list(sys.modules.items()):
            if name.startswith(('scripts.', 'jax_bomb.')):
                p = Path(module.__file__).resolve() if getattr(module, '__file__', None) else None
                if p and p.is_file():
                    relative = p.relative_to(root).as_posix()
                    used[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
        for f in ('scripts/diagnostic_reward.py', 'scripts/recompute_review_metrics.py',
                  'scripts/recompute_best_d_rewards.py'):
            used[f] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        doc['rewards'], doc['summary']['reward'] = sparse, totals.tolist()
        doc['meta'].update(reward_status='offline_recomputed_review_reward',
                           reward_components=list(M.COMPONENTS), reward_hashes=used,
                           review_reward=dict(schema='qqt.review-reward-only/v1', measured_count=len(infos),
                           component_names=list(M.COMPONENTS),
                           scope='现行Review JE10完整加法账本在生产Web状态/事件上的CPU离线复算；初始aux归零；不是ITER-PBT训练账本或历史rollout reward；没有Critic或反事实优势。',
                           limitations='清砖对拼按实际300tick结算；危险尚未消退的奖励保持pending，不在终局伪造额外结算tick；Web与JAX安全适配器存在跨引擎近似。'))
        # Final identity is regenerated by the publication script after reward reconstruction.
        file.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))
    print('VERIFIED' if verify else 'REWARD', file.stem, len(infos), totals.tolist(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for f in args.files:
        run(f, args.verify)
