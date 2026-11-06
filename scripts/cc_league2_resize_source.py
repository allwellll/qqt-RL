"""Derive a COMMITTED source checkpoint with N envs from a 2048-env one (CPU). Params/Adam/bank/RNG of the optimizer
unchanged; every per-env leaf (leading dim == source envs) of states/aux/keys is tiled; copies beyond the first get
keys folded with their copy index (n need not be a multiple: tile then truncate), so duplicated in-flight episodes diverge from the next tick.
usage: python -m scripts.cc_league2_resize_source SRC_CHECKPOINT DST_CHECKPOINT N"""
import json, os, pickle, shutil, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json, sha


def main(src, dst, n):
    src, dst = Path(src), Path(dst); cm = json.loads((src / 'COMMITTED.json').read_text())
    assert sha(src / 'state.pkl') == cm['state_sha256']
    payload = pickle.loads((src / 'state.pkl').read_bytes())
    import jax
    m = int(np.asarray(payload['keys']).shape[0]); r = -(-n // m)  # copies, last one truncated
    tile = lambda a: np.concatenate([np.asarray(a)] * r, 0)[:n] if np.ndim(a) and np.shape(a)[0] == m else a
    keys = np.asarray(payload['keys'])
    payload['keys'] = np.concatenate([keys] + [np.asarray(jax.vmap(lambda k: jax.random.fold_in(k, 0x5E7 + c))(keys)) for c in range(1, r)], 0)[:n]
    for name in ('states', 'aux'):
        payload[name] = jax.tree.map(tile, payload[name])
    leaves = [np.shape(x)[0] for x in jax.tree.leaves((payload['states'], payload['aux'])) if np.ndim(x)]
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / 'state.pkl').open('wb') as f:
        pickle.dump(payload, f, protocol=5)
    for name in ('policy.pt', 'policy_bun_mode0.pt'):
        shutil.copyfile(src / name, dst / name)
    receipt = dict(cm, checkpoint=str(dst), state_sha256=sha(dst / 'state.pkl'), policy_sha256=sha(dst / 'policy.pt'),
                   derived=dict(source=str(src), source_state_sha256=cm['state_sha256'], envs=[m, n], method='tile per-env leaves, fold keys per copy',
                                leading_dims=sorted(set(leaves)), script_sha256=sha('scripts/cc_league2_resize_source.py')), unix=time.time())
    assert receipt['policy_sha256'] == cm['policy_sha256']
    atomic_json(dst / 'COMMITTED.json', receipt); print(dst, receipt['state_sha256'][:12], sorted(set(leaves)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
