"""Inactive CPU-only fixed own-label fit before/after the declared switch."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import time
import pickle
import numpy as np
import jax
import jax.numpy as j
from scripts.dagger_rl_r16_run import B, read
from scripts.population_run_state import atomic_json, sha, lock
from scripts import combat_item_engine as T, carrier_mechanism_engine as E
from scripts.terminal_checkpoint_restore import digest


@jax.jit
def predict(params, o, g, mm, bm, label):
    mv, ab, _, _ = E.train.net_forward(params, 'transformer', o, g)
    lm = jax.nn.log_softmax(j.where(mm, mv, -1e9))
    la = jax.nn.log_softmax(j.where(bm, ab, -1e9))
    mass = j.exp(jax.nn.logsumexp(j.where(label, lm[:, :, None]+la[:, None, :], -1e9), (1, 2)))
    ability = j.argmax(la, -1)
    return label[j.arange(len(o)), j.argmax(lm, -1), ability], mass, label.any(1)[j.arange(len(o)), ability]


def run():
    unique = lock(B/'fixed_fit.lock')
    T.configure()
    p = read(B/'PROTOCOL.json')
    parent = pickle.load((B/'inputs/parent.pkl').open('rb'))['params']
    while not (B/'USER_STOP.json').exists():
        for seed in p['paired_seeds']:
            for arm in p['arms']:
                base = B/'formal'/f'{arm}_s{seed}'
                out = base/'FIXED_PRE_POST_SWITCH_FIT.json'
                if out.exists() or not (base/'u032/COMMITTED.json').exists():
                    continue
                xs = {}
                for u in [1, 16, 17, 32]:
                    cp = base/f'u{u:03d}'
                    assert sha(cp/'state.pkl') == read(cp/'COMMITTED.json')['state_sha256']
                    xs[u] = pickle.load((cp/'state.pkl').open('rb'))
                groups = {}
                for ref in [1, 16]:
                    h = xs[ref]['history']
                    rows = h['rows']
                    for source, origin in [('natural', h['source'] < 0), ('real_pickup_bank', h['source'] >= 0)]:
                        families = dict(route=~rows[-1], release=rows[-1] & rows[-2][:, :, 2].any(1),
                                        contact=rows[-1] & rows[-2][:, :, 0].any(1))
                        for family, eligible in families.items():
                            ix = np.flatnonzero(origin & eligible)
                            ix = ix[np.argsort(h['arrival_id'][ix])][:64]
                            key = f'u{ref:03d}/{source}/{family}'
                            result = dict(rows=len(ix), eligible_memory_rows=int((origin & eligible).sum()), fit={})
                            if len(ix):
                                result['fixed_input_digest'] = digest(tuple(a[ix] for a in rows))
                                padded = np.pad(ix, (0, 64-len(ix)), mode='edge')
                                for name, params in [('parent', parent)]+[(str(u), xs[u]['params']) for u in [1, 16, 17, 32]]:
                                    match, mass, ability_match = map(np.asarray, predict(params, *[a[padded] for a in rows[:5]]))
                                    assert np.isfinite(mass).all()
                                    result['fit'][name] = dict(joint_matches=int(match[:len(ix)].sum()),
                                        ability_matches=int(ability_match[:len(ix)].sum()), joint_mass_mean=float(mass[:len(ix)].mean()))
                            groups[key] = result
                atomic_json(out, dict(passed=True, groups=groups,
                    reference_state_sha256={str(u): sha(base/f'u{u:03d}/state.pkl') for u in [1, 16, 17, 32]},
                    selection='own u1/u16 retained memory first<=64 arrivals per source/family, not fit/outcome selected',
                    scope='Same input/mask/label functional fit; not independent natural ability, Q or causal forgetting.',
                    environment_steps=0, optimizer_steps=0))
        outputs = [B/'formal'/f'{a}_s{s}'/'FIXED_PRE_POST_SWITCH_FIT.json' for s in p['paired_seeds'] for a in p['arms']]
        saved = sum(x.exists() for x in outputs)
        atomic_json(B/'FIXED_FIT_STATUS.json', dict(saved=saved, expected=9, controls_training=False))
        if saved == 9:
            atomic_json(B/'FIXED_FIT_COMPLETE.json', dict(passed=True, runs=9, environment_steps=0, optimizer_steps=0))
            return
        if (B/'TECHNICAL_FAILURE.json').exists():
            return
        time.sleep(10)


if __name__ == '__main__':
    run()
