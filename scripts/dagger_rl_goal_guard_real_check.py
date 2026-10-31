"""Reproduce goal0 replay exposure and verify the inactive kind-only guard."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import jax
from scripts import combat_item_engine as T, carrier_mechanism_engine as E, je10_route_oracle as O
from scripts.dagger_rl_route_goal_eligibility import guarded_teacher
from scripts.population_run_state import atomic_json, sha
from scripts.terminal_checkpoint_restore import digest


def run(base):
    base = Path(base)
    T.configure()
    runbase = base/'formal/sequential_anchor_s2026102661'
    items, refs = [], []
    for update, index in [(8, 12), (1, 18)]:
        cp = runbase/f'u{update:03d}'
        receipt = json.loads((cp/'COMMITTED.json').read_text())
        assert sha(cp/'state.pkl') == receipt['state_sha256']
        x = pickle.load((cp/'state.pkl').open('rb'))['auxiliary']
        take = lambda tr: jax.tree.map(lambda a: np.asarray(a).reshape((-1,)+a.shape[2:])[index:index+1], tr)
        items.append((take(x['states']), take(x['aux'])))
        refs.append(dict(checkpoint=str(cp.relative_to(base)), stream='auxiliary',
                         env_index=index, state_sha256=receipt['state_sha256']))
    states, aux = [jax.tree.map(lambda *a: np.concatenate(a), *[item[k] for item in items]) for k in [0, 1]]
    obs, glob, mm, bm = map(np.asarray, jax.jit(E.inputs)(states))
    raw = np.asarray(jax.jit(O.labels)(states, aux, mm, np.zeros(2, np.int32), obs[:, 5]))
    original = np.asarray(jax.jit(T.route_teacher)(states, aux))
    guarded = np.asarray(jax.jit(guarded_teacher(T.route_teacher, E.inputs, O.labels))(states, aux))
    assert raw[0, 6] == 0 and original[0].any() and not guarded[0].any()
    assert raw[1, 6] == 3 and raw[1, 5] == 0 and original[1].any()
    assert np.array_equal(original[1], guarded[1]) and not (guarded & ~original).any()
    cells = np.floor(states.pos[:, 0]).astype(int)
    danger = obs[np.arange(2), 5, cells[:, 0], cells[:, 1]]
    atomic_json(base/'REAL_GOAL_GUARD_CHECK.json', dict(passed=True, activated_in_R16=False,
        kind_only=True, inputs=refs, input_state_digest=digest(states), input_aux_digest=digest(aux),
        raw_labels=raw.tolist(), current_cell_danger=danger.tolist(),
        original_joint_actions=[np.argwhere(a).tolist() for a in original],
        guarded_joint_actions=[np.argwhere(a).tolist() for a in guarded],
        invalid_goal_removed=True, valid_real_cargo_preserved_exact=True,
        whole_legacy_active_guard_rejected=True, source_sha256=sha(Path(__file__)),
        environment_steps=0, optimizer_steps=0))
    cp = runbase/'u009'
    receipt = json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'state.pkl') == receipt['state_sha256']
    history = pickle.load((cp/'state.pkl').open('rb'))['history']
    rows = history['rows']
    candidates = np.flatnonzero((history['source'] == 28) & (history['arrival_update'] == 9) & ~rows[-1])
    matches = []
    for index in candidates:
        diff = [float(np.max(np.abs(a[index].astype(float)-b[0].astype(float))))
                for a, b in zip(rows[:4], [obs, glob, mm, bm])]
        if max(diff) < 1e-6:
            assert np.array_equal(rows[-2][index], original[0])
            matches.append((int(index), diff))
    assert matches
    ii, rr = np.flatnonzero(rows[-1]), np.flatnonzero(~rows[-1])
    rng = np.random.default_rng(2026102661*1000+9)
    if len(ii) and len(rr):
        draws = np.concatenate([rng.choice(ii, 4096), rng.choice(rr, 4096)])
        rng.shuffle(draws)
    else:
        draws = rng.integers(0, len(rows[0]), 8192)
    witness = [dict(history_index=i, input_max_difference=d,
                    arrival_id=int(history['arrival_id'][i]), actual_replay_draws=int((draws == i).sum()))
               for i, d in matches]
    assert sum(x['actual_replay_draws'] for x in witness) > 0
    atomic_json(base/'GOAL0_REPLAY_WITNESS_v2.json', dict(passed=True, source=refs[0],
        history_checkpoint_sha256=receipt['state_sha256'], kind0=True,
        full_joint_label_match=True, history_input_matches=witness,
        stage='u9 DAgger.2; each saved replay position used once in the declared epoch',
        scope='A specific real collector-to-history-to-supervised-loss witness, not total contamination rate or per-row Adam effect.',
        source_sha256=sha(Path(__file__)), environment_steps=0, optimizer_steps=0))
    print(json.dumps(dict(passed=True, kind0_removed=True, valid_real_cargo_exact=True,
                         raw_labels=raw.tolist(), current_cell_danger=danger.tolist(), replay_witness=witness)), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
