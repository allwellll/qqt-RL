"""Actual CPU reset/ledger/collector checks before the main owner's handoff."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JAXBOMB_RULE='bun')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from scripts.population_run_state import atomic_json, sha
from scripts import combat_specialist_g_league_train as worker
from scripts import combat_regions_mixed_v2 as mixed
from scripts.bomb_threat_a3sr_spawn import base_mask

B = mixed.B


def finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))


def main():
    started = time.monotonic()
    M = worker.setup()
    protocol = json.loads((B / 'inputs/source_protocol.json').read_text())
    bs, ba = M.banks(None, None, json.loads(Path(protocol['r1_bank_manifest']).read_text()))
    spawns = j.asarray(json.loads(Path(protocol['spawn_manifest']).read_text())['spawn_cells'])
    n = 64; keys = jax.random.split(jax.random.PRNGKey(2026547100), n); seats = j.arange(n) % 2
    bun = mixed.bun_backend()
    st, ax = jax.jit(lambda ks: bun.fresh(ks, bs, ba, j.zeros(n, bool), spawns, seats))(keys)
    jax.block_until_ready(st)
    course = np.asarray(ax['a3sr']['mr_region']) > 0
    assert course.any() and (~course).any()
    level = next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x.get('qqt_id') == 806)
    bases = base_mask(level)
    cells = np.floor(np.asarray(st.pos)).astype(int)
    for i in np.flatnonzero(course):
        assert not np.array_equal(cells[i, 0], cells[i, 1])
        assert np.array_equal(np.asarray(st.brick[i]), np.asarray(level['brick'], bool).reshape(13, 15))
        for r, c in cells[i]:
            assert not bool(st.wall[i, r, c] | st.brick[i, r, c] | st.core.pushable[i, r, c])
            assert not bases[r, c]
    assert np.all(np.asarray(st.bun_carried)[course] < 0)
    assert np.all(np.asarray(st.fuse)[course] <= 0)
    assert np.all(np.asarray(ax['provenance_source'])[course] < 0)
    meta = bun.metadata(st, np.zeros(n, np.int8), j.zeros(n, bool), seats, ax['r1_attribute_highwater'])
    ax = dict(ax, combat_mix=dict(meta, opponent_generation=j.zeros(n, j.int32)))
    # Sample every frozen row by exhaustive metadata/geometry audit; runtime
    # distribution is checked separately rather than inferred from the config.
    for split in ('train', 'dev', 'holdout'):
        m = json.loads((B / (split + '_starts.json')).read_text())
        for region in set(row['region'] for row in m['rows']):
            rows = [row for row in m['rows'] if row['region'] == region]
            assert len(rows) == 48
            assert sum(r['learner_seat'] for r in rows) == 24
            for distance in (2, 4, 6):
                assert sum(r['distance'] == distance for r in rows) == 16
        assert all(row['region'] == 'center' for row in m['missing'])
    # Migration preserves all existing reward leaves and continuing episodes.
    old = dict(ax, a3sr={k: v for k, v in ax['a3sr'].items() if not k.startswith(('mr_', 'ca_'))})
    migrated = mixed.migrate_aux(old)
    from scripts.terminal_checkpoint_restore import digest
    assert digest({k: v for k, v in migrated.items() if k != 'a3sr'}) == digest({k: v for k, v in old.items() if k != 'a3sr'})
    assert not np.asarray(migrated['a3sr']['mr_region']).any()
    # Real transition, repeated beyond the 30-tick reward quarantine. Regional
    # stationary starts outside cages must never invent pickup or history pay.
    ids = j.asarray(np.flatnonzero(course)[:8])
    ss, aa, kk = jax.tree.map(lambda a: a[ids], (st, ax, keys))
    actions = j.zeros((len(ids), 2, 2), j.int32).at[:, :, 0].set(4)
    def rollout(ss, aa, kk):
        def step(carry, _):
            s, a, k = carry
            nk, ek = j.swapaxes(jax.vmap(lambda x: jax.random.split(x, 2))(k), 0, 1)
            after, na, r, done, info, parts, metrics = bun.transition(s, a, actions, ek, 0.)
            return (after, na, nk), (r, info['steal'], info['death'], metrics['r3_pay'], metrics['approach'])
        return jax.lax.scan(step, (ss, aa, kk), None, length=32)
    result, ledger = jax.jit(rollout)(ss, aa, kk)
    jax.block_until_ready(result)
    rewards, steals, deaths, r3_pay, approach = map(np.asarray, ledger)
    assert finite((result, ledger))
    assert not steals.any() and not deaths.any()
    # The inherited R1 discounted potential pays (gamma-1)*negative_phi
    # even while stationary. Identify it explicitly; do not mislabel that
    # known baseline term as a course relocation/history payment.
    expected = np.zeros_like(rewards)
    expected[:, np.arange(len(ids)), np.asarray(seats[ids])] = approach
    assert np.max(np.abs(rewards - expected)) <= 1e-6, ('unexplained stationary reward', rewards.max())
    assert np.max(r3_pay) <= 1e-6, ('free historical R3 reward', r3_pay.max())
    atomic_json(B / 'CPU_VALIDATION.json', dict(status='PASS', samples=n, regional=int(course.sum()),
        stationary_samples=len(ids), stationary_ticks=32, max_stationary_reward=float(rewards.max()),
        inherited_stationary_discount_max=float(approach.max()),
        unexplained_stationary_reward_max=float(np.abs(rewards-expected).max()),
        max_stationary_r3_pay=float(r3_pay.max()), finite=True, no_free_history=True,
        no_initial_death=True, no_fabricated_pickup=True, no_cage_starts=True,
        original_native_regional_map=True, existing_reward_ledgers_unchanged=True,
        migrated_reward_digest=digest({k: v for k, v in migrated.items() if k != 'a3sr'}),
        train_manifest_sha256=sha(B / 'train_starts.json'),
        region_counts=np.bincount(np.asarray(ax['a3sr']['mr_region']), minlength=9).tolist(),
        seconds=time.monotonic()-started, unix=time.time()))
    print((B / 'CPU_VALIDATION.json').read_text())


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(B / 'CPU_FAILURE.json', dict(status='FAIL', type=type(error).__name__,
            detail=str(error).replace(str(Path.cwd()), '<WORKSPACE>'), unix=time.time()))
        raise
