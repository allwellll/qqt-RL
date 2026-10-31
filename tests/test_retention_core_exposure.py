import jax
import jax.numpy as jnp
import numpy as np
from test_retention_core_replay import rows
from scripts import retention_core_replay as core


def test_full_exposure_preserves_all_nonreturn_history_current_and_joint_pairs():
    from scripts.retention_core_exposure import replace_history
    data = rows(12)
    bank = core.merge(core.empty(data, per_stratum=4), data)
    replay = {k: v for k, v in data.items() if k not in ['scene', 'custody']}
    replay['obs'] = replay['obs'] + 100
    replay['kind'] = replay['kind'].at[6].set(1).at[7].set(2)
    replay['preparation'] = replay['preparation'].at[8].set(True)
    replay['valid'] = replay['valid'].at[9].set(False)
    result, count = jax.jit(replace_history, static_argnames='history_start')(
        replay, bank, jax.random.PRNGKey(5), history_start=6)
    assert int(count) == 2
    for name in replay:
        np.testing.assert_array_equal(result[name][:10], replay[name][:10])
    assert (result['obs'][10:] < 100).all()
    np.testing.assert_array_equal(result['joint_target'][10:], bank['joint_target'][:2])
    assert set(result) == set(replay)


def test_empty_or_unproved_bank_cannot_replace_history():
    from scripts.retention_core_exposure import replace_history
    data = rows(8)
    replay = {k: v for k, v in data.items() if k not in ['scene', 'custody']}
    for bank in [core.empty(data, per_stratum=2), dict(data, custody=jnp.zeros(8, bool))]:
        result, count = replace_history(replay, bank, jax.random.PRNGKey(5), 4)
        assert int(count) == 0
        for name in replay:
            np.testing.assert_array_equal(result[name], replay[name])


def test_full_aggregate_reports_actual_all_slot_probability_and_keeps_fifo(monkeypatch):
    from scripts import retention_core_exposure as full
    from scripts import return_error_replay as baseline
    from scripts.prestock_joint_replay import buffer
    from test_prestock_joint_replay import batch
    monkeypatch.setattr(baseline, 'target_mass', lambda actor, d: jnp.full(d['valid'].size, .5))
    monkeypatch.setattr(core, 'replace_history', full.replace_history)
    data = list(batch())
    data[1] = jnp.zeros((1,4,24)).at[:,:,9].set(1).at[:,:,13:15].set(1)
    data[-1] = data[-1].at[:,:,0].set(8).at[:,:,58].set(1)
    data = tuple(data)
    fifo = buffer(capacity=4, additions=2, obs_shape=(1,1,1), global_size=24)
    template = {k: v for k, v in fifo.items() if k not in ['cursor', 'additions']}
    template.update(scene=jnp.zeros(4,jnp.int32), custody=jnp.zeros(4,bool))
    extended = dict(fifo, retention_core=core.empty(template, per_stratum=2))
    for generation in [1, 2, 3]:
        key = jax.random.PRNGKey(generation)
        expected, old_replay, _ = core._BASE_AGGREGATE(fifo, data, key, jnp.int32(generation), {})
        updated, replay, stats = jax.jit(full.aggregate)(extended, data, key, jnp.int32(generation), {})
        for name in expected:
            np.testing.assert_array_equal(expected[name], updated[name])
        np.testing.assert_array_equal(replay['obs'][:2], old_replay['obs'][:2])
        count = int(stats['core_replaced_history_return'])
        assert count == (0 if generation == 1 else 2)
        assert float(stats['core_selected_joint_mass_sum']) == .5 * count
        fifo, extended = expected, updated
