"""Do not teach pursuit with action-independent setup kill reward."""
import jax
import jax.numpy as jnp
import numpy as np


def native_fixture(n=4):
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    env.prepare()
    key = jax.random.PRNGKey(2026101519)
    states = scenes.init_batch(key, n, jax.nn.one_hot(9, 10))
    actions = jnp.tile(jnp.array([[[4, 0], [4, 0]]]), (n, 1, 1))
    after, _, info = jax.jit(jax.vmap(lambda state, action, k:
        env.step(state, action, k, False, True)))(states, actions, jax.random.split(key, n))
    return states, after, info


def test_only_declared_initial_fixture_kill_is_filtered():
    from scripts.je09_bun_ops import fixture_retention_info
    states, _, info = native_fixture()
    original = np.asarray(info['surviving_kill']).copy()
    adjusted, suppressed = jax.jit(fixture_retention_info)(states, info)
    assert int(suppressed.sum()) == 4
    assert int(adjusted['surviving_kill'].sum()) == 0
    np.testing.assert_array_equal(info['surviving_kill'], original)
    for name in info:
        if name != 'surviving_kill':
            np.testing.assert_array_equal(adjusted[name], info[name])
    for changed in [states._replace(core=states.core._replace(t=states.t + 1)),
                    states._replace(scene=jnp.zeros_like(states.scene)),
                    states._replace(trapped=states.trapped * 2),
                    states._replace(trapped_causal=jnp.zeros_like(states.trapped_causal))]:
        adjusted, suppressed = fixture_retention_info(changed, info)
        assert not bool(suppressed.any())
        np.testing.assert_array_equal(adjusted['surviving_kill'], original)


def test_production_reward_filters_fixture_kill_but_preserves_real_drop(monkeypatch):
    from scripts import je09_bun_ops as ops
    states, after, info = native_fixture()
    settled = jnp.ones(states.pos.shape[0], bool)
    aux = ops.reward_state(states.pos.shape[0])
    reward, ledger, parts, metrics = jax.jit(ops.combined_reward)(states, after, info, aux, settled)
    assert int(metrics['fixture_kill_suppressed'].sum()) == 4
    assert float(parts[..., 2].sum()) == 0
    assert bool(ledger['bridge']['active'].any())
    assert bool((ledger['drop']['credit'] >= 0).any())
    # Counterfactual changes only retention's surviving_kill input. Native
    # drop/credit, potential, milestone and safety inputs are identical.
    monkeypatch.setattr(ops, 'fixture_retention_info',
                        lambda state, event: (event, jnp.zeros_like(event['surviving_kill'])))
    old_reward, _, old_parts, old_metrics = jax.jit(lambda s, ns, i, a, d:
        ops.combined_reward(s, ns, i, a, d))(states, after, info, aux, settled)
    np.testing.assert_allclose(old_reward - reward, 16. * info['surviving_kill'], atol=1e-5)
    np.testing.assert_array_equal(parts[..., :2], old_parts[..., :2])
    for name in ['bridge_drop', 'bridge_pickup', 'drop_bonus', 'bridge_reward', 'bridge_clawback']:
        np.testing.assert_array_equal(metrics[name], old_metrics[name])
