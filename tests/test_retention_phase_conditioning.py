"""Deployment-visible phase features; no teacher/context authority is read."""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from scripts import retention_phase_conditioning as phase
from jax_bomb import jax_net as net


def inputs():
    obs = np.zeros((1, 24, 13, 15), np.float32)
    obs[0, 0, 8, 10] = 1
    obs[0, 14, 1:4, 1:4] = 1
    state = np.zeros((1, 24), np.float32)
    state[:, 9:11] = 1
    state[:, 20] = 1
    return obs, state


def params():
    return net.init_transformer(jax.random.PRNGKey(71), 24, 13, 15,
                                embed=8, depth=1, heads=4)


def test_phase_truth_table_current_observables_and_overlap_priority():
    obs, gv = inputs()
    assert int(phase.phase_id(obs, gv)[0]) == phase.COMBAT_CONTROL
    gv[:, 10] = 0
    assert int(phase.phase_id(obs, gv)[0]) == phase.STOCK_WINDOW
    gv[:, 10] = 1 + 1/60
    assert int(phase.phase_id(obs, gv)[0]) == phase.STOCK_WINDOW
    gv[:, 10] = 1  # no historical window/credit latch is available
    assert int(phase.phase_id(obs, gv)[0]) == phase.COMBAT_CONTROL
    obs[:, 17, 8, 9] = 1
    assert int(phase.phase_id(obs, gv)[0]) == phase.DROP_RECOVERY
    gv[:, 10] = 0
    assert int(phase.phase_id(obs, gv)[0]) == phase.DROP_RECOVERY
    gv[:, 13:15] = 1
    assert int(phase.phase_id(obs, gv)[0]) == phase.CARRY_RETURN
    obs[:, 0] = 0
    obs[:, 0, 4, 2] = 1
    assert int(phase.phase_id(obs, gv)[0]) == phase.CAGE_DELIVERY
    gv[:, 9] = 0
    assert int(phase.phase_id(obs, gv)[0]) == phase.COMBAT_CONTROL
    gv[:, 9] = 1.5
    gv[:, 13:15] = 0
    assert int(phase.phase_id(obs, gv)[0]) == phase.COMBAT_CONTROL
    gv[:, 9] = 1
    obs[:, 17] = 0
    gv[:, 20] = 0
    assert int(phase.phase_id(obs, gv)[0]) == phase.COMBAT_CONTROL


def test_delivery_mass_boundary_both_views_and_jit():
    obs, gv = inputs()
    gv[:, 13:15] = 1
    obs[:, 0] = 0
    obs[:, 0, 4, 2] = .5
    obs[:, 0, 5, 2] = .5
    assert int(phase.phase_id(obs, gv)[0]) == phase.CAGE_DELIVERY
    obs[:, 0, 4, 2] = .499
    obs[:, 0, 5, 2] = .501
    assert int(phase.phase_id(obs, gv)[0]) == phase.CARRY_RETURN
    encoded = phase.features(np.concatenate([obs, obs[:, :, ::-1, ::-1]]),
                             np.concatenate([gv, gv]))
    np.testing.assert_array_equal(encoded, [[0, 0, 0, 1, 0]] * 2)
    np.testing.assert_array_equal(jax.jit(phase.features)(obs, gv), encoded[:1])
    assert phase.features(obs, gv).shape == (1, 5)


def test_old_model_exact_and_migration_is_zero_only_for_actor_and_critic():
    obs, gv = inputs()
    p = params()
    before = pickle.dumps(p)
    reference = net.transformer_forward(p, jnp.asarray(obs), jnp.asarray(gv))
    actual = phase.forward(p, obs, gv)
    for a, b in zip(reference, actual):
        np.testing.assert_array_equal(a, b)
    pair = phase.migrate({'actor': p, 'critic': p})
    assert pickle.dumps(p) == before
    for role in ['actor', 'critic']:
        q = pair[role]
        assert q['state_w'].shape == (29, 8)
        np.testing.assert_array_equal(q['state_w'][:24], p['state_w'])
        np.testing.assert_array_equal(q['state_w'][24:], 0)
        for a, b in zip(reference, phase.forward(q, obs, gv)):
            np.testing.assert_array_equal(a, b)
        for name in p.keys() - {'state_w'}:
            for a, b in zip(jax.tree.leaves(q[name]), jax.tree.leaves(p[name])):
                np.testing.assert_array_equal(a, b)
    recovered = pickle.loads(pickle.dumps(pair))
    for a, b in zip(jax.tree.leaves(pair), jax.tree.leaves(recovered)):
        np.testing.assert_array_equal(a, b)


def test_both_networks_consume_phase_and_phase_rows_receive_gradient():
    obs, gv = inputs()
    pair = phase.migrate({'actor': params(), 'critic': params()})
    combat = jnp.array([[1, 0, 0, 0, 0]], jnp.float32)
    carry = jnp.array([[0, 0, 0, 1, 0]], jnp.float32)
    for role, head in [('actor', 0), ('critic', 3)]:
        q = pair[role]
        def loss(weight):
            test = dict(q, state_w=weight)
            logits = phase.forward(test, obs, gv)[head]
            return logits[0, 0]
        grad = jax.grad(loss)(q['state_w'])
        assert np.linalg.norm(np.asarray(grad[24:])) > 0
        q = dict(q, state_w=q['state_w'].at[24:].set(
            jnp.arange(40, dtype=jnp.float32).reshape(5, 8) * .1))
        first = phase.forward(q, obs, gv, feature_override=combat)[head]
        other = phase.forward(q, obs, gv, feature_override=carry)[head]
        assert np.max(np.abs(np.asarray(first-other))) > 1e-6


def test_unrelated_fields_cannot_change_phase_or_mutate_input():
    obs, gv = inputs()
    original = (obs.copy(), gv.copy())
    expected = phase.features(obs, gv)
    for index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 21, 22, 23]:
        changed = gv.copy(); changed[:, index] = 99
        np.testing.assert_array_equal(phase.features(obs, changed), expected)
    np.testing.assert_array_equal(obs, original[0])
    np.testing.assert_array_equal(gv, original[1])


def test_native_encoder_both_seats_and_real_stock_pickup_transition():
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    env.prepare()
    batch = scenes.init_batch(jax.random.PRNGKey(701), 2, jax.nn.one_hot(8, len(scenes.NAMES)))
    start = jax.tree.map(lambda x:x[0], batch)
    start = start._replace(trapped=jnp.array([0, 10], start.trapped.dtype))
    for seat, expected in [(0, phase.STOCK_WINDOW), (1, phase.COMBAT_CONTROL)]:
        obs=env.make_obs(start,seat)[None];gv=env.global_vec(start,seat)[None]
        assert int(phase.phase_id(obs,gv)[0])==expected
    # Use the real step boundary already covered by causal-init regressions;
    # the feature transitions on the resulting native cargo, never a scene ID.
    base=env._BUN_BASES[0]
    pos=batch.pos.at[:,1].set(base.astype(jnp.float32)+jnp.array([1.5,3.05]))
    before=batch._replace(core=batch.core._replace(pos=pos,
        brick=jnp.zeros_like(batch.brick),wall=jnp.zeros_like(batch.wall)))
    joint=jnp.zeros((2,2,2),jnp.int32).at[:,0,0].set(4).at[:,1,0].set(2)
    after,_,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        before,joint,jax.random.split(jax.random.PRNGKey(702),2))
    assert bool(info['steal'][:,1].all())
    assert bool((after.bun_carried[:,1]==0).all())
    sample=jax.tree.map(lambda x:x[0],after)
    encoded=phase.phase_id(env.make_obs(sample,1)[None],env.global_vec(sample,1)[None])
    assert int(encoded[0])==phase.CARRY_RETURN
