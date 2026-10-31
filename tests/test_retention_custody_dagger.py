import jax
import jax.numpy as jnp
import numpy as np


def _batch(custody, age, valid=True):
    n = len(custody)
    obs = jnp.arange(n, dtype=jnp.float32).reshape(1, n, 1, 1, 1)
    gv = jnp.zeros((1, n, 24), jnp.float32)
    legal = jnp.ones((1, n, 5), bool)
    ability = jnp.ones((1, n, 3), bool)
    context = jnp.zeros((1, n, 115), jnp.float32)
    context = context.at[:, :, 53].set(jnp.asarray(valid, jnp.float32))
    context = context.at[:, :, 54].set(3)
    context = context.at[:, :, 58].set(jnp.asarray(custody, jnp.float32))
    context = context.at[:, :, 95].set(1)
    context = context.at[:, :, 96].set(1)
    context = context.at[:, :, 114].set(jnp.asarray(age, jnp.float32))
    return (obs, gv, None, None, None, None, (legal, ability), context)


def test_short_window_requires_native_custody_and_expires_at_120_ticks():
    from scripts.retention_custody_dagger import return_window_mask

    valid = jnp.ones(5, bool)
    kind = jnp.array([3, 3, 3, 2, 3])
    preparation = jnp.array([False, False, False, False, True])
    custody = jnp.array([True, True, False, True, True])
    age = jnp.array([0, 119, 0, 0, 0])
    np.testing.assert_array_equal(
        return_window_mask(valid, kind, preparation, custody, age),
        [True, True, False, False, False],
    )
    assert not bool(return_window_mask(valid[:1], kind[:1], preparation[:1], custody[:1], jnp.array([120]))[0])


def test_fifo_preserves_actual_custody_age_and_only_qualifies_real_short_returns():
    from scripts.prestock_joint_replay import aggregate, buffer

    old = buffer(capacity=4, additions=4, obs_shape=(1, 1, 1), global_size=24)
    batch = _batch([True, True, False, True], [119, 120, 2, 4], [True, True, True, False])
    _, replay, _ = aggregate(old, batch, jax.random.PRNGKey(4), jnp.int32(1))
    assert "actual_custody" in replay and "custody_age" in replay and "return_window" in replay
    for i in np.flatnonzero(np.asarray(replay["valid"])):
        expected = bool(replay["actual_custody"][i]) and int(replay["custody_age"][i]) < 120
        assert bool(replay["return_window"][i]) == expected
    assert not bool(replay["return_window"][replay["obs"][:, 0, 0, 0] == 1].any())
    assert not bool(replay["return_window"][replay["obs"][:, 0, 0, 0] == 2].any())


def test_multiplier_only_scales_the_set_valued_real_return_rows(monkeypatch):
    from jax_bomb import jax_train as train
    from scripts import je09_stage_h5c_dagger as legacy
    from scripts.retention_custody_dagger import make_replay_loss
    from scripts.return_dagger_weight import weighted_joint_loss

    move = jnp.array([[2.0, 1.0, 0.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0, 0.0]])
    ability = jnp.array([[1.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    target = jnp.zeros((2, 5, 3), bool).at[0, 0, 0].set(True).at[0, 0, 1].set(True)
    target = target.at[1, 1, 1].set(True)
    replay = dict(
        obs=jnp.zeros((2, 1)),
        gv=jnp.zeros((2, 24)),
        legal=jnp.ones((2, 5), bool),
        ability_legal=jnp.ones((2, 3), bool),
        joint_target=target,
        valid=jnp.array([True, True]),
        kind=jnp.array([3, 3]),
        preparation=jnp.array([False, False]),
        actual_custody=jnp.array([True, False]),
        custody_age=jnp.array([12, 12]),
        return_window=jnp.array([True, False]),
    )
    monkeypatch.setattr(train, "net_forward", lambda params, *args: (params, ability, None, None))
    monkeypatch.setattr(legacy, "COEFFICIENT", 0.2)

    loss2 = make_replay_loss(2.0)(move, replay)
    loss4 = make_replay_loss(4.0)(move, replay)
    returning = replay["return_window"]
    expected4, _ = weighted_joint_loss(
        move / 1.0,
        ability / 1.0,
        replay["legal"],
        replay["ability_legal"],
        replay["joint_target"],
        replay["valid"],
        returning,
        4.0,
        legacy.COEFFICIENT,
    )
    np.testing.assert_allclose(loss4, expected4, rtol=1e-6)
    assert float(loss4) > float(loss2)
    grad2 = jax.grad(lambda p: make_replay_loss(2.0)(p, replay))(move)
    grad4 = jax.grad(lambda p: make_replay_loss(4.0)(p, replay))(move)
    np.testing.assert_allclose(grad4[1], grad2[1], rtol=1e-7, atol=1e-7)
    assert float(jnp.linalg.norm(grad4[0] - grad2[0])) > 0

    replay["return_window"] = jnp.zeros(2, bool)
    np.testing.assert_allclose(make_replay_loss(2.0)(move, replay), make_replay_loss(4.0)(move, replay), rtol=1e-7)
