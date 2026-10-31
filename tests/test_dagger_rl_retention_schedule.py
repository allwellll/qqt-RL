"""Inactive preparation: loss schedule with a common combat-retention term."""
import jax
import jax.numpy as j
import numpy as np
import optax
import pytest


def test_fixed_switch_and_unknown_schedule_are_explicit():
    from scripts.dagger_rl_retention_schedule import branch_for_update
    assert branch_for_update('sequential_anchor', 1) == 'dagger_anchor'
    assert branch_for_update('sequential_anchor', 16) == 'dagger_anchor'
    assert branch_for_update('sequential_anchor', 17) == 'ppo_anchor'
    assert branch_for_update('sequential_anchor', 32) == 'ppo_anchor'
    for u in [1, 16, 17, 32]:
        assert branch_for_update('joint_anchor', u) == 'joint_anchor'
        assert branch_for_update('ppo_anchor', u) == 'ppo_anchor'
    for arm, u in [('unknown', 1), ('joint_anchor', 0), ('joint_anchor', 33)]:
        with pytest.raises(ValueError):
            branch_for_update(arm, u)


def objective(p, batch, replay, mode):
    w, c = p['actor']['heads']['wb'], p['critic']
    ppo, value, entropy = j.sum(w**2), j.sum(c**2), j.sum(w)/10
    sup, anchor = j.sum((w-replay[0].mean())**2), j.sum((w-.1)**2)
    metrics = j.array([ppo, value, entropy, sup, 1., 1., anchor, 1.])
    return ppo+.5*value-.005*entropy+.2*sup+.05*anchor, metrics


def test_pure_rl_has_no_supervised_replay_dependency_and_updates_both_modules():
    from scripts.dagger_rl_retention_schedule import loss_for_branch
    p = dict(actor=dict(heads=dict(wb=j.arange(8.)/10)), critic=j.array([.3]))
    def loss(q, scale, branch):
        value, metrics = objective(q, None, (j.array([scale]),), 'anchor')
        return loss_for_branch(value, metrics, branch)
    a, ga = jax.value_and_grad(loss)(p, 1., 'ppo_anchor')
    b, gb = jax.value_and_grad(loss)(p, 100., 'ppo_anchor')
    np.testing.assert_array_equal(a, b)
    for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
        np.testing.assert_array_equal(x, y)
    assert np.linalg.norm(ga['actor']['heads']['wb']) > 0
    assert np.linalg.norm(ga['critic']) > 0
    dagger = jax.grad(loss)(p, 1., 'dagger_anchor')
    np.testing.assert_array_equal(dagger['critic'], 0.)
    joint = loss(p, 1., 'joint_anchor')
    np.testing.assert_array_equal(joint, objective(p, None, (j.array([1.]),), 'anchor')[0])


def test_dagger_freezes_critic_parameters_and_nonzero_adam_moments():
    from scripts.dagger_rl_retention_schedule import make_update
    from scripts.carrier_mechanism_train import preserve_critic
    p = dict(actor=dict(heads=dict(wb=j.arange(8.)/10)), critic=j.array([.3]))
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5, eps=1e-8))
    adam = opt.init(p)
    # Seed real nonzero moments: a zero Critic gradient alone does not freeze it.
    _, adam = opt.update(jax.tree.map(j.ones_like, p), adam, p)
    assert np.linalg.norm(adam[1][0].mu['critic']) > 0
    rep = lambda tr: jax.tree.map(lambda x: x[None], tr)
    batch = tuple(j.ones((1, 2, 16)) for _ in range(12))
    replay = (j.ones((1, 32)),)
    update = make_update(objective, opt, 'dagger_anchor', preserve_critic)
    (pn, an), metrics = update(rep(p), rep(adam), batch, replay, jax.random.split(jax.random.PRNGKey(1), 1))
    np.testing.assert_array_equal(pn['critic'][0], p['critic'])
    for name in ['mu', 'nu']:
        np.testing.assert_array_equal(getattr(an[1][0], name)['critic'][0], getattr(adam[1][0], name)['critic'])
    assert int(an[1][0].count[0]) == int(adam[1][0].count)+1
    assert np.linalg.norm(np.asarray(pn['actor']['heads']['wb'][0]-p['actor']['heads']['wb'])) > 0
    assert np.isfinite(np.asarray(metrics)).all()
