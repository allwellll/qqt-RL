import jax.numpy as jnp
import jax
import numpy as np

from scripts.je09_route_pbrs import (
    GAMMA, PHASE_CARRY, PHASE_DROP, PHASE_STOCK, discounted_delta, state,
    transition,
)
from scripts import je09_route_pbrs as pbrs


def test_pbrs_discounted_delta_sets_terminal_potential_to_zero():
    assert float(discounted_delta(0.25, 0.5, False)) == jnp.float32(GAMMA * .5 - .25)
    assert float(discounted_delta(-.5, 100., True)) == .5


def test_phase_and_identity_changes_are_offset_continuous():
    ledger = state(1)
    phase = jnp.full((1, 2), PHASE_STOCK)
    identity = jnp.full((1, 2), 11)
    ledger, delta, before, after = transition(
        ledger, phase, identity, jnp.full((1, 2), -.5), phase, identity,
        jnp.full((1, 2), -.4), jnp.zeros((1, 2), bool))
    assert jnp.allclose(before, 0.)
    assert jnp.allclose(after, .1)
    assert jnp.allclose(delta, GAMMA * .1)

    next_phase = jnp.full((1, 2), PHASE_DROP)
    next_id = jnp.full((1, 2), 22)
    next_ledger, delta, before, after = transition(
        ledger, phase, identity, jnp.full((1, 2), -.3), next_phase, next_id,
        jnp.full((1, 2), -.8), jnp.zeros((1, 2), bool))
    assert jnp.allclose(before, .2)
    assert jnp.allclose(after, before)
    assert jnp.allclose(delta, (GAMMA - 1.) * before)
    assert jnp.allclose(next_ledger['phi'], after)


def test_discounted_round_trip_telescopes_to_zero_when_terminal_phi_is_zero():
    ledger = state(1)
    phase = jnp.full((1, 2), PHASE_CARRY)
    identity = jnp.full((1, 2), 7)
    bases = [-.75, -.5, -.25, -.5, -.75]
    discounted = jnp.zeros((1, 2), jnp.float32)
    weight = 1.
    for left, right in zip(bases, bases[1:]):
        ledger, reward, _, _ = transition(
            ledger, phase, identity, jnp.full((1, 2), left), phase, identity,
            jnp.full((1, 2), right), jnp.zeros((1, 2), bool))
        discounted = discounted + weight * reward
        weight *= GAMMA
    ledger, terminal_reward, _, phi_terminal = transition(
        ledger, phase, identity, jnp.full((1, 2), -.75), phase, identity,
        jnp.full((1, 2), -.75), jnp.ones((1, 2), bool))
    discounted = discounted + weight * terminal_reward
    assert jnp.allclose(phi_terminal, 0.)
    assert jnp.allclose(ledger['phi'], 0.)
    assert jnp.allclose(discounted, 0., atol=1e-6)


def test_route_phase_switch_never_rewards_boundary_as_progress():
    ledger = state(1)
    phase = jnp.full((1, 2), PHASE_DROP)
    identity = jnp.full((1, 2), 5)
    ledger, _, _, _ = transition(ledger, phase, identity,
        jnp.full((1, 2), -.5), phase, identity, jnp.full((1, 2), -.4),
        jnp.zeros((1, 2), bool))
    next_phase = jnp.full((1, 2), PHASE_CARRY)
    next_id = jnp.full((1, 2), 6)
    _, delta, before, after = transition(ledger, phase, identity,
        jnp.full((1, 2), -.25), next_phase, next_id,
        jnp.full((1, 2), -.9), jnp.zeros((1, 2), bool))
    assert jnp.allclose(after, before)
    assert jnp.allclose(delta, (GAMMA - 1.) * before)


def test_native_stock_window_excludes_slow_glue_but_accepts_trap_or_respawn():
    from scripts import je09_bun_scenes as scenes
    from scripts.je09_stage_h8_ops import reward_state
    states = scenes.init_batch(jax.random.PRNGKey(41), 2,
        jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))
    rows = jnp.arange(2)
    stock = states.bun_stored.at[rows, 1, 1].set(1).at[rows, 0, 0].set(1)
    aux = reward_state(2)
    route = dict(aux['routing'], stock_distance=jnp.broadcast_to(
        scenes.DISTANCE[0][None], (2, 2, 13, 15)))
    free = states._replace(bun_stored=stock,
        bun_respawn=jnp.zeros_like(states.bun_respawn),
        move_status=jnp.ones_like(states.move_status))
    phase, *_ = pbrs.phase_values(free, route, aux['drop'])
    np.testing.assert_array_equal(np.asarray(phase), np.zeros((2, 2), np.int8))
    sugar = free._replace(trapped=jnp.ones_like(free.trapped))
    phase, *_ = pbrs.phase_values(sugar, route, aux['drop'])
    np.testing.assert_array_equal(np.asarray(phase), np.full((2, 2), PHASE_STOCK, np.int8))
    death = free._replace(bun_respawn=jnp.ones_like(free.bun_respawn))
    phase, *_ = pbrs.phase_values(death, route, aux['drop'])
    np.testing.assert_array_equal(np.asarray(phase), np.full((2, 2), PHASE_STOCK, np.int8))


def test_enemy_origin_loose_drop_requires_learner_credit_and_identity():
    from scripts import je09_bun_scenes as scenes
    from scripts.je09_stage_h8_ops import reward_state
    states = scenes.init_batch(jax.random.PRNGKey(42), 2,
        jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]))
    aux = reward_state(2)
    target = jnp.array([5, 7])
    routing = dict(aux['routing'], target=aux['routing']['target'].at[0, 1].set(target))
    states = states._replace(bun_loose=states.bun_loose.at[0, 5, 7, 1].set(1))
    drop = dict(aux['drop'], nonce=aux['drop']['nonce'].at[0, 1].set(3),
        credit=aux['drop']['credit'].at[0, 1].set(0))
    route = dict(routing, stock_distance=jnp.full((2, 2, 13, 15), 8, jnp.int32))
    phase, identity, *_ = pbrs.phase_values(states, route, drop)
    assert int(phase[0, 0]) == PHASE_DROP
    assert int(identity[0, 0]) == 200003
    wrong = dict(drop, credit=drop['credit'].at[0, 1].set(1))
    phase, *_ = pbrs.phase_values(states, route, wrong)
    assert int(phase[0, 0]) != PHASE_DROP


def test_disabled_reward_adapter_is_bitwise_old_reward():
    from scripts import je09_stage_h8_pbrs_ops as ops
    reward = jnp.array([[1.25, -2.5]], jnp.float32)
    parts = jnp.array([[[3., 4., 7.], [5., 6., 11.]]], jnp.float32)
    expected_aux = {'sentinel': jnp.array([7])}
    original = ops._original_reward
    ops._original_reward = lambda *args: (reward, expected_aux, parts, {'base': reward})
    try:
        got, aux, expanded, _ = ops.combined_reward(None, None, None, None, None,
                                                     coefficient=0.)
    finally:
        ops._original_reward = original
    np.testing.assert_array_equal(np.asarray(got), np.asarray(reward))
    np.testing.assert_array_equal(np.asarray(expanded[..., :2]), np.asarray(parts[..., :2]))
    np.testing.assert_array_equal(np.asarray(expanded[..., -1]), np.asarray(parts[..., -1]))
    assert aux is expected_aux
