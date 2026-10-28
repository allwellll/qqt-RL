"""RED-first contracts for the JE09 real drop-to-capture bridge."""
import pytest

from scripts.je09_bun_bridge import (
    BRIDGE_MANIFEST,
    bridge_funnel,
    bridge_reward_ledger,
    validate_bridge_start,
)


def test_bridge_manifest_uses_disjoint_seed_namespaces_and_real_chain():
    assert BRIDGE_MANIFEST["assist"] is False
    assert BRIDGE_MANIFEST["start"] == "empty_learner_real_enemy_carrier"
    assert BRIDGE_MANIFEST["events"] == [
        "attributed_drop", "first_pickup", "pickup_return", "progress25",
        "progress50", "home_half", "capture",
    ]
    assert len(set(BRIDGE_MANIFEST["seeds"].values())) == 3
    assert BRIDGE_MANIFEST["holdout_seed"] not in BRIDGE_MANIFEST["seeds"].values()


def test_bridge_start_is_empty_learner_and_route_is_reachable():
    start = {"learner_carried": -1, "enemy_carried": 1,
             "drop_distance": 6, "legal_return_route": True,
             "enemy_can_open_route": False}
    validate_bridge_start(start)
    with pytest.raises(ValueError):
        validate_bridge_start(dict(start, learner_carried=1))
    with pytest.raises(ValueError):
        validate_bridge_start(dict(start, legal_return_route=False))


def test_bridge_ledger_deduplicates_milestones_and_claws_back_failed_chain():
    ledger = bridge_reward_ledger()
    paid = []
    for event in ["attributed_drop", "first_pickup", "pickup_return", "progress25",
                  "progress25", "progress50", "home_half", "capture"]:
        paid.append(ledger.apply(event))
    assert paid[1] > 0 and paid[3] > 0 and paid[4] == 0
    assert ledger.capture is True and ledger.balance > 0
    failed = ledger.fail("death")
    assert failed < 0 and ledger.balance == 0 and ledger.capture is False


def test_bridge_funnel_has_fixed_denominators_and_first_pickup_boundary():
    rows = [
        {"eligible_drop": True, "toward_drop": 2, "arrival": True,
         "first_pickup": True, "pickup_return": True, "progress25": True,
         "progress50": False, "home_half": False, "capture": False,
         "death": False, "self": False, "trade": False},
        {"eligible_drop": True, "toward_drop": 0, "arrival": True,
         "first_pickup": False, "pickup_return": False, "progress25": False,
         "progress50": False, "home_half": False, "capture": False,
         "death": True, "self": False, "trade": False},
    ]
    result = bridge_funnel(rows)
    assert result["eligible_drop"] == 2
    assert result["first_pickup"] == 1
    assert result["progress25"] == 1
    assert result["capture"] == 0
    assert result["death"] == 1


def test_bridge_training_scene_starts_empty_and_emits_real_drop():
    import jax
    import jax.numpy as jnp
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    states = scenes.init_batch(jax.random.PRNGKey(123), 32, jax.nn.one_hot(9, 10))
    for row in range(32):
        seat = row % 2
        assert int(states.bun_carried[row, seat]) == -1
        assert int(states.bun_carried[row, 1 - seat]) == 1 - seat
        assert int(states.trapped[row, 1 - seat]) == 1
        assert bool(states.trapped_source[row, 1 - seat, seat])
    actions = jnp.tile(jnp.array([[[4, 0], [4, 0]]]), (32, 1, 1))
    _, _, info = jax.vmap(lambda state, action, key: env.step(
        state, action, key, auto_reset=False, return_info=True))(
        states, actions, jax.random.split(jax.random.PRNGKey(124), 32))
    assert bool(info["drop"].any())
    assert bool(info["carrier_death"].any())
    assert bool((info["drop_credit"] >= 0).any())


def bridge_event(**changes):
    import jax.numpy as jnp
    names = ["drop", "pickup_loose", "pickup_return", "progress25", "progress50",
             "home_half", "capture", "death", "self", "trade", "timeout", "lost",
             "carried", "settled", "topology_changed"]
    event = {name: jnp.zeros((1, 2), bool) for name in names}
    event.update(before_distance=jnp.full((1, 2), 12), after_distance=jnp.full((1, 2), 12))
    event.update({name: jnp.full((1, 2), value) for name, value in changes.items()})
    return event


def test_jax_bridge_requires_real_drop_then_pickup_and_no_duplicate_native_bonus():
    from scripts.je09_bun_bridge import jax_bridge_reward, jax_bridge_state
    state = jax_bridge_state(1)
    reward, state, _ = jax_bridge_reward(bridge_event(pickup_return=True, capture=True), state)
    assert float(reward.sum()) == 0
    _, state, _ = jax_bridge_reward(bridge_event(drop=True), state)
    reward, state, _ = jax_bridge_reward(bridge_event(pickup_loose=True, carried=True), state)
    assert float(reward.sum()) == 0  # Native attribution ledger already pays +8.
    reward, state, metrics = jax_bridge_reward(bridge_event(carried=True, before_distance=12,
                                                          after_distance=8), state)
    assert float(reward[0, 0]) == 2  # First net return +1, quarter +1.
    assert bool(metrics["bridge_events"][0, 0, 3])
    reward, state, _ = jax_bridge_reward(bridge_event(carried=True, after_distance=8), state)
    assert float(reward.sum()) == 0
    reward, state, _ = jax_bridge_reward(bridge_event(lost=True), state)
    assert float(reward[0, 0]) == -2
    reward, _, _ = jax_bridge_reward(bridge_event(carried=True, after_distance=1), state)
    assert float(reward.sum()) == 0


def test_jax_bridge_snapshot_and_terminal_clawback_includes_same_tick_reward():
    import copy
    from scripts.je09_bun_bridge import jax_bridge_reward, jax_bridge_state
    _, state, _ = jax_bridge_reward(bridge_event(drop=True), jax_bridge_state(1))
    _, state, _ = jax_bridge_reward(bridge_event(pickup_loose=True, carried=True), state)
    saved = copy.deepcopy(state)
    event = bridge_event(carried=True, after_distance=5, home_half=True)
    reward, state, _ = jax_bridge_reward(event, state)
    again, restored, _ = jax_bridge_reward(event, saved)
    assert float(reward[0, 0]) == float(again[0, 0]) == 5.5
    reward, _, _ = jax_bridge_reward(bridge_event(timeout=True), state)
    assert float(reward[0, 0]) == -5.5


def test_bridge_scene_legal_topology_distance_is_two_to_eight():
    import jax
    import numpy as np
    from scripts import je09_bun_scenes as scenes
    from scripts.je08_bun_potential import cpu_shortest
    from jax_bomb import je07_bun_env as env
    env.prepare()
    states = scenes.init_batch(jax.random.PRNGKey(124), 32, jax.nn.one_hot(9, 10))
    for row in range(32):
        seat = row % 2
        cell = np.asarray(states.pos[row]).astype(int)
        goals = np.zeros((13, 15), bool)
        goals[tuple(cell[1-seat])] = True
        distance = cpu_shortest(states.wall[row], states.brick[row], goals)
        assert 2 <= distance[tuple(cell[seat])] <= 8


def test_je09_curriculum_keeps_declared_bridge_and_combat_mass():
    import jax.numpy as jnp
    from scripts.je09_bun_ops import curriculum_weights
    weights = curriculum_weights(jnp.zeros((10, 70)))
    assert float(weights[0]) == pytest.approx(.4)
    assert float(weights[9]) == pytest.approx(.35)
    assert float(weights.sum()) == pytest.approx(1.)


def test_parent_retention_gradient_only_noncarriers_and_not_item_logit():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scripts.je09_bun_ops import parent_retention_kl
    moves=jnp.array([[1.,0.,0.,0.,0.],[1.,0.,0.,0.,0.]])
    ability=jnp.array([[1.,0.,3.],[1.,0.,3.]])
    gv=jnp.zeros((2,24)).at[1,13].set(1.)
    loss=lambda m,b:parent_retention_kl(m,b,jnp.zeros_like(m),jnp.zeros_like(b),
        jnp.ones_like(m,bool),jnp.ones_like(b,bool),gv)
    dm,db=jax.grad(loss,argnums=(0,1))(moves,ability)
    assert float(jnp.linalg.norm(dm[0]))>0
    assert float(jnp.linalg.norm(db[0,:2]))>0
    np.testing.assert_array_equal(dm[1],0.)
    np.testing.assert_array_equal(db[1],0.)
    np.testing.assert_array_equal(db[:,2],0.)
