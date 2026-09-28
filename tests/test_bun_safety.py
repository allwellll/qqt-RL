import jax
import jax.numpy as jnp
import pytest

from jax_bomb import bun_env
from jax_bomb import bun_safety
from jax_bomb import levels


@pytest.fixture(autouse=True)
def bun_level():
    bun_env.prepare()
    bun_env.configure_training("full=1", 1)
    yield
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


def _open_state(seed=0):
    state = bun_env._fresh(jax.random.PRNGKey(seed))
    core = state.core._replace(
        wall=jnp.zeros_like(state.core.wall),
        brick=jnp.zeros_like(state.core.brick),
        pushable=jnp.zeros_like(state.core.pushable),
        brick_linger=jnp.zeros_like(state.core.brick_linger),
        fuse=jnp.zeros_like(state.core.fuse),
        owner=jnp.full_like(state.core.owner, -1),
        bomb_blast=jnp.zeros_like(state.core.bomb_blast),
        blast_linger=jnp.zeros_like(state.core.blast_linger),
        pos=jnp.asarray([[6.5, 6.5], [10.5, 12.5]], jnp.float32),
        hp=jnp.ones_like(state.core.hp),
        alive=jnp.ones_like(state.core.alive),
        blast_cap=jnp.asarray([2.0, 2.0], jnp.float32),
        bombs_cap=jnp.asarray([2.0, 2.0], jnp.float32),
        spd_g=jnp.ones_like(state.core.spd_g),
    )
    return state._replace(
        core=core,
        blast_owner_linger=jnp.zeros_like(state.blast_owner_linger),
        move_status=jnp.zeros_like(state.move_status),
        status_ticks=jnp.zeros_like(state.status_ticks),
    )


def _bomb(state, row, column, fuse, owner, blast=2):
    return state._replace(core=state.core._replace(
        fuse=state.core.fuse.at[row, column].set(fuse),
        owner=state.core.owner.at[row, column].set(owner),
        bomb_blast=state.core.bomb_blast.at[row, column].set(blast)))


def test_owner_deadline_tracks_chain_and_rejects_ambiguous_overlap():
    state = _open_state(10)
    state = _bomb(state, 6, 4, 1, 1, 2)
    state = _bomb(state, 6, 6, 20, 0, 2)

    deadlines, _, _ = bun_safety.owned_hazard_deadlines(state)

    assert int(deadlines[0, 6, 8]) == 0
    assert int(deadlines[0, 6, 5]) == int(bun_safety._INF_TICK)
    assert int(deadlines[1, 6, 3]) == 0


def test_enemy_triggered_chain_keeps_the_triggered_bombs_real_owner():
    state = _open_state(15)
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(jnp.asarray([6.5, 8.5], jnp.float32))))
    state = _bomb(state, 6, 4, 1, 1, 2)
    state = _bomb(state, 6, 6, 20, 0, 2)

    _, _, info = bun_env.step(
        state, jnp.asarray([[4, 0], [4, 0]], jnp.int32),
        jax.random.PRNGKey(16), auto_reset=False, return_info=True)

    assert info["death_source"][0].tolist() == [True, False]
    assert info["causal_death_source"][0].tolist() == [False, True]
    assert not bool(info["own_bomb_defeat"][0])
    assert not bool(info["opponent_physical_defeat"][0])
    assert bool(info["opponent_causal_defeat"][0])


def test_enemy_only_blast_is_not_marked_as_own_suicide():
    state = _bomb(_open_state(11), 6, 6, 4, 1, 3)
    deadlines, _, _ = bun_safety.owned_hazard_deadlines(state)

    assert bool((deadlines[0] >= bun_safety._INF_TICK).all())
    assert int(deadlines[1, 6, 8]) < int(bun_safety._INF_TICK)


def test_doomed_state_exempts_all_actions():
    state = _open_state(12)
    walls = jnp.ones_like(state.core.wall).at[6, 6].set(False)
    state = state._replace(core=state.core._replace(wall=walls))
    state = _bomb(state, 6, 6, 1, 0, 1)

    analysis = bun_safety.analyze_actions(state)

    assert bool(analysis.doomed[0])
    assert not bool(analysis.avoidable[0].any())


def test_avoidable_idle_keeps_safe_escape_and_necessary_bomb_available():
    state = _bomb(_open_state(13), 6, 6, 5, 0, 3)
    analysis = bun_safety.analyze_actions(state)

    assert bool(analysis.avoidable[0, 4, 0])
    assert bool((analysis.survivable[0] & analysis.legal[0]).any())

    clean = _open_state(14)
    clean_analysis = bun_safety.analyze_actions(clean)
    assert bool((clean_analysis.legal[0, :, 1]
                 & ~clean_analysis.avoidable[0, :, 1]).any())


def test_fixed_seed_safety_replay_is_deterministic():
    state_a = _open_state(20260925)
    state_b = _open_state(20260925)
    first = bun_safety.analyze_actions(state_a)
    second = bun_safety.analyze_actions(state_b)

    assert jnp.array_equal(first.avoidable, second.avoidable)
    assert jnp.array_equal(first.survivable, second.survivable)


def test_joint_hard_mask_and_soft_penalty_preserve_safe_bomb_action():
    move_logits = jnp.zeros((1, bun_env.N_MOVES), jnp.float32)
    ability_logits = jnp.zeros((1, bun_env.N_BOMB), jnp.float32)
    move_mask = jnp.ones_like(move_logits, jnp.bool_)
    ability_mask = jnp.ones_like(ability_logits, jnp.bool_)
    avoidable = jnp.zeros(
        (1, bun_env.N_MOVES, bun_env.N_BOMB), jnp.bool_)
    avoidable = avoidable.at[0, 4, 1].set(True)

    hard = bun_safety.adjusted_joint_logits(
        move_logits, ability_logits, move_mask, ability_mask, avoidable,
        "hard", 4.0).reshape(1, bun_env.N_MOVES, bun_env.N_BOMB)
    soft = bun_safety.adjusted_joint_logits(
        move_logits, ability_logits, move_mask, ability_mask, avoidable,
        "soft", 4.0).reshape(1, bun_env.N_MOVES, bun_env.N_BOMB)

    assert float(hard[0, 4, 1]) <= -1e8
    assert bool(jnp.isfinite(hard[0, 0, 1]))
    assert float(soft[0, 4, 1]) == -4.0
    assert float(soft[0, 0, 1]) == 0.0
