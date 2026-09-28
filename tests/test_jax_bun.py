import jax
import jax.numpy as jnp
import pytest

from jax_bomb import bun_env
from jax_bomb import levels


@pytest.fixture(autouse=True)
def bun_level():
    bun_env.prepare()
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    yield
    bun_env.configure_training("full=1", bun_env.MAX_HP)
    levels.clear()


def idle_actions():
    return jnp.asarray([[4, 0], [4, 0]], jnp.int32)


def test_bun_initial_state_and_observation_contract():
    state = bun_env._fresh(jax.random.PRNGKey(0))

    assert int(state.core.level_id) == 0
    assert state.bun_stored.tolist() == [[1, 0], [0, 1]]
    assert state.bun_carried.tolist() == [-1, -1]
    assert bun_env.make_obs(state, 0).shape == (24, 13, 15)
    assert bun_env.global_vec(state, 0).shape == (24,)
    move_mask, ability_mask = bun_env.legal_mask(state)
    assert move_mask.shape == (2, 5)
    assert ability_mask.shape == (2, 3)


def test_steal_then_capture_ends_round_and_rewards_capture():
    state = bun_env._fresh(jax.random.PRNGKey(1))
    enemy_base = bun_env._BUN_BASES[1].astype(jnp.float32) + 1.5
    core = state.core._replace(pos=state.core.pos.at[0].set(enemy_base))
    state = state._replace(core=core)

    state, done, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(2), auto_reset=False,
        return_info=True)
    assert not bool(done)
    assert bool(info["steal"][0])
    assert int(state.bun_carried[0]) == 1
    assert int(state.bun_stored[1, 1]) == 0

    own_base = bun_env._BUN_BASES[0].astype(jnp.float32) + 1.5
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(own_base)))
    before_alive = state.alive
    state, done, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(3), auto_reset=False,
        return_info=True)
    assert bool(done)
    assert int(info["winner"]) == 0
    assert bool(info["capture"][0])
    assert int(state.bun_stored[0, 1]) == 1
    assert "objective_progress" in info

    rewards = bun_env.reward_from_events(
        info["dmg"][None], before_alive[None], info["alive"][None],
        info["hp"][None], done[None], info["crate"][None],
        jnp.zeros((1, 2), jnp.bool_), info["walls"][None],
        0.0, 0.0, 0.0, 1.0, moves=idle_actions()[None, :, 0],
        bombs=idle_actions()[None, :, 1], rule_info={
            key: value[None] if getattr(value, "ndim", 0) > 0 else value[None]
            for key, value in info.items()
        })
    assert float(rewards[0, 0]) >= 29.0
    assert float(rewards[0, 1]) <= -19.0


def test_carrier_cannot_bomb_and_death_drops_bun_then_respawns():
    state = bun_env._fresh(jax.random.PRNGKey(4))
    core = state.core._replace(
        pos=jnp.asarray([[6.5, 6.5], [8.5, 8.5]], jnp.float32),
        hp=jnp.asarray([1, bun_env.MAX_HP], jnp.int32),
        wall=state.core.wall.at[6, 6].set(False),
        brick=state.core.brick.at[6, 6].set(False),
        fuse=state.core.fuse.at[6, 6].set(1),
        owner=state.core.owner.at[6, 6].set(1),
        bomb_blast=state.core.bomb_blast.at[6, 6].set(1),
    )
    state = state._replace(
        core=core,
        bun_stored=state.bun_stored.at[1, 1].set(0),
        bun_carried=state.bun_carried.at[0].set(1),
        held_item=state.held_item.at[0].set(bun_env.ITEM_BANANA),
    )
    _, ability_mask = bun_env.legal_mask(state)
    assert not bool(ability_mask[0, 1])
    assert bool(ability_mask[0, 2])

    state, done, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(5), auto_reset=False,
        return_info=True)
    assert not bool(done)
    assert bool(info["death"][0])
    assert bool(info["drop"][0])
    assert int(state.bun_carried[0]) == -1
    assert int(state.bun_loose[6, 6, 1]) == 1
    assert int(state.bun_respawn[0]) == bun_env.BUN_RESPAWN_TICKS
    assert int(state.held_item[0]) == bun_env.ITEM_NONE

    state = state._replace(bun_respawn=state.bun_respawn.at[0].set(1))
    state, _, _ = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(6), auto_reset=False,
        return_info=True)
    assert bool(state.alive[0])
    assert int(state.hp[0]) == bun_env.MAX_HP
    assert int(state.bun_respawn[0]) == 0


def test_explosion_owner_prevents_false_kill_credit():
    state = bun_env._fresh(jax.random.PRNGKey(40))
    core = state.core._replace(
        pos=jnp.asarray([[6.5, 6.5], [8.5, 8.5]], jnp.float32),
        hp=jnp.asarray([1, bun_env.MAX_HP], jnp.int32),
        wall=state.core.wall.at[6, 6].set(False),
        brick=state.core.brick.at[6, 6].set(False),
        fuse=state.core.fuse.at[6, 6].set(1),
        owner=state.core.owner.at[6, 6].set(0),
        bomb_blast=state.core.bomb_blast.at[6, 6].set(1),
    )
    state = state._replace(core=core)
    _, done, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(41), auto_reset=False,
        return_info=True)
    assert not bool(done)
    assert bool(info["death_source"][0, 0])
    assert not bool(info["death_source"][0, 1])

    batched = {
        key: value[None] if getattr(value, "ndim", 0) > 0 else value[None]
        for key, value in info.items()
    }
    rewards = bun_env.reward_from_events(
        info["dmg"][None], state.alive[None], info["alive"][None],
        info["hp"][None], done[None], info["crate"][None],
        jnp.zeros((1, 2), jnp.bool_), info["walls"][None],
        0.0, 0.0, 0.0, 1.0, rule_info=batched)
    assert float(rewards[0, 0]) <= -8.0
    assert float(rewards[0, 1]) == pytest.approx(0.0)


def test_overlapping_blast_sources_do_not_count_as_self_kill_penalty():
    state = bun_env._fresh(jax.random.PRNGKey(42))
    core = state.core._replace(
        pos=jnp.asarray([[6.5, 6.5], [10.5, 12.5]], jnp.float32),
        hp=jnp.asarray([1, 1], jnp.int32),
        wall=jnp.zeros_like(state.core.wall),
        brick=jnp.zeros_like(state.core.brick),
        fuse=jnp.zeros_like(state.core.fuse).at[6, 5].set(1).at[6, 7].set(1),
        owner=jnp.full_like(state.core.owner, -1).at[6, 5].set(0).at[6, 7].set(1),
        bomb_blast=jnp.zeros_like(state.core.bomb_blast).at[6, 5].set(2).at[6, 7].set(2),
    )
    state = state._replace(core=core)
    _, done, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(43), auto_reset=False,
        return_info=True)

    assert not bool(done)
    assert info["death_source"][0].tolist() == [True, True]
    assert not bool(info["credited_kill"].any())
    batched = {
        key: value[None] if getattr(value, "ndim", 0) > 0 else value[None]
        for key, value in info.items()
    }
    rewards = bun_env.reward_from_events(
        info["dmg"][None], state.alive[None], info["alive"][None],
        info["hp"][None], done[None], info["crate"][None],
        jnp.zeros((1, 2), jnp.bool_), info["walls"][None],
        0.0, 0.0, 0.0, 1.0, rule_info=batched)
    assert float(rewards[0, 0]) > -8.0


def _open_causal_state(seed=200):
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
        pos=jnp.asarray([[10.5, 1.5], [10.5, 13.5]], jnp.float32),
        hp=jnp.ones_like(state.core.hp),
        alive=jnp.ones_like(state.core.alive),
        invuln=jnp.zeros_like(state.core.invuln),
    )
    return state._replace(
        core=core,
        blast_owner_linger=jnp.zeros_like(state.blast_owner_linger),
        blast_causal_linger=jnp.zeros_like(state.blast_causal_linger),
        blast_trigger_linger=jnp.zeros_like(state.blast_trigger_linger),
        kill_window_ticks=jnp.zeros_like(state.kill_window_ticks),
    )


def _with_bomb(state, row, column, fuse, owner, blast=2):
    return state._replace(core=state.core._replace(
        fuse=state.core.fuse.at[row, column].set(fuse),
        owner=state.core.owner.at[row, column].set(owner),
        bomb_blast=state.core.bomb_blast.at[row, column].set(blast),
    ))


def test_causal_kill_direct_owner_preserves_physical_credit():
    state = _open_causal_state(201)
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[1].set(jnp.asarray([6.5, 8.5]))))
    state = _with_bomb(state, 6, 6, 1, 0, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(202),
        auto_reset=False, return_info=True)

    assert info["credited_kill"].tolist() == [True, False]
    assert info["causal_kill"].tolist() == [True, False]
    assert info["trigger_kill"].tolist() == [False, False]
    assert bool(info["opponent_physical_defeat"][1])
    assert bool(info["opponent_causal_defeat"][1])
    assert not bool(info["own_bomb_defeat"].any())
    assert not bool(info["triggered_early"].any())
    assert info["surviving_causal_kill"].tolist() == [True, False]
    assert info["surviving_physical_kill"].tolist() == [True, False]
    assert info["surviving_kill"].tolist() == [True, False]


def test_causal_trigger_kill_credits_actor_not_physical_owner():
    state = _open_causal_state(203)._replace(
        lesson=jnp.asarray(bun_env.LESSON_COMBAT, jnp.int8))
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[1].set(jnp.asarray([6.5, 8.5]))))
    state = _with_bomb(state, 6, 4, 1, 0, 2)
    state = _with_bomb(state, 6, 6, 10, 1, 2)

    _, done, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(204),
        auto_reset=False, return_info=True)

    assert bool(done)
    assert info["death_source"][1].tolist() == [False, True]
    assert info["credited_kill"].tolist() == [False, False]
    assert info["causal_death_source"][1].tolist() == [True, False]
    assert info["causal_kill"].tolist() == [True, False]
    assert info["trigger_kill"].tolist() == [True, False]
    assert not bool(info["own_bomb_defeat"][1])
    assert not bool(info["opponent_physical_defeat"][1])
    assert bool(info["opponent_causal_defeat"][1])
    assert int(info["trigger_actor"][6, 6]) == 0
    assert int(info["trigger_original_tick"][6, 6]) == 9
    assert int(info["trigger_actual_tick"][6, 6]) == 0
    assert bool(info["trigger_opponent_death"][6, 6])
    assert bool(info["kill_window_active"][0])
    assert int(info["kill_window_remaining"][0]) == bun_env.BUN_RESPAWN_TICKS
    assert info["surviving_causal_kill"].tolist() == [True, False]
    assert info["surviving_physical_kill"].tolist() == [False, False]
    assert info["surviving_kill"].tolist() == [True, False]

    batched = {
        key: value[None] if getattr(value, "ndim", 0) > 0 else value[None]
        for key, value in info.items()
    }
    reward = bun_env.reward_from_events(
        info["dmg"][None], state.alive[None], info["alive"][None],
        info["hp"][None], done[None], info["crate"][None],
        jnp.zeros((1, 2), jnp.bool_), info["walls"][None],
        0.0, 0.0, 0.0, 1.0, rule_info=batched)
    assert float(reward[0, 0]) > 0.0


def test_causal_actor_propagates_through_multi_bomb_chain():
    state = _open_causal_state(205)
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[1].set(jnp.asarray([6.5, 8.5]))))
    state = _with_bomb(state, 6, 2, 1, 0, 2)
    state = _with_bomb(state, 6, 4, 15, 1, 2)
    state = _with_bomb(state, 6, 6, 20, 1, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(206),
        auto_reset=False, return_info=True)

    assert int(info["trigger_actor"][6, 4]) == 0
    assert int(info["trigger_actor"][6, 6]) == 0
    assert int(info["trigger_original_tick"][6, 4]) == 14
    assert int(info["trigger_original_tick"][6, 6]) == 19
    assert info["causal_kill"].tolist() == [True, False]
    assert info["trigger_kill"].tolist() == [True, False]
    assert bool(info["trigger_opponent_death"][6, 6])


def test_trigger_time_change_without_death_records_event_only():
    state = _open_causal_state(207)
    state = _with_bomb(state, 6, 4, 1, 0, 2)
    state = _with_bomb(state, 6, 6, 10, 1, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(208),
        auto_reset=False, return_info=True)

    assert bool(info["triggered_early"][6, 6])
    assert int(info["trigger_actor"][6, 6]) == 0
    assert not bool(info["trigger_opponent_death"].any())
    assert not bool(info["causal_kill"].any())
    assert not bool(info["trigger_kill"].any())


def test_natural_own_bomb_defeat_is_not_relabelled_as_opponent_defeat():
    state = _open_causal_state(215)
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(jnp.asarray([6.5, 8.5]))))
    state = _with_bomb(state, 6, 6, 1, 0, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(216),
        auto_reset=False, return_info=True)

    assert bool(info["own_bomb_defeat"][0])
    assert not bool(info["opponent_physical_defeat"][0])
    assert not bool(info["opponent_causal_defeat"][0])


def test_same_tick_natural_explosion_is_not_trigger_credit():
    state = _open_causal_state(209)
    state = _with_bomb(state, 6, 4, 1, 0, 2)
    state = _with_bomb(state, 6, 6, 1, 1, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(210),
        auto_reset=False, return_info=True)

    assert not bool(info["triggered_early"].any())
    assert not bool(info["trigger_actor_mask"].any())
    assert not bool(info["trigger_kill"].any())


def test_overlapping_causal_sources_are_not_credited():
    state = _open_causal_state(211)
    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[1].set(jnp.asarray([6.5, 6.5]))))
    state = _with_bomb(state, 6, 4, 1, 0, 2)
    state = _with_bomb(state, 6, 8, 1, 1, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(212),
        auto_reset=False, return_info=True)

    assert info["causal_death_source"][1].tolist() == [True, True]
    assert not bool(info["causal_kill"].any())
    assert not bool(info["trigger_kill"].any())


def test_mutual_death_cancels_all_positive_kill_credit():
    state = _open_causal_state(213)
    state = state._replace(core=state.core._replace(
        pos=jnp.asarray([[8.5, 6.5], [6.5, 6.5]], jnp.float32)))
    state = _with_bomb(state, 6, 4, 1, 0, 2)
    state = _with_bomb(state, 8, 4, 1, 1, 2)

    _, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(214),
        auto_reset=False, return_info=True)

    assert info["death"].tolist() == [True, True]
    assert bool(info["mutual_death"])
    assert not bool(info["credited_kill"].any())
    assert not bool(info["causal_kill"].any())
    assert not bool(info["surviving_physical_kill"].any())
    assert not bool(info["surviving_causal_kill"].any())
    assert not bool(info["surviving_kill"].any())
    assert not bool(info["trigger_kill"].any())
    assert not bool(info["own_bomb_defeat"].any())
    assert not bool(info["opponent_physical_defeat"].any())
    assert not bool(info["opponent_causal_defeat"].any())
    assert not bool(info["kill_window_active"].any())


def test_curriculum_resets_and_milestones_are_one_shot():
    bun_env.configure_training("carry_home=1", 1)
    state = bun_env._fresh(jax.random.PRNGKey(50))
    assert int(state.lesson) == bun_env.LESSON_CARRY_HOME
    assert state.bun_carried.tolist() == [1, 0]
    assert state.bun_stored.tolist() == [[0, 0], [0, 0]]
    assert state.hp.tolist() == [1, 1]

    bun_env.configure_training("full=1", bun_env.MAX_HP)
    state = bun_env._fresh(jax.random.PRNGKey(51))
    core = state.core._replace(
        pos=state.core.pos.at[0].set(jnp.asarray([6.5, 8.5])),
        wall=state.core.wall.at[6, 8].set(False),
        brick=state.core.brick.at[6, 8].set(False),
    )
    state = state._replace(core=core)
    state, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(52), auto_reset=False,
        return_info=True)
    assert bool(info["enter_enemy_half"][0])
    state, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(53), auto_reset=False,
        return_info=True)
    assert not bool(info["enter_enemy_half"][0])


def test_tactical_item_use_and_enemy_trigger():
    state = bun_env._fresh(jax.random.PRNGKey(7))
    core = state.core._replace(
        pos=jnp.asarray([[7.5, 7.5], [7.5, 10.5]], jnp.float32),
        wall=state.core.wall.at[7, 7].set(False).at[7, 10].set(False),
        brick=state.core.brick.at[7, 7].set(False).at[7, 10].set(False),
    )
    state = state._replace(
        core=core,
        held_item=state.held_item.at[0].set(bun_env.ITEM_SLOW_GLUE))
    use_actions = jnp.asarray([[4, 2], [4, 0]], jnp.int32)
    state, _, _ = bun_env.step(
        state, use_actions, jax.random.PRNGKey(8), auto_reset=False,
        return_info=True)
    assert int(state.field_item[7, 7]) == bun_env.ITEM_SLOW_GLUE
    assert not bool(state.field_armed[7, 7])

    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[0].set(jnp.asarray([7.5, 5.5]))))
    state, _, _ = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(9), auto_reset=False,
        return_info=True)
    assert bool(state.field_armed[7, 7])

    state = state._replace(core=state.core._replace(
        pos=state.core.pos.at[1].set(jnp.asarray([7.5, 7.5]))))
    state, _, info = bun_env.step(
        state, idle_actions(), jax.random.PRNGKey(10), auto_reset=False,
        return_info=True)
    assert bool(info["trap_hit"][1])
    assert int(state.move_status[1]) == bun_env.STATUS_SLOW
    assert int(state.status_ticks[1]) == 100


def test_bun_step_jits():
    state = bun_env._fresh(jax.random.PRNGKey(11))
    compiled = jax.jit(lambda s, a, k: bun_env.step(
        s, a, k, auto_reset=True, return_info=True))
    next_state, done, info = compiled(
        state, idle_actions(), jax.random.PRNGKey(12))
    jax.block_until_ready(next_state)
    assert int(next_state.t) == 1
    assert not bool(done)
    assert int(info["winner"]) == -2
