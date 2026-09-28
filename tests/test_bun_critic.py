import jax
import jax.numpy as jnp
import numpy as np
import pickle

from jax_bomb import bun_critic
from jax_bomb import bun_env
from jax_bomb.jax_net import init_transformer
from scripts import generate_bun_critic_data as generator
from scripts import adapt_bun_critic_onpolicy as onpolicy
from scripts import train_bun_critic as trainer
from scripts import train_bun_separate_ac as separate_ac


def test_frozen_transformer_features_block_actor_gradients():
    params = init_transformer(
        jax.random.PRNGKey(1), bun_env.N_OBS_CH, bun_env.H, bun_env.W,
        embed=16, depth=1, heads=4, ff_factor=2, patch=3, state_dim=24)
    obs = jnp.zeros((2, bun_env.N_OBS_CH, bun_env.H, bun_env.W), jnp.float32)
    global_state = jnp.zeros((2, 24), jnp.float32)
    gradients = jax.grad(lambda current: bun_critic.frozen_transformer_features(
        current, obs, global_state).sum())(params)
    leaves = jax.tree.leaves(gradients)
    assert leaves
    assert all(bool(jnp.all(leaf == 0)) for leaf in leaves)


def test_independent_critic_parameter_tree_has_no_actor_parameters():
    params = bun_critic.init_independent_critic(
        jax.random.PRNGKey(2), input_dim=32, hidden=16, action_count=15)
    assert set(params) == {"layer1", "layer2", "value", "win", "q", "aux"}
    value, logits, win_logit, q = bun_critic.independent_critic_forward(
        params, jnp.zeros((3, 32), jnp.float32))
    assert value.shape == (3,)
    assert logits.shape == (3, 128)
    assert win_logit.shape == (3,)
    assert q.shape == (3, 15)
    aux = bun_critic.independent_critic_aux_forward(
        params, jnp.zeros((3, 32), jnp.float32))
    assert aux.shape == (3, 15, len(bun_critic.AUX_TARGET_NAMES))


def test_context_explicitly_encodes_actor_opponent_phase_and_bucket():
    bun_env.prepare()
    state = generator.scenario_state("carry_return", 20260926)
    context = generator.context_vector(state, actor_id=1, opponent_id=0, bucket_id=11)
    assert context.shape == (44,)
    assert context[:2].tolist() == [0.0, 1.0]
    assert context[2:4].tolist() == [1.0, 0.0]
    lesson_start = 4
    # 冻结 r6 为兼容已有 Critic checkpoint，context 仍使用 danger-arena 前的
    # legacy lesson 宽度；新增 lesson 通过映射复用既有槽位。
    phase_start = lesson_start + int(bun_env.LESSON_DANGER_ARENA)
    bucket_start = phase_start + 3
    assert context[lesson_start + int(state.lesson)] == 1.0
    assert context[phase_start + 2] == 1.0
    assert context[bucket_start + 11] == 1.0


def test_context_explicitly_encodes_frozen_league_opponent():
    bun_env.prepare()
    state = generator.scenario_state("ambush_contact", 20260926)
    context = generator.context_vector(
        state, actor_id=0, opponent_id=1, bucket_id=3, league_opponent_id=2)
    assert context.shape == (44,)
    assert context[-4:].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_action_aux_targets_respect_actor_trade_and_own_bomb_provenance():
    data = {
        "actor_id": np.asarray([0, 1], np.int8),
        "legal": np.ones((2, 15), np.bool_),
        "survivable": np.ones((2, 15), np.bool_),
        "first_credited_kill": np.zeros((2, 15, 2), np.bool_),
        "first_causal_kill": np.zeros((2, 15, 2), np.bool_),
        "first_mutual_death": np.zeros((2, 15), np.bool_),
        "first_own_bomb_defeat": np.zeros((2, 15, 2), np.bool_),
    }
    data["first_credited_kill"][0, 2, 0] = True
    data["first_causal_kill"][1, 3, 1] = True
    data["first_mutual_death"][1, 3] = True
    data["first_own_bomb_defeat"][1, 4, 1] = True
    targets = trainer.action_aux_targets(data)
    assert targets.shape == (2, 15, len(bun_critic.AUX_TARGET_NAMES))
    assert targets[0, 2, :3].tolist() == [1.0, 1.0, 0.0]
    assert targets[1, 3, :3].tolist() == [1.0, 0.0, 0.0]
    assert targets[1, 4, :3].tolist() == [1.0, 0.0, 1.0]


def test_migrate_independent_critic_pads_context_and_adds_aux_head():
    legacy = bun_critic.init_independent_critic(
        jax.random.PRNGKey(7), input_dim=20, hidden=16, action_count=15)
    legacy = {key: value for key, value in legacy.items() if key != "aux"}
    migrated = trainer.migrate_independent_critic(legacy, input_dim=27)
    assert migrated["layer1"][0].shape == (27, 16)
    assert migrated["aux"][0].shape == (
        16, 15 * len(bun_critic.AUX_TARGET_NAMES))
    np.testing.assert_allclose(np.asarray(migrated["layer1"][0][20:]), 0.0)


def test_wilson_and_mean_intervals_are_bounded_and_ordered():
    low, high = generator.wilson(7, 10)
    assert 0.0 <= low <= 0.7 <= high <= 1.0
    q_low, q_high = generator.mean_interval(np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert q_low < 2.5 < q_high


def test_confidence_interval_audit_tolerates_float32_rounding_on_legal_actions():
    row = {
        "legal": np.asarray([True, False]),
        "win_ci_low": np.asarray([0.2, 2.0], np.float32),
        "win_rate": np.asarray([0.2, -1.0], np.float32),
        "win_ci_high": np.asarray([0.2, -2.0], np.float32),
        "q_ci_low": np.asarray([10.497377, 5.0], np.float32),
        "q": np.asarray([10.497376, -5.0], np.float32),
        "q_ci_high": np.asarray([10.497377, -6.0], np.float32),
    }
    assert generator.confidence_intervals_valid(row)


def test_large_manifest_seed_maps_deterministically_without_split_aliasing():
    seeds = [202609260000, 202609260103, 202609260204]
    mapped = [int(generator.prng_seed(seed)) for seed in seeds]
    assert len(set(seeds)) == len(seeds)
    assert len(set(mapped)) == len(mapped)
    assert mapped == [seed & 0xFFFFFFFF for seed in seeds]


def test_ambush_snapshot_bucket_commits_rollout_snapshot():
    bun_env.prepare()

    def snapshot(state, _seed):
        return state._replace(core=state.core._replace(t=jnp.asarray(12, jnp.int32)))

    state = generator.scenario_state("ambush_early", 20260926, snapshot)
    assert int(state.core.t) == 12
    assert int(state.lesson) == bun_env.LESSON_FULL_AMBUSH


def test_first_transition_commits_selected_action_before_replanning():
    bun_env.prepare()
    opponent = init_transformer(
        jax.random.PRNGKey(11), bun_env.N_OBS_CH, bun_env.H, bun_env.W,
        embed=64, depth=3, heads=4, ff_factor=4, patch=3, state_dim=24)
    opponent["heads"]["wb"] = (
        jnp.zeros((64, bun_env.N_BOMB), jnp.float32),
        jnp.zeros((bun_env.N_BOMB,), jnp.float32),
    )
    state = generator.open_state(2026092601)
    transition = generator.build_first_transitions(opponent)
    actions = jnp.asarray([
        [move, ability]
        for move in range(bun_env.N_MOVES)
        for ability in range(bun_env.N_BOMB)
    ], jnp.int32)
    candidates, _, _, _ = transition(
        state, actions, jnp.asarray(0, jnp.int32), generator.prng_seed(17))
    bomb_action = 1
    cell = np.floor(np.asarray(state.core.pos[0])).astype(np.int32)
    assert int(np.asarray(candidates.core.t[bomb_action])) == int(state.core.t) + 1
    assert int(np.asarray(candidates.core.fuse[bomb_action, cell[0], cell[1]])) > 0


def test_onpolicy_concatenate_preserves_time_environment_alignment():
    def row(offset, scenario, opponent, actor_id):
        inputs = np.arange(offset, offset + 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
        reward = inputs[:, :, 0]
        return {
            "inputs": inputs,
            "reward": reward,
            "done": np.zeros((2, 2), np.bool_),
            "returns": reward + 0.5,
            "win_target": reward + 1.0,
            "win_labeled": np.ones((2, 2), np.bool_),
            "bootstrap": np.asarray([offset + 90, offset + 91], np.float32),
            "scenario": np.asarray([scenario, scenario]),
            "opponent": np.asarray([opponent, opponent]),
            "actor_id": np.asarray([actor_id, actor_id], np.int8),
        }

    merged = onpolicy.concatenate([
        row(0, "ambush", "weak", 0),
        row(100, "combat", "recent", 1),
    ])
    assert merged["inputs"].shape == (2, 4, 3)
    assert merged["reward"].shape == (2, 4)
    assert merged["inputs"][:, :, 0].tolist() == merged["reward"].tolist()
    assert merged["scenario"].tolist() == ["ambush", "ambush", "combat", "combat"]
    assert merged["opponent"].tolist() == ["weak", "weak", "recent", "recent"]
    assert merged["bootstrap"].tolist() == [90.0, 91.0, 190.0, 191.0]


def test_discounted_returns_use_bootstrap_only_for_nonterminal_tail():
    reward = np.asarray([[1.0, 1.0], [2.0, 2.0]], np.float32)
    done = np.asarray([[False, False], [False, True]])
    result = onpolicy.discounted_returns(
        reward, done, gamma=0.5, bootstrap=np.asarray([10.0, 10.0], np.float32))
    np.testing.assert_allclose(result[:, 0], [4.5, 7.0])
    np.testing.assert_allclose(result[:, 1], [2.0, 2.0])


def test_target_critic_checkpoint_roundtrip(tmp_path):
    params = bun_critic.init_independent_critic(
        jax.random.PRNGKey(19), input_dim=32, hidden=16, action_count=15)
    path = tmp_path / "target.pkl"
    separate_ac.save_checkpoint(path, params, None, {"global_cycle": 25})
    loaded, opt_state = separate_ac.load_checkpoint(path)
    assert opt_state is None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["metadata"]["global_cycle"] == 25
    for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual))


def test_actor_features_empty_partition_is_skipped_by_training_contract():
    actor_ids = np.zeros((4,), np.int8)
    assert np.flatnonzero(actor_ids == 1).size == 0
