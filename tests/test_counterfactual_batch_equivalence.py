import os

os.environ.setdefault("JAXBOMB_RULE", "bun")

import jax

from jax_bomb import bun_env
from jax_bomb.jax_net import init_transformer
from qqt_rl.training.counterfactual import StateRequest, compare_replay_arrays
from scripts.generate_bun_tactical_v2_critic_data import generate_replay_arrays


def _actor():
    bun_env.prepare()
    bun_env.configure_training("danger_arena=1", 1, reward_profile="danger_arena")
    return init_transformer(
        jax.random.PRNGKey(17), bun_env.N_OBS_CH, bun_env.H, bun_env.W,
        embed=16, depth=1, heads=4, ff_factor=2, patch=3, state_dim=24)


def test_batched_counterfactual_matches_legacy_for_fixed_seeds():
    actor = _actor()
    requests = [
        StateRequest("train", 2026092801, 0),
        StateRequest("train", 2026092802, 0),
    ]
    legacy, legacy_meta = generate_replay_arrays(
        actor, requests, mc_samples=1, mc_horizon=1,
        state_batch_size=1, engine="legacy")
    batched, batched_meta = generate_replay_arrays(
        actor, requests, mc_samples=1, mc_horizon=1,
        state_batch_size=2, engine="batched")
    report = compare_replay_arrays(legacy, batched)
    differences = {key: value for key, value in report["fields"].items()
                   if not value["exact"]}
    assert set(differences) <= {"policy_probability"}, differences
    if "policy_probability" in differences:
        assert differences["policy_probability"]["max_abs_error"] <= 2e-8
    assert legacy_meta["declared_seeds"] == batched_meta["declared_seeds"]
    assert legacy_meta["effective_seeds"] == batched_meta["effective_seeds"]
    assert batched_meta["fixed_state_batch_size"] == 2


def test_batched_batch1_matches_legacy_at_rollout_horizon():
    """生产等价保证: state_batch_size=1 的 batched 引擎在真实 rollout horizon 下与
    legacy 位精确 (含 deterministic/mc objective 标签)。这是生产选定配置的契约锁。"""
    actor = _actor()
    requests = [
        StateRequest("train", 2026092801, 0),
        StateRequest("train", 2026092802, 0),
        StateRequest("train", 2026092803, 0),
    ]
    legacy, _ = generate_replay_arrays(
        actor, requests, mc_samples=2, mc_horizon=8,
        state_batch_size=1, engine="legacy")
    batched, _ = generate_replay_arrays(
        actor, requests, mc_samples=2, mc_horizon=8,
        state_batch_size=1, engine="batched")
    report = compare_replay_arrays(legacy, batched)
    differences = {key: value for key, value in report["fields"].items()
                   if not value["exact"]}
    assert differences == {}, differences


def test_batched_multistate_batch_diverges_from_legacy_at_rollout_horizon():
    """已知数值限制: state_batch_size>1 改变 transformer matmul 分块形状, 前向舍入差
    经 rollout 放大, 使标签不再与 legacy 位精确。故生产禁用 batch>1, 固定 batch=1。
    这里锁定该发散信号, 防止有人误将默认改回 >1 并声称等价。"""
    actor = _actor()
    requests = [
        StateRequest("train", 2026092801, 0),
        StateRequest("train", 2026092802, 0),
        StateRequest("train", 2026092803, 0),
    ]
    legacy, _ = generate_replay_arrays(
        actor, requests, mc_samples=2, mc_horizon=8,
        state_batch_size=1, engine="legacy")
    multi, _ = generate_replay_arrays(
        actor, requests, mc_samples=2, mc_horizon=8,
        state_batch_size=3, engine="batched")
    report = compare_replay_arrays(legacy, multi)
    differences = {key: value for key, value in report["fields"].items()
                   if not value["exact"]}
    assert "policy_probability" in differences
    assert differences["policy_probability"]["max_abs_error"] > 1e-6
