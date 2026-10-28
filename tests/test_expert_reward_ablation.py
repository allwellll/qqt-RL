from tests.external_artifacts import requires_artifacts
import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp

from scripts.expert_reward_ablation import bind_reward_arm, output_component_names, select_reward
from scripts.expert_reward_ablation_prepare import reward_component_names
from scripts.combat_item_events import METRICS as ITEM_METRICS


def test_minimal_arm_uses_only_preregistered_events_and_one_strict_pbrs():
    names = (
        "global_safety",
        "transport_capture",
        "lesson_task",
        "settled_kill",
        "settled_shaping",
        "control_entry_reward",
        "control_residual_reward",
        "route_shaped_reward",
        "banana_effective_pay",
        "combined_total",
    )
    parts = np.array(
        [
            [[-4, 6, 1, 2, 9, 4, 0.25, 0.5, 7, 23.75],
             [-5, 7, 3, 5, 8, 0, 0, -0.25, 6, 21.75]],
        ],
        dtype=np.float32,
    )
    legacy = np.array([[23.75, 21.75]], dtype=np.float32)
    info = {
        "capture": np.array([[True, False]]),
        "surviving_kill": np.array([[False, True]]),
        "death": np.array([[False, True]]),
    }

    result, terms, selected_parts = select_reward("minimal", legacy, info, parts, names)

    np.testing.assert_allclose(terms["delivery"], [[2, 0]])
    np.testing.assert_allclose(terms["kill"], [[0, 16]])
    np.testing.assert_allclose(terms["control"], [[4.25, 0]])
    np.testing.assert_allclose(terms["death"], [[0, -12]])
    np.testing.assert_allclose(terms["pbrs"], [[0.5, -0.25]])
    np.testing.assert_allclose(result, [[6.75, 3.75]])
    np.testing.assert_allclose(selected_parts[..., -1], result)
    for name in ("lesson_task", "settled_shaping", "banana_effective_pay"):
        np.testing.assert_array_equal(selected_parts[..., names.index(name)], 0)


def test_minimal_arm_ignores_nonpotential_shaping_and_item_payments():
    names = (
        "global_safety", "settled_kill", "transport_capture",
        "control_entry_reward",
        "control_residual_reward",
        "route_shaped_reward",
        "banana_effective_pay",
        "glue_effective_pay",
        "settled_shaping",
        "combined_total",
    )
    info = {
        "capture": np.zeros((1, 2), dtype=bool),
        "surviving_kill": np.zeros((1, 2), dtype=bool),
        "death": np.zeros((1, 2), dtype=bool),
    }
    first = np.array([[[0, 0, 0, 0, 0, 0.25, 0, 100, 100, 100],
                       [0, 0, 0, 0, 0, 0.25, 0, 100, 100, 100]]], dtype=np.float32)
    second = first.copy()
    second[..., 6:8] = -800

    first_reward, _, _ = select_reward("minimal", np.array([[9, 9]]), info, first, names)
    second_reward, _, _ = select_reward("minimal", np.array([[-99, 401]]), info, second, names)

    np.testing.assert_array_equal(first_reward, [[0.25, 0.25]])
    np.testing.assert_array_equal(second_reward, first_reward)


def test_legacy_arm_and_zero_coefficient_preserve_input_reward_exactly():
    names = ("route_shaped_reward", "control_entry_reward", "control_residual_reward")
    info = {
        "capture": np.ones((1, 2), dtype=bool),
        "surviving_kill": np.ones((1, 2), dtype=bool),
        "death": np.ones((1, 2), dtype=bool),
    }
    parts = np.ones((1, 2, len(names)), dtype=np.float32)
    legacy = np.array([[np.float32(-0.0), np.float32(3.125)]], dtype=np.float32)

    unchanged, terms, _ = select_reward("legacy", legacy, info, parts, names)
    zero, zero_terms, _ = select_reward("minimal", legacy, info, parts, names, coefficient=0.0)

    assert unchanged is legacy
    assert zero is legacy
    np.testing.assert_array_equal(terms["total"], np.zeros_like(legacy))
    np.testing.assert_array_equal(zero_terms["total"], np.zeros_like(legacy))
    assert np.signbit(unchanged[0, 0])


def test_output_names_match_reward_part_layout_and_reject_ambiguous_pbrs():
    names = output_component_names(("base", "route_shaped_reward", "combined_total"))
    assert names == ("base", "route_shaped_reward", "combined_total")

    try:
        select_reward("minimal", np.zeros((1, 2)), {}, np.zeros((1, 2, 5)),
                      ("global_safety", "settled_kill", "transport_capture",
                       "control_entry_reward", "control_residual_reward",
                       "route_shaped_reward", "strict_pbrs_reward", "combined_total"))
    except ValueError as exc:
        assert "exactly one" in str(exc).lower() or "ambiguous" in str(exc).lower()
    else:
        raise AssertionError("multiple potential reward channels must not enter the treatment")


def test_component_tensor_must_match_schema_width():
    names = ("global_safety", "settled_kill", "transport_capture",
             "control_entry_reward", "control_residual_reward",
             "route_shaped_reward", "combined_total")
    try:
        select_reward("minimal", np.zeros((1, 2)),
                      {"capture": np.zeros((1, 2)), "death": np.zeros((1, 2)),
                       "surviving_kill": np.zeros((1, 2))},
                      np.zeros((1, 2, len(names) - 1)), names)
    except ValueError as exc:
        assert "width" in str(exc).lower()
    else:
        raise AssertionError("truncated component tensors must be rejected")


def test_prepared_component_schema_matches_collector_appends_exact_item_metric_names():
    from scripts import je10_ops

    names = reward_component_names(je10_ops.COMPONENTS)
    assert names == tuple(je10_ops.COMPONENTS[:-1]) + tuple(ITEM_METRICS) + ("combined_total",)
    assert names[-len(ITEM_METRICS) - 1:-1] == tuple(ITEM_METRICS)
    assert len(set(names)) == len(names)


@requires_artifacts('scripts/expert_reward_ablation_run.py', 'scripts/expert_reward_ablation_cpu.py')
def test_persistent_owners_retain_their_exclusive_lock_handles():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    gpu_owner = (root / "scripts/expert_reward_ablation_run.py").read_text()
    cpu_owner = (root / "scripts/expert_reward_ablation_cpu.py").read_text()
    assert "owner_lock = lock(B / \"owner.lock\")" in gpu_owner
    assert "owner_lock = lock(B / \"cpu_evaluator.lock\")" in cpu_owner


@requires_artifacts('runs/ppo_scale_20261025/r2_longrun/USER_ENDED_EARLY.json', 'scripts/expert_reward_ablation_run.py')
def test_predecessor_handoff_accepts_user_ended_early_without_full_budget_claim():
    from pathlib import Path
    from scripts.expert_reward_handoff import validate_user_ended_early

    root = Path(__file__).resolve().parents[1]
    receipt = validate_user_ended_early(root, "runs/ppo_scale_20261025/r2_longrun")
    assert receipt["kind"] == "USER_ENDED_EARLY"
    assert receipt["fixed_budget_complete"] is False
    assert receipt["last_atomic_new_update"] == 1664

    owner_source = (root / "scripts/expert_reward_ablation_run.py").read_text()
    assert "predecessor_terminal()" in owner_source
    assert "predecessor_fixed_budget_complete" in owner_source


def test_gpu_handoff_ignores_stale_compute_entries_for_zombie_processes():
    from scripts.expert_reward_handoff import filter_live_compute_apps

    apps = "4312, GPU-a\n4313, GPU-b\n4314, GPU-c"
    statuses = "4312 S\n4313 Zs\n4314 R"
    assert filter_live_compute_apps(apps, statuses) == "4312, GPU-a\n4314, GPU-c"


def test_arm_binding_filters_controlled_kill_and_keeps_parent_state_transition():
    names = (
        "global_safety",
        "settled_kill",
        "transport_capture",
        "control_entry_reward",
        "control_residual_reward",
        "control_clawback",
        "route_shaped_reward",
        "item_positive",
        "combined_total",
    )
    legacy = np.array([[1, 2]], dtype=np.float32)
    parts = np.array([[[10, 16, 2, 4, 0.25, -4, 0.5, 8, 37.75],
                       [10, 16, 2, 4, 0.25, -4, 0.5, 8, 37.75]]], dtype=np.float32)
    info = {"capture": np.zeros((1, 2), dtype=bool),
            "death": np.zeros((1, 2), dtype=bool),
            "surviving_kill": np.ones((1, 2), dtype=bool)}
    new_aux = {"same": np.array([1])}
    metrics = {"unchanged": True}

    def parent(*_args):
        return legacy, new_aux, parts, metrics

    ops = SimpleNamespace(combined_reward=parent, COMPONENTS=names)
    credit = SimpleNamespace(parent_reward_events=lambda _states, _info: {
        "surviving_kill": np.array([[False, True]])})
    restore = bind_reward_arm(ops, credit, "minimal")
    actual, aux, selected_parts, actual_metrics = ops.combined_reward(
        None, None, info, None, None, 1.0, None)

    assert restore is parent
    assert aux is new_aux
    assert actual_metrics["unchanged"] is True
    np.testing.assert_allclose(actual, [[4.75, 20.75]])
    np.testing.assert_allclose(selected_parts[..., names.index("settled_kill")], [[0, 16]])
    np.testing.assert_array_equal(selected_parts[..., names.index("item_positive")], 0)
    np.testing.assert_array_equal(selected_parts[..., names.index("control_clawback")], 0)


def test_minimal_reward_adapter_is_traceable_inside_jax_update():
    names = ("global_safety", "settled_kill", "transport_capture",
             "control_entry_reward", "control_residual_reward",
             "route_shaped_reward", "settled_shaping", "combined_total")

    def apply(base, parts, capture, kill, death):
        reward, terms, updated = select_reward(
            "minimal", base,
            {"capture": capture, "surviving_kill": kill, "death": death},
            parts, names)
        return reward, updated[..., -1], terms["pbrs"]

    compiled = jax.jit(apply)
    reward, total, pbrs = compiled(
        jnp.array([[8.0, -3.0]]),
        jnp.array([[[0.0, 0.0, 0.0, 4.0, 0.25, 0.5, 99.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -0.25, 99.0, 0.0]]]),
        jnp.array([[True, False]]),
        jnp.array([[False, True]]),
        jnp.array([[False, True]]),
    )
    np.testing.assert_allclose(reward, [[6.75, 3.75]])
    np.testing.assert_allclose(total, reward)
    np.testing.assert_allclose(pbrs, [[0.5, -0.25]])
