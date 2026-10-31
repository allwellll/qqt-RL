import json
import numpy as np
import pytest

from scripts.expert_validation_diagnostics import (
    build_horizon_sensitivity,
    component_schema,
    summarize,
)


def test_component_schema_matches_appended_item_metrics_and_total():
    assert component_schema(
        ("base_reward", "combined_total"),
        ("banana_completed", "item_replacement_delta"),
    ) == ("base_reward", "banana_completed", "item_replacement_delta", "combined_total")


def test_component_schema_rejects_ambiguous_total_or_duplicate_label():
    with pytest.raises(ValueError, match="end in combined_total"):
        component_schema(("base_reward",), ("item",))
    with pytest.raises(ValueError, match="duplicate"):
        component_schema(("route_shaped_reward", "combined_total"),
                         ("route_shaped_reward",))


def test_summarize_separates_native_terminal_from_128_tick_truncation():
    steps, episodes = 128, 2
    active = np.ones((steps, episodes), dtype=bool)
    active[64:, 0] = False
    terminal = np.zeros((steps, episodes), dtype=bool)
    terminal[63, 0] = True
    reward = np.zeros((steps, episodes), dtype=np.float32)
    reward[63, 0] = 1.0
    parts = np.zeros((steps, episodes, 1), dtype=np.float32)
    parts[63, 0, 0] = 1.0
    actions = np.zeros((steps, episodes, 2), dtype=np.int32)
    scalar = np.zeros((steps, episodes), dtype=np.int32)
    events = np.zeros((steps, episodes, 8), dtype=np.int32)
    ledger = np.zeros((steps, episodes, 17), dtype=np.float32)
    ledger_names = (
        "control_gross_p0", "control_balance_p0", "route_phi_p0", "route_offset_p0",
        "transport_balance_p0", "transport_pending_capture_p0", "transport_episode_gross_p0",
        "drop_balance_origin0", "drop_balance_origin1", "drop_pending_capture_origin0",
        "drop_pending_capture_origin1", "banana_balance_p0", "banana_legacy_balance_p0",
        "banana_gross_p0", "glue_balance_p0", "glue_legacy_balance_p0", "glue_gross_p0",
    )
    arrays = [
        reward, parts, actions, active, events, scalar, scalar,
        np.full((steps, episodes), -1, dtype=np.int32), scalar, scalar,
        scalar, scalar, scalar, scalar, ledger, terminal,
    ]

    result = summarize("candidate", {}, ["combined_total"], arrays)

    assert result["steps"] == 128
    assert result["valid_transition_count"] == 192
    assert result["per_episode"][0]["terminal_transition_count"] == 1
    assert result["per_episode"][0]["artificial_horizon_truncated"] is False
    assert result["per_episode"][1]["terminal_transition_count"] == 0
    assert result["per_episode"][1]["artificial_horizon_truncated"] is True
    assert "no V bootstrap used" in result["bootstrap_status"]


def test_horizon_sensitivity_includes_only_identity_checked_extended_artifacts(tmp_path):
    artifact = tmp_path / "parent-128.json"
    artifact.write_text(json.dumps({
        "schema": "qqt.expert-validation-reward-alignment-horizon/v1",
        "checkpoint": "parent",
        "horizon": 128,
        "result": {
            "checkpoint": "parent",
            "steps": 128,
            "state_sha256": "state",
            "valid_transition_count": 1024,
            "terminal_transition_count": 0,
            "event_sum_p0": [0] * 8,
            "undiscounted_return_p0": [0.0] * 8,
            "discounted_return_p0_gamma_0.995": [0.0] * 8,
            "bootstrap_status": "no V bootstrap used",
            "per_episode": [{"terminal_transition_count": 0,
                             "artificial_horizon_truncated": True} for _ in range(8)],
        },
    }))
    base = {"checkpoint": "parent", "steps": 64,
            "undiscounted_return_p0": [0.0] * 8,
            "discounted_return_p0_gamma_0.995": [0.0] * 8,
            "per_episode": [{"terminal_transition_count": 0,
                             "artificial_horizon_truncated": True} for _ in range(8)]}

    result = build_horizon_sensitivity([base], {"parent": artifact})

    assert result["base_rows"][0]["horizon"] == 64
    assert len(result["extended_rows"]) == 1
    assert result["extended_rows"][0]["checkpoint"] == "parent"
    assert result["extended_rows"][0]["horizon"] == 128
    assert result["extended_rows"][0]["artificial_horizon_truncated"] == [True] * 8
    assert result["extended_rows"][0]["no_v_bootstrap"] is True
    assert not result["extended_rows"][0]["artifact"].startswith("/")

    wrong_identity = tmp_path / "wrong.json"
    wrong_identity.write_text(artifact.read_text().replace('"checkpoint": "parent"', '"checkpoint": "other"', 1))
    with pytest.raises(ValueError, match="identity mismatch"):
        build_horizon_sensitivity([base], {"parent": wrong_identity})
