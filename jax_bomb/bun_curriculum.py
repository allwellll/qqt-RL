"""Bun 分阶段课程门禁。"""

from __future__ import annotations

from collections.abc import Mapping


STAGE_ORDER = (
    "route", "bridge", "bridge_relaxed", "bridge_capture", "combat",
    "objective", "full")

STAGE_GATES = {
    "route": {
        "minimum": {
            "target_wall_rate": 0.60,
            "safe_detonation_ratio": 0.80,
        },
        "maximum": {"self_death_rate": 0.15},
    },
    "bridge": {
        "minimum": {
            "bridge_success_rate": 0.60,
            "safe_detonation_ratio": 0.80,
        },
        "maximum": {"self_death_rate": 0.15},
    },
    "bridge_relaxed": {
        "minimum": {
            "bridge_success_rate": 0.60,
            "safe_detonation_ratio": 0.80,
        },
        "maximum": {"self_death_rate": 0.15},
    },
    "bridge_capture": {
        "minimum": {
            "bridge_success_rate": 0.60,
            "capture_rate": 0.60,
            "safe_detonation_ratio": 0.80,
        },
        "maximum": {"self_death_rate": 0.15},
    },
    "combat": {
        "minimum": {
            "static_hit_rate": 0.80,
            "moving_hit_rate": 0.60,
            "causal_kill_rate": 0.50,
            "safe_detonation_ratio": 0.80,
        },
        "maximum": {"self_death_rate": 0.20},
    },
    "objective": {
        "minimum": {
            "near_steal_rate": 0.80,
            "carry_capture_rate": 0.60,
            "carry_return_rate": 0.60,
        },
        "maximum": {"self_death_rate": 0.20},
    },
    "full": {
        "minimum": {
            "full_capture_rate": 0.15,
            "safe_detonation_ratio": 0.80,
        },
        "maximum": {"self_death_rate": 0.20},
    },
}


def combat_gate_metrics(static_summary: Mapping[str, float],
                        moving_summary: Mapping[str, float],
                        kill_summary: Mapping[str, float]):
    """构造 Combat 门禁指标；最终击杀必须来自活跃自博弈。"""
    summaries = (static_summary, moving_summary, kill_summary)
    return {
        "static_hit_rate": static_summary.get("p0_credited_hit_rate"),
        "moving_hit_rate": moving_summary.get("p0_credited_hit_rate"),
        "causal_kill_rate": kill_summary.get("p0_causal_kill_rate"),
        "safe_detonation_ratio": min(
            (item.get("p0_safe_detonation_ratio", float("-inf"))
             for item in summaries),
            default=float("-inf"),
        ),
        "self_death_rate": max(
            (item.get("p0_self_kill_rate", float("inf"))
             for item in summaries),
            default=float("inf"),
        ),
    }


def check_stage_gate(stage: str, metrics: Mapping[str, float]):
    """返回 ``(是否晋级, 未通过指标)``，缺失指标按失败处理。"""
    if stage not in STAGE_GATES:
        raise ValueError(f"未知 Bun 课程阶段: {stage}")
    failures = {}
    gate = STAGE_GATES[stage]
    for name, threshold in gate["minimum"].items():
        value = float(metrics.get(name, float("-inf")))
        if value < threshold:
            failures[name] = {"value": value, "required_min": threshold}
    for name, threshold in gate["maximum"].items():
        value = float(metrics.get(name, float("inf")))
        if value > threshold:
            failures[name] = {"value": value, "required_max": threshold}
    return not failures, failures
