"""Bun safe-aggression 训练配置与固定里程碑。"""

from __future__ import annotations

EVAL_CYCLES = frozenset({
    0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 90,
    120, 160, 200, 240, 280, 320, 360,
})

FULL_FAMILY = {"name": "full_v2", "horizon": 40, "aggression": 1.0,
               "escape_margin": 0, "tie_break": 0, "bomb_threshold": 6,
               "positioning_preference": "chase", "structural_mode": "v2"}
EASY_FAMILY = {"name": "easy", "horizon": 24, "aggression": 0.55,
               "escape_margin": 0, "tie_break": 1, "bomb_threshold": 3,
               "positioning_preference": "space", "structural_mode": "v2"}
NORMAL_FAMILY = {"name": "normal", "horizon": 32, "aggression": 0.78,
                 "escape_margin": 1, "tie_break": 2, "bomb_threshold": 5,
                 "positioning_preference": "flank", "structural_mode": "v2"}
HELDOUT_FAMILY = {"name": "heldout_evasive_counter", "horizon": 36,
                  "aggression": 0.65, "escape_margin": 2, "tie_break": 3,
                  "bomb_threshold": 4, "positioning_preference": "flank",
                  "structural_mode": "heldout_evasive_counter"}
UNSEEN_FAMILY = {"name": "unseen_balanced", "horizon": 35,
                 "aggression": 0.86, "escape_margin": 1, "tie_break": 4,
                 "bomb_threshold": 4, "positioning_preference": "space",
                 "structural_mode": "v2"}


def validate_cycle_count(cycles: int) -> int:
    """正式 runner 只接受全局 cycle 1..360，防止错误续跑越界。"""
    if cycles not in range(1, 361):
        raise ValueError("cycles must be in [1, 360]")
    return cycles


def family_for_cycle(config: dict, cycle: int) -> dict:
    if not config.get("family_curriculum"):
        return FULL_FAMILY
    fraction = cycle / max(int(config["total_cycles"]), 1)
    if fraction <= 0.4:
        pattern = (EASY_FAMILY, NORMAL_FAMILY, EASY_FAMILY, NORMAL_FAMILY, FULL_FAMILY)
    elif fraction <= 0.8:
        pattern = (EASY_FAMILY, NORMAL_FAMILY, FULL_FAMILY, NORMAL_FAMILY, FULL_FAMILY)
    else:
        pattern = (EASY_FAMILY, NORMAL_FAMILY, NORMAL_FAMILY, NORMAL_FAMILY,
                   FULL_FAMILY, FULL_FAMILY, FULL_FAMILY, FULL_FAMILY,
                   FULL_FAMILY, FULL_FAMILY)
    return pattern[(cycle - 1) % len(pattern)]
