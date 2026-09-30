"""Cross-language safe-teacher regression: frozen self-play combat states.

These cases are a balanced, deterministic snapshot of both-alive combat states
sampled from tactical rule-bot self-play (brick-ful danger_arena, fixed seeds).
They lock in the *safe tactical teacher* decisions — escape-under-fire,
doomed-max-survival, safe pressure attacks, and space control — beyond the
sparse FSM/routing cases in ``bun_rule_bot_v2_cases.json``. The identical
fixture is replayed by ``web/test_bun_rule_bot_selfplay_parity.js`` so any
Python/JS divergence in the survival planner is caught in both languages.
"""
import json
from pathlib import Path

from jax_bomb.bun_rule_bot import BunRuleBot

FIXTURE = (Path(__file__).parent / "fixtures"
           / "bun_rule_bot_selfplay_parity_cases.json")


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_selfplay_parity_cases_reproduce_exact_decisions():
    data = _fixture()
    bases = data["bun_bases"]
    assert data["cases"], "fixture must not be empty"
    for case in data["cases"]:
        state = dict(case["state"])
        state.setdefault("bun_bases", bases)
        decision = BunRuleBot().analyze(state, case["player_id"])
        action = [int(x) for x in decision.action]
        assert action == case["expected_action"], case["id"]
        assert decision.reason == case["expected_reason"], case["id"]
        assert decision.phase == case["expected_phase"], case["id"]


def test_selfplay_parity_fixture_covers_survival_critical_reasons():
    reasons = {case["expected_reason"] for case in _fixture()["cases"]}
    # The teacher's danger-facing behaviors must be represented so the parity
    # guard actually exercises the survival planner, not just idle routing.
    for required in ("escape_immediate", "doomed_max_survival",
                     "safe_pressure_attack"):
        assert required in reasons, required
