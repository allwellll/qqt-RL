from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from jax_bomb.bun_rule_bot import (
    HORIZON_STEPS,
    TICK_HZ,
    BunRuleBot,
    decide_batch,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bun_rule_bot_cases.json"


def _fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixed_four_second_window_and_action_contract():
    fixtures = _fixtures()
    assert TICK_HZ == fixtures["tick_hz"] == 10
    assert HORIZON_STEPS == fixtures["horizon_steps"] == 40
    action = BunRuleBot().decide(fixtures["cases"][0]["state"], player_id=0)
    assert action.shape == (2,)
    assert action.dtype == np.int32
    assert 0 <= int(action[0]) < 5
    assert 0 <= int(action[1]) < 3


def test_shared_fixtures_have_exact_actions_and_reasons():
    bot = BunRuleBot()
    for case in _fixtures()["cases"]:
        decision = bot.analyze(case["state"], player_id=0)
        assert decision.action.tolist() == case["expected_action"], case["id"]
        if case["id"] in {"own_safe_drop", "bun_cage_chokepoint", "safe_non_trade_kill"}:
            assert decision.safe_escape_after_bomb is True
        if case["id"] in {"no_safe_drop", "trade_attack_rejected"}:
            assert int(decision.action[1]) == 0
        if case["id"] == "already_doomed":
            assert decision.doomed is True
            assert decision.claimed_escape is False


def test_chain_prediction_preserves_physical_and_causal_owners():
    case = next(item for item in _fixtures()["cases"] if item["id"] == "chain_reaction_escape")
    analysis = BunRuleBot().analyze(case["state"], player_id=0)
    chained = [bomb for bomb in analysis.predicted_bombs if bomb.row == 6 and bomb.col == 8][0]
    assert chained.explode_step == 5
    assert chained.physical_owner == 0
    assert chained.causal_owner == 1


def test_batch_api_matches_scalar_and_is_precompute_friendly():
    states = [case["state"] for case in _fixtures()["cases"]]
    scalar = np.stack([BunRuleBot().decide(state, 0) for state in states])
    batch = decide_batch(states, player_ids=np.zeros(len(states), dtype=np.int32))
    np.testing.assert_array_equal(batch, scalar)
    assert batch.dtype == np.int32
    assert batch.shape == (len(states), 2)


def test_typical_map_decision_hard_limit_under_10ms_after_warmup():
    case = next(item for item in _fixtures()["cases"] if item["id"] == "safe_non_trade_kill")
    bot = BunRuleBot()
    for _ in range(20):
        bot.decide(case["state"], 0)
    elapsed_ms = []
    for _ in range(300):
        started = time.thread_time_ns()
        bot.decide(case["state"], 0)
        elapsed_ms.append((time.thread_time_ns() - started) / 1e6)
    assert max(elapsed_ms) < 10.0
    assert float(np.percentile(elapsed_ms, 50)) < 2.0
