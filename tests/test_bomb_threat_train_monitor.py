import json

from scripts.bomb_threat_train_monitor import (
    aggregate_window, detect_alerts, native_component_schema, read_updates,
)


def row(update=1, bombs=10, ticks=100, resets=1, finite=True, env_steps=1000):
    parts = [0.0] * 90
    parts[2] = 2.0
    parts[9] = 1.0
    parts[61] = 0.5
    return {
        "new_update": update, "finite": finite,
        "actual_environment_steps_this_update": env_steps,
        "cycle_seconds": 1.0, "update_l2": {"actor": 1.0, "critic": 2.0},
        "loss": [1, 2], "unix": 1000 + update,
        "exposure": {"learner_bombs_placed": bombs,
            "learner_bomb_requests_rejected": 1, "threat_events": 2,
            "topology_events": 3, "captures": 4, "learner_deaths": 1,
            "resets": resets, "clear_scene_ticks": 10,
            "clear_scene_resets": 1, "bank_resets": 1, "total_ticks": ticks,
            "chain_reward": 2, "absolute_total_reward": 20,
            "native_parts": parts, "pool_ticks": [1, 2, 3, 4, 5, 6],
            "pool_new_episodes": [1, 1, 1, 1, 1, 1]},
    }


def test_schema_names_90_are_explicit_and_stable():
    schema = native_component_schema(90)
    assert schema["status"] == "known_90_v1"
    assert schema["indices"]["settled_kill"] == 2
    assert schema["indices"]["transport_capture"] == 9
    assert schema["indices"]["banana_pay"] == 84
    assert schema["indices"]["combined_total"] == 89


def test_aggregate_uses_denominators_and_marks_unrecorded_policy_metrics():
    out = aggregate_window("run", [row(1), row(2)], 1)
    assert out["per_100_tick"]["bombs_placed"] == 10
    assert out["avg_episode_ticks"] == 100
    assert out["policy_entropy"] == "未记录"
    assert out["opponents"]["hunter_tier3a"]["wins"] == "未记录"
    assert out["reward"]["native_component_schema"]["status"] == "known_90_v1"


def test_alerts_do_not_treat_missing_entropy_as_zero():
    rows = [aggregate_window("run", [row(i, bombs=10 if i == 1 else 1)], i) for i in range(1, 4)]
    alerts = detect_alerts(rows)
    codes = {x["code"] for x in alerts}
    assert "LOW_BOMBS_THREE_WINDOWS" not in codes
    assert "ENTROPY_UNAVAILABLE" in codes


def test_delivery_and_kill_direction_alert_requires_real_numeric_kills():
    rows = []
    for i, (captures, kill) in enumerate(((1, 5), (2, 4)), 1):
        r = row(i)
        r["exposure"]["captures"] = captures
        r["exposure"]["learner_kills"] = kill
        rows.append(aggregate_window("run", [r], i))
    assert "DELIVERY_UP_KILLS_DOWN" in {x["code"] for x in detect_alerts(rows)}


def test_read_updates_ignores_truncated_tail_and_deduplicates(tmp_path):
    p = tmp_path / "updates.jsonl"
    p.write_text(json.dumps(row(1)) + "\n" + json.dumps(row(1, bombs=99)) + "\n{" )
    got = read_updates(p)
    assert len(got) == 1 and got[0]["exposure"]["learner_bombs_placed"] == 99
