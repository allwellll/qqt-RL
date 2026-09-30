import json
from types import SimpleNamespace

import pytest

from scripts import eval_transformer_checkpoint_sweep as sweep


def make_run(tmp_path, iters=(500, 1000), final=2000):
    (tmp_path / "phase1.pt").write_bytes(b"p1")
    for it in iters:
        (tmp_path / f"phase2_it{it}.pt").write_bytes(f"it{it}".encode())
    (tmp_path / "phase2.pt").write_bytes(b"final")
    (tmp_path / "phase2.json").write_text(json.dumps({"ppo_iterations": final}))
    (tmp_path / "phase2_it500.json").write_text("{}")
    return tmp_path


def fake_result(sha, kills, own, opp_phys, opp_causal, spawns):
    games = len(kills)
    return {
        "schema": sweep.EVAL_SCHEMA, "checkpoint_sha256": sha, "seed": 7, "max_steps": 300,
        "summary": {
            "games": games,
            "p0_surviving_causal_kill_rate": sum(kills) / games,
            "p0_own_bomb_defeat_rate": sum(own) / games,
            "p0_avoidable_danger_death_rate": 0.0, "p0_danger_to_death_rate": sum(own) / games,
            "mutual_death_rate": 0.0, "p0_safe_detonation_ratio": 0.9,
            "p0_tactical_resolution_ratio": 1.0, "p0_avg_bombs": 3.0,
            "p0_safe_tactical_placements_per_game": 1.0, "p0_avg_policy_entropy": 1.1,
        },
        "per_episode": {
            "spawn_cells": spawns, "surviving_causal_kill": kills, "own_bomb_defeat": own,
            "opponent_physical_defeat": opp_phys, "opponent_causal_defeat": opp_causal,
            "avoidable_danger_death": [False] * games, "danger_to_death": own,
            "mutual_death": [False] * games,
        },
    }


def test_discover_maps_phase1_to_zero_and_final_to_ppo_iterations(tmp_path):
    items = sweep.discover(make_run(tmp_path))
    assert [(i["name"], i["iteration"], i["final"]) for i in items] == [
        ("phase1", 0, False), ("phase2_it500", 500, False),
        ("phase2_it1000", 1000, False), ("phase2", 2000, True)]


def test_discover_rejects_final_colliding_with_intermediate(tmp_path):
    make_run(tmp_path, iters=(500, 2000), final=2000)
    with pytest.raises(SystemExit):
        sweep.discover(tmp_path)


def test_select_supports_names_iters_ranges_and_final(tmp_path):
    items = sweep.discover(make_run(tmp_path, iters=(500, 1000, 1500)))
    names = lambda spec: [i["name"] for i in sweep.select(items, spec)]
    assert names("phase1,final") == ["phase1", "phase2"]
    assert names("1000-2000") == ["phase2_it1000", "phase2_it1500", "phase2"]
    assert names("500,phase2_it500") == ["phase2_it500"]
    with pytest.raises(SystemExit):
        sweep.select(items, "999")


def test_wilson_matches_historical_eval_interval():
    lo, hi = sweep.wilson(20, 64)
    assert lo == pytest.approx(0.21230909502265743)
    assert hi == pytest.approx(0.43392566359152623)
    assert sweep.wilson(0, 64)[0] == 0.0


def test_reusable_requires_matching_sha_seed_games(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(fake_result("abc", [True, False], [False, False],
                                           [False] * 2, [False] * 2, [[0], [1]])))
    assert sweep.reusable(path, "abc", 7, 2, 300)
    assert not sweep.reusable(path, "abd", 7, 2, 300)
    assert not sweep.reusable(path, "abc", 8, 2, 300)
    assert not sweep.reusable(path, "abc", 7, 64, 300)


def test_killed_by_opponent_is_union_not_sum():
    data = fake_result("s", [False] * 4, [False] * 4,
                       [True, True, False, False], [True, False, True, False], [[0]] * 4)
    row = sweep.row_for({"name": "x", "iteration": 0, "final": False}, data)
    assert row["metrics"]["killed_by_opponent"]["count"] == 3


def test_summary_paired_deltas_and_spawn_consistency(tmp_path):
    spawns = [[[1, 1], [2, 2]]] * 4
    ref = fake_result("a", [False, False, True, False], [True, True, False, False],
                      [False] * 4, [False] * 4, spawns)
    new = fake_result("b", [True, True, True, True], [False] * 4,
                      [False] * 4, [False] * 4, spawns)
    items = [{"name": "phase1", "iteration": 0, "final": False},
             {"name": "phase2", "iteration": 100, "final": True}]
    args = SimpleNamespace(seed=7, games=4, max_steps=300, run_dir=tmp_path, reference="phase1")
    summary = sweep.summarize(items, {"phase1": ref, "phase2": new}, args)
    assert summary["spawns_identical_across_checkpoints"]
    delta = summary["checkpoints"][1]["paired_vs_reference"]
    assert delta["surviving_kill"]["delta"] == pytest.approx(0.75)
    assert delta["own_bomb_death"]["delta"] == pytest.approx(-0.5)
    sweep.write_csv(summary, tmp_path / "s.csv")
    assert (tmp_path / "s.csv").read_text().splitlines()[0].startswith("name,iteration")
    sweep.plot([("t", summary)], {"surviving_kill_player_rate": 0.0}, tmp_path / "c.png", "t")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_paired_delta_refuses_mismatched_spawns():
    a = fake_result("a", [True], [False], [False], [False], [[[1, 1], [2, 2]]])
    b = fake_result("b", [True], [False], [False], [False], [[[2, 2], [1, 1]]])
    assert sweep.paired_delta(a, b) is None
