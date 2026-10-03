import json
from types import SimpleNamespace

import pytest

from scripts import eval_four_bot_matrix as matrix


def test_bot_list_is_exactly_tactical_plus_three_web_hunters():
    assert matrix.BOTS == ("bun.tactical_v2", "bun.hunter@hard", "bun.hunter@normal", "bun.hunter@easy")


def test_outcome_flags_separate_trade_from_kill_and_loss():
    win = matrix.outcome_flags(True, False, False, False)
    assert win["win"] and not win["loss"]
    trade = matrix.outcome_flags(False, False, False, True)
    assert trade["loss"] and not trade["win"] and not trade["surviving_kill"]
    mixed = matrix.outcome_flags(True, True, False, False)
    assert not mixed["win"] and not mixed["loss"]


def _tactical(games=4, spawns=None, sha="a" * 64, seed=7):
    flags = lambda *v: list(v)
    return {
        "schema": matrix.TACTICAL_SCHEMA, "checkpoint_sha256": sha, "seed": seed, "max_steps": 300,
        "summary": {"games": games},
        "per_episode": {
            "spawn_cells": spawns or [[[1, 1], [2, 2]]] * games,
            "surviving_causal_kill": flags(True, False, False, False),
            "surviving_physical_kill": flags(False, False, False, False),
            "opponent_physical_defeat": flags(False, True, False, False),
            "opponent_causal_defeat": flags(False, False, False, False),
            "own_bomb_defeat": flags(False, False, True, False),
            "mutual_death": flags(False, False, False, True),
            "danger_to_death": flags(False, True, True, False),
            "bombs": [3, 1, 2, 0],
        },
    }


def test_tactical_and_hunter_episodes_share_definitions():
    tactical = matrix.aggregate(matrix.tactical_episodes(_tactical()))
    records = []
    for game, (k, d, o, m, b) in enumerate([(1, 0, 0, 0, 3), (0, 1, 0, 0, 1), (0, 0, 1, 0, 2), (0, 0, 0, 1, 0)]):
        records.append({"game": game, "counts": {
            "surviving_causal_kill": k, "surviving_physical_kill": 0, "killed_by_bot": d,
            "own_bomb_defeat": o, "mutual_death": m, "bombs": b, "bot_own_bomb_defeat": 0, "bot_bombs": 1}})
    hunter = matrix.aggregate(matrix.hunter_episodes(list(reversed(records))))
    for key in matrix.EPISODE_FLAGS:
        assert tactical[key] == hunter[key], key
    assert tactical["games"] == hunter["games"] == 4
    assert tactical["win"]["count"] == 1 and tactical["loss"]["count"] == 3
    assert tactical["kill_share"] == 0.5
    assert tactical["bombs_per_game"] == hunter["bombs_per_game"] == 1.5


def test_tactical_cell_requires_exact_protocol(tmp_path):
    spawns = [[[1, 1], [2, 2]]] * 4
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_tactical(spawns=spawns)))
    assert matrix.tactical_cell_ok(path, "a" * 64, 7, 4, 300, spawns)
    assert not matrix.tactical_cell_ok(path, "b" * 64, 7, 4, 300, spawns)
    assert not matrix.tactical_cell_ok(path, "a" * 64, 8, 4, 300, spawns)
    assert not matrix.tactical_cell_ok(path, "a" * 64, 7, 64, 300, spawns)
    assert not matrix.tactical_cell_ok(path, "a" * 64, 7, 4, 300, [[[0, 0], [2, 2]]] * 4)
    assert not matrix.tactical_cell_ok(tmp_path / "missing.json", "a" * 64, 7, 4, 300, spawns)


def test_trace_protocol_does_not_reuse_non_trace_cache(tmp_path):
    path = tmp_path / "t.json"
    spawns = [[[1, 1], [2, 2]]] * 4
    path.write_text(json.dumps(_tactical(spawns=spawns)))
    assert not matrix.tactical_cell_ok(path, "a" * 64, 7, 4, 300, spawns, trace=True)
    assert not matrix.shard_ok(_shard(tmp_path), **META, trace=True)


def _shard(tmp_path, **override):
    data = {"schema": matrix.SHARD_SCHEMA, "seed": 7, "max_steps": 300,
            "bot": {"config": {"difficulty": "hard"}}, "game_range": [0, 2],
            "episodes": [{}, {}], "model": {"model_json_sha256": "m"}, "checkpoint_sha256": "c",
            "source_hashes": {"web/sim.js": "s"}, "spawn_manifest_sha256": "p"}
    data.update(override)
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(data))
    return path


META = dict(seed=7, max_steps=300, difficulty="hard", start=0, end=2, model_sha="m",
            checkpoint_sha="c", sources={"web/sim.js": "s"}, spawn_sha="p")


def test_shard_validation_rejects_any_protocol_drift(tmp_path):
    assert matrix.shard_ok(_shard(tmp_path), **META)
    for override in ({"seed": 8}, {"bot": {"config": {"difficulty": "easy"}}}, {"episodes": [{}]},
                     {"checkpoint_sha256": "x"}, {"source_hashes": {"web/sim.js": "t"}},
                     {"spawn_manifest_sha256": "q"}, {"game_range": [0, 3]}):
        assert not matrix.shard_ok(_shard(tmp_path, **override), **META), override


def test_merge_hunter_cell_demands_exactly_all_games(tmp_path):
    cell_dir = tmp_path / "hunter_hard"
    cell_dir.mkdir()
    spawns = {"spawn_sha256": "p", "spawn_cells": [[[1, 1], [2, 2]], [[3, 3], [4, 4]]]}
    shard = {"bot": {"id": "bun.hunter", "config": {"difficulty": "hard"}}, "model": {}, "protocol": {},
             "node_version": "v18", "elapsed_seconds": 1.0,
             "episodes": [{"game": g, "spawn_cells": spawns["spawn_cells"][g], "ticks": 300,
                           "model_ms": 1.0, "bot_ms": 1.0, "counts": {}} for g in range(2)]}
    (cell_dir / "games_000_002.json").write_text(json.dumps(shard))
    args = SimpleNamespace(games=2, seed=7, max_steps=300)
    row = {"name": "ck", "source": "<repo>/ck.pt"}
    matrix.merge_hunter_cell(row, "hard", "c", "m", cell_dir, [(0, 2)], spawns, {}, args)
    merged = json.loads((tmp_path / "hunter_hard.json").read_text())
    assert merged["games"] == 2 and len(merged["episodes"]) == 2
    assert merged["checkpoint"] == "<repo>/ck.pt"
    with pytest.raises(SystemExit, match="games"):
        matrix.merge_hunter_cell(row, "hard", "c", "m", cell_dir, [(0, 2)], spawns, {},
                                 SimpleNamespace(games=3, seed=7, max_steps=300))
    spawns["spawn_cells"][1] = [[0, 0], [0, 1]]
    with pytest.raises(SystemExit, match="spawn mismatch"):
        matrix.merge_hunter_cell(row, "hard", "c", "m", cell_dir, [(0, 2)], spawns, {}, args)


def test_manifest_rejects_duplicates_and_missing_fields(tmp_path):
    path = tmp_path / "m.json"
    row = {"name": "a", "path": "x", "source": "<r>/x", "group": "g"}
    path.write_text(json.dumps({"checkpoints": [row, row]}))
    with pytest.raises(SystemExit, match="duplicate"):
        matrix.load_manifest(path)
    path.write_text(json.dumps({"checkpoints": [{"name": "a", "path": "x"}]}))
    with pytest.raises(SystemExit, match="missing"):
        matrix.load_manifest(path)


def test_spawn_manifest_matches_tactical_evaluator_reset(tmp_path):
    jax = pytest.importorskip("jax")
    import numpy as np
    from jax_bomb import bun_env as env
    from jax_bomb.bun_frozen_opponents import clear_destructible_bricks
    out = tmp_path / "spawns.json"
    matrix.make_spawns(SimpleNamespace(seed=20261001, games=8, out=str(out)))
    data = json.loads(out.read_text())
    assert data["games"] == 8 and len(data["spawn_cells"]) == 8
    states = clear_destructible_bricks(env.init_batch(jax.random.PRNGKey(np.uint32(20261001)), 8))
    assert np.floor(np.asarray(states.core.pos)).astype(int).tolist() == data["spawn_cells"]
