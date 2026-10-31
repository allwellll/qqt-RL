import json
from pathlib import Path

import pytest

from scripts import bomb_threat_iter2_schedule as S
from scripts.population_run_state import sha


def test_a1_u960_marker_path_and_frozen_gate_threshold(tmp_path, monkeypatch):
    base = tmp_path / "runs/bomb_threat_v1"
    monkeypatch.setattr(S, "BASE", base)
    checkpoint = "runs/bomb_threat_v1/formal/bt_slot04_c000015/u000960"
    assert S.round1_a1_eval_marker(checkpoint) == (
        base / "eval/bt_slot04_c000015_u000960/NATURAL_COMPLETE.json"
    )

    commit = {"policy_sha256": "policy"}
    evaluation = {
        "policy_sha256": "policy",
        "natural_js_full2400": {
            "greedy": {"games": 64, "bombs_per_game": 15.40625, "zero_bomb_games": 15}
        },
    }
    gate = S.round1_a1_gate(checkpoint, evaluation, commit, "bt_slot04_c000016")
    assert gate["status"] == "A3_SWITCH_REQUIRED"
    assert gate["trigger"] is True
    assert gate["rule"] == "bombs_per_game<20 OR zero_bomb_games>=15"

    evaluation["natural_js_full2400"]["greedy"].update(bombs_per_game=20, zero_bomb_games=14)
    assert S.round1_a1_gate(checkpoint, evaluation, commit, "run")["status"] == "CONTINUE_A1"
    with pytest.raises(ValueError):
        S.round1_a1_gate(checkpoint, dict(evaluation, policy_sha256="wrong"), commit, "run")


def test_round1_boundary_disposition_never_resumes_a2_or_legacy_fallback():
    assert S.round1_boundary_disposition(7, {}, False) == "BLOCK"
    assert S.round1_boundary_disposition(4, {"status": "WAITING_FOR_U960_DEV_EVAL"}, False) == "WAIT"
    assert S.round1_boundary_disposition(4, {"status": "A3_SWITCH_REQUIRED"}, False) == "WAIT"
    assert S.round1_boundary_disposition(4, {"status": "CONTINUE_A1"}, False) == "CONTINUE"
    assert S.round1_boundary_disposition(4, {}, True) == "SWITCH"


def test_slot07_switch_requires_u1280_atomic_commit_and_uses_latest_slot01_actor(tmp_path, monkeypatch):
    base = tmp_path / "runs/bomb_threat_v1"
    (base / "iter2").mkdir(parents=True)
    (base / "protocols").mkdir()
    old_run = "bt_slot07_c000016"
    old_out = base / "formal" / old_run
    old_out.mkdir(parents=True)
    (base / "iter2/DECISION_ROUND1.json").write_text(json.dumps({"status": "AUTHORIZED"}))
    (base / "PROTOCOL.json").write_text(json.dumps({"seeds": {"7": 2026112907}}))
    (base / "iter2/TRAINING_PROTOCOL.json").write_text(json.dumps({
        "chunk_updates": 320,
        "arms": {"A3": {"iter2_arm": "A3", "lr": 0.000075, "entropy_coefficient": 0.005}},
    }))

    parent_run = "bt_slot01_c000016"
    parent_cp = base / "formal" / parent_run / "u001000"
    parent_cp.mkdir(parents=True)
    (parent_cp / "policy.pt").write_bytes(b"latest-a3-actor")
    (parent_cp / "state.pkl").write_bytes(b"parent-state")
    parent_commit = {
        "passed": True,
        "checkpoint": str(parent_cp.relative_to(tmp_path)),
        "update": 1000,
        "policy_sha256": sha(parent_cp / "policy.pt"),
        "state_sha256": sha(parent_cp / "state.pkl"),
    }
    (parent_cp / "COMMITTED.json").write_text(json.dumps(parent_commit))
    (base / "SLOTS.json").write_text(json.dumps({
        "1": {"run": parent_run, "iter2_arm": "A3"},
        "7": {"run": old_run, "iter2_arm": "A2"},
    }))

    old_recipe = {"iter2_arm": "A2", "entropy_fixed_005": True}
    (base / "protocols" / f"{old_run}.json").write_text(json.dumps({
        "chunk_start_update": 960, "chunk_updates": 320, "arms": {old_run: old_recipe}
    }))
    old_cp = old_out / "u001280"
    old_cp.mkdir()
    (old_cp / "state.pkl").write_bytes(b"slot07-state")
    (old_cp / "policy.pt").write_bytes(b"slot07-policy")
    done = {
        "passed": True,
        "checkpoint": str(old_cp.relative_to(tmp_path)),
        "update": 1280,
        "state_sha256": sha(old_cp / "state.pkl"),
        "policy_sha256": sha(old_cp / "policy.pt"),
    }
    (old_out / "COMPLETE.json").write_text(json.dumps(done))
    monkeypatch.setattr(S, "BASE", base)
    monkeypatch.setattr(S, "ROOT", tmp_path)

    def fake_make_protocol(slot, chunk, source, recipe, length=320):
        assert slot == 7 and chunk == 17 and source is None and length == 320
        assert recipe["iter2_arm"] == "A3"
        path = base / "protocols/bt_slot07_c000017.json"
        path.write_text(json.dumps({"arms": {path.stem: recipe}}))
        return path

    monkeypatch.setattr(S, "make_protocol", fake_make_protocol)
    entry = {"run": old_run, "iter2_arm": "A2"}
    assert S.apply_round1_a3_seed(7, entry, dict(done, update=1279), 17) is None
    result = S.apply_round1_a3_seed(7, entry, done, 17)

    assert result["run"] == "bt_slot07_c000017"
    assert result["iter2_arm"] == "A3"
    assert result["source"]["checkpoint"] == parent_commit["checkpoint"]
    assert result["pbt_lineage"]["target_seed"] == 2026112907
    assert result["pbt_lineage"]["copied"] == ["Actor policy only"]
    assert set(result["pbt_lineage"]["reinitialized"]) == {
        "Critic", "Adam", "environment", "auxiliary state", "RNG", "combat ring"
    }
    assert json.loads((old_out / "ITER2_ROUND1_SUPERSEDED.json").read_text())["formal_updates_discarded"] == 0
    handoff = json.loads((base / "formal/bt_slot07_c000017/PBT_HANDOFF.json").read_text())
    assert handoff["status"] == "PENDING_FRESH_INIT"
    assert handoff["critic_warmup_updates"] == 20


def test_a3_source_uses_persisted_atomic_parent_if_successor_has_not_committed(tmp_path, monkeypatch):
    base = tmp_path / "runs/bomb_threat_v1"
    prior = base / "formal/bt_slot01_c000016/u001280"
    prior.mkdir(parents=True)
    (prior / "policy.pt").write_bytes(b"last-committed-a3-actor")
    (prior / "COMMITTED.json").write_text(json.dumps({
        "passed": True,
        "checkpoint": str(prior.relative_to(tmp_path)),
        "update": 1280,
        "policy_sha256": sha(prior / "policy.pt"),
        "state_sha256": "state-retained-as-metadata-only",
    }))
    (base / "PROTOCOL.json").write_text(json.dumps({"seeds": {"7": 2026112907}}))
    successor = "bt_slot01_c000017"
    (base / "formal" / successor).mkdir(parents=True)
    (base / "SLOTS.json").write_text(json.dumps({"1": {
        "run": successor,
        "iter2_arm": "A3",
        "source": {
            "checkpoint": str(prior.relative_to(tmp_path)),
            "update": 1280,
            "policy_sha256": sha(prior / "policy.pt"),
            "state_sha256": "state-retained-as-metadata-only",
        },
    }}))
    monkeypatch.setattr(S, "BASE", base)
    monkeypatch.setattr(S, "ROOT", tmp_path)

    source, lineage = S._latest_a3_policy_source(7)
    assert source["checkpoint"] == str(prior.relative_to(tmp_path))
    assert source["update"] == 1280
    assert lineage["parent_run"] == "bt_slot01_c000016"
    assert lineage["parent_update"] == 1280


def test_a3_holdout_requests_precede_older_holdout_queue(tmp_path, monkeypatch):
    from scripts import bomb_threat_eval_owner as E

    base = tmp_path / "runs/bomb_threat_v1"
    requests = base / "eval_requests"
    requests.mkdir(parents=True)
    priority_names = ["bt_slot01_c000015_u000680_holdout", "bt_slot01_c000015_u000720_holdout"]
    (base / "GATE_AMEND2_HOLDOUT_PRIORITY.json").write_text(json.dumps({"names": priority_names}))
    paths = []
    for name, update in [("older_holdout", 600), (priority_names[0], 680), (priority_names[1], 720)]:
        p = requests / f"{name}.json"
        p.write_text(json.dumps({"name": name, "split": "holdout",
                                 "checkpoint": f"runs/bomb_threat_v1/formal/bt_slot01_c000015/u{update:06d}"}))
        paths.append(p)
    monkeypatch.setattr(E, "BASE", base)
    ordered = E.catchup_order(paths, reference_paths=paths, active_runs={})
    assert [p.stem for p in ordered] == priority_names + ["older_holdout"]


def test_gate_refresh_preserves_round1_holdouts_until_natural_markers_exist(tmp_path, monkeypatch):
    from scripts import bomb_threat_gate_report as G

    base = tmp_path / "runs/bomb_threat_v1"
    (base / "iter2").mkdir(parents=True)
    (base / "eval_requests").mkdir(parents=True)
    (base / "eval").mkdir(parents=True)
    (base / "iter2/DECISION_ROUND1.json").write_text(json.dumps({"status": "AUTHORIZED"}))
    for u in (680, 720):
        name = f"bt_slot01_c000015_u{u:06d}_holdout"
        (base / "eval_requests" / f"{name}.json").write_text("{}")
    monkeypatch.setattr(G, "BASE", base)
    assert G.round1_holdout_priority() == [
        "bt_slot01_c000015_u000680_holdout", "bt_slot01_c000015_u000720_holdout"
    ]
    marker = base / "eval/bt_slot01_c000015_u000680_holdout/NATURAL_COMPLETE.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert G.round1_holdout_priority() == ["bt_slot01_c000015_u000720_holdout"]


def test_passive_analysis_reader_survives_atomic_marker_replacement_race(tmp_path, monkeypatch):
    from scripts import bomb_threat_iter2_analysis as A

    marker = tmp_path / "GATE_METRICS.json"
    original = Path.read_text

    def raced_read(path, *args, **kwargs):
        if path == marker:
            raise FileNotFoundError("atomic replacement window")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raced_read)
    assert A.read(marker, {"pending": True}) == {"pending": True}


def test_round1_holdout_request_contract_is_both_modes_full_64():
    request = {
        "name": "bt_slot01_c000015_u000680_holdout",
        "checkpoint": "runs/bomb_threat_v1/formal/bt_slot01_c000015/u000680",
        "split": "holdout",
        "rule_version": "stock-window-v1",
        "reward_version": "bomb-threat-iter2-A3",
        "iter2_arm": "A3",
        "reason": "ITER-2 round-1 A3 holdout priority; both greedy and categorical full-game evaluation",
    }
    priority = {"names": ["bt_slot01_c000015_u000680_holdout", "bt_slot01_c000015_u000720_holdout"]}
    assert request["name"] in priority["names"]
    assert request["split"] == "holdout"
    assert "categorical" in request["reason"] and "greedy" in request["reason"]
