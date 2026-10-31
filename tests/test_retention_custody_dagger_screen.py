import json


def test_notification_is_persisted_before_best_effort_delivery(monkeypatch, tmp_path):
    from scripts import retention_custody_dagger_screen as screen

    base = tmp_path / "r6"
    monkeypatch.setattr(screen, "BASE", base)
    delivered = []
    monkeypatch.setattr(screen.subprocess, "run", lambda args, **kwargs: delivered.append(args))

    screen.persist_notification("fixed run completed; successor is queued")

    events = list((base / "notifications").glob("event_*.json"))
    assert len(events) == 1
    record = json.loads(events[0].read_text())
    assert record["message"] == "fixed run completed; successor is queued"
    assert record["experiment_id"] == "RETENTION-20261022-R6-CUSTODY-RETURN-DAGGER"
    assert delivered and delivered[0][:3] == ["jdme-notify", "send", record["message"]]


def test_continuation_marker_records_terminal_runs_and_next_fixed_run(monkeypatch, tmp_path):
    from scripts import retention_custody_dagger_screen as screen
    from scripts.population_run_state import atomic_json

    base = tmp_path / "r6"
    base.mkdir()
    monkeypatch.setattr(screen, "BASE", base)
    atomic_json(base / "DECLARATION.json", {
        "runs": [
            {"namespace": "runs/retention_20261022/r6/A0_s1"},
            {"namespace": "runs/retention_20261022/r6/A1_s1"},
        ]
    })
    terminal = tmp_path / "runs/retention_20261022/r6/A0_s1"
    terminal.mkdir(parents=True)
    atomic_json(terminal / "FIXED_RUN_TERMINAL.json", {"generation": 32})
    monkeypatch.setattr(screen, "ROOT", tmp_path)

    screen.write_continuation("RUN_TERMINAL", next_run="runs/retention_20261022/r6/A1_s1")

    record = json.loads((base / "CONTINUATION.json").read_text())
    assert record["completed_runs"] == ["runs/retention_20261022/r6/A0_s1"]
    assert record["next_run"] == "runs/retention_20261022/r6/A1_s1"
    assert record["fixed_generations_per_run"] == 32
    assert record["no_capability_gate"] is True


def test_common_inputs_are_compared_across_r5_seeds_not_new_r6_seeds(monkeypatch, tmp_path):
    from scripts import retention_custody_dagger_screen as screen

    prior = tmp_path / "r5"
    source = prior / "A0_s7"
    source_inputs = source / "inputs"
    source_inputs.mkdir(parents=True)
    (source_inputs / "parent.pkl").write_bytes(b"same-parent-input")
    for seed in (7, 8, 9):
        if seed == 7:
            continue
        peer_inputs = prior / f"A0_s{seed}" / "inputs"
        peer_inputs.mkdir(parents=True)
        (peer_inputs / "parent.pkl").write_bytes(b"same-parent-input")
    monkeypatch.setattr(screen, "PRIOR", prior)
    monkeypatch.setattr(screen, "SOURCE", source)
    monkeypatch.setattr(screen, "R5_SEEDS", [7, 8, 9])
    monkeypatch.setattr(screen, "SEEDS", [10, 11, 12])

    hashes = screen._verify_common_r5_inputs()

    assert set(hashes) == {"parent.pkl"}
