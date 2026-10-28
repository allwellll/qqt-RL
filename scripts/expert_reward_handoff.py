"""Validation helpers for the D reward-ablation predecessor handoff."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from scripts.population_run_state import sha


def _read(path: Path):
    return json.loads(path.read_text())


def filter_live_compute_apps(apps_output: str, ps_output: str) -> str:
    """Drop stale nvidia-smi entries for exited or zombie PIDs."""
    states = {}
    for line in ps_output.splitlines():
        cells = line.split(None, 1)
        if len(cells) == 2:
            try:
                states[int(cells[0])] = cells[1].strip()
            except ValueError:
                continue
    live = []
    for line in apps_output.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        try:
            pid = int(cells[0])
        except (ValueError, IndexError):
            continue
        state = states.get(pid)
        if state and not state.startswith(("Z", "X")):
            live.append(line.strip())
    return "\n".join(live)


def validate_user_ended_early(repo_root: Path, r2_relative: str) -> dict:
    """Validate an explicit R2 early-end receipt without calling it complete."""
    repo_root = Path(repo_root)
    r2 = repo_root / r2_relative
    receipt_path = r2 / "USER_ENDED_EARLY.json"
    assert receipt_path.is_file(), "R2 early-end receipt is missing"
    assert (r2 / "USER_STOP.json").is_file(), "R2 stop request is missing"
    assert (r2 / "OWNER_EXIT.json").is_file(), "R2 owner exit is not recorded"
    assert not (r2 / "TRAINING_COMPLETE.json").exists(), (
        "early-end receipt conflicts with a full-budget completion marker")
    receipt = _read(receipt_path)
    protocol = _read(r2 / "PROTOCOL.json")
    assert receipt.get("status") == "USER_ENDED_EARLY"
    assert receipt.get("fixed_budget_complete") is False
    assert receipt.get("full_state_verified") is True
    assert receipt.get("owner_exit_verified") is True
    assert receipt.get("total_planned_environment_steps") == protocol["budget"]["total_new_environment_steps"]
    seeds = [int(seed) for seed in protocol["paired_seeds"]]
    rows = {int(row["seed"]): row for row in receipt["per_seed"]}
    assert set(rows) == set(seeds)

    checkpoint = Path(receipt["last_atomic_checkpoint"])
    assert not checkpoint.is_absolute() and ".." not in checkpoint.parts
    cp = repo_root / checkpoint
    committed = _read(cp / "COMMITTED.json")
    state_path, policy_path = cp / "state.pkl", cp / "policy.pt"
    assert sha(state_path) == receipt["state_sha256"] == committed["state_sha256"]
    assert sha(policy_path) == receipt["policy_sha256"] == committed["policy_sha256"]
    with state_path.open("rb") as stream:
        state = pickle.load(stream)
    required = {"params", "adam", "states", "aux", "keys", "history",
                "seed", "update", "new_update"}
    assert required <= set(state)
    assert state["seed"] == committed["seed"]
    assert state["new_update"] == committed["new_update"]

    active = rows[int(committed["seed"])]
    assert active["latest_atomic_checkpoint"] == receipt["last_atomic_checkpoint"]
    assert active["state_sha256"] == receipt["state_sha256"]
    assert active["policy_sha256"] == receipt["policy_sha256"]
    assert active["updates_atomically_committed"] == committed["new_update"]
    assert sum(row["unexecuted_planned_environment_steps"] for row in rows.values()) == \
        receipt["total_unexecuted_planned_environment_steps"]
    return dict(kind="USER_ENDED_EARLY", fixed_budget_complete=False,
                receipt_sha256=sha(receipt_path), state_sha256=receipt["state_sha256"],
                policy_sha256=receipt["policy_sha256"],
                checkpoint=str(checkpoint), last_atomic_new_update=committed["new_update"])


def valid_d_resume_authorization(namespace: Path, protocol_sha: str,
                                 seal_sha: str, amendment_sha: str) -> bool:
    """Allow the explicitly paused idle owner to resume only under a new seal."""
    namespace = Path(namespace)
    stop = namespace / "USER_STOP.json"
    resume = namespace / "D_RESUME_AUTHORIZATION.json"
    if not stop.exists() or not resume.exists():
        return False
    token = _read(resume)
    return bool(
        token.get("authorized_by") == "explicit_user_instruction"
        and token.get("user_stop_sha256") == sha(stop)
        and token.get("protocol_sha256") == protocol_sha
        and token.get("source_seal_sha256") == seal_sha
        and token.get("handoff_amendment_sha256") == amendment_sha
        and token.get("allow_new_gpu_run") is True
        and token.get("old_queues_remain_paused") is True
    )


def validated_predecessor_terminal(repo_root: Path, r2_relative: str,
                                  acceptance_path: Path | None = None) -> dict | None:
    """Validate either a completed R2 budget or its accepted early end."""
    repo_root = Path(repo_root)
    r2 = repo_root / r2_relative
    complete_path = r2 / "TRAINING_COMPLETE.json"
    if complete_path.exists():
        protocol = _read(r2 / "PROTOCOL.json")
        complete = _read(complete_path)
        assert complete.get("new_environment_steps") == \
            protocol["budget"]["total_new_environment_steps"]
        return dict(kind="TRAINING_COMPLETE", fixed_budget_complete=True,
                    terminal_sha256=sha(complete_path))
    early = validate_user_ended_early(repo_root, r2_relative)
    if acceptance_path is None:
        return dict(kind="USER_ENDED_EARLY", fixed_budget_complete=False,
                    terminal_sha256=sha(r2 / "USER_ENDED_EARLY.json"),
                    checkpoint=early["checkpoint"])
    accepted = _read(Path(acceptance_path))
    terminal_path = r2 / "USER_ENDED_EARLY.json"
    assert accepted.get("passed") is True
    assert accepted.get("predecessor_terminal") == "USER_ENDED_EARLY"
    assert accepted.get("predecessor_fixed_budget_complete") is False
    assert accepted.get("predecessor_terminal_sha256") == sha(terminal_path)
    assert accepted.get("accepted_checkpoint") == early["checkpoint"]
    assert accepted.get("gpu_released") is True
    return dict(kind="USER_ENDED_EARLY", fixed_budget_complete=False,
                terminal_sha256=sha(terminal_path), checkpoint=early["checkpoint"])
