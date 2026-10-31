"""Recompute a batch-sweep result from immutable per-update and checkpoint records."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    for _ in range(8):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return None


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.campaign_gpu_batch_sweep_finalize ATTEMPT_DIR")
    out = ROOT / sys.argv[1]
    protocol_path = out / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    gpu_rows = []
    samples_path = out / "GPU_SAMPLES.jsonl"
    if samples_path.exists():
        for line in samples_path.read_text().splitlines():
            try:
                gpu_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    card_for_name = {v["name"]: g for v, g in zip(protocol["variants"], protocol["gpus"])}
    runs = []
    for variant in protocol["variants"]:
        name = variant["name"]
        run_dir = out / f"{name}_s{protocol['seed']}"
        complete = read(run_dir / "COMPLETE.json")
        update_path = run_dir / "updates.jsonl"
        updates = []
        if update_path.exists():
            for line in update_path.read_text().splitlines():
                try:
                    updates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        commit_ok = bool(complete and (ROOT / complete["checkpoint"] / "state.pkl").is_file()
                         and sha(ROOT / complete["checkpoint"] / "state.pkl") == complete["state_sha256"])
        warm = updates[2:]
        wall = sum(float(x.get("cycle_seconds", 0)) for x in warm)
        steps = sum(int(x.get("actual_environment_steps_this_update", 0)) for x in warm)
        started = float(updates[0]["unix"]) if updates else 0
        ended = float(updates[-1]["unix"]) if updates else 0
        gpu = card_for_name[name]
        samples = [
            row["gpus"][gpu] for row in gpu_rows
            if started <= float(row.get("unix", -1)) <= ended
            and gpu < len(row.get("gpus", []))
        ]
        as_float = lambda key: [float(s[key]) for s in samples if s.get(key) not in (None, "N/A")]
        finite = len(updates) == int(protocol["updates_per_variant"]) and all(x.get("finite") for x in updates)
        nonzero = all(
            float(x.get("update_l2", {}).get(k, 0)) > 0
            for x in updates for k in ("actor", "critic")
        )
        runs.append({
            "name": name, "gpu": gpu, "checkpoint_sha256": complete.get("state_sha256") if complete else None,
            "policy_sha256": complete.get("policy_sha256") if complete else None,
            "complete_marker_and_state_hash_valid": commit_ok,
            "update_count": len(updates), "finite_all_updates": finite, "actor_critic_nonzero_all_updates": nonzero,
            "environment_steps_total": complete.get("actual_environment_steps") if complete else None,
            "optimizer_steps_total": complete.get("actual_optimizer_steps") if complete else None,
            "warm_updates_used": len(warm), "warm_cycle_seconds": wall,
            "warm_environment_steps": steps, "warm_environment_steps_per_second": steps / wall if wall else None,
            "warm_cycle_seconds_median": statistics.median([float(x["cycle_seconds"]) for x in warm]) if warm else None,
            "warm_collector_seconds_median": statistics.median([float(x["rollout_seconds"]) for x in warm]) if warm else None,
            "warm_update_plus_host_seconds_median": statistics.median([float(x["cycle_seconds"])-float(x["rollout_seconds"]) for x in warm]) if warm else None,
            "all_update_metrics": updates,
            "gpu_samples_during_update_window": len(samples),
            "gpu_util_mean_percent": statistics.mean(as_float("util_percent")) if as_float("util_percent") else None,
            "gpu_util_max_percent": max(as_float("util_percent"), default=None),
            "gpu_power_mean_W": statistics.mean(as_float("power_W")) if as_float("power_W") else None,
            "gpu_power_max_W": max(as_float("power_W"), default=None),
            "gpu_memory_max_MiB": max(as_float("memory_MiB"), default=None),
        })
    passed = all(r["complete_marker_and_state_hash_valid"] and r["update_count"] == protocol["updates_per_variant"] and r["finite_all_updates"] and r["actor_critic_nonzero_all_updates"] for r in runs)
    status = read(out / "SWEEP_STATUS.json") or {}
    result = {
        "status": "COMPLETE" if passed else "FAILED_OR_INCOMPLETE",
        "passed": passed,
        "derived_from": "per-update updates.jsonl + COMPLETE.json + atomic checkpoint hashes + 1-second GPU_SAMPLES.jsonl",
        "supervisor_status_before_finalize": status.get("status"),
        "supervisor_exit_issue": "parent supervisor encountered a progress-file replacement race; worker run artifacts were reconciled independently" if status.get("status") == "RUNNING" else None,
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": sha(protocol_path),
        "gpu_samples": str(samples_path.relative_to(ROOT)), "gpu_sample_count": len(gpu_rows),
        "training_measurement_only": True, "capability_selection": False, "runs": runs,
    }
    tmp = out / "SWEEP_RESULT.json.tmp"
    tmp.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    tmp.replace(out / "SWEEP_RESULT.json")
    tmp = out / "SWEEP_STATUS.json.tmp"
    tmp.write_text(json.dumps({"status": result["status"], "result": "SWEEP_RESULT.json", "finalized_from_worker_artifacts": True}, sort_keys=True, indent=2) + "\n")
    tmp.replace(out / "SWEEP_STATUS.json")
    print(json.dumps({"status": result["status"], "runs": [{k: r[k] for k in ("name", "update_count", "warm_environment_steps_per_second", "warm_cycle_seconds_median", "warm_collector_seconds_median", "gpu_util_mean_percent", "gpu_power_mean_W", "gpu_memory_max_MiB")} for r in runs]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
