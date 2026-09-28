#!/usr/bin/env python3
"""Completion-marker monitor and scorer for Bun safe-aggression v7."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

from qqt_rl.training.io import (
    atomic_copy,
    atomic_write_json as atomic,
    checkpoint_is_finite,
    sha256_file as digest,
)


def score(summary):
    rows = [row["metrics"] for row in summary["strata"].values()]
    mean = lambda key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    bombs, own, trade = mean("p0_avg_bombs"), mean("p0_own_bomb_defeat_rate"), mean("mutual_death_rate")
    eligible = bombs >= 0.5 and own <= 0.25 and trade <= 0.20
    value = (2 * mean("p0_surviving_causal_kill_rate")
             + mean("p0_surviving_physical_kill_rate")
             + 0.25 * mean("p0_conditional_tactical_placement_rate")
             + 0.25 * mean("p0_tactical_resolution_ratio")
             - own - trade - mean("p0_avoidable_danger_death_rate"))
    return {"eligible": eligible, "score": value if eligible else -1.0 + value,
            "mean_bombs": bombs, "mean_own_bomb": own, "mean_trade": trade}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True)
    args = parser.parse_args(); manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text()); root = Path(manifest["run_root"])
    ckpt = Path(manifest["checkpoint_root"]); target = int(manifest["target_cycles"])
    (root / "completion_monitor.pid").write_text(f"{os.getpid()}\n")
    while True:
        statuses = []
        for candidate in manifest["candidates"]:
            path = root / candidate["name"] / "status.json"
            statuses.append(json.loads(path.read_text()) if path.exists() else {})
        if all(row.get("status") == "complete" and row.get("cycle", 0) >= target
               for row in statuses):
            break
        time.sleep(20)

    results = []
    for candidate in manifest["candidates"]:
        name = candidate["name"]; summaries = []
        for cycle in manifest["milestones"]:
            path = root / name / f"cycle_{cycle:03d}/summary.json"
            if path.exists(): summaries.append((cycle, json.loads(path.read_text())))
        scored = [(cycle, score(summary)) for cycle, summary in summaries]
        early_cycle, early_score = max((row for row in scored if row[0] <= 90), key=lambda row: row[1]["score"])
        best_cycle, best_score = max(scored, key=lambda row: row[1]["score"])
        artifacts = {}
        for label, cycle in (("early_best", early_cycle), ("best", best_cycle), ("latest", target)):
            artifacts[label] = {"cycle": cycle}
            for kind, suffix in (("actor", "actor.pt"), ("critic", "critic.pkl"),
                                 ("target_critic", "target_critic.pkl")):
                source = ckpt / name / f"cycle_{cycle:03d}_{suffix}"
                destination = ckpt / name / f"{label}_{suffix}"
                if not source.is_file() or not checkpoint_is_finite(source):
                    raise RuntimeError(f"invalid final checkpoint: {source}")
                atomic_copy(source, destination)
                if digest(source) != digest(destination):
                    raise RuntimeError(f"final checkpoint hash mismatch: {destination}")
                artifacts[label][kind] = {"path": str(destination), "sha256": digest(destination)}
        results.append({"name": name, "early_best_score": early_score,
                        "best_score": best_score, "artifacts": artifacts,
                        "episodes_completed": next(row for row in statuses if row.get("candidate") == name).get("episodes_completed")})
    payload = {"schema": "bun_safe_aggression_v7_complete_v1", "status": "complete",
               "target_cycles": target, "manifest_sha256": digest(manifest_path),
               "verified_unix": time.time(), "candidates": results}
    atomic(root / "final_summary.json", payload); atomic(root / "COMPLETE.json", payload)


if __name__ == "__main__":
    main()
