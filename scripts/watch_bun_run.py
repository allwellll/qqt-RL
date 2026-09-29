#!/usr/bin/env python3
"""Lightweight non-LLM watchdog for a Bun safe-aggression run.

单一所有者 (flock) 周期性巡检: supervisor/worker PID 存活、各候选 status.json 阶段与
cycle、最新 checkpoint 的 cycle 号/mtime/可选有限性、GPU 利用率与显存, 写出 heartbeat 并在
状态跃迁 (完成/失败/停滞) 时追加 notify 事件。纯 stdlib, 不加载任何 LLM/训练依赖 (有限性
检查按需惰性导入)。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqt_rl.training.io import atomic_write_json

_CYCLE_ACTOR = re.compile(r"cycle_(\d+)_actor\.pt$")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def newest_checkpoint(candidate_ckpt: Path) -> dict:
    best_cycle, best_path = -1, None
    if candidate_ckpt.is_dir():
        for entry in candidate_ckpt.iterdir():
            match = _CYCLE_ACTOR.search(entry.name)
            if match and int(match.group(1)) > best_cycle:
                best_cycle, best_path = int(match.group(1)), entry
    if best_path is None:
        return {"cycle": None, "path": None, "mtime": None, "age_s": None}
    mtime = best_path.stat().st_mtime
    return {"cycle": best_cycle, "path": str(best_path),
            "mtime": mtime, "age_s": round(time.time() - mtime, 1)}


def checkpoint_finite(path: str | None) -> bool | None:
    if not path:
        return None
    try:
        from qqt_rl.training.io import checkpoint_is_finite
        return bool(checkpoint_is_finite(path))
    except Exception:
        return None


def gpu_snapshot() -> list[dict]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        out = subprocess.run(
            [binary, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": int(parts[0]), "util_pct": float(parts[1]),
                         "mem_used_mib": float(parts[2]), "mem_total_mib": float(parts[3])})
    return rows


def poll(manifest: dict, run_root: Path, checkpoint_root: Path,
         target_cycles: int, stall_seconds: float, check_finite: bool) -> dict:
    now = time.time()
    supervisor_pid = read_json(run_root / "supervisor.pid.json").get("pid")
    candidates = []
    for record in manifest.get("candidates", []):
        name = record["name"]
        status = read_json(run_root / name / "status.json")
        worker_pid = status.get("worker_pid")
        ckpt = newest_checkpoint(checkpoint_root / name)
        updated = status.get("updated_unix")
        stalled = (status.get("status") == "running" and updated is not None
                   and (now - updated) > stall_seconds)
        candidates.append({
            "name": name,
            "status": status.get("status"),
            "phase": status.get("phase"),
            "cycle": status.get("cycle"),
            "tactical_transitions": status.get("tactical_transitions_completed"),
            "error": status.get("error"),
            "worker_pid": worker_pid,
            "worker_alive": pid_alive(worker_pid),
            "status_age_s": round(now - updated, 1) if updated else None,
            "stalled": stalled,
            "checkpoint": ckpt,
            "checkpoint_finite": (checkpoint_finite(ckpt["path"])
                                  if check_finite else None),
        })
    failed = [c["name"] for c in candidates if c["status"] == "failed"]
    stalled = [c["name"] for c in candidates if c["stalled"]]
    complete = [c for c in candidates
                if c["status"] == "complete" and (c["cycle"] or 0) >= target_cycles]
    all_complete = len(complete) == len(candidates) and candidates
    if failed or (run_root / "TECHNICAL_FAILURE.json").exists():
        health = "ALERT_FAILED"
    elif stalled:
        health = "ALERT_STALLED"
    elif all_complete or (run_root / "COMPLETE.json").exists():
        health = "COMPLETE"
    elif not pid_alive(supervisor_pid):
        health = "ALERT_NO_SUPERVISOR"
    else:
        health = "OK"
    return {
        "schema": "qqt_rl_bun_watchdog_heartbeat_v1",
        "checked_unix": now,
        "run_root": str(run_root),
        "target_cycles": target_cycles,
        "supervisor_pid": supervisor_pid,
        "supervisor_alive": pid_alive(supervisor_pid),
        "health": health,
        "failed": failed,
        "stalled": stalled,
        "complete_count": len(complete),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "gpu": gpu_snapshot(),
    }


def signature(beat: dict) -> str:
    parts = [beat["health"], f"sup={beat['supervisor_alive']}"]
    for c in beat["candidates"]:
        parts.append(f"{c['name']}:{c['status']}:{c['cycle']}:{c['phase']}")
    return "|".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--stall-seconds", type=float, default=1800.0)
    parser.add_argument("--check-finite", action="store_true",
                        help="每轮对最新 checkpoint 做有限性校验 (惰性导入 torch, 略重)")
    parser.add_argument("--once", action="store_true",
                        help="只巡检一次即退出 (用于自检/CI)")
    args = parser.parse_args()

    manifest = read_json(args.manifest.resolve())
    if not manifest:
        raise SystemExit(f"cannot read manifest: {args.manifest}")
    run_root = Path(manifest["run_root"]).resolve()
    checkpoint_root = Path(manifest["checkpoint_root"]).resolve()
    target_cycles = int(manifest["target_cycles"])
    run_root.mkdir(parents=True, exist_ok=True)

    lock_handle = (run_root / "watchdog.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another watchdog already owns this run") from error
    atomic_write_json(run_root / "watchdog.pid.json", {"pid": os.getpid()})

    heartbeat_path = run_root / "watchdog_heartbeat.json"
    notify_path = run_root / "watchdog_notify.jsonl"
    last_sig = None
    while True:
        beat = poll(manifest, run_root, checkpoint_root, target_cycles,
                    args.stall_seconds, args.check_finite)
        atomic_write_json(heartbeat_path, beat)
        sig = signature(beat)
        if sig != last_sig:
            event = {"unix": beat["checked_unix"], "health": beat["health"],
                     "failed": beat["failed"], "stalled": beat["stalled"],
                     "complete_count": beat["complete_count"],
                     "candidates": [{"name": c["name"], "status": c["status"],
                                     "cycle": c["cycle"], "phase": c["phase"]}
                                    for c in beat["candidates"]]}
            with notify_path.open("a") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"NOTIFY {stamp} {beat['health']} "
                  f"complete={beat['complete_count']}/{beat['candidate_count']} "
                  f"failed={beat['failed']} stalled={beat['stalled']}", flush=True)
            last_sig = sig
        if args.once or beat["health"] in {"COMPLETE", "ALERT_FAILED"}:
            print(json.dumps({"final_health": beat["health"]}), flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
