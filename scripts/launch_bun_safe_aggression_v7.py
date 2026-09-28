#!/usr/bin/env python3
"""Durable, restart-safe supervisor for Bun safe-aggression training."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from qqt_rl.training.io import atomic_write_json, sha256_file


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def process_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (OSError, ValueError, IndexError):
        return None


def process_matches(record: dict, expected_config: Path) -> bool:
    pid = int(record.get("pid", -1))
    if not process_alive(pid) or process_start_ticks(pid) != record.get("start_ticks"):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    decoded = [item.decode(errors="replace") for item in command if item]
    return (str(expected_config) in decoded
            and any(item.endswith("run_bun_tactical_v2_candidate.py") for item in decoded))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def require_under(path: str | Path, root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must stay under repository root")
    return resolved


def verify_frozen_inputs(manifest: dict, run_root: Path,
                         frozen_root: Path) -> None:
    for row in manifest.get("frozen_files", []):
        path = require_under(frozen_root / row["relative_path"], frozen_root, "frozen file")
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen runtime hash mismatch: {path}")
    input_root = run_root / "frozen_inputs"
    for row in manifest.get("bootstrap", []):
        path = require_under(input_root / row["name"], input_root, "bootstrap file")
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"bootstrap hash mismatch: {path}")
    for candidate in manifest["candidates"]:
        path = require_under(candidate["config"], run_root, "candidate config")
        if sha256_file(path) != candidate.get("config_sha256"):
            raise RuntimeError(f"candidate config hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    repo_root = Path(manifest["repo_root"]).resolve()
    root = require_under(manifest["run_root"], repo_root, "run_root")
    require_under(manifest["checkpoint_root"], repo_root, "checkpoint_root")
    require_under(manifest["data_root"], repo_root, "data_root")
    frozen = require_under(manifest["frozen_root"], repo_root, "frozen_root")
    require_under(manifest_path, repo_root, "manifest")
    verify_frozen_inputs(manifest, root, frozen)
    python = manifest["python"]
    root.mkdir(parents=True, exist_ok=True)

    # flock 随进程退出自动释放；同一 run 同时只能有一个 supervisor。
    lock_handle = (root / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another supervisor already owns this run") from error

    atomic_write_json(root / "supervisor.pid.json", {"pid": os.getpid()})
    environment = os.environ.copy()
    environment.update({
        "QQT_REPO_ROOT": str(repo_root),
        "BUN_V2_RUN_ROOT": manifest["run_root"],
        "BUN_V2_CKPT_ROOT": manifest["checkpoint_root"],
        "BUN_V2_DATA_ROOT": manifest["data_root"],
        "BUN_V2_FROZEN_ROOT": manifest["frozen_root"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(frozen),
    })
    state_path = root / "supervisor_state.json"
    prior_state = read_json(state_path)
    attempts = dict(prior_state.get("attempts", {}))
    workers: dict[str, dict] = {}

    def status_for(candidate: dict) -> dict:
        return read_json(root / candidate["name"] / "status.json")

    def complete(candidate: dict) -> bool:
        status = status_for(candidate)
        return (status.get("status") == "complete"
                and status.get("cycle", 0) >= manifest["target_cycles"])

    def publish(phase: str = "running") -> None:
        rows = []
        for candidate in manifest["candidates"]:
            name = candidate["name"]
            worker = workers.get(name, {})
            rows.append({
                "name": name, "gpu": candidate["gpu"], "seed": candidate["seed"],
                "pid": worker.get("pid"), "attempt": attempts.get(name, 0),
                "adopted": bool(worker.get("adopted")), "status": status_for(candidate),
            })
        atomic_write_json(root / "worker_registry.json", {
            "schema": "qqt_rl_workers_v2", "supervisor_pid": os.getpid(),
            "workers": rows, "updated_unix": time.time(),
        })
        atomic_write_json(state_path, {
            "schema": "qqt_rl_supervisor_state_v2", "status": phase,
            "attempts": attempts, "target_cycles": manifest["target_cycles"],
            "updated_unix": time.time(),
        })

    def spawn(candidate: dict) -> None:
        name = candidate["name"]
        config_path = require_under(candidate["config"], root, "candidate config")
        attempts[name] = int(attempts.get(name, 0)) + 1
        log = root / name / f"worker_attempt_{attempts[name]:02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        output = log.open("a", encoding="utf-8")
        child = environment.copy()
        child["CUDA_VISIBLE_DEVICES"] = str(candidate["gpu"])
        child["JAX_COMPILATION_CACHE_DIR"] = str(root / name / "cache")
        process = subprocess.Popen([
            python, "-P", "-B", "-u",
            str(frozen / "scripts/run_bun_tactical_v2_candidate.py"),
            "--config", str(config_path),
        ], cwd=repo_root, env=child, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
        workers[name] = {"pid": process.pid, "popen": process,
                         "output": output, "adopted": False,
                         "config": config_path}
        publish()

    def adopt_or_spawn(candidate: dict) -> None:
        name = candidate["name"]
        if complete(candidate):
            return
        pid_path = root / name / "worker.pid.json"
        if pid_path.exists():
            try:
                record = read_json(pid_path)
                config_path = require_under(candidate["config"], root, "candidate config")
            except (ValueError, json.JSONDecodeError):
                record = {}
                config_path = Path(candidate["config"]).resolve()
            if process_matches(record, config_path):
                pid = int(record["pid"])
                workers[name] = {"pid": pid, "popen": None,
                                 "output": None, "adopted": True,
                                 "record": record, "config": config_path}
                attempts[name] = max(int(attempts.get(name, 0)), 1)
                return
        spawn(candidate)

    def stop_managed_workers() -> None:
        for worker in workers.values():
            process = worker.get("popen")
            pid = int(worker.get("pid", -1))
            config_path = worker.get("config")
            owned = process is not None and process.poll() is None
            if process is None and config_path is not None:
                owned = process_matches(worker.get("record", {}), config_path)
            if owned:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for worker in workers.values():
            process = worker.get("popen")
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            output = worker.get("output")
            if output is not None and not output.closed:
                output.close()

    try:
        for candidate in manifest["candidates"]:
            adopt_or_spawn(candidate)
        publish()
        while not all(complete(candidate) for candidate in manifest["candidates"]):
            for candidate in manifest["candidates"]:
                name = candidate["name"]
                if complete(candidate):
                    continue
                worker = workers[name]
                process = worker.get("popen")
                alive = process.poll() is None if process is not None else process_alive(worker["pid"])
                if alive:
                    continue
                output = worker.get("output")
                if output is not None and not output.closed:
                    output.close()
                if attempts.get(name, 0) >= manifest["max_attempts"]:
                    atomic_write_json(root / "TECHNICAL_FAILURE.json", {
                        "candidate": name, "attempts": attempts[name],
                        "updated_unix": time.time(),
                    })
                    raise RuntimeError(f"technical retry budget exhausted: {name}")
                time.sleep(20)
                spawn(candidate)
            publish()
            time.sleep(20)

        publish("workers_complete")
        finalizer_log = (root / "finalizer.log").open("a", encoding="utf-8")
        try:
            finalizer = subprocess.run([
                python, "-P", "-B", "-u",
                str(frozen / "scripts/finalize_bun_safe_aggression_v7.py"),
                "--manifest", str(manifest_path),
            ], cwd=repo_root, env=environment,
                stdout=finalizer_log, stderr=subprocess.STDOUT, check=False)
        finally:
            finalizer_log.close()
        if finalizer.returncode != 0 or not (root / "COMPLETE.json").is_file():
            raise RuntimeError(f"finalizer failed rc={finalizer.returncode}")
        publish("complete")
    except BaseException:
        stop_managed_workers()
        publish("failed")
        raise


if __name__ == "__main__":
    main()
