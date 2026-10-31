"""Run the authorized single-GPU batch-size technical sweep."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def gpu_snapshot() -> list[dict[str, str]]:
    query = "index,memory.used,utilization.gpu,power.draw"
    p = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    result = []
    for line in p.stdout.splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) == 4:
            result.append(dict(zip(("index", "memory_MiB", "util_percent", "power_W"), cols)))
    return result


def read_json_snapshot(path: Path) -> object | None:
    # Workers atomically replace progress markers; exists()+read_text() is racy.
    for _ in range(4):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
    return None


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.campaign_gpu_batch_sweep PROTOCOL.json")
    protocol_path = ROOT / sys.argv[1]
    protocol = json.loads(protocol_path.read_text())
    out = ROOT / protocol["namespace"]
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "SWEEP_STATUS.json"
    if (out / "SWEEP_RESULT.json").exists():
        raise SystemExit("sweep already has a terminal result")
    variants = protocol["variants"]
    cards = protocol["gpus"]
    if len(variants) != len(cards) or len(set(cards)) != len(cards):
        raise ValueError("this frozen sweep requires one unique variant per GPU")

    children: list[dict] = []
    for variant, gpu in zip(variants, cards):
        name = variant["name"]
        run_protocol = {
            k: protocol[k] for k in ("namespace", "source_sha256", "sources")
        }
        run_protocol["common"] = {
            "devices": 1,
            "envs_per_device": variant["envs_per_device"],
            "rollout_ticks": protocol["rollout_ticks"],
            "global_minibatch": variant["global_minibatch"],
            "epochs": 1,
            "lr": protocol["lr"],
            "dagger": 0,
            "combat_kl": 0,
            "reward": "immutable legacy task and shaping",
            "opponent": "fixed JAX rule3",
            "bank_probability": 0.25,
        }
        run_protocol["arms"] = {name: {}}
        run_protocol["paired_seeds"] = [protocol["seed"]]
        batch = variant["envs_per_device"] * protocol["rollout_ticks"]
        run_protocol["environment_steps_per_run"] = batch * protocol["updates_per_variant"]
        run_protocol["checkpoints_environment_steps"] = [batch, batch * 3, batch * protocol["updates_per_variant"]]
        run_protocol["single_variable"] = "envs_per_device; global_minibatch=4096 variants are technical minibatch controls"
        run_protocol["no_holdout_tuning"] = True
        run_protocol["stopping"] = "exactly six updates; technical failures only"
        run_protocol["evaluation"] = "none; throughput-only technical measurement"
        run_protocol["history"] = "source history retained; pure PPO objective"
        variant_protocol = out / "protocols" / f"{name}.json"
        atomic_json(variant_protocol, run_protocol)
        run_dir = out / f"{name}_s{protocol['seed']}"
        log_path = out / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        affinity = protocol["cpu_affinity"][str(gpu)]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAXBOMB_DEVICE_OVERRIDE": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "JAX_PLATFORM_NAME": "cuda",
            "JAXBOMB_RULE": "bun",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        })
        cmd = [
            "taskset", "-c", affinity, sys.executable, "-m", "scripts.campaign_scale_train",
            str(variant_protocol.relative_to(ROOT)), name, str(protocol["seed"]),
        ]
        log = log_path.open("w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append({
            "name": name, "gpu": gpu, "pid": proc.pid, "process": proc,
            "log": log, "log_path": str(log_path.relative_to(ROOT)),
            "protocol": str(variant_protocol.relative_to(ROOT)),
            "run_dir": str(run_dir.relative_to(ROOT)), "started": time.time(),
            "last_exit": None,
        })
        print(f"started {name} on GPU {gpu} pid {proc.pid}", flush=True)

    atomic_json(status_path, {
        "status": "RUNNING", "protocol": str(protocol_path.relative_to(ROOT)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": [{k: c[k] for k in ("name", "gpu", "pid", "log_path", "protocol", "run_dir", "started")} for c in children],
    })
    samples = out / "GPU_SAMPLES.jsonl"
    while any(c["process"].poll() is None for c in children):
        snapshot = {"unix": time.time(), "gpus": gpu_snapshot()}
        with samples.open("a") as f:
            f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
            f.flush()
        rows = []
        for c in children:
            rc = c["process"].poll()
            progress_file = ROOT / c["run_dir"] / "PROGRESS.json"
            progress = read_json_snapshot(progress_file)
            c["last_exit"] = rc
            rows.append({"name": c["name"], "gpu": c["gpu"], "pid": c["pid"], "exit_code": rc, "progress": progress})
        atomic_json(status_path, {"status": "RUNNING", "workers": rows, "last_gpu_sample": snapshot})
        time.sleep(1)
    for c in children:
        c["log"].close()

    runs = []
    for c in children:
        run_dir = ROOT / c["run_dir"]
        complete_path = run_dir / "COMPLETE.json"
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
        updates = []
        update_path = run_dir / "updates.jsonl"
        if update_path.exists():
            updates = [json.loads(line) for line in update_path.read_text().splitlines() if line]
        elapsed = time.time() - c["started"]
        steps = complete.get("actual_environment_steps", 0) if complete else 0
        runs.append({
            "name": c["name"], "gpu": c["gpu"], "pid": c["pid"],
            "exit_code": c["process"].returncode, "complete": complete is not None,
            "checkpoint_sha256": complete.get("state_sha256") if complete else None,
            "policy_sha256": complete.get("policy_sha256") if complete else None,
            "updates": len(updates), "finite": all(x.get("finite", False) for x in updates),
            "environment_steps": steps, "elapsed_seconds": elapsed,
            "environment_steps_per_second": steps / elapsed if elapsed else None,
            "update_metrics": updates,
            "log": c["log_path"],
        })
    passed = all(r["exit_code"] == 0 and r["complete"] and r["updates"] == protocol["updates_per_variant"] and r["finite"] for r in runs)
    result = {
        "status": "COMPLETE" if passed else "FAILED",
        "passed": passed, "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha(protocol_path), "gpus": cards,
        "gpu_samples": str(samples.relative_to(ROOT)), "gpu_sample_count": samples.read_text().count("\n") if samples.exists() else 0,
        "runs": runs, "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(out / "SWEEP_RESULT.json", result)
    atomic_json(status_path, {"status": result["status"], "result": "SWEEP_RESULT.json", "finished_utc": result["finished_utc"]})
    print(json.dumps({"status": result["status"], "runs": [{k: r[k] for k in ("name", "gpu", "exit_code", "complete", "updates", "environment_steps_per_second")} for r in runs]}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
