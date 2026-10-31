"""Run authorized GPU capacity scans and the paired env1024 PPO pilot."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
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


def read_json_snapshot(path: Path):
    for _ in range(8):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
    return None


def gpu_snapshot():
    fields = ("index", "memory_MiB", "util_percent", "power_W")
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return [dict(zip(fields, (x.strip() for x in line.split(",")))) for line in p.stdout.splitlines() if len(line.split(",")) == 4]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.campaign_gpu_followup PROTOCOL.json")
    path = ROOT / sys.argv[1]
    contract = json.loads(path.read_text())
    out = ROOT / contract["namespace"]
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "STATUS.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") not in ("PREPARED", "BLOCKED"):
        raise SystemExit("followup namespace has already been claimed")
    for name, expected in contract["source_sha256"].items():
        actual = sha(ROOT / name)
        if actual != expected:
            raise RuntimeError(f"frozen source hash mismatch: {name}")

    tasks = []
    for task in contract["tasks"]:
        run_protocol = {
            "namespace": task["run_namespace"],
            "common": {
                "devices": 1,
                "envs_per_device": task["envs_per_device"],
                "rollout_ticks": contract["rollout_ticks"],
                "global_minibatch": task["global_minibatch"],
                "epochs": 1,
                "lr": contract["lr"],
                "dagger": 0,
                "combat_kl": 0,
                "reward": "immutable legacy task and shaping",
                "opponent": "fixed JAX rule3",
                "bank_probability": 0.25,
            },
            "arms": {task["arm"]: {}},
            "paired_seeds": [task["seed"]],
            "sources": {str(task["seed"]): contract["sources"][str(task["seed"])]},
            "environment_steps_per_run": task["envs_per_device"] * contract["rollout_ticks"] * task["updates"],
            "checkpoints_environment_steps": task.get("checkpoints_environment_steps", []),
            "single_variable": task.get("single_variable", "environment count; all other PPO fields frozen"),
            "source_sha256": contract["source_sha256"],
            "no_holdout_tuning": True,
            "stopping": f"fixed {task['updates']} optimizer updates; technical errors only",
            "evaluation": task.get("evaluation", "none; technical throughput measurement"),
            "history": "complete source optimizer and checkpoint provenance retained; PPO objective uses no DAgger",
        }
        protocol_path = out / "protocols" / f"{task['task_id']}.json"
        atomic_json(protocol_path, run_protocol)
        run_dir = ROOT / task["run_namespace"] / f"{task['arm']}_s{task['seed']}"
        if (run_dir / "COMPLETE.json").exists():
            raise RuntimeError(f"run already complete; refusing rerun: {task['task_id']}")
        log_path = out / "logs" / f"{task['task_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cpu = contract["cpu_affinity"][str(task["gpu"])]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(task["gpu"]),
            "JAXBOMB_DEVICE_OVERRIDE": str(task["gpu"]),
            "JAX_PLATFORMS": "cuda", "JAX_PLATFORM_NAME": "cuda", "JAXBOMB_RULE": "bun",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        })
        cmd = ["taskset", "-c", cpu, sys.executable, "-m", "scripts.campaign_scale_train",
               str(protocol_path.relative_to(ROOT)), task["arm"], str(task["seed"])]
        log = log_path.open("w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        tasks.append({**task, "pid": proc.pid, "process": proc, "log": log,
                      "log_path": str(log_path.relative_to(ROOT)), "protocol_path": str(protocol_path.relative_to(ROOT)),
                      "run_dir": str(run_dir.relative_to(ROOT)), "started": time.time()})
        print(f"started {task['task_id']} gpu={task['gpu']} pid={proc.pid}", flush=True)
    atomic_json(status_path, {
        "status": "RUNNING", "protocol": str(path.relative_to(ROOT)), "protocol_sha256": sha(path),
        "workers": [{k: t[k] for k in ("task_id", "gpu", "seed", "arm", "pid", "run_dir", "protocol_path", "log_path")} for t in tasks],
    })

    sample_path = out / "GPU_SAMPLES.jsonl"
    while any(t["process"].poll() is None for t in tasks):
        snap = {"unix": time.time(), "gpus": gpu_snapshot()}
        with sample_path.open("a") as f:
            f.write(json.dumps(snap, separators=(",", ":")) + "\n"); f.flush(); os.fsync(f.fileno())
        rows = []
        for t in tasks:
            rc = t["process"].poll()
            progress = read_json_snapshot(ROOT / t["run_dir"] / "PROGRESS.json")
            rows.append({"task_id": t["task_id"], "gpu": t["gpu"], "seed": t["seed"], "pid": t["pid"], "exit_code": rc, "progress": progress})
        atomic_json(status_path, {"status": "RUNNING", "protocol": str(path.relative_to(ROOT)), "workers": rows, "last_gpu_sample": snap})
        time.sleep(1)
    for t in tasks:
        t["log"].close()
    results = []
    for t in tasks:
        run_dir = ROOT / t["run_dir"]
        complete = read_json_snapshot(run_dir / "COMPLETE.json")
        updates = []
        update_path = run_dir / "updates.jsonl"
        if update_path.exists():
            for line in update_path.read_text().splitlines():
                try: updates.append(json.loads(line))
                except json.JSONDecodeError: pass
        commit_ok = bool(complete and (ROOT / complete["checkpoint"] / "state.pkl").is_file()
                         and sha(ROOT / complete["checkpoint"] / "state.pkl") == complete["state_sha256"])
        results.append({
            "task_id": t["task_id"], "gpu": t["gpu"], "seed": t["seed"], "arm": t["arm"],
            "exit_code": t["process"].returncode, "complete_marker_and_state_hash_valid": commit_ok,
            "complete": complete, "update_count": len(updates), "updates": updates,
            "environment_steps_per_second_wall": (complete.get("actual_environment_steps", 0) / (time.time()-t["started"])) if complete else None,
            "log": t["log_path"], "run_dir": t["run_dir"],
        })
    passed = all(r["exit_code"] == 0 and r["complete_marker_and_state_hash_valid"] and r["update_count"] == next(t["updates"] for t in tasks if t["task_id"] == r["task_id"]) for r in results)
    result = {"status": "COMPLETE" if passed else "COMPLETE_WITH_TECHNICAL_FAILURES", "passed": passed,
              "protocol": str(path.relative_to(ROOT)), "protocol_sha256": sha(path), "results": results,
              "gpu_samples": str(sample_path.relative_to(ROOT)),
              "gpu_sample_count": sample_path.read_text().count("\n") if sample_path.exists() else 0}
    atomic_json(out / "RESULT.json", result)
    atomic_json(status_path, {"status": result["status"], "result": "RESULT.json", "protocol_sha256": sha(path)})
    print(json.dumps({"status": result["status"], "results": [{k:r[k] for k in ("task_id","gpu","seed","exit_code","update_count","environment_steps_per_second_wall")} for r in results]}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
