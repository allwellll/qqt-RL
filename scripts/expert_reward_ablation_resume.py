"""Compatibility owner for continuing the frozen D1 schedule after a writer error.

The scientific protocol and its sealed training sources stay unchanged.  This
owner supplies the generic PPO writer's legacy budget-key aliases in memory,
verifies every completed arm, and resumes only the next preregistered run.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts.expert_reward_handoff import (
    valid_d_resume_authorization,
    validated_predecessor_terminal,
)

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_reward_20261025/D_ablation_v1"
SEEDS = (2026102851, 2026102852, 2026102853)
ARMS = ("legacy", "minimal")
ENV_STEPS_PER_UPDATE = 16 * 128 * 8


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def compatible_protocol(protocol: dict) -> dict:
    """Expose old generic writer keys without mutating frozen protocol JSON."""
    if protocol.get("id") != "EXPERT-REWARD-ABLATION-20261025-D1":
        raise ValueError("unexpected D protocol identity")
    result = copy.deepcopy(protocol)
    budget = result["budget"]
    budget.setdefault("new_environment_steps_per_seed",
                      budget["new_environment_steps_per_arm_seed"])
    budget.setdefault("new_optimizer_steps_per_seed",
                      budget["new_optimizer_steps_per_arm_seed"])
    return result


def verify_or_repair_terminal(arm: str, seed: int) -> dict | None:
    run = B / "formal" / f"{arm}_s{seed}"
    complete = run / "COMPLETE.json"
    protocol = read(B / "PROTOCOL.json")
    if complete.exists():
        record = read(complete)
        if record.get("new_update") != protocol["updates"]:
            raise RuntimeError(f"incomplete terminal marker: {arm}/{seed}")
        cp = R / record["saved_checkpoint"]
        committed = read(cp / "COMMITTED.json")
        if sha(cp / "state.pkl") != record["state_sha256"] or \
                sha(cp / "policy.pt") != record["policy_sha256"] or \
                committed["state_sha256"] != record["state_sha256"] or \
                committed["policy_sha256"] != record["policy_sha256"]:
            raise RuntimeError(f"terminal checkpoint hash mismatch: {arm}/{seed}")
        return record

    checkpoints = sorted(run.glob("u*/COMMITTED.json"),
                         key=lambda p: int(read(p)["new_update"]))
    if not checkpoints:
        return None
    cp = checkpoints[-1].parent
    committed = read(cp / "COMMITTED.json")
    if int(committed.get("new_update", -1)) != int(protocol["updates"]):
        return None
    expected_steps = int(protocol["budget"]["new_environment_steps_per_arm_seed"])
    expected_optim = int(protocol["budget"]["new_optimizer_steps_per_arm_seed"])
    if committed.get("environment_steps") != expected_steps or \
            committed.get("new_optimizer_steps") != expected_optim or \
            not committed.get("passed") or not committed.get("metrics", {}).get("finite"):
        raise RuntimeError(f"final committed point does not match D budget: {arm}/{seed}")
    state_sha = sha(cp / "state.pkl")
    policy_sha = sha(cp / "policy.pt")
    if state_sha != committed.get("state_sha256") or policy_sha != committed.get("policy_sha256"):
        raise RuntimeError(f"final committed point hash mismatch: {arm}/{seed}")
    record = dict(
        passed=True, arm=arm, seed=seed, source_update=64,
        update=64 + int(protocol["updates"]), new_update=int(protocol["updates"]),
        environment_steps=expected_steps, optimizer_steps=expected_optim,
        lineage_environment_steps=1_048_576 + expected_steps,
        state_sha256=state_sha, policy_sha256=policy_sha,
        saved_checkpoint=cp.relative_to(R).as_posix(), no_peak_selection=True,
        terminal_marker_repaired_from_final_committed_checkpoint=True,
    )
    atomic_json(complete, record)
    atomic_json(B / "milestones" / f"{arm}_s{seed}.json", dict(
        passed=True, run=f"{arm}_s{seed}", checkpoint=record["saved_checkpoint"],
        state_sha256=state_sha, policy_sha256=policy_sha,
        updates=record["new_update"], environment_steps=expected_steps,
        optimizer_steps=expected_optim, no_peak_selection=True,
        terminal_marker_repaired=True, unix=time.time()))
    return record


def train_one(arm: str, seed: int) -> int:
    """Run/resume one scheduled arm with only in-memory budget-name aliases."""
    import scripts.expert_reward_ablation_train as worker

    worker.configure(arm)
    base = worker.base
    original_read = base.read

    def read_compatible(path):
        value = original_read(path)
        if Path(path).name == "PROTOCOL.json":
            return compatible_protocol(value)
        return value

    base.read = read_compatible
    items, bank_states, bank_aux = base.setup()
    complete = verify_or_repair_terminal(arm, seed)
    if complete is not None:
        return 0
    return 0 if base.train_seed(seed, items, bank_states, bank_aux) else 2


def gpu_snapshot() -> tuple[list[list[int]], str]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    devices = [[int(cell.strip()) for cell in row.split(",")]
               for row in raw.strip().splitlines()]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True).strip()
    return devices, apps


def verify_terminal(arm: str, seed: int) -> dict:
    record = verify_or_repair_terminal(arm, seed)
    if record is None:
        raise RuntimeError(f"run has no verified terminal: {arm}/{seed}")
    cp = R / record["saved_checkpoint"]
    protocol = read(B / "PROTOCOL.json")
    if record["new_update"] != protocol["updates"] or \
            record["environment_steps"] != protocol["budget"]["new_environment_steps_per_arm_seed"] or \
            record["optimizer_steps"] != protocol["budget"]["new_optimizer_steps_per_arm_seed"]:
        raise RuntimeError(f"fixed budget mismatch: {arm}/{seed}")
    return dict(run=f"{arm}_s{seed}", new_update=record["new_update"],
                environment_steps=record["environment_steps"],
                state_sha256=record["state_sha256"],
                policy_sha256=record["policy_sha256"],
                checkpoint=record["saved_checkpoint"],
                cpu_holdout=(B / "eval_done" / f"{arm}_s{seed}_u{record['new_update']:06d}.json").exists())


def notify(message: str) -> None:
    print(message, flush=True)
    atomic_json(B / "notifications" / f"{time.time_ns()}.json",
                {"message": message, "unix": time.time()})
    try:
        subprocess.run(["jdme-notify", "send", message], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def ensure_cpu_owner() -> None:
    if (B / "CPU_EVALUATION_COMPLETE.json").exists() or (B / "CPU_STOP.json").exists():
        return
    lines = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    if any("-m scripts.expert_reward_ablation_cpu" in line and
           "expert_reward_ablation_resume" not in line
           for line in lines):
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    with (B / "cpu_owner.log").open("a") as stream:
        subprocess.Popen([sys.executable, "-m", "scripts.expert_reward_ablation_cpu"],
                         cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT,
                         start_new_session=True)
    atomic_json(B / "CPU_OWNER_LAUNCH.json", dict(
        launched=True, unique_owner_lock="cpu_evaluator.lock",
        controls_gpu=False, unix=time.time()))


def owner() -> None:
    owner_lock = lock(B / "owner.lock")
    protocol = read(B / "PROTOCOL.json")
    scoped = read(B / "SCOPED_GPU_AUTHORIZATION.json")
    seal = read(B / "SOURCE_SEAL.json")
    if sha(B / "PROTOCOL.json") != scoped["protocol_sha256"] or \
            sha(B / "SOURCE_SEAL.json") != scoped["source_seal_sha256"]:
        raise RuntimeError("frozen D authorization hash mismatch")
    if not (B / "TECHNICAL_SMOKE.json").exists() or \
            not read(B / "TECHNICAL_SMOKE.json").get("passed") or \
            not read(B / "FRESH_RESTORE.json").get("passed"):
        raise RuntimeError("previously completed D technical checks are missing")
    if not (B / "CONTINUATION_AUTHORIZATION_20261026.json").exists():
        raise RuntimeError("explicit D continuation authorization is missing")
    if not valid_d_resume_authorization(
            B, scoped["protocol_sha256"], scoped["source_seal_sha256"],
            seal["predecessor_handoff_amendment_sha256"]):
        raise RuntimeError("hash-bound D resume authorization is invalid")
    if sha(R / "runs/retention_20261021/USER_GPU_PAUSE.json") != \
            scoped["global_pause_sha256"]:
        raise RuntimeError("global GPU pause authorization changed")
    predecessor = validated_predecessor_terminal(
        R, "runs/ppo_scale_20261025/r2_longrun", B / "PREDECESSOR_ACCEPTANCE.json")
    if predecessor is None or predecessor["kind"] != "USER_ENDED_EARLY":
        raise RuntimeError("the accepted R2 early-terminal handoff is not valid")
    ensure_cpu_owner()
    atomic_json(B / "ORCHESTRATION_REPAIR_OWNER.json", dict(
        unique=True, namespace="runs/expert_reward_20261025/D_ablation_v1",
        protocol_sha256=sha(B / "PROTOCOL.json"), source_seal_sha256=sha(B / "SOURCE_SEAL.json"),
        protocol_unchanged=True, reward_coefficients_unchanged=True,
        previous_owner_exited=True, old_queues_remain_paused=True,
        started_unix=time.time()))

    sequence = [(seed, arm) for seed in SEEDS for arm in ARMS]
    completed = []
    for seed, arm in sequence:
        if verify_or_repair_terminal(arm, seed) is not None:
            completed.append(verify_terminal(arm, seed))
            continue
        devices, apps = gpu_snapshot()
        if apps or any(device[1] != 0 for device in devices):
            raise RuntimeError("GPU is not fully released; refusing to start next D arm")
        if (B / "USER_STOP.json").exists() and \
                not read(B / "D_RESUME_AUTHORIZATION.json").get("allow_new_gpu_run"):
            atomic_json(B / "ORCHESTRATION_PAUSED_20261026.json", dict(
                no_next_run_started=True, last_verified_runs=completed,
                old_user_stop_preserved=True, unix=time.time()))
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                   JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="cuda", JAXBOMB_RULE="bun",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true",
                   JAX_COMPILATION_CACHE_DIR=str(B / "cache"))
        log_path = B / f"resume_{arm}_{seed}.log"
        atomic_json(B / "ORCHESTRATION_REPAIR_STATUS.json", dict(
            stage="RUNNING_NEXT_PREREGISTERED_RUN", arm=arm, seed=seed,
            completed_before_launch=completed, gpu_snapshot=devices,
            source_seal_sha256=sha(B / "SOURCE_SEAL.json"), unix=time.time()))
        with log_path.open("a") as stream:
            worker_proc = subprocess.Popen(
                [sys.executable, "-m", "scripts.expert_reward_ablation_resume",
                 "worker", arm, str(seed)], cwd=R, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            while worker_proc.poll() is None:
                ensure_cpu_owner()
                devices, apps = gpu_snapshot()
                atomic_json(B / "GPU_ACTIVITY.json", dict(
                    stage="formal", args=[arm, seed], devices=devices,
                    compute_apps_present=bool(apps), controls_gpu=True, unix=time.time()))
                time.sleep(10)
        if worker_proc.returncode != 0:
            atomic_json(B / "ORCHESTRATION_REPAIR_FAILURE.json", dict(
                passed=False, arm=arm, seed=seed, worker_returncode=worker_proc.returncode,
                last_verified_runs=completed, log=log_path.relative_to(R).as_posix(),
                fixed_protocol_preserved=True, unix=time.time()))
            raise RuntimeError(f"D worker failed: {arm}/{seed}, rc={worker_proc.returncode}")
        completed.append(verify_terminal(arm, seed))
        notify(f"D 固定run终态 {arm}/seed{seed}：u320、5,242,880 on-policy步，state/policy hash 已核验；按预注册顺序接续。")

    atomic_json(B / "TRAINING_COMPLETE.json", dict(
        passed=True, runs=6, arms=list(ARMS), paired_seeds=list(SEEDS),
        environment_steps=protocol["budget"]["total_environment_steps"],
        optimizer_steps=protocol["budget"]["total_optimizer_steps"],
        fixed_budget=True, completed_runs=completed,
        no_peak_selection=True, old_queues_remain_paused=True, unix=time.time()))
    devices, apps = gpu_snapshot()
    atomic_json(B / "GPU_COMPLETE.json", dict(passed=True, runs=6,
        controls_gpu=False, devices=devices, compute_apps_present=bool(apps), unix=time.time()))
    atomic_json(B / "ORCHESTRATION_REPAIR_COMPLETE.json", dict(
        passed=True, all_six_fixed_runs_complete=True, protocol_unchanged=True,
        reward_coefficients_unchanged=True, completed_runs=completed,
        controls_gpu=False, old_queues_remain_paused=True, unix=time.time()))
    notify("D1六个预注册固定run已完成并核验；CPU/JS继续异步消费，R2与旧储备保持停止/暂停。")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    mode = args.pop(0) if args else "owner"
    if mode == "worker":
        if len(args) != 2:
            raise SystemExit("usage: worker <legacy|minimal> <seed>")
        return train_one(args[0], int(args[1]))
    if mode != "owner":
        raise SystemExit("usage: [owner | worker <legacy|minimal> <seed>]")
    owner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
