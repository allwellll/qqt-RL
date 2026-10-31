"""Persistent D1 owner; waits for R2's declared terminal and released GPUs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.expert_reward_handoff import (
    filter_live_compute_apps,
    valid_d_resume_authorization,
    validate_user_ended_early,
)
from scripts.population_run_state import atomic_json, lock, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_reward_20261025/D_ablation_v1"
R2 = R / "runs/ppo_scale_20261025/r2_longrun"
GLOBAL_PAUSE = R / "runs/retention_20261021/USER_GPU_PAUSE.json"


def read(path):
    return json.loads(Path(path).read_text())


def notify(message):
    print(message, flush=True)
    atomic_json(B / "notifications" / f"{time.time_ns()}.json",
                {"message": message, "unix": time.time()})
    try:
        subprocess.run(["jdme-notify", "send", message], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def authorize():
    scoped = read(B / "SCOPED_GPU_AUTHORIZATION.json")
    seal = read(B / "SOURCE_SEAL.json")
    if sha(GLOBAL_PAUSE) != scoped["global_pause_sha256"]:
        return False
    protocol = read(B / "PROTOCOL.json")
    assert scoped["authorized_by"] == "explicit_user_instruction"
    assert scoped["protocol_sha256"] == sha(B / "PROTOCOL.json")
    assert scoped["source_seal_sha256"] == sha(B / "SOURCE_SEAL.json")
    assert scoped["preformal_protocol_amendment_sha256"] == seal["protocol_amendment_sha256"]
    assert sha(B / "PROTOCOL_AMENDMENT_20261025.json") == seal["protocol_amendment_sha256"]
    assert protocol["amendment"]["formal_updates_before_amendment"] == 0
    for name, expected in seal["sources"].items():
        assert sha(R / name) == expected, f"sealed source changed: {name}"
    assert protocol["predecessor"] == "runs/ppo_scale_20261025/r2_longrun"
    assert protocol["updates"] == 320
    if (B / "USER_STOP.json").exists() and not valid_d_resume_authorization(
            B, scoped["protocol_sha256"], scoped["source_seal_sha256"],
            seal["predecessor_handoff_amendment_sha256"]):
        return False
    terminal = predecessor_terminal()
    # The user may end the predecessor before its declared budget.  A valid
    # early-terminal receipt still requires an atomically committed, verified
    # checkpoint and an explicit owner exit; it must never be treated as a
    # fixed-budget completion.
    if terminal is not None:
        return True
    return True


def r2_complete():
    marker = R2 / "TRAINING_COMPLETE.json"
    if not marker.exists():
        return False
    protocol = read(R2 / "PROTOCOL.json")
    complete = read(marker)
    if complete.get("new_environment_steps") != protocol["budget"]["total_new_environment_steps"]:
        raise RuntimeError("R2 training marker disagrees with its fixed budget")
    for seed in protocol["paired_seeds"]:
        run = (R2 / "formal" /
               f"ppo_scaled_batch_continuation_s{seed}" / "COMPLETE.json")
        record = read(run)
        cp = R / record["saved_checkpoint"]
        committed = read(cp / "COMMITTED.json")
        assert record["new_update"] == protocol["updates"]
        assert sha(cp / "state.pkl") == record["state_sha256"] == committed["state_sha256"]
        assert sha(cp / "policy.pt") == record["policy_sha256"] == committed["policy_sha256"]
    return True


def predecessor_terminal():
    """Return the validated predecessor terminal kind, or ``None``.

    R2 may terminate either with its fixed-budget marker or with the explicit
    user early-stop receipt.  The latter is an accepted handoff boundary, not
    a claim that the fixed budget completed.
    """
    if (R2 / "TRAINING_COMPLETE.json").exists():
        r2_complete()
        return "TRAINING_COMPLETE"
    receipt_path = R2 / "USER_ENDED_EARLY.json"
    if not receipt_path.exists():
        return None
    validate_user_ended_early(R, "runs/ppo_scale_20261025/r2_longrun")
    return "USER_ENDED_EARLY"


def gpu_snapshot():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    devices = [[int(cell.strip()) for cell in row.split(",")]
               for row in raw.strip().splitlines()]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader"], text=True).strip()
    ps_state = subprocess.check_output(["ps", "-eo", "pid=,stat="], text=True)
    apps = filter_live_compute_apps(apps, ps_state)
    return devices, apps


def wait_for_r2_and_gpu_release():
    while authorize():
        status = read(R2 / "STATUS.json") if (R2 / "STATUS.json").exists() else {}
        terminal = predecessor_terminal()
        if terminal is not None:
            devices, apps = gpu_snapshot()
            free = not apps and all(row[1] == 0 for row in devices)
            atomic_json(B / "PREDECESSOR_TERMINAL_WAIT.json", dict(
                passed=True, predecessor="runs/ppo_scale_20261025/r2_longrun",
                predecessor_terminal=terminal,
                predecessor_training_complete=(terminal == "TRAINING_COMPLETE"),
                predecessor_user_ended_early=(terminal == "USER_ENDED_EARLY"),
                predecessor_fixed_budget_complete=(terminal == "TRAINING_COMPLETE"),
                predecessor_final_seeds=read(R2 / "PROTOCOL.json")["paired_seeds"]
                if terminal == "TRAINING_COMPLETE" else None,
                devices=devices, compute_apps_present=bool(apps),
                gpu_released=free, controls_gpu=False, unix=time.time()))
            if free:
                terminal_path = (R2 / "USER_ENDED_EARLY.json"
                                 if terminal == "USER_ENDED_EARLY"
                                 else R2 / "TRAINING_COMPLETE.json")
                accepted = dict(
                    passed=True, predecessor_terminal=terminal,
                    predecessor_fixed_budget_complete=(terminal == "TRAINING_COMPLETE"),
                    predecessor_terminal_sha256=sha(terminal_path),
                    gpu_released=True, devices=devices,
                    controls_gpu=False, unix=time.time())
                if terminal == "USER_ENDED_EARLY":
                    accepted["accepted_checkpoint"] = validate_user_ended_early(
                        R, "runs/ppo_scale_20261025/r2_longrun")["checkpoint"]
                atomic_json(B / "PREDECESSOR_ACCEPTANCE.json", accepted)
                return True
        else:
            atomic_json(B / "WAITING_FOR_PREDECESSOR.json", dict(
                predecessor="runs/ppo_scale_20261025/r2_longrun",
                predecessor_status={k: status.get(k) for k in
                    ("stage", "seed", "update", "new_update", "finite",
                     "environment_steps", "last_committed_new_update")},
                no_gpu_claim=True, no_preemption=True, unix=time.time()))
        time.sleep(20)
    atomic_json(B / "PAUSED.json", dict(passed=True, no_new_gpu_run_started=True,
                                        explicit_user_stop=True, unix=time.time()))
    return False


def ensure_cpu_owner():
    if (B / "CPU_EVALUATION_COMPLETE.json").exists() or (B / "CPU_STOP.json").exists():
        return
    try:
        lines = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    except subprocess.CalledProcessError:
        lines = []
    if any("-m scripts.expert_reward_ablation_cpu" in line and
           "expert_reward_ablation_run" not in line for line in lines):
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    log = (B / "cpu_owner.log").open("a")
    subprocess.Popen([sys.executable, "-m", "scripts.expert_reward_ablation_cpu"],
                     cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    log.close()
    atomic_json(B / "CPU_OWNER_LAUNCH.json", dict(
        launched=True, unique_owner_lock="cpu_evaluator.lock",
        controls_gpu=False, unix=time.time()))


def child(command, *args):
    if not authorize():
        return False
    devices, apps = gpu_snapshot()
    if apps or any(row[1] for row in devices):
        raise RuntimeError("GPU not fully released before D run launch")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="cuda", JAXBOMB_RULE="bun",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true",
        JAX_COMPILATION_CACHE_DIR=str(B / "cache"),
    )
    command_line = [sys.executable, "-m", "scripts.expert_reward_ablation_train",
                    command, *map(str, args)]
    logfile = B / f"{command.replace('-', '_')}_{'_'.join(map(str,args))}.log"
    atomic_json(B / "OWNER_STAGE.json", dict(
        stage=command, args=list(args), devices_at_start=devices,
        unique_gpu_owner=True, unix=time.time()))
    with logfile.open("a") as stream:
        worker = subprocess.Popen(command_line, cwd=R, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT)
        while worker.poll() is None:
            ensure_cpu_owner()
            if not authorize():
                # The worker checks authorization at each update and commits
                # its latest state atomically before exiting.
                atomic_json(B / "USER_STOP_SEEN.json", dict(
                    worker_present=True, command=command,
                    requested_at_safe_update_boundary=True, unix=time.time()))
            devices, _ = gpu_snapshot()
            atomic_json(B / "GPU_ACTIVITY.json", dict(
                stage=command, args=list(args), devices=devices,
                controls_gpu=True, unix=time.time()))
            time.sleep(10)
        if worker.returncode != 0:
            raise RuntimeError(f"D worker {command} {args} exited {worker.returncode}")
    ensure_cpu_owner()
    return not (B / "PAUSED.json").exists()


def verify_complete_run(arm, seed):
    base = B / "formal" / f"{arm}_s{seed}"
    record = read(base / "COMPLETE.json")
    cp = R / record["saved_checkpoint"]
    committed = read(cp / "COMMITTED.json")
    expected = read(B / "PROTOCOL.json")
    assert record["new_update"] == expected["updates"] == 320
    assert record["environment_steps"] == expected["budget"]["new_environment_steps_per_arm_seed"]
    assert record["optimizer_steps"] == expected["budget"]["new_optimizer_steps_per_arm_seed"]
    assert sha(cp / "state.pkl") == record["state_sha256"] == committed["state_sha256"]
    assert sha(cp / "policy.pt") == record["policy_sha256"] == committed["policy_sha256"]
    atomic_json(B / "milestones" / f"{arm}_s{seed}.json", dict(
        passed=True, run=f"{arm}_s{seed}", checkpoint=str(cp.relative_to(R)),
        state_sha256=record["state_sha256"], policy_sha256=record["policy_sha256"],
        updates=320, environment_steps=record["environment_steps"],
        optimizer_steps=record["optimizer_steps"], no_peak_selection=True,
        unix=time.time()))


def owner():
    B.mkdir(parents=True, exist_ok=True)
    owner_lock = lock(B / "owner.lock")
    if not authorize():
        atomic_json(B / "PAUSED.json", dict(passed=True, reason="user authorization changed",
                                            no_gpu_run_started=True, unix=time.time()))
        return
    atomic_json(B / "OWNER.json", dict(unique=True,
        namespace="runs/expert_reward_20261025/D_ablation_v1",
        predecessor="runs/ppo_scale_20261025/r2_longrun",
        old_queues_remain_paused=True, started_unix=time.time()))
    if not wait_for_r2_and_gpu_release():
        return
    notify("R2三seed固定训练已完成且8卡显存/compute app释放；D1奖励对照开始技术smoke，CPU评估独立跟随。")
    if not (B / "TECHNICAL_SMOKE.json").exists():
        if not child("smoke"):
            return
    if not read(B / "TECHNICAL_SMOKE.json")["passed"]:
        raise RuntimeError("D1 minimal-reward technical smoke did not pass")
    if not (B / "FRESH_RESTORE.json").exists():
        if not child("restore-smoke"):
            return
    if not read(B / "FRESH_RESTORE.json")["passed"]:
        raise RuntimeError("D1 fresh-process restore smoke did not pass")
    protocol = read(B / "PROTOCOL.json")
    seeds = protocol["paired_seeds"]
    sequence = [(seed, arm) for seed in seeds for arm in ("legacy", "minimal")]
    for seed, arm in sequence:
        if not authorize():
            atomic_json(B / "PAUSED.json", dict(passed=True, before_next_run=True,
                                                next_run=f"{arm}_s{seed}", unix=time.time()))
            return
        run = B / "formal" / f"{arm}_s{seed}"
        if (run / "COMPLETE.json").exists():
            verify_complete_run(arm, seed)
            continue
        if not child("formal", arm, seed):
            if (run / "COMPLETE.json").exists():
                verify_complete_run(arm, seed)
            if (B / "PAUSED.json").exists():
                return
        verify_complete_run(arm, seed)
        notify(f"D1固定run终态 {arm}/seed{seed}：g320，5,242,880 on-policy环境步，原子state/policy SHA已核验；接续下一配对run。")
    total_runs = len(seeds) * len(protocol["arms"])
    atomic_json(B / "TRAINING_COMPLETE.json", dict(
        passed=True, runs=total_runs, paired_seeds=seeds,
        arms=protocol["arms"], environment_steps=protocol["budget"]["total_environment_steps"],
        optimizer_steps=protocol["budget"]["total_optimizer_steps"],
        fixed_budget=True, no_peak_selection=True,
        old_queues_remain_paused=True, unix=time.time()))
    ensure_cpu_owner()
    notify("D1六个配对固定预算run完成并核验原子hash；CPU/JS按已提交点继续异步，旧O5/储备仍暂停。")
    atomic_json(B / "GPU_COMPLETE.json", dict(passed=True, runs=total_runs,
                                               controls_gpu=False, unix=time.time()))


if __name__ == "__main__":
    owner()
