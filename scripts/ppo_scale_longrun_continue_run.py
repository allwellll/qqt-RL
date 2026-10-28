"""Persistent unique GPU owner for the R2 fixed long-horizon continuation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/ppo_scale_20261025/r2_longrun"
GLOBAL_PAUSE = R / "runs/retention_20261021/USER_GPU_PAUSE.json"


def read(path):
    return json.loads(Path(path).read_text())


class UserStopped(Exception):
    pass


def authorize():
    protocol = read(B / "PROTOCOL.json")
    auth = read(B / "SCOPED_GPU_AUTHORIZATION.json")
    assert auth["authorized_by"] == "explicit_user_instruction"
    assert auth["protocol_sha256"] == sha(B / "PROTOCOL.json")
    assert protocol["old_queues_remain_paused"] is True
    assert protocol["r1_immutable"] is True
    if sha(GLOBAL_PAUSE) != auth["global_pause_sha256"]:
        raise UserStopped("global user pause changed; safe-boundary stop requested")
    if (B / "USER_STOP.json").exists():
        raise UserStopped("explicit user stop; next update is not claimed")
    for name, expected in read(B / "SOURCE_SEAL.json")["sources"].items():
        assert sha(R / name) == expected, f"sealed source changed: {name}"


def notify(message):
    print(message, flush=True)
    atomic_json(B / "notifications" / f"{time.time_ns()}.json",
                {"message": message, "unix": time.time()})
    try:
        subprocess.run(["jdme-notify", "send", message], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def child(command):
    authorize()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        JAX_PLATFORMS="cuda",
        JAX_PLATFORM_NAME="cuda",
        JAXBOMB_RULE="bun",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        JAX_COMPILATION_CACHE_DIR=str(B / "cache"),
    )
    while True:
        authorize()
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not apps:
            break
        atomic_json(B / "RESOURCE_WAIT.json", dict(
            command=command, no_preempt=True, controls_gpu=True, unix=time.time()))
        time.sleep(5)
    atomic_json(B / "OWNER_STAGE.json", dict(stage=command, unix=time.time()))
    log = (B / f"{command.replace('-', '_')}.log").open("a")
    try:
        worker = subprocess.Popen(
            [sys.executable, "-m", "scripts.ppo_scale_longrun_continue_train", command],
            cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
        while worker.poll() is None:
            ensure_cpu_owner()
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True)
            devices = [[int(x.strip()) for x in row.split(",")]
                       for row in raw.strip().splitlines()]
            atomic_json(B / "GPU_ACTIVITY.json", dict(
                stage=command, devices=devices, unix=time.time()))
            with (B / "GPU_ACTIVITY.jsonl").open("a") as activity:
                activity.write(json.dumps(dict(stage=command, devices=devices,
                                               unix=time.time())) + "\n")
                activity.flush()
            time.sleep(10)
        if worker.returncode != 0:
            raise RuntimeError(f"{command} exit={worker.returncode}")
    finally:
        log.close()


def ensure_cpu_owner():
    if (B / "CPU_EVALUATION_COMPLETE.json").exists() or (B / "CPU_STOP.json").exists():
        return
    lines = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    matches = [line for line in lines
               if "-m scripts.ppo_scale_longrun_continue_cpu" in line
               and "continue_run" not in line]
    if matches:
        return
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES="",
               JAX_PLATFORMS="cpu",
               JAX_PLATFORM_NAME="cpu",
               OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    log = (B / "cpu_owner.log").open("a")
    subprocess.Popen([sys.executable, "-m", "scripts.ppo_scale_longrun_continue_cpu"],
                     cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    log.close()
    atomic_json(B / "CPU_OWNER_LAUNCH.json", dict(
        launched=True, controls_gpu=False, unique_owner_lock="cpu_evaluator.lock",
        unix=time.time()))


def owner():
    unique = lock(B / "owner.lock")
    authorize()
    atomic_json(B / "OWNER.json", dict(
        unique=True, namespace=str(B.relative_to(R)),
        source_namespace="runs/ppo_scale_20261025/r1_longrun",
        old_queues_remain_paused=True, started_unix=time.time()))
    try:
        ensure_cpu_owner()
        if not (B / "TECHNICAL_SMOKE.json").exists():
            child("smoke")
        if not read(B / "TECHNICAL_SMOKE.json")["passed"]:
            raise RuntimeError("technical smoke marker not passed")
        if not (B / "FRESH_RESTORE.json").exists():
            child("restore-smoke")
        if not read(B / "FRESH_RESTORE.json")["passed"]:
            raise RuntimeError("fresh restore marker not passed")
        if not (B / "TRAINING_COMPLETE.json").exists():
            child("formal")
        if (B / "PAUSED.json").exists():
            atomic_json(B / "PAUSED_OWNER.json", dict(
                passed=True, safe_boundary=True,
                last_new_update=read(B / "PAUSED.json").get("last_new_update"),
                current_seed=read(B / "PAUSED.json").get("seed"),
                old_queues_remain_paused=True, unix=time.time()))
            notify("R2在用户暂停的安全原子边界停止；未启动后续更新/seed，原子恢复点已写收据。")
            return
        if (B / "TRAINING_COMPLETE.json").exists():
            atomic_json(B / "GPU_COMPLETE.json", dict(
                passed=True,
                seeds=read(B / "PROTOCOL.json")["paired_seeds"],
                new_environment_steps=read(B / "PROTOCOL.json")["budget"]["total_new_environment_steps"],
                new_optimizer_steps=read(B / "PROTOCOL.json")["budget"]["total_new_optimizer_steps"],
                old_queues_remain_paused=True,
                cpu_js_independent=True,
                unix=time.time(),
            ))
            notify("R2三seed固定长程GPU预算已完成并原子验收；CPU/JS继续异步消费，旧O5/储备仍暂停。")
            while not (B / "CPU_EVALUATION_COMPLETE.json").exists():
                ensure_cpu_owner()
                if (B / "CPU_TECHNICAL_FAILURE.json").exists():
                    notify("R2固定CPU/JS评估有独立技术失败，GPU预算不受影响；保留队列并持续重试。")
                atomic_json(B / "OWNER_STAGE.json", dict(
                    stage="WAITING_FOR_ASYNC_EVALUATION_AFTER_GPU_COMPLETE",
                    gpu_training_complete=True, controls_gpu=False,
                    unix=time.time()))
                time.sleep(30)
            atomic_json(B / "FINAL_OWNER_NOTICE.json", dict(
                passed=True,
                training_complete=True,
                cpu_evaluation_complete=True,
                all_three_seed_terminals=True,
                old_queues_remain_paused=True,
                unix=time.time()))
            notify("R2三seed长程训练与固定CPU/JS/最终holdout均完成；完整多seed证据已写入R2 namespace和当日报告。")
    except UserStopped as exc:
        atomic_json(B / "PAUSED_OWNER.json", dict(
            passed=True, safe_boundary=True, reason=str(exc),
            old_queues_remain_paused=True, unix=time.time()))
        notify("R2收到用户暂停，已保留最近安全原子checkpoint，不领取下一更新或seed。")
    except Exception as exc:
        reason = str(exc).replace(str(R), "<WORKTREE>")
        atomic_json(B / "TECHNICAL_FAILURE.json", dict(
            passed=False, reason=reason, recent_atomic_preserved=True,
            old_queues_remain_paused=True, unix=time.time()))
        notify(f"R2 GPU技术故障：{reason}；最近原子checkpoint保留，旧队列未恢复。")
        raise
    finally:
        atomic_json(B / "OWNER_EXIT.json", dict(
            unix=time.time(), old_queues_remain_paused=True,
            training_complete=(B / "TRAINING_COMPLETE.json").exists()))


if __name__ == "__main__":
    owner()
