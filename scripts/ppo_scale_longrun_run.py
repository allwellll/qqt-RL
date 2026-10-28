"""Persistent owner for the bounded PPO batch-scale/long-horizon experiment."""
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/ppo_scale_20261025/r1_longrun"


def read(path):
    return json.loads(Path(path).read_text())


class UserStopped(Exception):
    pass


def authorize():
    if (B / "USER_STOP.json").exists():
        raise UserStopped("safe atomic user stop")
    p = read(B / "PROTOCOL.json")
    a = read(B / "SCOPED_GPU_AUTHORIZATION.json")
    assert a["authorized_by"] == "explicit_user_instruction"
    assert not a["old_continuous_resume"] and a["old_queues_remain_paused"]
    assert a["protocol_sha256"] == sha(B / "PROTOCOL.json")
    assert sha(B / "inputs/parent.pkl") == p["parent_input_sha256"]
    assert read(B / "PREREGISTRATION.json")["passed"]
    assert p["id"] == "PPO-SCALE-LONGRUN-20261025-R1"
    assert p["arms"] == ["ppo_base_batch", "ppo_scaled_batch"]
    assert len(p["paired_seeds"]) == 3


def notify(message):
    print(message, flush=True)
    atomic_json(B / "notifications" / f"{time.time_ns()}.json",
                {"message": message, "unix": time.time()})
    try:
        subprocess.run(["jdme-notify", "send", message], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def child(args, stage):
    authorize()
    seal = read(B / "SOURCE_SEAL.json")["sources"]
    for name, expected in seal.items():
        assert sha(R / name) == expected, f"sealed source changed: {name}"
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="cuda",
               JAXBOMB_RULE="bun", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true",
               CUBLAS_WORKSPACE_CONFIG=":4096:8",
               JAX_COMPILATION_CACHE_DIR=str(B / "cache"))
    while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            text=True).strip():
        authorize()
        atomic_json(B / "RESOURCE_WAIT.json", {"stage": stage, "no_preempt": True,
                                                 "unix": time.time()})
        time.sleep(5)
    atomic_json(B / "OWNER_STAGE.json", {"stage": stage, "unix": time.time()})
    log = B / f"{stage.split('/')[0]}.log"
    with log.open("a") as stream:
        worker = subprocess.Popen([os.environ.get("PYTHON", sys.executable),
                                   "-m", *args], cwd=R, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT)
        while worker.poll() is None:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True)
            with (B / "GPU_ACTIVITY.jsonl").open("a") as output:
                rows = []
                for row in raw.strip().splitlines():
                    vals = [int(x.strip()) for x in row.split(",")]
                    rows.append(vals)
                output.write(json.dumps({"unix": time.time(), "stage": stage,
                                         "devices": rows}) + "\n")
            time.sleep(5)
    if (B / "USER_STOP.json").exists():
        raise UserStopped("worker stopped after last atomic checkpoint")
    if worker.returncode != 0:
        raise RuntimeError(f"{stage} exit={worker.returncode}")


def owner():
    owner_lock = lock(B / "owner.lock")
    authorize()
    try:
        if not (B / "ACTION_SAMPLING_AUDIT.json").exists():
            child(["scripts.ppo_scale_longrun_train", "audit"], "audit")
        if not (B / "TECHNICAL_SMOKE.json").exists():
            child(["scripts.ppo_scale_longrun_train", "smoke"], "smoke")
        assert read(B / "TECHNICAL_SMOKE.json")["passed"]
        if not (B / "FRESH_RESTORE.json").exists():
            if not (B / "EXPECTED_RESTORE.json").exists():
                child(["scripts.ppo_scale_longrun_train", "restore-reference"], "restore-reference")
            child(["scripts.ppo_scale_longrun_train", "restore"], "restore")
        assert read(B / "FRESH_RESTORE.json")["passed"]
        if not (B / "TRAINING_COMPLETE.json").exists():
            child(["scripts.ppo_scale_longrun_train", "formal"], "formal")
        assert read(B / "TRAINING_COMPLETE.json")["passed"]
        atomic_json(B / "GPU_COMPLETE.json", {
            "passed": True, "formal_runs": 6,
            "formal_environment_steps": 6291456,
            "formal_optimizer_steps": 18432,
            "old_continuous_resume": False,
            "cpu_js_independent": True,
        })
        notify("PPO scale R1 六个等环境预算固定run GPU终态已保存；CPU/JS异步评估继续，旧队列保持暂停。")
    except UserStopped as error:
        atomic_json(B / "PAUSED_OWNER.json", {"safe_atomic": True,
                                               "reason": str(error),
                                               "old_continuous_resume": False})
        notify("PPO scale R1 已在原子边界安全暂停，保存点和恢复入口保留。")
    except Exception as error:
        reason = str(error).replace(str(R), "<WORKTREE>")
        atomic_json(B / "TECHNICAL_FAILURE.json", {"reason": reason, "unix": time.time()})
        notify(f"PPO scale R1 技术故障：{reason}；最近原子点保留，旧队列未恢复。")
        raise
    finally:
        atomic_json(B / "OWNER_EXIT.json", {"unix": time.time(),
                                              "old_continuous_resume": False})


if __name__ == "__main__":
    owner()
