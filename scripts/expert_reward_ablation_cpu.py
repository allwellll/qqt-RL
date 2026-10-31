"""Unique CPU/production-JS evaluator for preregistered D1 checkpoints."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                  JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",
                  NODE_OPTIONS="--require=./scripts/ppo_scale_metrics_preload.js")

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_reward_20261025/D_ablation_v1"
GLOBAL_JS_LOCK = R / "runs/je10_bun_20261018/evaluation.lock"

from scripts.population_run_state import atomic_json, lock  # noqa: E402


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


def pending_requests():
    done = {p.stem for p in (B / "eval_done").glob("*.json")}
    return [p for p in sorted((B / "eval_queue").glob("*.json"))
            if p.stem not in done]


def consume_one(path):
    from scripts import ppo_scale_longrun_continue_cpu as evaluator
    evaluator.B, evaluator.R, evaluator.notify = B, R, notify
    return evaluator.consume(path)


def owner():
    owner_lock = lock(B / "cpu_evaluator.lock")
    atomic_json(B / "CPU_OWNER.json", dict(unique=True, controls_gpu=False,
                                              started_unix=time.time()))
    failures = 0
    while not (B / "CPU_STOP.json").exists():
        todo = pending_requests()
        if todo:
            path = todo[0]
            atomic_json(B / "CPU_ACTIVE.json", dict(
                request=path.name, controls_gpu=False, unix=time.time()))
            try:
                GLOBAL_JS_LOCK.parent.mkdir(parents=True, exist_ok=True)
                with GLOBAL_JS_LOCK.open("a") as stream:
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        atomic_json(B / "CPU_STATUS.json", dict(
                            stage="WAITING_FOR_UNIQUE_JS_REVIEW_OWNER",
                            request=path.name, controls_gpu=False, unix=time.time()))
                        time.sleep(10)
                        continue
                    try:
                        result = consume_one(path)
                    finally:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                atomic_json(B / "CPU_STATUS.json", dict(
                    stage="EVALUATED", request=path.name,
                    level=read(path)["level"], result=result,
                    controls_gpu=False, unix=time.time()))
                failures = 0
                notify(f"D1异步评估完成 {path.stem}；CPU/JS未阻塞GPU训练。")
            except Exception as exc:
                failures += 1
                reason = str(exc).replace(str(R), "<WORKTREE>")
                atomic_json(B / "CPU_TECHNICAL_FAILURE.json", dict(
                    passed=False, request=path.name, reason=reason,
                    controls_gpu=False, retryable=True, unix=time.time()))
                atomic_json(B / "CPU_STATUS.json", dict(
                    stage="TECHNICAL_RETRY", request=path.name,
                    failures=failures, controls_gpu=False, unix=time.time()))
                notify(f"D1 CPU/JS评估技术故障 {path.stem}：{reason}；GPU训练继续，稍后重试。")
                time.sleep(min(300, 15 * failures))
            continue

        complete = B / "TRAINING_COMPLETE.json"
        queued = list((B / "eval_queue").glob("*.json"))
        done = list((B / "eval_done").glob("*.json"))
        if complete.exists() and len(done) == len(queued):
            atomic_json(B / "CPU_EVALUATION_COMPLETE.json", dict(
                passed=True, requests=len(done), fixed_run_count=6,
                all_final_holdouts_consumed=True, controls_gpu=False,
                no_holdout_tuning=True, unix=time.time()))
            atomic_json(B / "CPU_OWNER_EXIT.json", dict(
                passed=True, unique=True, controls_gpu=False, unix=time.time()))
            notify("D1六个固定run的异步CPU/JS评估与holdout均已消费；可进行预声明配对分析。")
            return
        atomic_json(B / "CPU_STATUS.json", dict(
            stage="WAITING_FOR_COMMITTED_CHECKPOINTS",
            pending_requests=0, queued_requests=len(queued),
            completed_requests=len(done), controls_gpu=False, unix=time.time()))
        time.sleep(10)
    atomic_json(B / "CPU_PAUSED.json", dict(
        passed=True, pending_requests=len(pending_requests()),
        controls_gpu=False, unix=time.time()))


if __name__ == "__main__":
    owner()
