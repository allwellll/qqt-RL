"""Unique asynchronous CPU/production-JS evaluator for R2 checkpoints."""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    JAX_PLATFORMS="cpu",
    JAX_PLATFORM_NAME="cpu",
    JAXBOMB_RULE="bun",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NODE_OPTIONS="--require=./scripts/ppo_scale_metrics_preload.js",
)

R = Path(__file__).resolve().parents[1]
B = R / "runs/ppo_scale_20261025/r2_longrun"

from scripts.population_run_state import atomic_json, lock, sha  # noqa: E402


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


def export_model(cp: Path) -> Path:
    from scripts.retention_phase_export import authority_export
    out = cp / "retention" / "model"
    authority_export(cp / "policy.pt", out)
    return out / "model.json"


def natural(cp: Path, games: int, seed_start: int, output: Path):
    model = export_model(cp)
    regions = B / "cpu_eval_inputs" / "regions.json"
    env = dict(os.environ,
               JE10_ROUTE_TASK="natural_full2400",
               JE10_ROUTE_GAMES=str(games),
               JE10_ROUTE_START="0",
               JE10_ROUTE_END=str(games),
               JE10_ROUTE_SEED_START=str(seed_start))
    with output.with_suffix(".log").open("a") as stream:
        subprocess.run(
            ["node", "scripts/dagger_rl_route.js", str(model.relative_to(R)),
             str(output.relative_to(R)), str(regions.relative_to(R))],
            cwd=R, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return read(output)


def light_eval(cp: Path, request: dict):
    out = cp / "retention" / "light_dev16.json"
    if out.exists():
        result = read(out)
    else:
        result = natural(cp, 16, read(B / "CPU_EVAL_PROTOCOL.json")["natural_seed_start"], out)
        result = result["results"]["natural_full2400"]
        atomic_json(out, dict(
            passed=True,
            policy_sha256=request["checkpoint_policy_sha256"],
            games=16,
            seed_start=read(B / "CPU_EVAL_PROTOCOL.json")["natural_seed_start"],
            engine="production JS natural full2400; greedy action selection",
            results=result,
            controls_gpu=False,
            holdout_not_used=True,
        ))
        result = read(out)
    return dict(level="light", checkpoint=request["checkpoint"],
                seed=request["seed"], new_update=request["new_update"],
                policy_sha256=request["checkpoint_policy_sha256"],
                games=result.get("games"),
                delivery_episodes=result.get("delivery_episodes"),
                deaths=result.get("death"), controls_gpu=False)


def full_eval(cp: Path, request: dict):
    import scripts.ppo_scale_longrun_cpu as prior_cpu
    import concurrent.futures
    from scripts import je08_bun_authority as authority
    prior_cpu.B = B
    prior_cpu.R = R
    prior_cpu.notify = notify
    out = cp / "retention"
    out.mkdir(parents=True, exist_ok=True)
    policy_sha = sha(cp / "policy.pt")
    assert policy_sha == request["checkpoint_policy_sha256"]
    model = export_model(cp)
    protocol = read(B / "CPU_EVAL_PROTOCOL.json")
    spawn = B / "cpu_eval_inputs" / "spawns.json"

    def hunter(job):
        tier, start = job
        target = out / f"hunter_{tier}_{start:03d}.json"
        if not target.exists():
            subprocess.run([
                "node", "scripts/je10_web_hunter.js",
                "--model", str(model.relative_to(R)),
                "--difficulty", tier,
                "--spawns", str(spawn.relative_to(R)),
                "--start", str(start), "--end", str(start + 8),
                "--seed", str(protocol["hunter_seed"]),
                "--checkpoint-sha", policy_sha,
                "--out", str(target.relative_to(R)),
            ], cwd=R, env=os.environ.copy(), check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        part = read(target)
        assert len(part["episodes"]) == 8
        return part

    tactical_path = out / "tactical_v2.json"
    authority.OUT = B / "cpu_eval_inputs"

    def tactical():
        if not tactical_path.exists():
            authority.tactical(cp / "policy.pt", tactical_path)
        return read(tactical_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        natural_future = pool.submit(prior_cpu.natural, cp)
        tactical_future = pool.submit(tactical)
        hunter_parts = list(pool.map(hunter, [
            (tier, start) for tier in ["hard", "normal", "easy"]
            for start in range(0, 64, 8)
        ]))
        natural_future.result()
        tactical_result = tactical_future.result()

    from scripts.je08_bun_authority import summarize_hunter
    hunter_summary = {
        tier: summarize_hunter([
            episode for start in range(0, 64, 8)
            for episode in read(out / f"hunter_{tier}_{start:03d}.json")["episodes"]
        ])
        for tier in ["hard", "normal", "easy"]
    }
    atomic_json(out / "COMPLETE.json", dict(
        passed=True,
        policy_sha256=policy_sha,
        protocol_sha256=sha(B / "CPU_EVAL_PROTOCOL.json"),
        hunter=hunter_summary,
        tactical=tactical_result["settled_style_summary"],
        natural=read(out / "natural_full64.json"),
        engines_separate=True,
        controls_gpu=False,
        holdout_not_used_for_tuning=True,
        unix=time.time(),
    ))
    receipt = read(out / "COMPLETE.json")
    return dict(level="full", checkpoint=request["checkpoint"],
                seed=request["seed"], new_update=request["new_update"],
                policy_sha256=request["checkpoint_policy_sha256"],
                natural_games=receipt["natural"]["results"]["natural_full2400"].get("games"),
                natural_delivery_episodes=receipt["natural"]["results"]["natural_full2400"].get("delivery_episodes"),
                tactical=receipt["tactical"], hunter=receipt["hunter"],
                controls_gpu=False)


def holdout_eval(cp: Path, request: dict):
    import scripts.ppo_scale_longrun_cpu as prior_cpu
    prior_cpu.B = B
    prior_cpu.R = R
    holdout = cp / "holdout"
    (holdout / "retention").mkdir(parents=True, exist_ok=True)
    model = holdout / "retention" / "model"
    if not (model / "model.json").exists():
        from scripts.retention_phase_export import authority_export
        authority_export(cp / "policy.pt", model)
    if not (holdout / "retention" / "natural_full64.json").exists():
        prior_cpu.natural(
            holdout,
            seed_start=read(B / "CPU_EVAL_PROTOCOL.json")["holdout_seed_start"],
        )
    source = holdout / "retention" / "natural_full64.json"
    target = cp / "retention" / "natural_holdout64.json"
    if not target.exists():
        shutil.copyfile(source, target)
    result = read(target)
    summary = result["results"]["natural_full2400"]
    atomic_json(cp / "retention" / "HOLDOUT_COMPLETE.json", dict(
        passed=True,
        policy_sha256=request["checkpoint_policy_sha256"],
        seed_start=read(B / "CPU_EVAL_PROTOCOL.json")["holdout_seed_start"],
        games=64,
        delivery_episodes=summary.get("delivery_episodes"),
        death_events=summary.get("death"),
        no_holdout_tuning=True,
        controls_gpu=False,
    ))
    return dict(level="holdout", checkpoint=request["checkpoint"],
                seed=request["seed"], new_update=request["new_update"],
                policy_sha256=request["checkpoint_policy_sha256"],
                games=summary.get("games"),
                delivery_episodes=summary.get("delivery_episodes"),
                deaths=summary.get("death"), controls_gpu=False)


def consume(request_path: Path):
    request = read(request_path)
    cp = R / request["checkpoint"]
    receipt = read(cp / "COMMITTED.json")
    assert sha(cp / "state.pkl") == request["checkpoint_state_sha256"] == receipt["state_sha256"]
    assert sha(cp / "policy.pt") == request["checkpoint_policy_sha256"] == receipt["policy_sha256"]
    if request["level"] == "full":
        result = full_eval(cp, request)
    elif request["level"] == "light":
        result = light_eval(cp, request)
    else:
        raise ValueError(f"unknown evaluation level: {request['level']}")
    if request.get("holdout"):
        result["holdout"] = holdout_eval(cp, request)
    atomic_json(B / "eval_done" / f"{request_path.stem}.json",
                dict(passed=True, request=request, result=result, unix=time.time()))
    atomic_json(B / "CPU_STATUS.json", dict(
        stage="EVALUATED", checkpoint=request["checkpoint"],
        seed=request["seed"], new_update=request["new_update"],
        level=request["level"], controls_gpu=False, unix=time.time(),
    ))
    notify(f"R2异步评估完成 {request['checkpoint']} ({request['level']})；CPU/JS未阻塞GPU训练。")


def pending_requests():
    done = {p.stem for p in (B / "eval_done").glob("*.json")}
    return [p for p in sorted((B / "eval_queue").glob("*.json"))
            if p.stem not in done]


def owner():
    unique = lock(B / "cpu_evaluator.lock")
    import fcntl
    global_lock = R / "runs/je10_bun_20261018/evaluation.lock"
    with global_lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        atomic_json(B / "CPU_OWNER.json", dict(unique=True, controls_gpu=False,
                                                 started_unix=time.time()))
        failures = 0
        while not (B / "CPU_STOP.json").exists():
            todo = pending_requests()
            if todo:
                request_path = todo[0]
                atomic_json(B / "CPU_ACTIVE.json", dict(
                    unique_owner=True,
                    request=request_path.name,
                    controls_gpu=False,
                    max_concurrent_checkpoints=1,
                    started_unix=time.time(),
                ))
                try:
                    consume(request_path)
                    failures = 0
                    continue
                except Exception as exc:
                    failures += 1
                    atomic_json(B / "CPU_RETRY.json", dict(
                        request=request_path.name,
                        failures=failures,
                        reason=str(exc).replace(str(R), "<WORKTREE>"),
                        controls_gpu=False,
                        next_retry_seconds=min(300, 15 * failures),
                        unix=time.time(),
                    ))
                    notify(f"R2 CPU/JS评估技术问题（不影响GPU）：{request_path.name}，保留请求并重试。")
                    time.sleep(min(300, 15 * failures))
                    continue
            atomic_json(B / "CPU_STATUS.json", dict(
                stage="WAITING_FOR_SAVED_CHECKPOINTS",
                queued_requests=len(list((B / "eval_queue").glob("*.json"))),
                pending_requests=0,
                controls_gpu=False,
                unix=time.time(),
            ))
            complete = B / "TRAINING_COMPLETE.json"
            if complete.exists():
                seeds = read(B / "PROTOCOL.json")["paired_seeds"]
                expected = [f"s{seed}_u{read(B / 'PROTOCOL.json')['updates']:06d}"
                            for seed in seeds]
                if all((B / "eval_done" / f"{stem}.json").exists() for stem in expected):
                    atomic_json(B / "CPU_EVALUATION_COMPLETE.json", dict(
                        passed=True,
                        completed_requests=len(list((B / "eval_queue").glob("*.json"))),
                        final_holdouts=len(seeds),
                        controls_gpu=False,
                        no_holdout_tuning=True,
                        unix=time.time(),
                    ))
                    notify("R2全部声明CPU/JS评估和三seed最终holdout完成；holdout仅报告，未参与调参。")
                    atomic_json(B / "CPU_OWNER_EXIT.json", dict(
                        unix=time.time(), controls_gpu=False,
                        completed=True, unique_owner=True))
                    return
            if (B / "USER_STOP.json").exists() and not todo:
                atomic_json(B / "CPU_PAUSED.json", dict(
                    passed=True, saved_requests_remaining=0, controls_gpu=False,
                    unix=time.time()))
                return
            time.sleep(10)
    atomic_json(B / "CPU_OWNER_EXIT.json", dict(unix=time.time(), controls_gpu=False))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        owner()
    elif sys.argv[1] == "evaluate":
        consume(Path(sys.argv[2]))
    else:
        raise SystemExit(sys.argv[1])
