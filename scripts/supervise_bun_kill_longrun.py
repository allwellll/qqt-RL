#!/usr/bin/env python3
"""Durable, restart-safe supervisor for the 4-GPU DP kill-ability long run.

Single-instance (flock), detached (its own session), and restart-safe: on a
technical crash it resumes from the newest `actor_it{N}.pt` checkpoint in the
run directory (via `--load` + `--iter-offset N` + remaining iters), so global
checkpoint numbering (it200, it400, ... it1400) stays contiguous and existing
history is never overwritten. Completion = final `actor.pt` present.

Training path is the canonical device-side JAX/JIT pmap DP self-play
(`python -m jax_bomb.bun_train --devices >1`). NO Python rule bot, NO host
callback, NO reward-v2 — the DP path only ever uses the device-side
`flee_bot_actions` anchor via `--flee-bot-ratio` and the legacy reward profile.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqt_rl.training.io import atomic_write_json

_CKPT_RE = re.compile(r"actor_it(\d+)\.pt$")


def latest_checkpoint(run_root: Path) -> tuple[Path | None, int]:
    """Return (path, global_iter) of the newest actor_it{N}.pt, else (None, 0)."""
    best_n, best_path = 0, None
    for path in run_root.glob("actor_it*.pt"):
        match = _CKPT_RE.search(path.name)
        if match:
            n = int(match.group(1))
            if n > best_n:
                best_n, best_path = n, path
    return best_path, best_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--devices-list", default="0,1,2,3")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1600)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--arch", default="mlp4")
    parser.add_argument("--bun-hp", type=int, default=1)
    parser.add_argument("--bun-curriculum",
                        default="combat_static=0.2,combat_moving=0.2,"
                                "combat_kill=0.3,combat=0.3")
    parser.add_argument("--reward-profile", default="legacy")
    parser.add_argument("--flee-bot-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--obs-quant", action="store_true", default=True)
    parser.add_argument("--max-attempts", type=int, default=6)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    warm_start = Path(args.warm_start).resolve()
    forbidden = (repo_root / "checkpoints" / "bun_kill_dp_20260929").resolve()
    if run_root == forbidden:
        raise SystemExit("refusing to reuse bun_kill_dp_20260929 as run dir")
    if not run_root.is_relative_to(repo_root):
        raise SystemExit("run-root must live under the repository root")
    if not warm_start.is_file():
        raise SystemExit(f"warm-start checkpoint missing: {warm_start}")
    run_root.mkdir(parents=True, exist_ok=True)
    final_ckpt = run_root / "actor.pt"

    lock_handle = (run_root / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another supervisor already owns this run") from error

    atomic_write_json(run_root / "supervisor.pid.json", {
        "pid": os.getpid(), "run_root": str(run_root),
        "started_unix": time.time(),
    })

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.devices_list,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_COMPILATION_CACHE_DIR": str(run_root / "xla_cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    resolved = {
        "warm_start": str(warm_start), "devices": args.devices,
        "devices_list": args.devices_list, "iters_total": args.iters,
        "save_every": args.save_every, "seed": args.seed,
        "num_envs": args.num_envs, "num_steps": args.num_steps,
        "minibatch": args.minibatch, "epochs": args.epochs, "arch": args.arch,
        "bun_hp": args.bun_hp, "bun_curriculum": args.bun_curriculum,
        "reward_profile": args.reward_profile,
        "flee_bot_ratio": args.flee_bot_ratio, "lr": args.lr,
        "ent_coef": args.ent_coef, "obs_quant": bool(args.obs_quant),
        "opponent": "device-side JIT flee_bot_actions anchor + pure self-play",
        "forbidden": "no python rule bot / no host callback / no reward-v2",
    }

    def publish(phase: str, attempt: int, extra: dict | None = None) -> None:
        _, done_n = latest_checkpoint(run_root)
        atomic_write_json(run_root / "supervisor_state.json", {
            "schema": "bun_kill_longrun_supervisor_v1", "status": phase,
            "supervisor_pid": os.getpid(), "attempt": attempt,
            "latest_checkpoint_iter": done_n,
            "final_present": final_ckpt.is_file(),
            "resolved": resolved, "updated_unix": time.time(),
            **(extra or {}),
        })

    attempt = 0
    try:
        while not final_ckpt.is_file():
            resume_path, done_n = latest_checkpoint(run_root)
            load_path = resume_path if resume_path is not None else warm_start
            remaining = args.iters - done_n
            if remaining <= 0:
                remaining = args.save_every  # finish the tail into final actor.pt
            attempt += 1
            if attempt > args.max_attempts:
                atomic_write_json(run_root / "TECHNICAL_FAILURE.json", {
                    "attempts": attempt - 1, "latest_checkpoint_iter": done_n,
                    "updated_unix": time.time(),
                })
                raise RuntimeError("technical retry budget exhausted")

            cmd = [
                args.python, "-B", "-u", "-m", "jax_bomb.bun_train",
                "--arch", args.arch,
                "--devices", str(args.devices),
                "--num-envs", str(args.num_envs),
                "--num-steps", str(args.num_steps),
                "--minibatch", str(args.minibatch),
                "--epochs", str(args.epochs),
                "--iters", str(remaining),
                "--iter-offset", str(done_n),
                "--flee-bot-ratio", str(args.flee_bot_ratio),
                "--bun-curriculum", args.bun_curriculum,
                "--bun-hp", str(args.bun_hp),
                "--bun-reward-profile", args.reward_profile,
                "--lr", str(args.lr),
                "--ent-coef", str(args.ent_coef),
                "--seed", str(args.seed),
                "--load", str(load_path),
                "--save", str(final_ckpt),
                "--save-every", str(args.save_every),
            ]
            if args.obs_quant:
                cmd.append("--obs-quant")

            log = run_root / f"train_attempt_{attempt:02d}.log"
            output = log.open("a", encoding="utf-8")
            output.write(
                f"\n=== attempt {attempt} resume_from={load_path} "
                f"offset={done_n} remaining={remaining} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            output.flush()
            publish("running", attempt, {"resume_from": str(load_path),
                                         "iter_offset": done_n,
                                         "remaining_iters": remaining,
                                         "log": str(log)})
            process = subprocess.Popen(
                cmd, cwd=repo_root, env=environment, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
            atomic_write_json(run_root / "worker.pid.json", {
                "pid": process.pid, "attempt": attempt,
                "cmd": cmd, "updated_unix": time.time(),
            })
            returncode = process.wait()
            output.close()
            if final_ckpt.is_file():
                break
            if returncode == 0:
                # clean exit but no final ckpt -> loop will resume tail
                continue
            publish("retrying", attempt, {"last_returncode": returncode})
            time.sleep(20)

        publish("complete", attempt)
        atomic_write_json(run_root / "COMPLETE.json", {
            "final": str(final_ckpt), "attempts": attempt,
            "updated_unix": time.time(),
        })
    except BaseException:
        # best-effort stop of the managed training process group
        try:
            record = worker = None
            pid_file = run_root / "worker.pid.json"
            if pid_file.is_file():
                import json
                record = json.loads(pid_file.read_text())
                worker = int(record.get("pid", -1))
            if worker and worker > 0:
                os.killpg(worker, signal.SIGTERM)
        except (ProcessLookupError, OSError, ValueError):
            pass
        publish("failed", attempt)
        raise


if __name__ == "__main__":
    main()
