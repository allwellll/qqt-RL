#!/usr/bin/env python3
"""Launch and supervise the fixed CX-29 trap-route A/B experiment."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qqt_rl.training.io import atomic_write_json, checkpoint_is_finite, sha256_file  # noqa: E402
from qqt_rl.training.jax_cache import cache_environment  # noqa: E402

INIT_SHA256 = "6e190b009f180f2e449fd19941c930c0d1a86d680071eeb02262aadb6808ebd0"
DEFAULT_EXP = "EXP-CX-20261003-29"
EXPERIMENTS = {
    DEFAULT_EXP: {
        "dir": "cx20261003_29_trap_routes",
        "seeds": [20264001, 20264002, 20264003, 20264004],
        "iters": 101,
        "save_every": 50,
        "smoke_iters": 3,
        "module": "scripts.cx_train_trap_routes",
        "base_flags": ["--reward-shaping-scale", "0.6", "--hunter-fraction", "0.5"],
        "arms": {"control": ["--trap-route-mode", "current"],
                 "trap16": ["--trap-route-mode", "trap16"]},
    },
}
DEPS = [
    "jax_bomb/*.py", "qqt_rl/training/*.py", "web/sim.js", "web/bun_hunter_bot.js",
    "scripts/cx_train_frozen.py", "scripts/cx_train_opponents.py",
    "scripts/cx_train_terrain.py", "scripts/cx_train_clear_opponents.py",
    "scripts/cx_train_reward_scale.py", "scripts/cx_train_ema.py",
    "scripts/cx_train_kill_reward.py", "scripts/cx_train_value_projection.py",
    "scripts/cx_train_trap_routes.py", "scripts/cx_counterbomb_bot.py",
    "scripts/cx_trap_route_bot.py", "scripts/eval_four_bot_matrix.py",
    "scripts/eval_web_hunter.js", "scripts/score_eval_screen.py",
    "scripts/cx_screen_round.py", "scripts/cx_confirm_round.py", "scripts/launch_r1_ab.py",
]
BASE_FLAGS = [
    "--devices", "1", "--arch", "transformer", "--embed", "192", "--depth", "4",
    "--num-envs", "256", "--num-steps", "64", "--minibatch", "512", "--epochs", "1",
    "--flee-bot-ratio", "0.3",
    "--jax-bot-pool", "dodge=1,bomber_easy=2,hunter=2,hunter_hard=1,legacy_flee=1",
    "--jax-bot-adaptive", "--bun-curriculum", "danger_arena=1", "--bun-reward-profile", "danger_arena",
    "--bun-base-bomb-reward", "0.06", "--bun-tactical-bomb-placement-reward", "0.6",
    "--bun-forced-kill-reward", "2.0", "--bun-tactical-bomb-resolution-reward", "1.0",
    "--bun-enemy-threat-reward", "0.4",
    "--bun-spawn-buckets", "native=0.3,near=0.1,mid=0.15,far=0.1,below=0.15,upper_left=0.1,upper_right=0.1",
]
POLL_SECONDS = 20.0
ITER_RE = re.compile(r"\[iter (\d+)\]\s+[\d.]+s\s+sps=([\d,]+)")


def plan(exp=DEFAULT_EXP):
    e = EXPERIMENTS[exp]
    return [(arm, seed, gpu) for gpu, (arm, seed) in
            enumerate((arm, seed) for arm in e["arms"] for seed in e["seeds"])]


def train_argv(py, arm, seed, init, run_dir, iters, save_every, exp=DEFAULT_EXP):
    e = EXPERIMENTS[exp]
    return [py, "-m", e["module"], *e.get("base_flags", BASE_FLAGS), "--iters", str(iters),
            "--save-every", str(save_every), "--seed", str(seed), *e["arms"][arm],
            "--load", str(init), "--save", str(Path(run_dir) / "ckpt" / "final.pt")]


def freeze_deps():
    files = sorted({path for pattern in DEPS for path in ROOT.glob(pattern) if path.is_file()})
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout
    return {"files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
            "git_head": git("rev-parse", "HEAD").strip(), "git_status": git("status", "--porcelain"),
            "git_diff": git("diff", "HEAD")}


def prepare(run_dir, arm, seed, gpu, init, iters, save_every, deps, exp=DEFAULT_EXP):
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run namespace {run_dir}")
    (run_dir / "ckpt").mkdir(parents=True)
    argv = train_argv(os.environ.get("QQT_PY", sys.executable), arm, seed, init, run_dir, iters, save_every, exp)
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "JAXBOMB_RULE": "bun", "PYTHONPATH": str(ROOT),
           "PYTHONUNBUFFERED": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", **cache_environment(run_dir)}
    (run_dir / "git_diff.patch").write_text(deps["git_diff"])
    manifest = {"experiment": exp, "arm": arm, "seed": seed, "gpu": gpu, "arm_flags": EXPERIMENTS[exp]["arms"][arm],
                "iters": iters, "save_every": save_every, "init_sha256": sha256_file(init), "argv": argv, "env": env,
                "git_head": deps["git_head"], "git_status": deps["git_status"], "deps_sha256": deps["files"],
                "source_root": str(ROOT), "interpreter": argv[0], "interpreter_realpath": str(Path(argv[0]).resolve())}
    atomic_write_json(run_dir / "manifest.json", manifest)
    (run_dir / "command.txt").write_text(" ".join(f"{key}={value}" for key, value in env.items()) + " " + " ".join(argv) + "\n")
    return manifest


def start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _progress(log):
    last = None
    if log.exists():
        for match in ITER_RE.finditer(log.read_text(errors="replace")[-20000:]):
            last = {"iter": int(match.group(1)), "sps": int(match.group(2).replace(",", ""))}
    return last


def _check_ckpts(run_dir, checked, final=False):
    for point in sorted((Path(run_dir) / "ckpt").glob("*.pt")):
        if point.name not in checked and (final or point.with_suffix(".json").exists()):
            checked[point.name] = {"sha256": sha256_file(point), "finite": bool(checkpoint_is_finite(point))}
    return all(result["finite"] for result in checked.values())


def supervise(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    log, status_path = run_dir / "train.log", run_dir / "status.json"
    status = {"state": "starting", "supervisor_pid": os.getpid(), "started": time.time(), "checkpoints": {}}
    with open(log, "ab") as output:
        proc = subprocess.Popen(manifest["argv"], cwd=ROOT, env={**os.environ, **manifest["env"]}, stdout=output,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    status.update(state="running", trainer_pid=proc.pid, trainer_start_ticks=start_ticks(proc.pid))
    atomic_write_json(status_path, status)
    finite = True
    while (rc := proc.poll()) is None:
        time.sleep(POLL_SECONDS)
        finite = _check_ckpts(run_dir, status["checkpoints"]) and finite
        status.update(progress=_progress(log), updated=time.time())
        if not finite:
            proc.terminate()
            status["state"] = "stopping_nonfinite"
        atomic_write_json(status_path, status)
    finite = _check_ckpts(run_dir, status["checkpoints"], final=True) and finite
    ok = rc == 0 and finite and (run_dir / "ckpt" / "final.pt").exists()
    status.update(state="complete" if ok else "failed", rc=rc, finite=finite, progress=_progress(log), ended=time.time())
    atomic_write_json(status_path, status)
    (run_dir / ("COMPLETE" if ok else "FAILED")).write_text(json.dumps({"rc": rc, "finite": finite}) + "\n")
    return 0 if ok else 1


def spawn_supervisor(run_dir):
    with open(Path(run_dir) / "supervisor.log", "ab") as output:
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "supervise", str(run_dir)], cwd=ROOT,
                                stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True).pid


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--exp", choices=sorted(EXPERIMENTS), default=DEFAULT_EXP)
        command.add_argument("--arm")
        command.add_argument("--init", default=os.environ.get("QQT_INIT_CKPT"))
        command.add_argument("--out")
        command.add_argument("--gpu", type=int, default=0)
        command.add_argument("--wait", action="store_true")
    sub.add_parser("supervise").add_argument("run_dir")
    sub.add_parser("status").add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "supervise":
        return supervise(args.run_dir)
    if args.cmd == "status":
        for status_path in sorted(Path(args.out).glob("*/status.json")):
            status = json.loads(status_path.read_text())
            print(status_path.parent.name, status["state"], status.get("trainer_pid"), status.get("progress"), status.get("rc"))
        return 0
    init = Path(args.init or "").resolve()
    if not init.is_file() or sha256_file(init) != INIT_SHA256:
        raise SystemExit(f"init checkpoint missing or SHA256 != {INIT_SHA256}")
    experiment = EXPERIMENTS[args.exp]
    out, deps = Path(args.out or ROOT / "runs" / experiment["dir"]).resolve(), freeze_deps()
    if args.cmd == "smoke":
        arm = args.arm or next(iter(experiment["arms"]))
        jobs = [(out / f"smoke_{arm}_s{experiment['seeds'][0]}", arm, experiment["seeds"][0], args.gpu,
                 experiment["smoke_iters"], 1)]
    else:
        jobs = [(out / f"{arm}_s{seed}", arm, seed, gpu, experiment["iters"], experiment["save_every"])
                for arm, seed, gpu in plan(args.exp)]
    pids = {}
    for run_dir, arm, seed, gpu, iters, save_every in jobs:
        prepare(run_dir, arm, seed, gpu, init, iters, save_every, deps, args.exp)
        pids[run_dir.name] = spawn_supervisor(run_dir)
    atomic_write_json(out / f"launch_{args.cmd}_{int(time.time())}.json", {"supervisors": pids})
    print(json.dumps(pids))
    if args.wait:
        while not all(any((out / name / marker).exists() for marker in ("COMPLETE", "FAILED")) for name in pids):
            time.sleep(5)
        return int(any((out / name / "FAILED").exists() for name in pids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
