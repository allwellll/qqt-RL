#!/usr/bin/env python3
"""Single-GPU sequential reward-v2 coefficient sweep.

Trains a Control arm plus reward-coefficient candidate arms from an IDENTICAL
base bootstrap checkpoint, with an IDENTICAL pre-declared learner seed and
candidate_slot, varying ONLY the danger_arena reward coefficients. Each arm is
then evaluated by the runner's fixed-seed harness. Because seed / slot / base
checkpoint / budget / opponent are all held constant, each (Control, candidate)
pair is atomic — the sole free variable is the reward shaping.

The heavy per-cycle recipe (counterfactual data + critic calibration + joint
micro updates + fixed-seed eval) is reused unchanged via
``run_bun_tactical_v2_candidate.py`` so the sweep is directly comparable to the
eventual 4-GPU production run. Arms run sequentially on one GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import prepare_bun_safe_aggression_v7 as prep
from qqt_rl.training.io import atomic_write_json, sha256_file

# 预声明种子与 slot 对所有 arm 完全一致 —— critic_seed_base_v7 只取决于
# (slot, cycle)，故同 slot+seed 的两条 arm 产生逐位一致的 rollout / critic 数据，
# 唯一变量是 reward 系数，从而构成原子对照。
SWEEP_SEED = 202609280000
SLOT = 0

# 缩减首轮：forced_kill=2.0、resolution=1.0 固定，扫 base_bomb × placement。
REDUCED_ARMS = (
    ("control", 0.0, 0.0, 0.0, 0.0),
    ("p03_b03", 0.3, 1.0, 0.03, 2.0),
    ("p03_b06", 0.3, 1.0, 0.06, 2.0),
    ("p03_b10", 0.3, 1.0, 0.10, 2.0),
    ("p06_b03", 0.6, 1.0, 0.03, 2.0),
    ("p06_b06", 0.6, 1.0, 0.06, 2.0),
    ("p06_b10", 0.6, 1.0, 0.10, 2.0),
)


def build_arm_config(common: dict, name: str, gpu: int, placement: float,
                     resolution: float, base_bomb: float, forced_kill: float,
                     cycles: int, config_path: Path) -> dict:
    transition_budget = cycles * prep.UPDATES_PER_CYCLE * prep.TRANSITIONS_PER_UPDATE
    return dict(
        common, name=name, candidate_slot=SLOT, gpu=gpu, seed=SWEEP_SEED,
        tactical_bomb_placement_reward=placement,
        tactical_bomb_resolution_reward=resolution,
        base_bomb_reward=base_bomb,
        forced_kill_reward=forced_kill,
        reward_variant=(
            f"placement={placement},resolution={resolution},"
            f"base_bomb={base_bomb},forced_kill={forced_kill}"),
        required_tactical_transitions=transition_budget,
        total_transition_budget=transition_budget,
        tactical_transition_budget=transition_budget,
        config=str(config_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir", type=Path,
        default=Path("/mnt/jpfs/afs/wangyaqi/code_room/qqt-RL/runs/"
                     "bun_high_throughput_20260929/frozen_inputs"),
        help="dir holding base_actor.pt / base_critic.pkl / etc.")
    parser.add_argument("--run-name",
                        default=f"reward_v2_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=8,
                        help="per-arm training budget in cycles (1..360)")
    parser.add_argument("--counterfactual-state-batch-size", type=int, default=2)
    parser.add_argument("--critic-batch-size", type=int, default=64)
    parser.add_argument(
        "--python", type=Path,
        default=Path("/mnt/jpfs/afs/wangyaqi/code_room/qqt-gpu-sim/.venv/bin/python"),
        help="CUDA-enabled interpreter for the worker/eval subprocess "
             "(the sweep driver itself may run under a CPU-only venv)")
    args = parser.parse_args()

    worker_python = args.python.expanduser().absolute()
    if not worker_python.is_file():
        raise SystemExit(f"--python not found: {worker_python}")

    if args.cycles not in range(1, 361):
        raise SystemExit("--cycles must be in [1, 360]")
    bootstrap = args.bootstrap_dir.resolve()
    missing = [name for name in prep.BOOTSTRAP_FILES
               if not (bootstrap / name).is_file()]
    if missing:
        raise SystemExit(f"bootstrap dir missing files: {missing}")

    base = (ROOT / "runs" / args.run_name).resolve()
    if base.exists():
        raise SystemExit(f"refusing to reuse existing run path: {base}")
    frozen = base / "frozen_runtime"
    inputs = base / "frozen_inputs"
    run_root = base / "run"
    ckpt_root = base / "checkpoints"
    data_root = base / "data"
    cache_root = base / "cache"
    for path in (run_root, ckpt_root, data_root, cache_root):
        path.mkdir(parents=True)

    prep.copy_runtime(frozen)
    inputs.mkdir(parents=True)
    for name in prep.BOOTSTRAP_FILES:
        shutil.copy2(bootstrap / name, inputs / name)

    common = prep.build_common_config(
        args.cycles, args.cycles,
        jax_cache_dir=str(cache_root / "jax"),
        counterfactual_state_batch_size=args.counterfactual_state_batch_size,
        critic_batch_size=args.critic_batch_size,
        persistent_worker=True)

    environment = os.environ.copy()
    environment.update({
        "QQT_REPO_ROOT": str(ROOT),
        "BUN_V2_RUN_ROOT": str(run_root),
        "BUN_V2_CKPT_ROOT": str(ckpt_root),
        "BUN_V2_DATA_ROOT": str(data_root),
        "BUN_V2_FROZEN_ROOT": str(frozen),
        "BUN_V2_INPUT_ROOT": str(inputs),
        "BUN_V2_PYTHON": str(worker_python),
        "PYTHONPATH": str(frozen),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
    })

    arms = []
    for name, placement, resolution, base_bomb, forced_kill in REDUCED_ARMS:
        candidate_dir = run_root / name
        candidate_dir.mkdir(parents=True)
        config_path = candidate_dir / "config.v1.json"
        config = build_arm_config(
            common, name, args.gpu, placement, resolution, base_bomb,
            forced_kill, args.cycles, config_path)
        atomic_write_json(config_path, config)
        arms.append({"name": name, "config": str(config_path),
                     "reward_variant": config["reward_variant"],
                     "config_sha256": sha256_file(config_path)})

    manifest = {
        "schema": "reward_v2_coefficient_sweep_v1",
        "run_name": args.run_name, "run_root": str(run_root),
        "checkpoint_root": str(ckpt_root), "data_root": str(data_root),
        "frozen_root": str(frozen), "input_root": str(inputs),
        "gpu": args.gpu, "cycles": args.cycles,
        "worker_python": str(worker_python),
        "seed": SWEEP_SEED, "candidate_slot": SLOT,
        "bootstrap_dir": str(bootstrap),
        "bootstrap_hashes": {name: sha256_file(bootstrap / name)
                             for name in prep.BOOTSTRAP_FILES},
        "paired_eval_seed": common["paired_eval_seed"],
        "arms": arms, "started_unix": time.time(),
    }
    atomic_write_json(base / "sweep_manifest.json", manifest)

    runner = frozen / "scripts/run_bun_tactical_v2_candidate.py"
    print(f"[sweep] run_dir={base}", flush=True)
    print(f"[sweep] arms={[a['name'] for a in arms]}", flush=True)
    results = []
    for arm in arms:
        started = time.time()
        log_path = run_root / arm["name"] / "worker.log"
        print(f"[sweep] START arm={arm['name']} "
              f"variant=({arm['reward_variant']}) log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                [str(worker_python), "-P", "-B", "-u", str(runner),
                 "--config", arm["config"]],
                cwd=str(ROOT), env=environment, stdout=log,
                stderr=subprocess.STDOUT)
        status = "complete" if process.returncode == 0 else "failed"
        elapsed = time.time() - started
        print(f"[sweep] END   arm={arm['name']} status={status} "
              f"rc={process.returncode} elapsed={elapsed:.1f}s", flush=True)
        results.append({"name": arm["name"], "status": status,
                        "returncode": process.returncode,
                        "elapsed_seconds": elapsed})
        if process.returncode != 0:
            print(f"[sweep] ABORT: arm {arm['name']} failed; "
                  f"see {log_path}", flush=True)
            break

    atomic_write_json(base / "sweep_results.json", {
        "schema": "reward_v2_coefficient_sweep_results_v1",
        "run_name": args.run_name, "results": results,
        "finished_unix": time.time()})
    print(f"[sweep] DONE results={base / 'sweep_results.json'}", flush=True)


if __name__ == "__main__":
    main()
