#!/usr/bin/env python3
"""Prepare a clean, immutable Bun safe-aggression run (1..360 cycles)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb.bun_seed_namespace import critic_seed_plan_v7
from qqt_rl.training.config import EVAL_CYCLES, validate_cycle_count
from qqt_rl.training.io import atomic_copy, atomic_write_json, sha256_file

UPDATES_PER_CYCLE = 4
TRANSITIONS_PER_UPDATE = 4096
BOOTSTRAP_FILES = (
    "base_actor.pt",
    "base_critic.pkl",
    "base_target_critic.pkl",
    "reference_actor.pt",
    "bc_v2_escape_selective_bc.npz",
    "generic_critic_replay_aux.npz",
    "weak_actor.pt",
    "old_actor.pt",
)
RUNTIME_SCRIPTS = (
    "run_bun_tactical_v2_candidate.py",
    "launch_bun_safe_aggression_v7.py",
    "finalize_bun_safe_aggression_v7.py",
    "generate_bun_tactical_v2_critic_data.py",
    "concat_bun_critic_replay.py",
    "train_bun_critic.py",
    "adapt_bun_critic_onpolicy.py",
    "train_bun_bc.py",
    "train_bun_separate_ac.py",
    "eval_bun_tactical_opponent.py",
    "generate_bun_critic_data.py",
    "augment_bun_critic_aux_labels.py",
)


def copy_runtime(destination: Path) -> None:
    shutil.copytree(ROOT / "jax_bomb", destination / "jax_bomb",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "qqt_rl", destination / "qqt_rl",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "web/assets/maps", destination / "web/assets/maps")
    (destination / "scripts").mkdir(parents=True)
    for name in RUNTIME_SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, destination / "scripts" / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="bun_safe_aggression")
    parser.add_argument("--cycles", type=int, default=360)
    parser.add_argument("--segment-size", type=int, default=90)
    parser.add_argument("--gpus", default="0,1,2,3")
    return parser.parse_args()


def build_common_config(cycles: int, segment_size: int) -> dict:
    return dict(
        initial_bc="bc_v2_escape", initial_actor="base_actor.pt",
        initial_critic="base_critic.pkl",
        initial_target_critic="base_target_critic.pkl",
        reference_actor="reference_actor.pt", actor_lr=1e-5, critic_lr=3e-5,
        entropy=0.003, kl=0.05, bc_coef=0.12, target_tau=0.04,
        counterfactual_coef=0.30, danger_escape_reward=0.75,
        avoidable_penalty=4.0, critic_epochs=20, critic_calibration_lr=1e-4,
        cf_train_states=2, cf_mc_samples=2, cf_horizon=40,
        family_curriculum=True, historical_mix=False, use_bc=True,
        aux_coef=0.20, total_cycles=cycles, segment_size=segment_size,
        paired_eval_seed=202609280000,
        transitions_per_update=TRANSITIONS_PER_UPDATE,
        critic_seed_namespace_version="v7",
        bot_sha256=sha256_file(ROOT / "jax_bomb/bun_rule_bot.py"),
    )


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.run_name):
        raise SystemExit("run-name must be a simple 1..80 character directory name")
    cycles = validate_cycle_count(args.cycles)
    if args.segment_size < 1 or args.segment_size > cycles:
        raise SystemExit("segment-size must be in [1, cycles]")
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit("safe-aggression baseline requires exactly four GPU ids")

    bootstrap = args.bootstrap_dir.resolve()
    missing = [name for name in BOOTSTRAP_FILES if not (bootstrap / name).is_file()]
    if missing:
        raise SystemExit(f"missing bootstrap files: {', '.join(missing)}")

    run_root = ROOT / "runs" / args.run_name
    checkpoint_root = ROOT / "checkpoints" / args.run_name
    data_root = ROOT / "data" / args.run_name
    frozen_root = run_root / "frozen_runtime"
    if any(path.exists() for path in (run_root, checkpoint_root, data_root)):
        raise SystemExit("refusing to reuse an existing run/data/checkpoint path")

    copy_runtime(frozen_root)
    inputs = run_root / "frozen_inputs"
    inputs.mkdir(parents=True)
    for name in BOOTSTRAP_FILES:
        shutil.copy2(bootstrap / name, inputs / name)

    common = build_common_config(cycles, args.segment_size)
    arm_specs = (
        ("control", 202609280001, 0.0, 0.0),
        ("safe_low", 202609280101, 0.05, 0.35),
        ("safe_medium", 202609280201, 0.10, 0.75),
        ("safe_high", 202609280301, 0.20, 1.50),
    )
    candidates = []
    for slot, ((name, seed, placement, resolution), gpu) in enumerate(zip(arm_specs, gpu_ids)):
        # 准备阶段扫描完整 horizon；runner 生成每个 cycle 数据后仍会严格检查
        # train/validation/test 集合交叉，禁止 raw seed 与 uint32 seed 泄漏。
        for cycle in range(1, cycles + 1):
            critic_seed_plan_v7(slot, seed, cycle, train_states=2)
        candidate_dir = run_root / name
        candidate_dir.mkdir(parents=True)
        config_path = candidate_dir / "config.v1.json"
        transition_budget = cycles * UPDATES_PER_CYCLE * TRANSITIONS_PER_UPDATE
        config = dict(
            common, name=name, candidate_slot=slot, gpu=gpu, seed=seed,
            tactical_bomb_placement_reward=placement,
            tactical_bomb_resolution_reward=resolution,
            reward_variant=f"placement={placement},resolution={resolution}",
            required_tactical_transitions=transition_budget,
            total_transition_budget=transition_budget,
            tactical_transition_budget=transition_budget,
            config=str(config_path),
        )
        atomic_write_json(config_path, config)
        candidate_checkpoints = checkpoint_root / name
        atomic_copy(inputs / "base_actor.pt", candidate_checkpoints / "cycle_000_actor.pt")
        atomic_copy(inputs / "base_critic.pkl", candidate_checkpoints / "cycle_000_critic.pkl")
        atomic_copy(inputs / "base_target_critic.pkl",
                    candidate_checkpoints / "cycle_000_target_critic.pkl")
        candidates.append({**config, "config_sha256": sha256_file(config_path)})

    data_root.mkdir(parents=True)
    segments = [
        [start, min(start + args.segment_size - 1, cycles)]
        for start in range(1, cycles + 1, args.segment_size)
    ]
    manifest = {
        "schema": "qqt_rl_bun_safe_aggression_manifest_v1",
        "created_unix": time.time(),
        "repo_root": str(ROOT),
        "run_root": str(run_root),
        "checkpoint_root": str(checkpoint_root),
        "data_root": str(data_root),
        "frozen_root": str(frozen_root),
        "python": sys.executable,
        "target_cycles": cycles,
        "segment_size": args.segment_size,
        "segments": segments,
        "updates_per_cycle": UPDATES_PER_CYCLE,
        "transitions_per_update": TRANSITIONS_PER_UPDATE,
        "milestones": sorted(cycle for cycle in EVAL_CYCLES if cycle <= cycles),
        "max_attempts": 8,
        "strict_seed_leakage_check": True,
        "candidates": candidates,
        "bootstrap": [
            {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(inputs.iterdir())
        ],
        "frozen_files": [
            {"relative_path": str(path.relative_to(frozen_root)),
             "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(frozen_root.rglob("*")) if path.is_file()
        ],
    }
    manifest_path = run_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "cycles": cycles,
                      "candidates": [row["name"] for row in candidates]}, indent=2))


if __name__ == "__main__":
    main()
