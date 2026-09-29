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
from qqt_rl.bots import create_default_registry
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
ARM_NAMES = ("control", "safe_low", "safe_medium", "safe_high")
ARM_INIT_FILES = {
    "actor": "latest_actor.pt",
    "critic": "latest_critic.pkl",
    "target_critic": "latest_target_critic.pkl",
}
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
    "bun_training_worker.py",
)


def resolve_arm_init(init_run: Path, arm: str) -> dict[str, Path]:
    """将某个 arm 的 latest_{actor,critic,target_critic} 解析为源路径。

    用于新 lineage 各档续自旧 run 同名 arm 的最新 checkpoint（warm-start）。
    任一缺失即拒绝，避免用错误起点污染新实验。
    """
    base = Path(init_run) / arm
    resolved = {key: base / name for key, name in ARM_INIT_FILES.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing per-arm init checkpoints: {', '.join(missing)}")
    return resolved


def copy_runtime(destination: Path) -> None:
    shutil.copytree(ROOT / "jax_bomb", destination / "jax_bomb",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "qqt_rl", destination / "qqt_rl",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "web/assets/maps", destination / "web/assets/maps")
    # create_default_registry 运行时按 frozen-runtime 根解析该 fixture 计算 bot 指纹哈希，
    # 故必须随运行时一起冻结，否则 eval/生成阶段启动即缺文件。
    fixture = destination / "tests/fixtures/bot_contract_cases.json"
    fixture.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "tests/fixtures/bot_contract_cases.json", fixture)
    (destination / "scripts").mkdir(parents=True)
    for name in RUNTIME_SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, destination / "scripts" / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--init-from-run", type=Path,
                        help="旧 run production 目录；各档续自其同名 arm 的 "
                             "latest_{actor,critic,target_critic}（warm-start 新实验）")
    parser.add_argument("--run-name", default="bun_safe_aggression")
    parser.add_argument("--cycles", type=int, default=360)
    parser.add_argument("--segment-size", type=int, default=90)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--jax-cache-dir", type=Path)
    parser.add_argument("--counterfactual-state-batch-size", type=int, default=1)
    parser.add_argument("--critic-batch-size", type=int, default=64)
    parser.add_argument("--no-persistent-worker", action="store_true")
    return parser.parse_args()


def build_common_config(cycles: int, segment_size: int, *,
                        jax_cache_dir: str | None = None,
                        counterfactual_state_batch_size: int = 1,
                        critic_batch_size: int = 64,
                        persistent_worker: bool = True) -> dict:
    bot_record = create_default_registry("python").describe(
        "bun.tactical_v2", {})
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
        opponent_bot=bot_record,
        opponent_bot_config_source="family",
        counterfactual_engine="batched",
        counterfactual_state_batch_size=counterfactual_state_batch_size,
        critic_batch_size=critic_batch_size,
        persistent_worker=persistent_worker,
        jax_cache_dir=jax_cache_dir,
        jax_cache_min_compile_time_secs=0.0,
        jax_cache_min_entry_size_bytes=0,
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
    if args.counterfactual_state_batch_size < 1:
        raise SystemExit("counterfactual-state-batch-size must be positive")
    if args.critic_batch_size not in (64, 128, 256):
        raise SystemExit("critic-batch-size must be one of 64,128,256")

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

    # 各档续自旧 run 同名 arm 的 latest checkpoint（warm-start）。拷入 frozen_inputs
    # 并以 <arm>_{actor,critic,target_critic} 命名，manifest 的 bootstrap 段自动登记其
    # 哈希；每档 config 的 initial_* 指向自己的文件。未指定则回退单一 base_* 起点。
    arm_init: dict[str, dict] = {}
    if args.init_from_run:
        init_run = args.init_from_run.resolve()
        for arm in ARM_NAMES:
            resolved = resolve_arm_init(init_run, arm)
            filenames, provenance = {}, {}
            for role, source in resolved.items():
                suffix = ARM_INIT_FILES[role].split("latest_", 1)[1]
                target_name = f"{arm}_{suffix}"
                shutil.copy2(source, inputs / target_name)
                filenames[role] = target_name
                provenance[role] = {"source": str(source),
                                    "sha256": sha256_file(source)}
            arm_init[arm] = {"filenames": filenames, "provenance": provenance}

    common = build_common_config(
        cycles, args.segment_size,
        jax_cache_dir=(str(args.jax_cache_dir.expanduser().resolve())
                       if args.jax_cache_dir else None),
        counterfactual_state_batch_size=args.counterfactual_state_batch_size,
        critic_batch_size=args.critic_batch_size,
        persistent_worker=not args.no_persistent_worker)
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
        init_overrides = {}
        if name in arm_init:
            init_overrides = {
                "initial_actor": arm_init[name]["filenames"]["actor"],
                "initial_critic": arm_init[name]["filenames"]["critic"],
                "initial_target_critic": arm_init[name]["filenames"]["target_critic"],
                "init_provenance": arm_init[name]["provenance"],
                "init_scheme": "per_arm_latest_warm_start",
            }
        config = dict(
            common, name=name, candidate_slot=slot, gpu=gpu, seed=seed,
            tactical_bomb_placement_reward=placement,
            tactical_bomb_resolution_reward=resolution,
            reward_variant=f"placement={placement},resolution={resolution}",
            required_tactical_transitions=transition_budget,
            total_transition_budget=transition_budget,
            tactical_transition_budget=transition_budget,
            config=str(config_path), **init_overrides,
        )
        atomic_write_json(config_path, config)
        candidate_checkpoints = checkpoint_root / name
        actor_src = inputs / config["initial_actor"]
        critic_src = inputs / config["initial_critic"]
        target_src = inputs / config["initial_target_critic"]
        atomic_copy(actor_src, candidate_checkpoints / "cycle_000_actor.pt")
        atomic_copy(critic_src, candidate_checkpoints / "cycle_000_critic.pkl")
        atomic_copy(target_src, candidate_checkpoints / "cycle_000_target_critic.pkl")
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
        "init_scheme": ("per_arm_latest_warm_start" if arm_init
                        else "single_base"),
        "init_from_run": (str(args.init_from_run.resolve())
                          if args.init_from_run else None),
        "bot_registry": create_default_registry("python").list(),
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
