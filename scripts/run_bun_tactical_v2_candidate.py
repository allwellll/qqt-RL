#!/usr/bin/env python3
"""Run one append-only Ambush learner against frozen tactical-v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from jax_bomb.bun_seed_namespace import V7_SPLIT_STRIDE, critic_seed_base_v7
from qqt_rl.training.config import (
    EVAL_CYCLES,
    FULL_FAMILY,
    HELDOUT_FAMILY,
    UNSEEN_FAMILY,
    family_for_cycle,
    validate_cycle_count,
)
from qqt_rl.training.io import (
    atomic_copy,
    atomic_write_json as atomic,
    checkpoint_is_finite as finite,
    sha256_file as digest,
)
from qqt_rl.training.process import run_logged

REPO = Path(os.environ.get("QQT_REPO_ROOT", Path.cwd())).resolve()
RUN = Path(os.environ["BUN_V2_RUN_ROOT"]).resolve()
CKPT = Path(os.environ["BUN_V2_CKPT_ROOT"]).resolve()
DATA = Path(os.environ["BUN_V2_DATA_ROOT"]).resolve()
FROZEN = Path(os.environ["BUN_V2_FROZEN_ROOT"]).resolve()
PYTHON = str(REPO / ".venv/bin/python")
def run(command, log, environment):
    run_logged(command, log, environment, REPO)


def family_identity(base_hash, family_hash, config):
    payload = {"base_sha256": base_hash, "family_sha256": family_hash,
               "config": config}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def mixed_schedule(cycle):
    history = ("old", "recent", "rule_combat")[(cycle - 1) % 3]
    if cycle % 10 in (9, 0):
        return ["tactical", "tactical", history, history]
    return ["tactical", "tactical", "tactical", history]


def tactical_updates_through(config, cycle):
    return sum(
        item == "tactical"
        for completed_cycle in range(1, cycle + 1)
        for item in (mixed_schedule(completed_cycle)
                     if config["historical_mix"] else ["tactical"] * 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_cycle_count(int(config["total_cycles"]))
    if config.get("critic_seed_namespace_version") != "v7":
        raise ValueError("clean qqt-RL runs require critic_seed_namespace_version=v7")
    name, gpu, seed = config["name"], str(config["gpu"]), int(config["seed"])
    root, ckpt, data, cache = RUN / name, CKPT / name, DATA / name, RUN / name / "cache"
    for path in (root, ckpt, data, cache):
        path.mkdir(parents=True, exist_ok=True)
    repair_data = data / config.get("repair_data_namespace", "")
    repair_data.mkdir(parents=True, exist_ok=True)
    status_path, resume_path = root / "status.json", root / "resume_state.json"
    stat_fields = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    atomic(root / "worker.pid.json", {
        "pid": os.getpid(), "start_ticks": int(stat_fields[21]),
        "config": str(Path(args.config).resolve()), "candidate": name,
    })
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu, "JAX_PLATFORM_NAME": "gpu",
        "JAXBOMB_RULE": "bun", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_COMPILATION_CACHE_DIR": str(cache), "PYTHONPATH": str(FROZEN),
        "BUN_TACTICAL_BOT_PATH": str(FROZEN / "jax_bomb/bun_rule_bot.py"),
        "BUN_TACTICAL_BOT_EXPECTED_SHA256": config["bot_sha256"],
    })
    scripts = FROZEN / "scripts"
    inputs = Path(os.environ.get("BUN_V2_INPUT_ROOT", RUN / "frozen_inputs")).resolve()
    prefix = config["initial_bc"]
    reference = str(inputs / config.get("reference_actor", "ambush_c2_actor.pt"))
    generic = str(inputs / "generic_critic_replay_aux.npz")
    weak, old, recent = (str(inputs / "weak_actor.pt"),
                         str(inputs / "old_actor.pt"), reference)
    bc_data = str(inputs / f"{prefix}_selective_bc.npz")
    if resume_path.exists():
        resume = json.loads(resume_path.read_text())
        last, actor, critic, target = (int(resume["last_completed_cycle"]),
                                       resume["actor"], resume["critic"],
                                       resume["target_critic"])
        if resume.get("candidate") != name or not 0 <= last <= int(config["total_cycles"]):
            raise RuntimeError("resume state candidate/cycle mismatch")
        expected_hashes = resume.get("hashes", {})
        resume_paths = {"actor": actor, "critic": critic,
                        "target_critic": target, "reference_actor": reference}
        for key, path in resume_paths.items():
            if not Path(path).is_file() or digest(path) != expected_hashes.get(key):
                raise RuntimeError(f"resume checkpoint hash mismatch: {key}")
        if not all(finite(path) for path in (actor, critic, target)):
            raise RuntimeError("resume checkpoint contains non-finite values")
    else:
        last = 0
        actor = str(inputs / config.get(
            "initial_actor", f"{prefix}_selective_bc_actor.pt"))
        critic = str(inputs / config.get("initial_critic", "ambush_c2_critic.pkl"))
        target = str(inputs / config.get("initial_target_critic",
                                         config.get("initial_critic", "ambush_c2_critic.pkl")))
        resume = {"schema": "bun_tactical_v2_resume_v2",
                  "candidate": name, "last_completed_cycle": 0,
                  "actor": actor, "critic": critic,
                  "target_critic": target,
                  "reference_actor": reference,
                  "episodes_completed": 0,
                  "hashes": {key: digest(value) for key, value in {
                      "actor": actor, "critic": critic,
                      "target_critic": target,
                      "reference_actor": reference}.items()},
                  "updated_unix": time.time()}
        atomic(resume_path, resume)

    segment_size = int(config.get("segment_size", 90))
    segment_root = root / "segments"
    segment_root.mkdir(exist_ok=True)

    family_module_hash = digest(FROZEN / "jax_bomb/bun_tactical_family.py")

    def write_status(state, cycle, phase, **extra):
        tactical_updates = tactical_updates_through(config, cycle)
        atomic(status_path, {"schema": "bun_tactical_v2_candidate_status_v2",
                             "candidate": name, "gpu": int(gpu),
                             "worker_pid": os.getpid(), "status": state,
                             "cycle": cycle, "phase": phase,
                             "transitions_completed": cycle * 4 * config["transitions_per_update"],
                             "tactical_transitions_completed": tactical_updates * config["transitions_per_update"],
                             "updated_unix": time.time(), **extra})

    def evaluate(cycle, actor_path, critic_path, matched_family):
        cycle_dir = root / f"cycle_{cycle:03d}"
        paired_eval_seed = int(config.get("paired_eval_seed", seed))
        strata = (("v2_matched", 700000, matched_family),
                  ("v2_full", 1700000, FULL_FAMILY),
                  ("v2_unseen_family", 2700000, UNSEEN_FAMILY),
                  ("v2_structural_heldout", 3700000, HELDOUT_FAMILY))
        summaries = {}
        for label, offset, family in strata:
            eval_environment = environment.copy()
            eval_environment["JAX_COMPILATION_CACHE_DIR"] = str(
                cache / f"cycle_{cycle:03d}" / "eval")
            eval_environment["BUN_TACTICAL_FAMILY_JSON"] = json.dumps(family, sort_keys=True)
            output = cycle_dir / f"{label}.json"
            run([PYTHON, str(scripts / "eval_bun_tactical_opponent.py"), actor_path,
                 "--games", "64", "--seed", str(paired_eval_seed + offset),
                 "--max-steps", "300", "--json-out", str(output)],
                cycle_dir / f"{label}.log", eval_environment)
            payload = json.loads(output.read_text())
            summaries[label] = {"family": family,
                                "opponent_identity_hash": payload["opponent"]["opponent_identity_hash"],
                                "metrics": payload["summary"]}
        train_diag = (json.loads((cycle_dir / "train.json").read_text())["heldout"]
                      if (cycle_dir / "train.json").exists() else None)
        summary = {"schema": "bun_tactical_v2_milestone_v2",
                   "candidate": name, "cycle": cycle,
                   "actor": actor_path, "actor_sha256": digest(actor_path),
                   "critic": critic_path, "critic_sha256": digest(critic_path),
                   "base_bot_hash": config["bot_sha256"],
                   "matched_family_identity": family_identity(
                       config["bot_sha256"], family_module_hash, matched_family),
                   "games_per_stratum": 64,
                   "games_total": 256,
                   "critic_tactical_diagnostics": train_diag,
                   "strata": summaries, "updated_unix": time.time()}
        atomic(cycle_dir / "summary.json", summary)

    try:
        if last == 0 and not (root / "cycle_000/summary.json").exists():
            write_status("running", 0, "cycle0_fixed256")
            evaluate(0, actor, critic, family_for_cycle(config, 1))
        for cycle in range(last + 1, int(config["total_cycles"]) + 1):
            segment = (cycle - 1) // segment_size + 1
            segment_start = (segment - 1) * segment_size + 1
            segment_end = min(segment * segment_size, int(config["total_cycles"]))
            segment_manifest = segment_root / f"segment_{segment:02d}_manifest.json"
            if not segment_manifest.exists():
                atomic(segment_manifest, {
                    "schema": "bun_safe_aggression_segment_v1",
                    "candidate": name, "segment": segment,
                    "global_cycle_start": segment_start,
                    "global_cycle_end": segment_end,
                    "config_sha256": digest(args.config),
                    "input_hashes": {key: digest(value) for key, value in {
                        "actor": actor, "critic": critic,
                        "target_critic": target, "reference_actor": reference}.items()},
                    "created_unix": time.time(),
                })
            cycle_dir = root / f"cycle_{cycle:03d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            attempt = cycle_dir / f"attempt_{len(list(cycle_dir.glob('attempt_*'))) + 1:02d}"
            attempt.mkdir()
            family = family_for_cycle(config, cycle)
            identity = family_identity(config["bot_sha256"], family_module_hash, family)
            cycle_environment = environment.copy()
            cycle_environment["JAX_COMPILATION_CACHE_DIR"] = str(
                cache / f"cycle_{cycle:03d}" / "training")
            cycle_environment["BUN_TACTICAL_FAMILY_JSON"] = json.dumps(family, sort_keys=True)
            schedule = mixed_schedule(cycle) if config["historical_mix"] else ["tactical"] * 4
            critic_seed_base = critic_seed_base_v7(
                int(config.get("candidate_slot", config["gpu"])), seed, cycle)
            split_stride = V7_SPLIT_STRIDE
            atomic(attempt / "freeze_manifest.json", {
                "schema": "bun_tactical_v2_cycle_freeze_v2", "cycle": cycle,
                "actor": actor, "actor_sha256": digest(actor),
                "critic": critic, "critic_sha256": digest(critic),
                "target_critic": target, "target_critic_sha256": digest(target),
                "opponent_schedule": schedule, "family": family,
                "rollout_opponent_hash": identity,
                "critic_data_opponent_hash": identity,
                "eval_opponent_hash": identity, "hash_alignment": True,
                "critic_seed_namespace_version": config.get(
                    "critic_seed_namespace_version", "legacy_v6"),
                "critic_seed_base": critic_seed_base,
                "critic_seed_split_stride": split_stride,
                "repair_data_namespace": config.get("repair_data_namespace", ""),
                "created_unix": time.time()})
            write_status("running", cycle - 1, "tactical_v2_counterfactual",
                         attempt=attempt.name, family=family, schedule=schedule)
            cf = repair_data / f"cycle_{cycle:03d}_tactical_v2.npz"
            cf_manifest = attempt / "critic_data_manifest.json"
            run([PYTHON, str(scripts / "generate_bun_tactical_v2_critic_data.py"),
                 "--actor", actor, "--output", str(cf), "--manifest", str(cf_manifest),
                 "--seed-base", str(critic_seed_base),
                 "--split-stride", str(split_stride),
                 "--train-states", str(config["cf_train_states"]),
                 "--validation-states", "1", "--test-states", "1",
                 "--mc-samples", str(config["cf_mc_samples"]),
                 "--mc-horizon", str(config["cf_horizon"])],
                attempt / "critic_data.log", cycle_environment)
            critic_manifest = json.loads(cf_manifest.read_text())
            hashes = {critic_manifest["rollout_opponent_hash"],
                      critic_manifest["critic_data_opponent_hash"],
                      critic_manifest["eval_opponent_hash"], identity}
            if hashes != {identity}:
                raise RuntimeError(f"opponent hash mismatch: {hashes}")
            merged = repair_data / f"cycle_{cycle:03d}_replay.npz"
            run([PYTHON, str(scripts / "concat_bun_critic_replay.py"),
                 "--input", generic, "--input", str(cf), "--output", str(merged),
                 "--manifest", str(attempt / "replay_manifest.json")],
                attempt / "replay.log", cycle_environment)
            write_status("running", cycle - 1, "critic_only_calibration",
                         attempt=attempt.name, family=family)
            actor_hash = digest(actor)
            calibration = attempt / "critic_calibration"
            run([PYTHON, str(scripts / "train_bun_critic.py"), "--data", str(merged),
                 "--actor", actor, "--initial-critic", critic,
                 "--output-dir", str(calibration), "--seed", str(seed + cycle * 10000 + 101),
                 "--epochs", str(config["critic_epochs"]), "--batch-size", "64",
                 "--eval-every", "10", "--lr", str(config["critic_calibration_lr"]),
                 "--aux-coef", str(config["aux_coef"])],
                attempt / "critic_calibration.log", cycle_environment)
            if digest(actor) != actor_hash:
                raise RuntimeError("critic-only calibration changed actor")
            calibrated = str(calibration / "independent_critic.pkl")
            write_status("running", cycle - 1, "joint_micro_updates",
                         attempt=attempt.name, family=family, schedule=schedule)
            out_actor, out_critic, out_target = (str(attempt / "actor.pt"),
                                                  str(attempt / "critic.pkl"),
                                                  str(attempt / "target_critic.pkl"))
            command = [PYTHON, "-u", str(scripts / "train_bun_separate_ac.py"),
                       "--actor", actor, "--critic", calibrated,
                       "--target-critic", target, "--reference-actor", reference,
                       "--weak", weak, "--old", old, "--recent", recent,
                       "--counterfactual", str(merged), "--bc-data", bc_data,
                       "--save-actor", out_actor, "--save-critic", out_critic,
                       "--save-target-critic", out_target,
                       "--json-out", str(cycle_dir / "train.json"),
                       "--curriculum", "danger_arena=1", "--opponent", "tactical",
                       "--opponent-schedule", ",".join(schedule),
                       "--seed", str(seed + cycle * 10000 + 500), "--updates", "4",
                       "--num-envs", "32", "--num-steps", "128", "--carry-rollout-state",
                       "--actor-lr", str(config["actor_lr"]),
                       "--critic-lr", str(config["critic_lr"]),
                       "--entropy", str(config["entropy"]), "--kl", str(config["kl"]),
                       "--bc-coef", str(config["bc_coef"]),
                       "--target-tau", str(config["target_tau"]),
                       "--counterfactual-coef", str(config["counterfactual_coef"]),
                       "--counterfactual-aux-coef", str(config["aux_coef"]),
                       "--reward-profile", "danger_arena",
                       "--danger-escape-reward", str(config["danger_escape_reward"]),
                       "--avoidable-danger-death-penalty", str(config["avoidable_penalty"]),
                       "--tactical-bomb-placement-reward", str(
                           config.get("tactical_bomb_placement_reward", 0.0)),
                       "--tactical-bomb-resolution-reward", str(
                           config.get("tactical_bomb_resolution_reward", 0.0))]
            run(command, cycle_dir / "train.log", cycle_environment)
            if not all(finite(path) for path in (out_actor, out_critic, out_target)):
                raise RuntimeError("non-finite cycle checkpoint")
            if cycle in EVAL_CYCLES:
                write_status("running", cycle, "fixed256", attempt=attempt.name)
                evaluate(cycle, out_actor, out_critic, family)
            actor_final = ckpt / f"cycle_{cycle:03d}_actor.pt"
            critic_final = ckpt / f"cycle_{cycle:03d}_critic.pkl"
            target_final = ckpt / f"cycle_{cycle:03d}_target_critic.pkl"
            # cycle checkpoint 只有在三份文件均完成且有限值检查通过后才发布；
            # 临时文件 + os.replace 避免监督器或恢复逻辑读到半写文件。
            for source, destination in ((out_actor, actor_final),
                                        (out_critic, critic_final),
                                        (out_target, target_final)):
                atomic_copy(source, destination)
            actor, critic, target = map(str, (actor_final, critic_final, target_final))
            train = json.loads((cycle_dir / "train.json").read_text())
            cycle_episodes = sum(
                int(row.get("episode_resets", 0)) for row in train["history"])
            previous_episodes = int(resume.get("episodes_completed", 0))
            episodes_completed = previous_episodes + cycle_episodes
            completed_updates = cycle * 4
            tactical_updates = tactical_updates_through(config, cycle)
            resume = {"schema": "bun_tactical_v2_resume_v3", "candidate": name,
                      "last_completed_cycle": cycle, "actor": actor,
                      "critic": critic, "target_critic": target,
                      "reference_actor": reference, "family": family,
                      "opponent_schedule": schedule, "attempt": attempt.name,
                      "updates_completed": completed_updates,
                      "tactical_updates_completed": tactical_updates,
                      "transitions_completed": completed_updates * config["transitions_per_update"],
                      "tactical_transitions_completed": tactical_updates * config["transitions_per_update"],
                      "episodes_completed": episodes_completed,
                      "equivalent_full_300_tick_games": (
                          completed_updates * config["transitions_per_update"] / 300.0),
                      "last_update_tps": train["history"][-1]["transitions_per_second"],
                      "hashes": {key: digest(value) for key, value in {
                          "actor": actor, "critic": critic,
                          "target_critic": target, "reference_actor": reference}.items()},
                      "updated_unix": time.time()}
            atomic(resume_path, resume)
            # resume_state 是恢复提交点；只有它成功后才切换 latest 指针，避免
            # 崩溃重跑同一 cycle 时向外暴露跨版本 checkpoint 组合。
            atomic(ckpt / "latest.json", {
                "cycle": cycle, "actor": actor, "critic": critic,
                "target_critic": target, "hashes": resume["hashes"],
            })
            atomic(attempt / "COMPLETE.json", resume)
            if cycle == segment_end:
                atomic(segment_root / f"segment_{segment:02d}_resume.json", {
                    "schema": "bun_safe_aggression_segment_resume_v1",
                    "candidate": name, "segment": segment,
                    "global_cycle_completed": cycle,
                    "episodes_completed": episodes_completed,
                    "equivalent_full_300_tick_games": resume[
                        "equivalent_full_300_tick_games"],
                    "hashes": resume["hashes"],
                    "completed_unix": time.time(),
                })
            write_status("running", cycle, "cycle_complete",
                         last_update_tps=resume["last_update_tps"])
        write_status("complete", int(config["total_cycles"]), "budget_complete")
    except Exception as error:
        write_status("failed", int(locals().get("cycle", last)),
                     "technical_failure", error=repr(error))
        raise


if __name__ == "__main__":
    main()
