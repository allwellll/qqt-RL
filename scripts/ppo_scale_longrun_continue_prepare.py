"""Register the long-horizon continuation from the completed scaled R1 runs.

This preparation step only creates a new, immutable namespace and records the
three per-seed source checkpoint hashes.  It never starts a worker.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, sha

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

R = Path(__file__).resolve().parents[1]
R1 = R / "runs/ppo_scale_20261025/r1_longrun"
B = R / "runs/ppo_scale_20261025/r2_longrun"
SEEDS = [2026102751, 2026102752, 2026102753]
SOURCE_UPDATES = {seed: 64 for seed in SEEDS}
NEW_UPDATES = 2048
ENV_STEPS_PER_UPDATE = 16 * 128 * 8
NEW_STEPS_PER_SEED = NEW_UPDATES * ENV_STEPS_PER_UPDATE
OLD_STEPS_PER_SEED = 1_048_576
OPTIMIZER_STEPS_PER_UPDATE = 32
EXPECTED_STEADY_STEPS_PER_SECOND = 2622.0


def read(path: Path):
    return json.loads(path.read_text())


def audit_state(path: Path, seed: int):
    import jax
    import numpy as np

    with path.open("rb") as stream:
        state = pickle.load(stream)
    required = {"params", "adam", "states", "aux", "keys", "history", "update", "seed", "arm"}
    assert required <= set(state)
    assert (state["update"], state["seed"], state["arm"]) == (64, seed, "ppo_scaled_batch")
    assert np.asarray(state["keys"]).shape == (8, 16, 2)
    finite = True
    for key in ("params", "adam", "states", "aux", "keys", "history"):
        for leaf in jax.tree.leaves(state[key]):
            value = np.asarray(leaf)
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                finite = False
                break
    counts = [int(np.asarray(leaf)) for leaf in jax.tree.leaves(state["adam"])
              if np.asarray(leaf).shape == () and np.asarray(leaf).dtype.kind in "iu"]
    assert finite and counts
    summary = dict(
        passed=True,
        seed=seed,
        fields=sorted(state),
        required_fields=sorted(required),
        state_finite=finite,
        keys_shape=list(np.asarray(state["keys"]).shape),
        adam_scalar_counters=counts,
        auxiliary_fields=sorted(state["aux"]),
        history_fields=sorted(state["history"]),
    )
    del state
    return summary


def prepare() -> None:
    if (B / "PROTOCOL.json").exists():
        raise RuntimeError("continuation namespace already registered")
    assert (R1 / "GPU_COMPLETE.json").exists()
    assert (R1 / "CPU_EVALUATION_COMPLETE.json").exists()
    assert (R1 / "OWNER_EXIT.json").exists()
    assert (R1 / "CPU_OWNER_EXIT.json").exists()
    old_auth = read(R1 / "SCOPED_GPU_AUTHORIZATION.json")
    old_pause = R / "runs/retention_20261021/USER_GPU_PAUSE.json"
    assert sha(old_pause) == old_auth["old_global_pause_sha256"]

    sources = {}
    source_audits = {}
    for seed in SEEDS:
        cp = R1 / "formal" / f"ppo_scaled_batch_s{seed}" / "u064"
        receipt = read(cp / "COMMITTED.json")
        assert receipt["passed"] and receipt["update"] == 64
        assert receipt["environment_steps"] == OLD_STEPS_PER_SEED
        assert sha(cp / "state.pkl") == receipt["state_sha256"]
        assert sha(cp / "policy.pt") == receipt["policy_sha256"]
        source_audits[str(seed)] = audit_state(cp / "state.pkl", seed)
        sources[str(seed)] = dict(
            checkpoint=str(cp.relative_to(R)),
            update=64,
            environment_steps=OLD_STEPS_PER_SEED,
            optimizer_steps_cumulative=receipt["metrics"]["cumulative_optimizer_steps"],
            state_sha256=receipt["state_sha256"],
            policy_sha256=receipt["policy_sha256"],
        )

    B.mkdir(parents=True, exist_ok=True)
    for folder in ["inputs", "cpu_eval_inputs", "cache", "CPU_QUEUE", "notifications", "eval_queue", "eval_done"]:
        (B / folder).mkdir(parents=True, exist_ok=True)
    for name in ["spawns.json", "regions.json"]:
        shutil.copyfile(R1 / "cpu_eval_inputs" / name, B / "inputs" / name)
        shutil.copyfile(R1 / "cpu_eval_inputs" / name, B / "cpu_eval_inputs" / name)

    protocol = dict(
        id="PPO-SCALE-LONGRUN-20261025-R2",
        namespace=str(B.relative_to(R)),
        mode="ppo_only_scaled_batch_long_horizon_continuation",
        parent=dict(
            namespace=str(R1.relative_to(R)),
            arm="ppo_scaled_batch",
            source_update=64,
            per_seed=sources,
            no_peak_selection=True,
            complete_state_required=["params", "adam", "states", "aux", "keys", "history"],
        ),
        arms=["ppo_scaled_batch_continuation"],
        paired_seeds=SEEDS,
        source_update=64,
        updates=NEW_UPDATES,
        single_variable="long-horizon observation at the already validated scaled PPO batch; no recipe change",
        hypothesis="The validated large batch needs a materially longer on-policy horizon before short-run regressions can be interpreted.",
        falsification="Report numerical failure, damaged restoration, or reliable severe safety regression; ordinary low scores and seed variation do not stop a healthy fixed budget.",
        common=dict(
            reward="R1 scaled reward unchanged",
            input="R1 public observation unchanged",
            mask="R1 action masks unchanged",
            opponents="fixed JAX rule3",
            network="R1 network unchanged",
            optimizer="continue each seed's complete Adam state",
            gamma=.995,
            gae_lambda=.95,
            clip=.1,
            value=.5,
            entropy=.005,
            gradient_clip=.5,
            epochs=1,
            lr=7.5e-5,
            dagger_coefficient=0.0,
            combat_kl_coefficient=0.0,
            no_offline_behavior_as_ppo=True,
            no_synthetic_carry=True,
        ),
        batch=dict(
            devices=8,
            envs_per_device=16,
            rollout_steps=128,
            global_rollout_batch=16_384,
            global_minibatch=512,
            local_minibatch=64,
            optimizer_steps_per_update=OPTIMIZER_STEPS_PER_UPDATE,
        ),
        budget=dict(
            new_updates_per_seed=NEW_UPDATES,
            new_environment_steps_per_seed=NEW_STEPS_PER_SEED,
            new_optimizer_steps_per_seed=NEW_UPDATES * OPTIMIZER_STEPS_PER_UPDATE,
            total_new_environment_steps=NEW_STEPS_PER_SEED * len(SEEDS),
            total_new_optimizer_steps=NEW_UPDATES * OPTIMIZER_STEPS_PER_UPDATE * len(SEEDS),
            old_environment_steps_per_seed=OLD_STEPS_PER_SEED,
            fixed_total_environment_steps=True,
            no_short_budget_fallback=True,
        ),
        checkpointing=dict(
            atomic_every_update=False,
            checkpoint_every_updates=64,
            light_dev_every_updates=64,
            light_dev_every_environment_steps=64 * ENV_STEPS_PER_UPDATE,
            full_eval_every_updates=512,
            final_holdout=True,
            save_after_current_update_on_user_pause=True,
            cpu_owner="scripts.ppo_scale_longrun_continue_cpu",
            gpu_owner="scripts.ppo_scale_longrun_continue_run",
            no_evaluation_gate=True,
        ),
        evaluation=dict(
            light_dev_games=16,
            full_natural_games=64,
            tactical_games=64,
            hunter_games_per_tier=64,
            hunter_tiers=["hard", "normal", "easy"],
            engines=["Python Tactical v2", "production JS Hunter", "native JS full2400"],
            holdout_seed_start=2026112700,
            dev_seed_start=2026102480,
            no_holdout_tuning=True,
            categorical_training_and_greedy_production_separate=True,
            bank_regression_only=True,
        ),
        expected_throughput=dict(
            source_steps_per_second=EXPECTED_STEADY_STEPS_PER_SECOND,
            estimated_hours_per_seed=NEW_STEPS_PER_SEED / EXPECTED_STEADY_STEPS_PER_SECOND / 3600,
            estimated_hours_total=NEW_STEPS_PER_SEED * len(SEEDS) / EXPECTED_STEADY_STEPS_PER_SECOND / 3600,
            estimate_only=True,
        ),
        stopping="fixed 2048 updates per seed; only technical failure, explicit user pause, or reliable severe safety anomaly",
        old_queues_remain_paused=True,
        r1_immutable=True,
        created_unix=time.time(),
    )
    atomic_json(B / "PROTOCOL.json", protocol)
    atomic_json(B / "SOURCE_MANIFEST.json", dict(
        r1_namespace=str(R1.relative_to(R)),
        sources=sources,
        state_audits=source_audits,
        r1_gpu_complete_sha256=sha(R1 / "GPU_COMPLETE.json"),
        r1_cpu_complete_sha256=sha(R1 / "CPU_EVALUATION_COMPLETE.json"),
        no_source_copy_mutation=True,
    ))
    ev = dict(
        namespace=str(B.relative_to(R)),
        hunter_seed=20261024,
        natural_seed_start=2026102480,
        holdout_seed_start=2026112700,
        light_dev_games=16,
        full_games=64,
        tactical_games=64,
        hunter_games_per_tier=64,
        no_training_launch=True,
        no_holdout_tuning=True,
        engines=["Python Tactical v2", "production JS Hunter", "native JS full2400"],
        spawns_sha256=sha(B / "inputs/spawns.json"),
        regions_sha256=sha(B / "inputs/regions.json"),
        controls_gpu=False,
    )
    atomic_json(B / "CPU_EVAL_PROTOCOL.json", ev)
    atomic_json(B / "SCOPED_GPU_AUTHORIZATION.json", dict(
        authorized_by="explicit_user_instruction",
        protocol_sha256=sha(B / "PROTOCOL.json"),
        global_pause_sha256=old_auth["old_global_pause_sha256"],
        old_queues_remain_paused=True,
        old_o5_resume=False,
        r1_must_remain_immutable=True,
        no_push=True,
        no_publish=True,
    ))
    atomic_json(B / "PREREGISTRATION.json", dict(
        passed=True,
        protocol_sha256=sha(B / "PROTOCOL.json"),
        parent_selection="all three completed scaled u064 states; no peak selection",
        capability_gate=False,
        short_point_is_observation_only=True,
        long_budget_fixed=True,
    ))
    seal_names = [
        "scripts/ppo_scale_longrun_continue_prepare.py",
        "scripts/ppo_scale_longrun_continue_train.py",
        "scripts/ppo_scale_longrun_continue_run.py",
        "scripts/ppo_scale_longrun_continue_cpu.py",
        "scripts/ppo_scale_longrun_train.py",
        "scripts/dagger_rl_r4_train.py",
        "scripts/carrier_mechanism_engine.py",
        "scripts/combat_item_engine.py",
        "scripts/carrier_mechanism_analysis.py",
        "scripts/carrier_mechanism_stats.py",
        "scripts/je10_route_oracle.py",
    ]
    atomic_json(B / "SOURCE_SEAL.json", dict(
        protocol_sha256=sha(B / "PROTOCOL.json"),
        sources={name: sha(R / name) for name in seal_names},
        no_run_or_checkpoint_files=True,
    ))
    print(sha(B / "PROTOCOL.json"), flush=True)


if __name__ == "__main__":
    prepare()
