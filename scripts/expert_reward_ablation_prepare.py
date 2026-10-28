"""Freeze the executable D1 paired reward-ablation protocol before GPU use."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, sha
from scripts.combat_item_events import METRICS as ITEM_COMPONENTS

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_reward_20261025/D_ablation_v1"
R1 = R / "runs/ppo_scale_20261025/r1_longrun"
R2 = R / "runs/ppo_scale_20261025/r2_longrun"
D_SEEDS = [2026102851, 2026102852, 2026102853]
SOURCE_SEEDS = [2026102751, 2026102752, 2026102753]


def reward_component_names(base_names, item_names=ITEM_COMPONENTS):
    """Match combat_item_engine's passive metric append order exactly."""
    base_names = tuple(base_names)
    if not base_names or base_names[-1] != "combined_total":
        raise ValueError("base reward schema must end in combined_total")
    combined = (*base_names[:-1], *tuple(item_names), "combined_total")
    if len(set(combined)) != len(combined):
        raise ValueError("reward component schema contains duplicate labels")
    return combined


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    if (B / "PROTOCOL.json").exists():
        raise RuntimeError("D1 already has a frozen protocol")
    assert (R2 / "PROTOCOL.json").exists()
    source_protocol = read(R1 / "PROTOCOL.json")
    source_arm = "ppo_scaled_batch"
    source_map = dict(zip(D_SEEDS, SOURCE_SEEDS))
    sources = {}
    for seed in D_SEEDS:
        parent_seed = source_map[seed]
        cp = R1 / "formal" / f"{source_arm}_s{parent_seed}" / "u064"
        receipt = read(cp / "COMMITTED.json")
        assert receipt["passed"] and receipt["update"] == 64
        assert sha(cp / "state.pkl") == receipt["state_sha256"]
        assert sha(cp / "policy.pt") == receipt["policy_sha256"]
        sources[str(seed)] = dict(
            checkpoint=str(cp.relative_to(R)), source_seed=parent_seed,
            update=64, environment_steps=1_048_576,
            state_sha256=receipt["state_sha256"],
            policy_sha256=receipt["policy_sha256"],
            optimizer_state="complete Adam carried forward exactly",
            env_aux_rng_history="complete paired R1 state carried forward exactly",
        )

    # Keep the R1 component-name ABI, including passive item diagnostics.
    from scripts import carrier_mechanism_engine as engine
    component_names = reward_component_names(engine.ops.COMPONENTS)
    route_coefficient = float(engine.ops.ROUTE_COEFFICIENT)

    protocol = dict(
        id="EXPERT-REWARD-ABLATION-20261025-D1",
        namespace="runs/expert_reward_20261025/D_ablation_v1",
        predecessor="runs/ppo_scale_20261025/r2_longrun",
        predecessor_required_terminal=True,
        predecessor_gpu_release_required=True,
        predecessor_cpu_evaluation_nonblocking=True,
        parent=dict(
            namespace="runs/ppo_scale_20261025/r1_longrun",
            arm=source_arm, update=64, paired_source_map={str(k): v for k,v in source_map.items()},
            no_peak_selection=True,
        ),
        arms=["legacy", "minimal"],
        paired_seeds=D_SEEDS,
        arm_order=[f"{arm}_s{seed}" for seed in D_SEEDS
                   for arm in ("legacy", "minimal")],
        updates=320,
        single_variable="reward package only; no network, optimizer, input, action-mask, opponent, curriculum, or batch changes",
        reward_package=dict(
            legacy="sealed R1 scaled reward composition byte-for-byte at reward output",
            reward_component_names=list(component_names),
            reward_component_schema_source=(
                "je10_ops.COMPONENTS[:-1] + combat_item_events.METRICS + combined_total"),
            minimal=dict(
                delivery="+2.0 per native capture event",
                direct_kill="+16.0 per surviving causal kill after controlled-kill duplicate suppression",
                native_control="existing edge-triggered control entry/residual ledger only",
                death="-12.0 per learner death",
                strict_pbrs=dict(
                    component="route_shaped_reward",
                    emitted_value="the frozen je10_ops output; no second scaling in the adapter",
                    route_coefficient=route_coefficient,
                    discounted_with_gamma=0.995,
                    terminal_potential_zero=True,
                    hidden_history_ledger=True,
                    partial_observability_policy_invariance_claimed=False,
                ),
                removed=["all non-potential route/bridge/prestock shaping",
                         "requested glue/banana replacement reward or clawback",
                         "item placement/tick inventory/request payments",
                         "unconditional steal/progress/region/approach bonuses",
                         "legacy shaping and reward clawbacks"],
                controls=["PPO advantage/GAE receives the selected reward directly",
                          "reward state bookkeeping remains shared and does not change observations or actions",
                          "reward coefficient zero returns legacy array exactly"],
            ),
        ),
        common=dict(
            devices=8, envs_per_device=16, rollout_steps=128,
            global_rollout_batch=16384, global_minibatch=512,
            optimizer_steps_per_update=32, lr=7.5e-5, epochs=1,
            gamma=0.995, gae_lambda=0.95, clip=0.1, value=0.5,
            entropy=0.005, gradient_clip=0.5,
            dagger_coefficient=0.0, combat_kl_coefficient=0.0,
            opponent="fixed JAX rule3", teacher_not_applied=True,
            offline_behavior_not_ppo=True, synthetic_carry_training=False,
            source_network="seed-paired R1 scaled u064 actor/critic exact",
            source_optimizer="matching R1 u064 Adam state exact",
            source_env_rng="matching R1 u064 environment/aux/RNG/history exact",
            no_other_code_or_recipe_change=True,
        ),
        budget=dict(
            new_updates_per_arm_seed=320,
            new_environment_steps_per_arm_seed=5_242_880,
            new_optimizer_steps_per_arm_seed=10_240,
            total_environment_steps=31_457_280,
            total_optimizer_steps=61_440,
            runs=6, checkpoint_every_updates=64,
            checkpoint_every_environment_steps=1_048_576,
            fixed_total_budget=True,
        ),
        evaluation=dict(
            light_each_checkpoint=dict(engine="production JS natural full2400; greedy", games=16,
                                        seed_start=2026102480),
            final_each_arm_seed=dict(
                natural_games=64, tactical_v2_games=64,
                hunter_games_per_tier=64, hunter_tiers=["hard","normal","easy"],
                engines=["production JS natural full2400", "Python Tactical v2", "production JS Hunter"],
                categorical_training_and_greedy_production_separate=True,
                teacher_off=True, fixed_dev_seed_start=2026102480,
                fixed_holdout_seed_start=2026112700, no_holdout_tuning=True,
            ),
            same_fixed_cpu_js_review_owner=True,
            may_not_block_gpu=True,
            report_action_opportunity_and_reward_components=True,
        ),
        stopping="complete all six fixed runs; short-point ability does not stop training; only technical failure or explicit user stop",
        result_selection="paired seed contrasts, full trajectories, worst seed and independent holdout; no peak selection",
        old_queues_remain_paused=True,
        no_push=True, no_publish=True,
        source_protocol_sha256=sha(R1 / "PROTOCOL.json"),
        created_unix=time.time(),
    )

    B.mkdir(parents=True, exist_ok=True)
    for folder in ("inputs", "cache", "formal", "smoke", "eval_queue", "eval_done",
                   "milestones", "notifications", "cpu_eval_inputs"):
        (B / folder).mkdir(parents=True, exist_ok=True)
    atomic_json(B / "PROTOCOL.json", protocol)
    atomic_json(B / "SOURCE_MANIFEST.json", dict(sources=sources,
        no_peak_selection=True, source_protocol_sha256=protocol["source_protocol_sha256"]))
    for name in ("spawns.json", "regions.json"):
        shutil.copyfile(R2 / "cpu_eval_inputs" / name, B / "cpu_eval_inputs" / name)
    r2eval = read(R2 / "CPU_EVAL_PROTOCOL.json")
    eval_protocol = dict(namespace=protocol["namespace"],
        hunter_seed=r2eval["hunter_seed"], natural_seed_start=2026102480,
        holdout_seed_start=2026112700, light_dev_games=16, full_games=64,
        tactical_games=64, hunter_games_per_tier=64,
        spawns_sha256=sha(B / "cpu_eval_inputs" / "spawns.json"),
        regions_sha256=sha(B / "cpu_eval_inputs" / "regions.json"),
        controls_gpu=False, no_training_launch=True, no_holdout_tuning=True)
    atomic_json(B / "CPU_EVAL_PROTOCOL.json", eval_protocol)

    seal = read(R2 / "SOURCE_SEAL.json")["sources"]
    seal.update({
        "scripts/expert_reward_ablation.py": sha(R / "scripts/expert_reward_ablation.py"),
        "scripts/expert_reward_ablation_train.py": sha(R / "scripts/expert_reward_ablation_train.py"),
        "scripts/expert_reward_ablation_run.py": sha(R / "scripts/expert_reward_ablation_run.py"),
        "scripts/expert_reward_ablation_cpu.py": sha(R / "scripts/expert_reward_ablation_cpu.py"),
        "scripts/expert_reward_ablation_prepare.py": sha(R / "scripts/expert_reward_ablation_prepare.py"),
        "tests/test_expert_reward_ablation.py": sha(R / "tests/test_expert_reward_ablation.py"),
    })
    atomic_json(B / "SOURCE_SEAL.json", dict(
        protocol_sha256=sha(B / "PROTOCOL.json"), sources=seal,
        r2_source_seal_sha256=sha(R2 / "SOURCE_SEAL.json"),
        no_run_or_checkpoint_files=True))
    atomic_json(B / "SCOPED_GPU_AUTHORIZATION.json", dict(
        authorized_by="explicit_user_instruction",
        protocol_sha256=sha(B / "PROTOCOL.json"),
        source_seal_sha256=sha(B / "SOURCE_SEAL.json"),
        predecessor_protocol_sha256=sha(R2 / "PROTOCOL.json"),
        global_pause_sha256=sha(R / "runs/retention_20261021/USER_GPU_PAUSE.json"),
        only_after_predecessor_fixed_training_complete=True,
        only_after_eight_gpu_release=True,
        old_o5_continuous_reservoir_queues_remain_paused=True,
    ))
    atomic_json(B / "PREREGISTRATION.json", dict(
        passed=True, protocol_sha256=sha(B / "PROTOCOL.json"),
        source_manifest_sha256=sha(B / "SOURCE_MANIFEST.json"),
        source_seal_sha256=sha(B / "SOURCE_SEAL.json"),
        paired_runs=6, formal_updates=0, no_holdout_tuning=True,
        capability_gate=False, training_waits_for_r2_terminal=True,
        created_unix=time.time()))
    atomic_json(B / "PREPARATION.json", dict(
        passed=True, formal_updates=0,
        state="PRE_REGISTERED_WAITING_FOR_R2_FIXED_TRAINING_AND_GPU_RELEASE",
        controls_gpu=False, no_old_queue_resume=True, unix=time.time()))
    print(json.dumps(dict(protocol=protocol["id"], sha256=sha(B / "PROTOCOL.json"),
                          sources=len(sources), formal_updates=0), indent=2))


if __name__ == "__main__":
    prepare()
