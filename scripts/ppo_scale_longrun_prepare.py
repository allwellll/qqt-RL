"""Pre-register the PPO-only batch-scale and long-horizon experiment."""
import copy
import shutil
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/ppo_scale_20261025/r1_longrun"
PARENT = R / "runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032"
R16 = R / "runs/dagger_rl_20261025/r16_retention_schedule"
SEEDS = [2026102751, 2026102752, 2026102753]


def prepare():
    if (B / "PROTOCOL.json").exists():
        raise RuntimeError("PPO scale namespace already registered")
    old = __import__("json").loads((R16 / "PROTOCOL.json").read_text())
    parent_input = R16 / "inputs/parent.pkl"
    assert sha(PARENT / "state.pkl") == old["parent"]["state_sha256"]
    assert sha(PARENT / "policy.pt") == old["parent"]["policy_sha256"]
    p = dict(
        id="PPO-SCALE-LONGRUN-20261025-R1",
        namespace="runs/ppo_scale_20261025/r1_longrun",
        mode="ppo_only_batch_scale_long_horizon",
        parent=dict(checkpoint=str(PARENT.relative_to(R)),
                    state_sha256=sha(PARENT / "state.pkl"),
                    policy_sha256=sha(PARENT / "policy.pt")),
        parent_input_sha256=sha(parent_input),
        arms=["ppo_base_batch", "ppo_scaled_batch"],
        paired_seeds=SEEDS, updates_by_arm={"ppo_base_batch": 128, "ppo_scaled_batch": 64},
        checkpoints_by_arm={"ppo_base_batch": [0, 1, 16, 32, 64, 96, 128],
                           "ppo_scaled_batch": [0, 1, 8, 16, 32, 48, 64]},
        single_variable="global on-policy PPO rollout/minibatch scale; no DAgger loss",
        hypothesis="A larger matched-reuse batch can improve long-horizon PPO stability without changing policy objective or action semantics.",
        falsification="If scaled throughput is not effective, numerical stability fails, or natural/battle safety tails degrade against the matched base arm, retain the base configuration and do not promote either by peak.",
        common=dict(reward="R16 legacy reward unchanged", input="R16 public observation unchanged",
                    mask="R16 action masks unchanged", opponents="fixed JAX rule3",
                    network="R16 network unchanged", optimizer="Adam semantics restored from common parent",
                    gamma=.995, gae_lambda=.95, clip=.1, value=.5, entropy=.005,
                    gradient_clip=.5, epochs=1, lr=7.5e-5,
                    dagger_coefficient=0., combat_kl_coefficient=0.,
                    no_offline_behavior_as_ppo=True, no_synthetic_carry=True),
        batch=dict(ppo_base_batch=dict(envs_per_device=8, devices=8, rollout_steps=128,
                                       global_rollout_batch=8192, global_minibatch=256,
                                       local_minibatch=32, optimizer_steps_per_update=32),
                   ppo_scaled_batch=dict(envs_per_device=16, devices=8, rollout_steps=128,
                                         global_rollout_batch=16384, global_minibatch=512,
                                         local_minibatch=64, optimizer_steps_per_update=32)),
        budget=dict(updates_by_arm={"ppo_base_batch": 128, "ppo_scaled_batch": 64},
              environment_steps_per_run={"ppo_base_batch": 1048576,
              "ppo_scaled_batch": 1048576}, total_environment_steps=6291456,
              optimizer_steps_per_run={"ppo_base_batch": 4096, "ppo_scaled_batch": 2048},
              total_optimizer_steps=18432,
              saved_at_environment_steps=True),
        sampling_audit=dict(student_categorical=True, deployment_greedy_separate=True,
                            natural_and_real_bank_sources=True, finite_budget_steps=8192,
                            no_capability_gate=True),
        evaluation=dict(natural_games=64, tactical_games=64,
                        hunter_games_per_tier=64, tiers=["hard", "normal", "easy"],
                        checkpoints="u1/u8/u16/u32/u48/u64; parent and both matched arms",
                        holdout_seed_start=2026112600,
                        engines=["Python Tactical v2", "production JS Hunter", "native JS full2400"],
                        bank="regression only; existing single-map P0 allfar limitation disclosed",
                        no_holdout_tuning=True),
        async_owner="scripts.ppo_scale_longrun_cpu; no GPU launch",
        successor="After all six fixed runs and full curve/holdout/tail evidence, choose only a separately preregistered bounded question; no automatic DAgger or old queue.",
        stopping="fixed six equal-environment-budget runs; only technical failure or explicit user stop interrupts healthy training",
        old_queues_remain_paused=True, old_r18_formal_steps=0, unix=time.time())
    B.mkdir(parents=True, exist_ok=True)
    for folder in ["inputs", "cpu_eval_inputs", "inputs/sealed_sources", "cache", "CPU_QUEUE", "notifications"]:
        (B / folder).mkdir(parents=True, exist_ok=True)
    atomic_json(B / "PROTOCOL.json", p)
    shutil.copyfile(parent_input, B / "inputs/parent.pkl")
    for name in ["spawns.json", "regions.json"]:
        shutil.copyfile(R16 / "cpu_eval_inputs" / name, B / "cpu_eval_inputs" / name)
    ev = copy.deepcopy(__import__("json").loads((R16 / "CPU_EVAL_PROTOCOL.json").read_text()))
    ev.update(holdout_seed_start=2026112600, no_training_launch=True,
              evaluation_namespace="ppo_scale_20261025/r1_longrun")
    atomic_json(B / "CPU_EVAL_PROTOCOL.json", ev)
    atomic_json(B / "SCOPED_GPU_AUTHORIZATION.json", dict(
        authorized_by="explicit_user_instruction",
        protocol_sha256=sha(B / "PROTOCOL.json"),
        old_global_pause_sha256=sha(R / "runs/retention_20261021/USER_GPU_PAUSE.json"),
        old_continuous_resume=False, old_queues_remain_paused=True,
        r18_formal_not_started=True))
    atomic_json(B / "PREREGISTRATION.json", dict(
        passed=True, protocol_sha256=sha(B / "PROTOCOL.json"),
        parent_clean=True, selection_pending=False, capability_gate=False,
        no_holdout_tuning=True))
    seal_names = [
        "scripts/ppo_scale_longrun_run.py", "scripts/ppo_scale_longrun_train.py",
        "scripts/ppo_scale_longrun_cpu.py", "scripts/ppo_scale_longrun_prepare.py",
        "scripts/dagger_rl_r4_train.py", "scripts/carrier_mechanism_engine.py",
        "scripts/combat_item_engine.py", "scripts/dagger_rl_oracle.py",
        "scripts/retention_phase_export.py", "scripts/je10_route_oracle.py",
        "scripts/dagger_rl_metrics_v2.js", "scripts/dagger_rl_metrics_v2_preload.js",
        "tests/test_ppo_scale_longrun.py",
    ]
    atomic_json(B / "SOURCE_SEAL.json", dict(
        protocol_sha256=sha(B / "PROTOCOL.json"),
        sources={name: sha(R / name) for name in seal_names},
        no_run_or_checkpoint_files=True))
    for name in seal_names:
        target = B / "inputs/sealed_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R / name, target)
    atomic_json(B / "R18_HANDOFF_RECEIPT.json", dict(
        r18_namespace="runs/dagger_rl_20261025/r18_item_chain",
        r18_formal_environment_steps=0, r18_formal_optimizer_steps=0,
        r18_technical_smoke_preserved=True, r18_user_stop_sha256=sha(R / "runs/dagger_rl_20261025/r18_item_chain/USER_STOP.json")))
    print(sha(B / "PROTOCOL.json"), flush=True)


if __name__ == "__main__":
    prepare()
