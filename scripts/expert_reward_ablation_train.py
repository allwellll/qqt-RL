"""Isolated D-reward worker; never changes the sealed R2 long-run worker."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from scripts import expert_reward_ablation as reward_adapter
from scripts import ppo_scale_longrun_continue_train as base
from scripts.expert_reward_handoff import (
    valid_d_resume_authorization,
    validated_predecessor_terminal,
)
from scripts.population_run_state import atomic_json, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_reward_20261025/D_ablation_v1"
R2 = R / "runs/ppo_scale_20261025/r2_longrun"
GLOBAL_PAUSE = R / "runs/retention_20261021/USER_GPU_PAUSE.json"


def read(path):
    return json.loads(Path(path).read_text())


def notify(message):
    message = message.replace("R2", "D1")
    print(message, flush=True)
    atomic_json(B / "notifications" / f"{time.time_ns()}.json",
                {"message": message, "unix": time.time()})


def authorize():
    protocol = read(B / "PROTOCOL.json")
    scoped = read(B / "SCOPED_GPU_AUTHORIZATION.json")
    seal = read(B / "SOURCE_SEAL.json")
    if (B / "USER_STOP.json").exists() and not valid_d_resume_authorization(
            B, scoped["protocol_sha256"], scoped["source_seal_sha256"],
            seal["predecessor_handoff_amendment_sha256"]):
        return False
    if sha(GLOBAL_PAUSE) != scoped["global_pause_sha256"]:
        return False
    assert scoped["authorized_by"] == "explicit_user_instruction"
    assert scoped["protocol_sha256"] == sha(B / "PROTOCOL.json")
    assert scoped["source_seal_sha256"] == sha(B / "SOURCE_SEAL.json")
    assert scoped["preformal_protocol_amendment_sha256"] == seal["protocol_amendment_sha256"]
    assert sha(B / "PROTOCOL_AMENDMENT_20261025.json") == seal["protocol_amendment_sha256"]
    assert protocol["amendment"]["formal_updates_before_amendment"] == 0
    assert protocol["id"] == "EXPERT-REWARD-ABLATION-20261025-D1"
    assert protocol["updates"] == 320
    assert protocol["budget"]["new_environment_steps_per_arm_seed"] == 5_242_880
    assert protocol["predecessor"] == "runs/ppo_scale_20261025/r2_longrun"
    terminal = validated_predecessor_terminal(
        R, "runs/ppo_scale_20261025/r2_longrun",
        B / "PREDECESSOR_ACCEPTANCE.json")
    assert terminal["kind"] in {"TRAINING_COMPLETE", "USER_ENDED_EARLY"}
    for name, expected in read(B / "SOURCE_SEAL.json")["sources"].items():
        assert sha(R / name) == expected, f"sealed source changed: {name}"
    return True


def configure(arm):
    base.B, base.R, base.read, base.authorize, base.notify = B, R, read, authorize, notify
    base.ARM = arm
    original_setup = base.setup

    def setup_with_reward_arm():
        result = original_setup()
        if arm in {"legacy", "minimal", "minimal_smoke"}:
            treatment = "minimal" if arm in {"minimal", "minimal_smoke"} else "legacy"
            ops = base.P.T.E.ops
            from scripts import je10_control_credit as control_credit
            reward_adapter.bind_reward_arm(ops, control_credit, treatment)
        return result

    base.setup = setup_with_reward_arm
    base.source_for_seed = source_for_seed

    original_checkpoint = base.checkpoint

    def checkpoint_with_arm_queue(base_path, state, metrics, *, formal=True):
        cp, record = original_checkpoint(base_path, state, metrics, formal=False)
        if formal:
            new_update = int(state["new_update"])
            request = dict(
                checkpoint=str(cp.relative_to(R)),
                checkpoint_state_sha256=record["state_sha256"],
                checkpoint_policy_sha256=record["policy_sha256"],
                seed=int(state["seed"]), source_update=64,
                new_update=new_update, environment_steps=record["environment_steps"],
                level="full" if new_update == 320 else "light",
                holdout=new_update == 320, controls_gpu=False,
                reward_arm=arm,
            )
            atomic_json(B / "eval_queue" /
                        f"{base_path.name}_u{new_update:06d}.json", request)
        return cp, record

    base.checkpoint = checkpoint_with_arm_queue
    return original_setup


def source_for_seed(seed):
    """Load the fixed seed-paired R1 u064 parent without selecting a peak."""
    manifest = read(B / "SOURCE_MANIFEST.json")["sources"][str(seed)]
    parent_cp = R / manifest["checkpoint"]
    committed = read(parent_cp / "COMMITTED.json")
    assert sha(parent_cp / "state.pkl") == manifest["state_sha256"] == committed["state_sha256"]
    assert sha(parent_cp / "policy.pt") == manifest["policy_sha256"] == committed["policy_sha256"]
    with (parent_cp / "state.pkl").open("rb") as stream:
        source = pickle.load(stream)
    assert source["seed"] == manifest["source_seed"]
    assert source["arm"] == base.SOURCE_ARM and source["update"] == 64
    assert set(source) >= {"params", "adam", "states", "aux", "keys", "history"}
    assert np.asarray(source["keys"]).shape == (8, 16, 2)
    # The D1 seed maps one-to-one to the corresponding R1 paired seed. Change
    # only the run label; params/Adam/env/aux/RNG/history retain the sealed bytes.
    source["seed"] = seed
    return source, parent_cp, manifest


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args else "formal"
    if command in {"smoke", "restore-smoke"}:
        arm = "minimal_smoke"
    elif command == "formal":
        if len(args) != 2 or args[0] not in {"legacy", "minimal"}:
            raise SystemExit("usage: formal <legacy|minimal> <paired-seed>")
        arm, seed = args[0], int(args[1])
    else:
        raise SystemExit(f"unknown command: {command}")
    configure(arm)
    if command == "smoke":
        base.smoke()
    elif command == "restore-smoke":
        base.restore_smoke()
    else:
        result = base.train_seed(seed, *base.setup())
        if not result and (B / "PAUSED.json").exists():
            notify(f"D实验在安全原子边界暂停：{arm}/seed{seed}。")


if __name__ == "__main__":
    main()
