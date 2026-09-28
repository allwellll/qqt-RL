#!/usr/bin/env python3
"""Train and gate a Bun transformer with deterministic successful demonstrations."""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jax_bomb import bun_env as env
from jax_bomb.bun_expert import PHASE_NAMES
from jax_bomb.jax_net import transformer_forward


BC_GATE = {
    "move_accuracy": 0.90,
    "ability_accuracy": 0.99,
    "bomb_recall": 0.90,
    "phase_move_accuracy": 0.80,
    "lesson_move_accuracy": 0.80,
    "lesson_ability_accuracy": 0.95,
    "rollout_capture_rate": 0.80,
    "rollout_self_death_rate": 0.0,
    "rollout_safe_detonation_rate": 1.0,
}


def _expand(patterns: list[str]) -> list[str]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise ValueError(f"no data matched: {patterns}")
    return paths


def _load(paths: list[str]) -> dict[str, np.ndarray]:
    keys = ("obs", "state", "move_mask", "ability_mask", "move_action",
            "ability_action", "phase")
    chunks = {key: [] for key in keys}
    episode_seeds = []
    for path in paths:
        with np.load(path) as data:
            for key in keys:
                chunks[key].append(data[key])
            chunks.setdefault("lesson", []).append(
                data["lesson"] if "lesson" in data
                else np.zeros(len(data["obs"]), np.uint8))
            chunks.setdefault("sample_weight", []).append(
                data["sample_weight"] if "sample_weight" in data
                else np.ones(len(data["obs"]), np.float32))
            episode_seeds.append(data["episode_seeds"])
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    result["episode_seeds"] = np.concatenate(episode_seeds)
    return result


def _lesson_weights(spec: str) -> np.ndarray:
    weights = np.ones(len(env.LESSON_NAMES), np.float32)
    if not spec:
        return weights
    name_to_index = {name: index for index, name in enumerate(env.LESSON_NAMES)}
    for item in spec.split(","):
        name, raw = item.strip().split("=", 1)
        if name not in name_to_index:
            raise ValueError(f"unknown Bun lesson weight: {name}")
        value = float(raw)
        if value <= 0:
            raise ValueError(f"lesson weight must be positive: {item}")
        weights[name_to_index[name]] = value
    return weights


def _masked_logits(logits, mask):
    return jnp.where(mask, logits, jnp.full_like(logits, -1e9))


def _phase_weights(phases: np.ndarray) -> np.ndarray:
    counts = np.bincount(phases, minlength=len(PHASE_NAMES)).astype(np.float64)
    weights = np.sqrt(phases.size / np.maximum(counts, 1.0))
    weights /= np.mean(weights[phases])
    return np.clip(weights, 0.5, 4.0).astype(np.float32)


def _phase_multipliers(spec: str) -> np.ndarray:
    weights = np.ones(len(PHASE_NAMES), np.float32)
    if not spec:
        return weights
    name_to_index = {name: index for index, name in enumerate(PHASE_NAMES)}
    for item in spec.split(","):
        name, raw = item.strip().split("=", 1)
        if name not in name_to_index:
            raise ValueError(f"unknown Bun phase weight: {name}")
        value = float(raw)
        if value <= 0:
            raise ValueError(f"phase weight must be positive: {item}")
        weights[name_to_index[name]] = value
    return weights


def _build_update(optimizer, temperature: float, ability_coef: float,
                  bomb_weight: float, anchor_params, anchor_coef: float):
    @jax.jit
    def update(params, opt_state, obs, state, move_mask, ability_mask,
               move_action, ability_action, phase_weight):
        def loss_fn(current):
            move_logits, ability_logits, _, _ = transformer_forward(
                current, obs, state)
            move_logits = _masked_logits(move_logits / temperature, move_mask)
            ability_logits = _masked_logits(
                ability_logits / temperature, ability_mask)
            rows = jnp.arange(obs.shape[0])
            move_ce = -jax.nn.log_softmax(move_logits)[rows, move_action]
            ability_ce = -jax.nn.log_softmax(ability_logits)[rows, ability_action]
            ability_weight = phase_weight * jnp.where(
                ability_action == 1, bomb_weight, 1.0)
            move_loss = jnp.sum(move_ce * phase_weight) / jnp.sum(phase_weight)
            ability_loss = (jnp.sum(ability_ce * ability_weight)
                            / jnp.sum(ability_weight))
            if anchor_coef:
                squared_sum = sum(
                    jnp.sum(jnp.square(value - anchor))
                    for value, anchor in zip(
                        jax.tree.leaves(current), jax.tree.leaves(anchor_params)))
                parameter_count = sum(
                    value.size for value in jax.tree.leaves(current))
                anchor_loss = squared_sum / parameter_count
            else:
                anchor_loss = jnp.asarray(0.0, jnp.float32)
            loss = (move_loss + ability_coef * ability_loss
                    + anchor_coef * anchor_loss)
            metrics = jnp.asarray([
                loss, move_loss, ability_loss,
                jnp.mean(jnp.argmax(move_logits, axis=-1) == move_action),
                jnp.mean(jnp.argmax(ability_logits, axis=-1) == ability_action),
                anchor_loss,
            ])
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    return update


def _frame_metrics(params, data: dict[str, np.ndarray], batch_size: int) -> dict:
    totals = {
        "frames": 0, "move_correct": 0, "ability_correct": 0,
        "bomb_frames": 0, "bomb_correct": 0,
    }
    phase_total = np.zeros(len(PHASE_NAMES), np.int64)
    phase_correct = np.zeros(len(PHASE_NAMES), np.int64)
    lesson_total = np.zeros(len(env.LESSON_NAMES), np.int64)
    lesson_move_correct = np.zeros(len(env.LESSON_NAMES), np.int64)
    lesson_ability_correct = np.zeros(len(env.LESSON_NAMES), np.int64)

    @jax.jit
    def predict(obs, state, move_mask, ability_mask):
        move_logits, ability_logits, _, _ = transformer_forward(
            params, obs, state)
        return (jnp.argmax(_masked_logits(move_logits, move_mask), axis=-1),
                jnp.argmax(_masked_logits(ability_logits, ability_mask), axis=-1))

    for start in range(0, len(data["obs"]), batch_size):
        stop = min(start + batch_size, len(data["obs"]))
        move_pred, ability_pred = jax.device_get(predict(
            jnp.asarray(data["obs"][start:stop], jnp.float32) / 255.0,
            jnp.asarray(data["state"][start:stop], jnp.float32) / 255.0,
            jnp.asarray(data["move_mask"][start:stop]),
            jnp.asarray(data["ability_mask"][start:stop])))
        move_action = data["move_action"][start:stop]
        ability_action = data["ability_action"][start:stop]
        phases = data["phase"][start:stop]
        lessons = data["lesson"][start:stop]
        move_ok = move_pred == move_action
        ability_ok = ability_pred == ability_action
        bomb = ability_action == 1
        totals["frames"] += stop - start
        totals["move_correct"] += int(move_ok.sum())
        totals["ability_correct"] += int(ability_ok.sum())
        totals["bomb_frames"] += int(bomb.sum())
        totals["bomb_correct"] += int((ability_ok & bomb).sum())
        for phase in range(len(PHASE_NAMES)):
            selected = phases == phase
            phase_total[phase] += int(selected.sum())
            phase_correct[phase] += int((move_ok & selected).sum())
        for lesson in range(len(env.LESSON_NAMES)):
            selected = lessons == lesson
            lesson_total[lesson] += int(selected.sum())
            lesson_move_correct[lesson] += int((move_ok & selected).sum())
            lesson_ability_correct[lesson] += int((ability_ok & selected).sum())
    return {
        "frames": totals["frames"],
        "move_accuracy": totals["move_correct"] / totals["frames"],
        "ability_accuracy": totals["ability_correct"] / totals["frames"],
        "bomb_frames": totals["bomb_frames"],
        "bomb_recall": totals["bomb_correct"] / totals["bomb_frames"],
        "phase_move_accuracy": {
            PHASE_NAMES[index]: phase_correct[index] / phase_total[index]
            for index in range(len(PHASE_NAMES)) if phase_total[index]
        },
        "phase_frames": {
            PHASE_NAMES[index]: int(phase_total[index])
            for index in range(len(PHASE_NAMES))
        },
        "lesson_move_accuracy": {
            env.LESSON_NAMES[index]: lesson_move_correct[index] / lesson_total[index]
            for index in range(len(env.LESSON_NAMES)) if lesson_total[index]
        },
        "lesson_ability_accuracy": {
            env.LESSON_NAMES[index]: lesson_ability_correct[index] / lesson_total[index]
            for index in range(len(env.LESSON_NAMES)) if lesson_total[index]
        },
        "lesson_frames": {
            env.LESSON_NAMES[index]: int(lesson_total[index])
            for index in range(len(env.LESSON_NAMES)) if lesson_total[index]
        },
    }


def _rollout_metrics(params, seeds: np.ndarray) -> dict:
    env.prepare()
    env.configure_training("full=1", 1)
    seed_keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seeds])
    step_keys = jnp.stack([
        jax.random.PRNGKey(int(seed) ^ 0x5A17) for seed in seeds])
    states = jax.vmap(env._fresh)(seed_keys)
    batch = len(seeds)

    def observe(current):
        danger = jax.vmap(
            lambda state: env._danger_map(
                state.core.fuse, state.core.wall,
                state.core.bomb_blast, state.core.brick))(current)
        obs = jax.vmap(lambda state, danger_map: env.make_obs(
            state, 0, danger_map))(current, danger)
        global_state = jax.vmap(lambda state: env.global_vec(state, 0))(current)
        move_mask, ability_mask = jax.vmap(env.legal_mask)(current)
        return obs, global_state, move_mask[:, 0], ability_mask[:, 0]

    def select_state(old, new, active):
        shape = (active.shape[0],) + (1,) * (old.ndim - 1)
        return jnp.where(active.reshape(shape), new, old)

    @jax.jit
    def run(current, keys):
        initial = (
            current, keys, jnp.ones((batch,), jnp.bool_),
            jnp.zeros((batch,), jnp.bool_),
            jnp.zeros((batch,), jnp.bool_),
            jnp.zeros((batch,), jnp.bool_),
            jnp.zeros((batch,), jnp.bool_),
            jnp.zeros((batch,), jnp.int32),
            jnp.zeros((batch,), jnp.int32),
            jnp.zeros((batch,), jnp.int32),
        )

        def body(_, carry):
            (state, rng, active, captured, stole, crossed, died,
             own_detonations, safe_detonations, owner_walls) = carry
            obs, global_state, move_mask, ability_mask = observe(state)
            move_logits, ability_logits, _, _ = transformer_forward(
                params, obs, global_state)
            move = jnp.argmax(_masked_logits(move_logits, move_mask), axis=-1)
            ability = jnp.argmax(
                _masked_logits(ability_logits, ability_mask), axis=-1)
            actions = jnp.stack([
                jnp.stack([move, ability], axis=-1),
                jnp.tile(jnp.asarray([[4, 0]], jnp.int32), (batch, 1)),
            ], axis=1)
            split = jax.vmap(jax.random.split)(rng)
            next_rng, step_rng = split[:, 0], split[:, 1]
            candidate, done, info = jax.vmap(
                lambda one, action, key: env.step(
                    one, action, key, auto_reset=False, return_info=True)
            )(state, actions, step_rng)
            live = active[:, None]
            captured = captured | (active & info["capture"][:, 0])
            stole = stole | (active & info["steal"][:, 0])
            crossed = crossed | (active & info["carry_cross_home"][:, 0])
            died = died | (active & info["death"][:, 0])
            own_detonations += (live * info["own_detonation"]).astype(
                jnp.int32)[:, 0]
            safe_detonations += (live * info["safe_bomb_escape"]).astype(
                jnp.int32)[:, 0]
            owner_walls += (live * info["walls_by_owner"]).astype(
                jnp.int32)[:, 0]
            candidate = jax.tree.map(
                lambda old, new: select_state(old, new, active), state, candidate)
            return (
                candidate, next_rng, active & ~done, captured, stole, crossed,
                died, own_detonations, safe_detonations, owner_walls,
            )

        return jax.lax.fori_loop(0, env.MAX_STEPS, body, initial)

    (_, _, _, captured, stole, crossed, died, own_detonations,
     safe_detonations, owner_walls) = jax.device_get(run(states, step_keys))
    total_detonations = int(own_detonations.sum())
    return {
        "episode_seeds": seeds.astype(np.int64).tolist(),
        "episodes": batch,
        "capture_rate": float(captured.mean()),
        "steal_rate": float(stole.mean()),
        "carry_cross_home_rate": float(crossed.mean()),
        "self_death_rate": float(died.mean()),
        "episodes_with_owner_wall": float((owner_walls > 0).mean()),
        "own_detonations": total_detonations,
        "safe_detonations": int(safe_detonations.sum()),
        "safe_detonation_rate": (
            float(safe_detonations.sum() / total_detonations)
            if total_detonations else 0.0),
    }


def _gate(frame: dict, rollout: dict) -> dict:
    phase_pass = all(
        accuracy >= BC_GATE["phase_move_accuracy"]
        for accuracy in frame["phase_move_accuracy"].values())
    lesson_move_pass = all(
        accuracy >= BC_GATE["lesson_move_accuracy"]
        for accuracy in frame["lesson_move_accuracy"].values())
    lesson_ability_pass = all(
        accuracy >= BC_GATE["lesson_ability_accuracy"]
        for accuracy in frame["lesson_ability_accuracy"].values())
    checks = {
        "move_accuracy": frame["move_accuracy"] >= BC_GATE["move_accuracy"],
        "ability_accuracy": (
            frame["ability_accuracy"] >= BC_GATE["ability_accuracy"]),
        "bomb_recall": frame["bomb_recall"] >= BC_GATE["bomb_recall"],
        "all_phase_move_accuracy": phase_pass,
        "all_lesson_move_accuracy": lesson_move_pass,
        "all_lesson_ability_accuracy": lesson_ability_pass,
        "rollout_capture_rate": (
            rollout["capture_rate"] >= BC_GATE["rollout_capture_rate"]),
        "rollout_self_death_rate": (
            rollout["self_death_rate"] <= BC_GATE["rollout_self_death_rate"]),
        "rollout_safe_detonation_rate": (
            rollout["safe_detonation_rate"]
            >= BC_GATE["rollout_safe_detonation_rate"]),
    }
    return {"thresholds": BC_GATE, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--validation-data", nargs="+", required=True)
    parser.add_argument("--rollout-seeds-data", nargs="+", default=None)
    parser.add_argument("--init", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ability-coef", type=float, default=1.0)
    parser.add_argument("--bomb-weight", type=float, default=8.0)
    parser.add_argument("--lesson-weights", default="")
    parser.add_argument("--phase-sampling", choices=["uniform", "balanced"],
                        default="balanced")
    parser.add_argument("--phase-weights", default="")
    parser.add_argument("--anchor-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()

    train_paths = _expand(args.train_data)
    validation_paths = _expand(args.validation_data)
    train = _load(train_paths)
    validation = _load(validation_paths)
    overlap = sorted(set(train["episode_seeds"].tolist()) &
                     set(validation["episode_seeds"].tolist()))
    if overlap:
        raise ValueError(f"train/validation seed overlap: {overlap}")

    with open(args.init, "rb") as file:
        checkpoint = pickle.load(file)
    params = checkpoint.get("params", checkpoint) if isinstance(
        checkpoint, dict) else checkpoint
    params = jax.tree.map(jnp.asarray, params)
    anchor_params = params
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    update = _build_update(
        optimizer, args.temperature, args.ability_coef, args.bomb_weight,
        anchor_params, args.anchor_coef)
    phase_weight_table = (
        _phase_weights(train["phase"])
        if args.phase_sampling == "balanced"
        else np.ones(len(PHASE_NAMES), np.float32))
    phase_weight_table *= _phase_multipliers(args.phase_weights)
    phase_weight_table /= np.mean(phase_weight_table[train["phase"]])
    sample_weight = phase_weight_table[train["phase"]]
    lesson_weight_table = _lesson_weights(args.lesson_weights)
    sample_probability = (sample_weight * train["sample_weight"]
                          * lesson_weight_table[train["lesson"]] * np.where(
        train["ability_action"] == 1, args.bomb_weight, 1.0)
    )
    sample_probability /= sample_probability.sum()
    rng = np.random.default_rng(args.seed)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        indices = rng.choice(len(train["obs"]), args.batch_size,
                             replace=True, p=sample_probability)
        params, opt_state, metrics = update(
            params, opt_state,
            jnp.asarray(train["obs"][indices], jnp.float32) / 255.0,
            jnp.asarray(train["state"][indices], jnp.float32) / 255.0,
            jnp.asarray(train["move_mask"][indices]),
            jnp.asarray(train["ability_mask"][indices]),
            jnp.asarray(train["move_action"][indices], jnp.int32),
            jnp.asarray(train["ability_action"][indices], jnp.int32),
            jnp.asarray(sample_weight[indices], jnp.float32))
        if step == 1 or step % 25 == 0 or step == args.steps:
            values = np.asarray(jax.device_get(metrics), np.float64)
            record = {
                "step": step, "loss": float(values[0]),
                "move_loss": float(values[1]),
                "ability_loss": float(values[2]),
                "batch_move_accuracy": float(values[3]),
                "batch_ability_accuracy": float(values[4]),
                "anchor_loss": float(values[5]),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "wb") as file:
        pickle.dump(jax.device_get(params), file)
    frame = _frame_metrics(params, validation, args.eval_batch_size)
    rollout_data = (_load(_expand(args.rollout_seeds_data))
                    if args.rollout_seeds_data else validation)
    rollout = _rollout_metrics(params, rollout_data["episode_seeds"])
    gate = _gate(frame, rollout)
    result = {
        "schema": "bun_bc_train_v1",
        "config": vars(args),
        "train_files": train_paths,
        "validation_files": validation_paths,
        "train_frames": int(len(train["obs"])),
        "validation_frames": int(len(validation["obs"])),
        "seed_overlap": overlap,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "validation_frame_metrics": frame,
        "validation_rollout_metrics": rollout,
        "gate": gate,
        "checkpoint": args.save,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_gate and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
