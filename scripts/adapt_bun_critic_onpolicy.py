#!/usr/bin/env python3
"""Four-cycle frozen-actor on-policy adaptation for the independent Bun critic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_env as env
from jax_bomb import jax_train
from jax_bomb.bun_critic import (
    AUX_TARGET_NAMES,
    independent_critic_aux_forward,
    independent_critic_forward,
    weighted_hl_gauss_loss,
)

SCENARIOS = {
    "full": ("full=1", env.LESSON_FULL, 12),
    "ambush": ("full_ambush=1", env.LESSON_FULL_AMBUSH, 8),
    "combat": ("combat_kill=1", env.LESSON_COMBAT_KILL, 4),
    "post_kill": ("kill_rush=1", env.LESSON_KILL_RUSH, 9),
    "steal": ("near_steal=1", env.LESSON_NEAR_STEAL, 10),
    "return": ("carry_return=1", env.LESSON_CARRY_RETURN, 11),
    "danger_arena": ("danger_arena=1", env.LESSON_DANGER_ARENA, 8),
}
OPPONENTS = {
    "rule_combat": 2,
    "weak": 3,
    "old": 4,
    "recent": 5,
}


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_pickle(path):
    with open(path, "rb") as file:
        value = pickle.load(file)
    return value.get("params", value) if isinstance(value, dict) else value


def migrate_critic(params, input_dim):
    params = jax.tree.map(jnp.asarray, params)
    weight, bias = params["layer1"]
    expected_aux = 15 * len(AUX_TARGET_NAMES)
    if "aux" not in params:
        hidden = weight.shape[1]
        params = {**params, "aux": (
            jnp.zeros((hidden, expected_aux), jnp.float32),
            jnp.zeros((expected_aux,), jnp.float32))}
    elif params["aux"][1].shape[0] < expected_aux:
        aux_weight, aux_bias = params["aux"]
        params = {**params, "aux": (
            jnp.pad(aux_weight, ((0, 0), (0, expected_aux - aux_weight.shape[1]))),
            jnp.pad(aux_bias, ((0, expected_aux - aux_bias.shape[0]),)))}
    if weight.shape[0] == input_dim:
        return params
    if weight.shape[0] > input_dim:
        raise ValueError(f"critic input {weight.shape[0]} exceeds requested {input_dim}")
    padded = jnp.pad(weight, ((0, input_dim - weight.shape[0]), (0, 0)))
    return {**params, "layer1": (padded, bias)}


def context_from_global(global_state, actor_id, opponent_name, lesson_id, bucket_id):
    count = global_state.shape[0]
    context = np.zeros((count, 44), np.float32)
    context[:, actor_id] = 1.0
    coarse_opponent = 0 if opponent_name == "rule_combat" else 1
    context[:, 2 + coarse_opponent] = 1.0
    encoded_lesson = (env.LESSON_FULL_AMBUSH
                      if lesson_id == env.LESSON_DANGER_ARENA else lesson_id)
    context[:, 4 + encoded_lesson] = 1.0
    phase_start = 4 + env.LESSON_DANGER_ARENA
    phase = np.where(global_state[:, 13] >= 0.5, 2,
                     np.where(global_state[:, 12] > 0, 1, 0))
    context[np.arange(count), phase_start + phase] = 1.0
    bucket_start = phase_start + 3
    context[:, bucket_start + bucket_id] = 1.0
    league_start = 37
    opponent_index = tuple(OPPONENTS).index(opponent_name)
    context[:, league_start + opponent_index] = 1.0
    return context


def discounted_returns(reward, done, gamma, bootstrap):
    result = np.zeros_like(reward, np.float32)
    running = np.asarray(bootstrap, np.float32)
    for tick in range(reward.shape[0] - 1, -1, -1):
        running = reward[tick] + gamma * running * (~done[tick])
        result[tick] = running
    return result


def resolved_win_targets(reward, done):
    targets = np.zeros_like(reward, np.float32)
    labeled = np.zeros_like(done, np.bool_)
    next_target = np.zeros((reward.shape[1],), np.float32)
    next_labeled = np.zeros((reward.shape[1],), np.bool_)
    for tick in range(reward.shape[0] - 1, -1, -1):
        terminal = done[tick]
        next_target = np.where(terminal, reward[tick] > 0, next_target)
        next_labeled = terminal | next_labeled
        targets[tick] = next_target
        labeled[tick] = next_labeled
    return targets, labeled


def diagnostics(reward, done, target_return, value, bootstrap, gamma, lam):
    next_value = np.concatenate([value[1:], bootstrap[None]], axis=0)
    delta = reward + gamma * next_value * (~done) - value
    gae = np.zeros_like(reward)
    running = np.zeros((reward.shape[1],), np.float32)
    for tick in range(reward.shape[0] - 1, -1, -1):
        running = delta[tick] + gamma * lam * running * (~done[tick])
        gae[tick] = running
    residual = target_return - value
    variance = float(np.var(target_return))
    correlation = spearmanr(target_return.reshape(-1), value.reshape(-1)).statistic
    true_advantage = residual
    confident = np.abs(true_advantage) >= 0.25
    return {
        "mse": float(np.mean(residual ** 2)),
        "value_bias": float(np.mean(value - target_return)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "ev": float(1.0 - np.var(residual) / variance) if variance > 1e-8 else 0.0,
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "td_variance": float(np.var(delta)),
        "gae_variance": float(np.var(gae - true_advantage)),
        "advantage_sign": float(
            (np.sign(gae[confident]) == np.sign(true_advantage[confident])).mean()
        ) if confident.any() else None,
        "advantage_count": int(confident.sum()),
    }


def make_collect_fn(num_steps, num_envs, weak, old, recent):
    @jax.jit
    def collect(actor, states, key, weights):
        return jax_train.collect_rollout(
            actor, "transformer", states, key, num_steps,
            obs_quant=False, checkpoint=False, flee_bot_ratio=0.0,
            safety_mode="off", bun_opponent_weights=weights,
            bun_opponent_weak_params=weak,
            bun_opponent_old_params=old,
            bun_opponent_recent_params=recent)
    return collect


def collect_one(collect_fn, actor, critic, actor_id, scenario_name, opponent_name,
                split, cycle, seed, num_envs, gamma):
    key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
    states = env.init_batch(key, num_envs)
    weights = np.zeros((len(jax_train.BUN_OPPONENT_NAMES),), np.float32)
    weights[OPPONENTS[opponent_name]] = 1.0
    started = time.time()
    final_states, batch, _, _ = collect_fn(
        actor, states, key, jnp.asarray(weights))
    jax.block_until_ready(batch[5])
    elapsed = time.time() - started
    obs, global_state, _, _, _, reward, done, _ = batch
    obs = np.asarray(obs[:, :num_envs], np.float32)
    global_state = np.asarray(global_state[:, :num_envs], np.float32)
    reward = np.asarray(reward[:, :num_envs], np.float32)
    done = np.asarray(done[:, :num_envs], np.bool_)
    curriculum, lesson_id, bucket_id = SCENARIOS[scenario_name]
    time_steps = obs.shape[0]
    flat_global = global_state.reshape(-1, global_state.shape[-1])
    context = context_from_global(
        flat_global, actor_id, opponent_name, lesson_id, bucket_id
    ).reshape(time_steps, num_envs, -1)
    inputs = np.concatenate([
        obs.reshape(time_steps, num_envs, -1), global_state, context], axis=2)
    final_obs = np.asarray(jax_train.both_perspectives(final_states)[:num_envs], np.float32)
    final_global = np.asarray(jax_train.both_states(final_states)[:num_envs], np.float32)
    final_context = context_from_global(
        final_global, actor_id, opponent_name, lesson_id, bucket_id)
    final_inputs = np.concatenate([
        final_obs.reshape(num_envs, -1), final_global, final_context], axis=1)
    bootstrap, _, _, _ = independent_critic_forward(
        critic, jnp.asarray(final_inputs))
    bootstrap = np.asarray(bootstrap, np.float32)
    returns = discounted_returns(reward, done, gamma, bootstrap)
    win_target, win_labeled = resolved_win_targets(reward, done)
    return {
        "inputs": inputs.astype(np.float32), "bootstrap": bootstrap,
        "reward": reward, "done": done, "returns": returns,
        "win_target": win_target, "win_labeled": win_labeled,
        "scenario": np.full((num_envs,), scenario_name),
        "opponent": np.full((num_envs,), opponent_name),
        "actor_id": np.full((num_envs,), actor_id, np.int8),
        "seed": seed, "split": split, "cycle": cycle,
        "elapsed": elapsed, "curriculum": curriculum,
    }


def concatenate(rows):
    time_keys = ("inputs", "reward", "done", "returns", "win_target",
                 "win_labeled")
    env_keys = ("bootstrap", "scenario", "opponent", "actor_id")
    result = {}
    for key in time_keys:
        result[key] = np.concatenate([row[key] for row in rows], axis=1)
    for key in env_keys:
        result[key] = np.concatenate([row[key] for row in rows], axis=0)
    return result


def load_counterfactual(path, new_context_dim=44):
    loaded = np.load(path)
    obs = loaded["obs"].astype(np.float32).reshape(len(loaded["obs"]), -1) / 255.0
    old_context = loaded["context"].astype(np.float32)
    context = np.pad(old_context, ((0, 0), (0, new_context_dim - old_context.shape[1])))
    inputs = np.concatenate([obs, loaded["global"], context], axis=1)
    actor_ids = loaded["actor_id"].astype(np.int32)
    action_count = loaded["legal"].shape[1]
    rows = np.broadcast_to(np.arange(len(actor_ids))[:, None], (len(actor_ids), action_count))
    actions = np.broadcast_to(np.arange(action_count)[None, :], rows.shape)
    actors = np.broadcast_to(actor_ids[:, None], rows.shape)
    kill = (loaded["first_credited_kill"][rows, actions, actors]
            | loaded["first_causal_kill"][rows, actions, actors])
    kill &= ~loaded["first_mutual_death"]
    survival = loaded["survivable"]
    own = loaded["first_own_bomb_defeat"][rows, actions, actors]
    safe_escape = loaded["safe_escape_exists"] if "safe_escape_exists" in loaded else survival
    forced = loaded["forced_kill_state"] if "forced_kill_state" in loaded else kill
    bomb_forced = loaded["bomb_creates_forced_kill"] if "bomb_creates_forced_kill" in loaded else forced
    bomb_trade = loaded["bomb_creates_trade"] if "bomb_creates_trade" in loaded else loaded["first_mutual_death"]
    own_risk = loaded["own_bomb_death_risk"] if "own_bomb_death_risk" in loaded else own
    minimum = loaded["minimum_escape_time"] if "minimum_escape_time" in loaded else np.where(safe_escape,1.0,40.0)
    aux = np.stack([survival, kill, own_risk, safe_escape, forced,
                    bomb_forced, bomb_trade,
                    np.where(safe_escape, np.exp(-minimum/40.0), 0.0)], axis=-1).astype(np.float32)
    return {
        "inputs": inputs.astype(np.float32),
        "value": loaded["value_target"].astype(np.float32),
        "q": loaded["q"].astype(np.float32),
        "legal": loaded["legal"].astype(np.float32),
        "good": loaded["good_action"].astype(np.int32),
        "bad": loaded["bad_action"].astype(np.int32),
        "aux": aux,
        "split": loaded["split_name"],
    }


def evaluate(params, data, gamma, lam):
    flat_inputs = data["inputs"].reshape(-1, data["inputs"].shape[-1])
    value, _, win_logit, _ = independent_critic_forward(
        params, jnp.asarray(flat_inputs))
    value = np.asarray(value).reshape(data["returns"].shape)
    win_probability = np.asarray(jax.nn.sigmoid(win_logit)).reshape(
        data["win_target"].shape)
    overall = diagnostics(
        data["reward"], data["done"], data["returns"], value,
        data["bootstrap"], gamma, lam)
    labeled = data["win_labeled"]
    overall["brier"] = float(np.mean(
        (win_probability[labeled] - data["win_target"][labeled]) ** 2)
    ) if labeled.any() else None
    result = {"overall": overall, "scenarios": {}, "opponents": {}, "layers": {}}
    env_scenario = data["scenario"]
    env_opponent = data["opponent"]
    for scenario in SCENARIOS:
        columns = np.flatnonzero(env_scenario == scenario)
        if len(columns):
            result["scenarios"][scenario] = diagnostics(
                data["reward"][:, columns], data["done"][:, columns],
                data["returns"][:, columns], value[:, columns],
                data["bootstrap"][columns], gamma, lam)
    for opponent in OPPONENTS:
        columns = np.flatnonzero(env_opponent == opponent)
        if len(columns):
            result["opponents"][opponent] = diagnostics(
                data["reward"][:, columns], data["done"][:, columns],
                data["returns"][:, columns], value[:, columns],
                data["bootstrap"][columns], gamma, lam)
    for scenario in ("full", "ambush"):
        for opponent in OPPONENTS:
            columns = np.flatnonzero(
                (env_scenario == scenario) & (env_opponent == opponent))
            if len(columns):
                result["layers"][f"{scenario}/{opponent}"] = diagnostics(
                    data["reward"][:, columns], data["done"][:, columns],
                    data["returns"][:, columns], value[:, columns],
                    data["bootstrap"][columns], gamma, lam)
    return result


def train_cycle(params, onpolicy, counterfactual, split, epochs, batch_size,
                lr, mix_ratio, seed):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(seed)
    cf_indices = np.flatnonzero(counterfactual["split"] == split)
    flat_inputs = onpolicy["inputs"].reshape(-1, onpolicy["inputs"].shape[-1])
    on_count = len(flat_inputs)
    cf_batch = max(1, int(batch_size * mix_ratio))
    on_batch = max(1, batch_size - cf_batch)

    @jax.jit
    def update(params, opt_state, on_idx, cf_idx):
        def loss_fn(current):
            _, on_logits, on_win, _ = independent_critic_forward(
                current, jnp.asarray(flat_inputs)[on_idx])
            on_value_loss = weighted_hl_gauss_loss(
                on_logits, jnp.asarray(onpolicy["returns"].reshape(-1))[on_idx],
                jnp.ones((len(on_idx),), jnp.float32))
            win_labeled = jnp.asarray(onpolicy["win_labeled"].reshape(-1))[on_idx]
            win_loss_raw = optax.sigmoid_binary_cross_entropy(
                on_win, jnp.asarray(onpolicy["win_target"].reshape(-1))[on_idx])
            win_loss = jnp.sum(win_loss_raw * win_labeled) / jnp.maximum(
                win_labeled.sum(), 1)
            _, cf_logits, _, cf_q = independent_critic_forward(
                current, jnp.asarray(counterfactual["inputs"])[cf_idx])
            cf_value_loss = weighted_hl_gauss_loss(
                cf_logits, jnp.asarray(counterfactual["value"])[cf_idx],
                jnp.ones((len(cf_idx),), jnp.float32))
            legal = jnp.asarray(counterfactual["legal"])[cf_idx]
            q_loss = jnp.sum(optax.huber_loss(
                cf_q, jnp.asarray(counterfactual["q"])[cf_idx], delta=2.0
            ) * legal) / jnp.maximum(legal.sum(), 1.0)
            good = jnp.asarray(counterfactual["good"])[cf_idx]
            bad = jnp.asarray(counterfactual["bad"])[cf_idx]
            pair_mask = (good >= 0) & (bad >= 0)
            rows = jnp.arange(len(cf_idx))
            pair_loss = jnp.sum(jax.nn.softplus(-(
                cf_q[rows, jnp.maximum(good, 0)]
                - cf_q[rows, jnp.maximum(bad, 0)])) * pair_mask
            ) / jnp.maximum(pair_mask.sum(), 1)
            aux_logits = independent_critic_aux_forward(
                current, jnp.asarray(counterfactual["inputs"])[cf_idx])
            aux_error = optax.sigmoid_binary_cross_entropy(
                aux_logits, jnp.asarray(counterfactual["aux"])[cf_idx])
            aux_loss = jnp.sum(aux_error * legal[:, :, None]) / jnp.maximum(
                legal.sum() * aux_error.shape[-1], 1.0)
            return (0.60 * on_value_loss + 0.10 * win_loss
                    + 0.30 * cf_value_loss + 0.15 * q_loss
                    + 0.10 * pair_loss + 0.10 * aux_loss)
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = []
    updates_per_epoch = max(1, int(np.ceil(on_count / on_batch)))
    for _ in range(epochs):
        for _ in range(updates_per_epoch):
            on_idx = jnp.asarray(rng.choice(on_count, on_batch, replace=on_count < on_batch))
            cf_idx = jnp.asarray(rng.choice(cf_indices, cf_batch, replace=len(cf_indices) < cf_batch))
            params, opt_state, loss = update(params, opt_state, on_idx, cf_idx)
            losses.append(float(loss))
    return params, {"loss_first": losses[0], "loss_last": losses[-1],
                    "updates": len(losses)}


def gate(metrics_history):
    current = metrics_history[-1]["validation"]
    ambush = current["scenarios"]["ambush"]
    layers = current["layers"]
    checks = {
        "ambush_ev": ambush["ev"] >= 0.25,
        "ambush_rmse": ambush["rmse"] <= 5.0,
        "ambush_td": ambush["td_variance"] <= 2.0,
        "ambush_gae": ambush["gae_variance"] <= 8.0,
        "ambush_advantage": (ambush["advantage_sign"] is not None
                              and ambush["advantage_sign"] >= 0.85),
        "ambush_layers_nonnegative": all(
            item["ev"] >= 0.0 for name, item in layers.items()
            if name.startswith("ambush/")),
        "combat_ev_nonnegative": current["scenarios"]["combat"]["ev"] >= 0.0,
    }
    if len(metrics_history) >= 2:
        previous = metrics_history[-2]["validation"]["scenarios"]["ambush"]
        checks["ambush_ev_two_cycles"] = previous["ev"] >= 0.25
    else:
        checks["ambush_ev_two_cycles"] = False
    return {"passed": all(checks.values()), "checks": checks,
            "ambush": ambush, "layers": layers}


def save_npz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", action="append", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--recent", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-base", type=int, default=202609261000)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mix-ratio", type=float, default=0.40)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--ignore-capability-gate", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actors = [jax.tree.map(jnp.asarray, load_pickle(path)) for path in args.actor]
    actor_hashes = [digest(path) for path in args.actor]
    weak = jax.tree.map(jnp.asarray, load_pickle(args.weak))
    old = jax.tree.map(jnp.asarray, load_pickle(args.old))
    recent = jax.tree.map(jnp.asarray, load_pickle(args.recent))
    counterfactual = load_counterfactual(args.counterfactual)
    input_dim = counterfactual["inputs"].shape[1]
    params = migrate_critic(load_pickle(args.critic), input_dim)
    env.prepare()

    seed_manifest = []
    history = []
    for cycle in range(1, args.cycles + 1):
        collected = {split: [] for split in ("train", "validation", "test")}
        for scenario_index, (scenario_name, (curriculum, _, _)) in enumerate(SCENARIOS.items()):
            env.configure_training(curriculum, 1, reward_profile="combat_evolution")
            collect_fn = make_collect_fn(
                args.num_steps, args.num_envs, weak, old, recent)
            for split_index, split in enumerate(("train", "validation", "test")):
                for actor_id, actor in enumerate(actors):
                    for opponent_index, opponent_name in enumerate(OPPONENTS):
                        seed = (args.seed_base + cycle * 1_000_000
                                + split_index * 100_000 + scenario_index * 10_000
                                + actor_id * 1_000 + opponent_index * 100)
                        row = collect_one(
                            collect_fn, actor, params, actor_id, scenario_name,
                            opponent_name, split, cycle, seed,
                            args.num_envs, args.gamma)
                        collected[split].append(row)
                        seed_manifest.append({
                            "cycle": cycle, "split": split,
                            "scenario": scenario_name, "actor_id": actor_id,
                            "opponent": opponent_name, "seed": seed})
            del collect_fn
        cycle_started = time.time()
        split_data = {split: concatenate(collected[split])
                      for split in ("train", "validation", "test")}
        for split, data in split_data.items():
            save_npz(output_dir / f"cycle_{cycle:03d}_{split}.npz", data)
        params, train_info = train_cycle(
            params, split_data["train"], counterfactual, "train",
            args.epochs, args.batch_size, args.lr, args.mix_ratio,
            args.seed_base + cycle)
        finite = all(bool(np.isfinite(np.asarray(leaf)).all())
                     for leaf in jax.tree.leaves(params))
        validation = evaluate(params, split_data["validation"], args.gamma, args.lam)
        test = evaluate(params, split_data["test"], args.gamma, args.lam)
        actor_hashes_after = [digest(path) for path in args.actor]
        result = {
            "cycle": cycle, "train": train_info,
            "validation": validation, "test": test,
            "critic_finite": finite,
            "actor_hash_unchanged": actor_hashes == actor_hashes_after,
            "wall_seconds": time.time() - cycle_started,
        }
        history.append(result)
        result["gate"] = gate(history)
        with (output_dir / f"cycle_{cycle:03d}.json").open("w") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        with (output_dir / f"cycle_{cycle:03d}.critic.pkl").open("wb") as file:
            pickle.dump(jax.device_get(params), file)
        print(json.dumps({"cycle": cycle, "gate": result["gate"],
                          "validation_ambush": validation["scenarios"]["ambush"]},
                         ensure_ascii=False), flush=True)
        if not finite or actor_hashes != actor_hashes_after:
            raise RuntimeError("finite/hash gate failed")

    seed_sets = {
        split: {item["seed"] for item in seed_manifest if item["split"] == split}
        for split in ("train", "validation", "test")}
    leakage = any(seed_sets[left] & seed_sets[right]
                  for left, right in (("train", "validation"),
                                      ("train", "test"),
                                      ("validation", "test")))
    if leakage:
        raise RuntimeError("seed leakage")

    final_gate = gate(history)
    manifest = {
        "schema": "bun_critic_onpolicy_adaptation_v1",
        "created_date": "2026-09-26", "config": vars(args),
        "actors": [{"path": path, "sha256": value}
                   for path, value in zip(args.actor, actor_hashes)],
        "seed_manifest": seed_manifest, "seed_leakage": leakage,
        "cycles": history, "final_gate": final_gate,
        "wall_seconds": time.time() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"final_gate": final_gate,
                      "wall_seconds": manifest["wall_seconds"]},
                     ensure_ascii=False, indent=2))
    if not final_gate["passed"] and not args.ignore_capability_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
