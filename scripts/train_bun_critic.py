#!/usr/bin/env python3
"""Train and evaluate Bun critics while keeping actor checkpoints immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb.bun_critic import (
    AUX_TARGET_NAMES,
    frozen_feature_critic_forward,
    frozen_transformer_features,
    independent_critic_aux_forward,
    independent_critic_forward,
    init_frozen_feature_critic,
    init_independent_critic,
    weighted_hl_gauss_loss,
)
from jax_bomb.jax_net import transformer_forward


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def padded_index_batches(indices, batch_size):
    """固定 batch shape；尾部 padding 的 loss 权重严格为零。"""
    indices = np.asarray(indices, np.int32)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        valid = np.ones((len(batch),), np.float32)
        if len(batch) < batch_size:
            if len(batch) == 0:
                continue
            padding = np.full((batch_size - len(batch),), batch[-1], np.int32)
            batch = np.concatenate([batch, padding])
            valid = np.concatenate([
                valid, np.zeros((len(padding),), np.float32)])
        yield batch, valid


def load_params(path):
    with open(path, "rb") as file:
        value = pickle.load(file)
    value = value.get("params", value) if isinstance(value, dict) else value
    return jax.tree.map(jnp.asarray, value)


def migrate_independent_critic(params, input_dim, action_count=15):
    """Pad legacy critic inputs and add the action-outcome auxiliary head."""
    params = jax.tree.map(jnp.asarray, params)
    layer1_weight, layer1_bias = params["layer1"]
    if layer1_weight.shape[0] > input_dim:
        raise ValueError(
            f"initial critic input {layer1_weight.shape[0]} exceeds dataset input {input_dim}")
    if layer1_weight.shape[0] < input_dim:
        layer1_weight = jnp.pad(
            layer1_weight, ((0, input_dim - layer1_weight.shape[0]), (0, 0)))
        params = {**params, "layer1": (layer1_weight, layer1_bias)}
    expected_aux = action_count * len(AUX_TARGET_NAMES)
    if "aux" not in params:
        hidden = layer1_weight.shape[1]
        params = {**params, "aux": (
            jnp.zeros((hidden, expected_aux), jnp.float32),
            jnp.zeros((expected_aux,), jnp.float32),
        )}
    elif params["aux"][1].shape[0] < expected_aux:
        weight, bias = params["aux"]
        params = {**params, "aux": (
            jnp.pad(weight, ((0, 0), (0, expected_aux - weight.shape[1]))),
            jnp.pad(bias, ((0, expected_aux - bias.shape[0]),)),
        )}
    return params


def action_aux_targets(data):
    actor_ids = np.asarray(data["actor_id"], np.int32)
    action_count = np.asarray(data["legal"]).shape[1]
    rows = np.broadcast_to(np.arange(len(actor_ids))[:, None], (len(actor_ids), action_count))
    actions = np.broadcast_to(np.arange(action_count)[None, :], rows.shape)
    actor_columns = np.broadcast_to(actor_ids[:, None], rows.shape)
    mutual = np.asarray(data["first_mutual_death"], np.bool_)
    credited = np.asarray(data["first_credited_kill"], np.bool_)[
        rows, actions, actor_columns]
    causal = np.asarray(data["first_causal_kill"], np.bool_)[
        rows, actions, actor_columns]
    own_bomb = np.asarray(data["first_own_bomb_defeat"], np.bool_)[
        rows, actions, actor_columns]
    survival = np.asarray(data["survivable"], np.bool_)
    nontrade_kill = (credited | causal) & ~mutual
    safe_escape = np.asarray(data.get("safe_escape_exists", survival), np.bool_)
    forced = np.asarray(data.get("forced_kill_state", nontrade_kill), np.bool_)
    bomb_forced = np.asarray(data.get("bomb_creates_forced_kill", forced), np.bool_)
    bomb_trade = np.asarray(data.get("bomb_creates_trade", mutual), np.bool_)
    own_risk = np.asarray(data.get("own_bomb_death_risk", own_bomb), np.bool_)
    minimum = np.asarray(data.get(
        "minimum_escape_time", np.where(safe_escape, 1.0, 40.0)), np.float32)
    minimum_score = np.where(safe_escape, np.exp(-minimum / 40.0), 0.0)
    return np.stack([
        np.asarray(data.get("self_survive_4s", survival), np.bool_),
        nontrade_kill, own_risk, safe_escape, forced, bomb_forced,
        bomb_trade, minimum_score,
    ], axis=-1).astype(np.float32)


def expected_calibration_error(probability, target, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(target), 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper)
        if mask.any():
            error += mask.sum() / total * abs(probability[mask].mean() - target[mask].mean())
    return float(error)


def regression_metrics(target, prediction):
    target = np.asarray(target, np.float64)
    prediction = np.asarray(prediction, np.float64)
    residual = target - prediction
    variance = float(np.var(target))
    corr = spearmanr(target, prediction).statistic if len(target) > 1 else math.nan
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "explained_variance": float(1.0 - np.var(residual) / variance) if variance > 1e-12 else 0.0,
        "spearman": float(corr) if np.isfinite(corr) else 0.0,
    }


def classification_metrics(target, probability):
    target = np.asarray(target, np.float64)
    probability = np.clip(np.asarray(probability, np.float64), 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((probability - target) ** 2)),
        "ece_10": expected_calibration_error(probability, target),
    }


def actor_features(actor_params, obs, global_state, batch_size=256):
    fn = jax.jit(lambda o, g: frozen_transformer_features(actor_params, o, g))
    rows = []
    for start in range(0, len(obs), batch_size):
        rows.append(np.asarray(fn(
            jnp.asarray(obs[start:start + batch_size], jnp.float32) / 255.0,
            jnp.asarray(global_state[start:start + batch_size], jnp.float32))))
    return np.concatenate(rows)


def actor_baseline_values(actor_params, actor_ids, obs, global_state):
    prediction = np.zeros((len(obs),), np.float32)
    for actor_id, params in enumerate(actor_params):
        indices = np.flatnonzero(actor_ids == actor_id)
        if not len(indices):
            continue
        forward = jax.jit(lambda o, g: transformer_forward(params, o, g)[2])
        prediction[indices] = np.asarray(forward(
            jnp.asarray(obs[indices], jnp.float32) / 255.0,
            jnp.asarray(global_state[indices], jnp.float32)))
    return prediction


def make_weights(bucket_ids, train_mask):
    counts = np.bincount(bucket_ids[train_mask])
    inverse = np.zeros_like(counts, np.float32)
    inverse[counts > 0] = 1.0 / counts[counts > 0]
    weights = inverse[bucket_ids]
    return weights / max(float(weights[train_mask].mean()), 1e-8)


def auxiliary_metrics(aux_probability, targets, legal):
    names = ("survival", "nontrade_kill", "own_bomb")
    result = {}
    for index, name in enumerate(names):
        selected_target = targets[..., index][legal]
        selected_probability = aux_probability[..., index][legal]
        result[name] = {
            **classification_metrics(selected_target, selected_probability),
            "accuracy": float(((selected_probability >= 0.5) == selected_target).mean()),
            "positive_rate": float(selected_target.mean()),
            "samples": int(len(selected_target)),
        }
    return result


def evaluate(name, value, win_probability, action_q, data, mask,
             aux_probability=None, aux_targets=None):
    indices = np.flatnonzero(mask)
    value_target = data["value_target"][indices]
    win_target = data["win_target"][indices]
    result = {
        "name": name,
        "states": int(len(indices)),
        "value": regression_metrics(value_target, value[indices]),
        "win": classification_metrics(win_target, win_probability[indices]),
        "buckets": {},
        "actors": {},
        "opponents": {},
    }
    for actor_id in np.unique(data["actor_id"][indices]):
        actor_indices = indices[data["actor_id"][indices] == actor_id]
        result["actors"][str(int(actor_id))] = {
            **regression_metrics(
                data["value_target"][actor_indices], value[actor_indices]),
            **classification_metrics(
                data["win_target"][actor_indices], win_probability[actor_indices]),
            "states": int(len(actor_indices)),
        }
    for bucket in np.unique(data["bucket_name"][indices]):
        bucket_indices = indices[data["bucket_name"][indices] == bucket]
        result["buckets"][str(bucket)] = {
            **regression_metrics(data["value_target"][bucket_indices], value[bucket_indices]),
            **classification_metrics(data["win_target"][bucket_indices], win_probability[bucket_indices]),
            "states": int(len(bucket_indices)),
        }
    if "league_opponent_name" in data:
        for opponent in np.unique(data["league_opponent_name"][indices]):
            opponent_indices = indices[data["league_opponent_name"][indices] == opponent]
            result["opponents"][str(opponent)] = {
                **regression_metrics(
                    data["value_target"][opponent_indices], value[opponent_indices]),
                **classification_metrics(
                    data["win_target"][opponent_indices],
                    win_probability[opponent_indices]),
                "states": int(len(opponent_indices)),
            }
    labeled = indices[(data["good_action"][indices] >= 0) & (data["bad_action"][indices] >= 0)]
    if len(labeled):
        good = data["good_action"][labeled].astype(np.int32)
        bad = data["bad_action"][labeled].astype(np.int32)
        pair_ok = action_q[labeled, good] > action_q[labeled, bad]
        result["pair_accuracy"] = float(pair_ok.mean())
        result["pair_count"] = int(len(labeled))
    else:
        result["pair_accuracy"] = None
        result["pair_count"] = 0
    legal = data["legal"][indices]
    true_advantage = data["q"][indices] - value_target[:, None]
    predicted_advantage = action_q[indices] - value[indices, None]
    confident = legal & (np.abs(true_advantage) >= 0.25)
    result["advantage_sign_accuracy"] = float(
        (np.sign(true_advantage[confident]) == np.sign(predicted_advantage[confident])).mean()
    ) if confident.any() else None
    td_error = (action_q[indices] - data["q"][indices])[legal]
    result["counterfactual_td_error_variance"] = float(np.var(td_error))
    result["counterfactual_td_rmse"] = float(np.sqrt(np.mean(td_error ** 2)))
    deterministic_key = (
        "deterministic_return" if "deterministic_return" in data
        else "exact_return")
    deterministic_return = data[deterministic_key][indices]
    policy_probability = data["policy_probability"][indices]
    chosen = np.argmax(policy_probability, axis=1)
    rows = np.arange(len(indices))
    realized_return = deterministic_return[rows, chosen]
    predicted_q = action_q[indices][rows, chosen]
    fixed_td_error = predicted_q - realized_return
    deterministic_value = np.sum(policy_probability * deterministic_return, axis=1)
    true_advantage = realized_return - deterministic_value
    predicted_advantage = predicted_q - value[indices]
    confident_rollout = np.abs(true_advantage) >= 0.25
    result["fixed_rollout_td_error_variance"] = float(np.var(fixed_td_error))
    result["fixed_rollout_td_rmse"] = float(np.sqrt(np.mean(fixed_td_error ** 2)))
    result["fixed_rollout_advantage_sign_accuracy"] = float(
        (np.sign(true_advantage[confident_rollout])
         == np.sign(predicted_advantage[confident_rollout])).mean()
    ) if confident_rollout.any() else None
    result["fixed_rollout_advantage_count"] = int(confident_rollout.sum())
    if aux_probability is not None and aux_targets is not None:
        result["action_outcomes"] = auxiliary_metrics(
            aux_probability[indices], aux_targets[indices], legal)
    return result


def train_head(features, next_features, data, train_mask, validation_mask, args,
               initial_value_head=None):
    key = jax.random.PRNGKey(args.seed)
    params = init_frozen_feature_critic(key, features.shape[1])
    if initial_value_head is not None:
        actor_weight, actor_bias = initial_value_head
        initialized_weight = params["value"]["weight"].at[
            :actor_weight.shape[0]].set(actor_weight)
        params = {
            **params,
            "value": {"weight": initialized_weight, "bias": actor_bias},
        }
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    weights = jnp.asarray(make_weights(data["bucket_id"], train_mask))
    train_indices = np.flatnonzero(train_mask)

    @jax.jit
    def step(params, opt_state, indices, valid_mask):
        def loss_fn(current):
            _, logits, win_logit = frozen_feature_critic_forward(current, features[indices])
            sample_weights = weights[indices] * valid_mask
            value_loss = weighted_hl_gauss_loss(
                logits, data["value_target"][indices], sample_weights)
            win_loss = optax.sigmoid_binary_cross_entropy(
                win_logit, data["win_target"][indices])
            win_loss = jnp.sum(win_loss * sample_weights) / jnp.maximum(
                sample_weights.sum(), 1e-8)
            return value_loss + args.win_coef * win_loss
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    best_params = params
    best_rmse = float("inf")
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        shuffled = rng.permutation(train_indices)
        for batch, valid_mask in padded_index_batches(shuffled, args.batch_size):
            params, opt_state, _ = step(
                params, opt_state, jnp.asarray(batch), jnp.asarray(valid_mask))
        if epoch % args.eval_every == 0 or epoch + 1 == args.epochs:
            prediction = np.asarray(frozen_feature_critic_forward(params, features)[0])
            rmse = regression_metrics(
                np.asarray(data["value_target"])[validation_mask], prediction[validation_mask])["rmse"]
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = jax.device_get(params)
    value, _, win_logit = frozen_feature_critic_forward(best_params, features)
    next_value = frozen_feature_critic_forward(
        best_params, next_features.reshape(-1, next_features.shape[-1]))[0].reshape(
            next_features.shape[:2])
    action_q = np.asarray(data["immediate_reward"]) + args.gamma * np.asarray(next_value) * (
        1.0 - np.asarray(data["immediate_done"], np.float32))
    return best_params, np.asarray(value), np.asarray(jax.nn.sigmoid(win_logit)), action_q


def train_independent(inputs, next_inputs, data, train_mask, validation_mask, args):
    key = jax.random.PRNGKey(args.seed + 1)
    if args.initial_critic:
        params = migrate_independent_critic(
            load_params(args.initial_critic), inputs.shape[1])
    else:
        params = init_independent_critic(key, inputs.shape[1], hidden=args.hidden)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    weights = jnp.asarray(make_weights(data["bucket_id"], train_mask))
    train_indices = np.flatnonzero(train_mask)
    legal = jnp.asarray(data["legal"], jnp.float32)
    aux_targets = jnp.asarray(action_aux_targets(data), jnp.float32)

    @jax.jit
    def step(params, opt_state, indices, valid_mask):
        def loss_fn(current):
            _, logits, win_logit, q = independent_critic_forward(current, inputs[indices])
            aux_logits = independent_critic_aux_forward(current, inputs[indices])
            sample_weights = weights[indices] * valid_mask
            value_loss = weighted_hl_gauss_loss(
                logits, data["value_target"][indices], sample_weights)
            win_loss = optax.sigmoid_binary_cross_entropy(
                win_logit, data["win_target"][indices])
            q_error = optax.huber_loss(q, data["q"][indices], delta=2.0)
            legal_weights = legal[indices] * valid_mask[:, None]
            q_loss = jnp.sum(q_error * legal_weights) / jnp.maximum(
                legal_weights.sum(), 1.0)
            good = data["good_action"][indices]
            bad = data["bad_action"][indices]
            pair_mask = ((good >= 0) & (bad >= 0)
                         & valid_mask.astype(jnp.bool_))
            safe_good = jnp.maximum(good, 0)
            safe_bad = jnp.maximum(bad, 0)
            rows = jnp.arange(len(indices))
            pair_loss = jnp.sum(
                jax.nn.softplus(-(q[rows, safe_good] - q[rows, safe_bad])) * pair_mask
            ) / jnp.maximum(pair_mask.sum(), 1)
            aux_error = optax.sigmoid_binary_cross_entropy(
                aux_logits, aux_targets[indices])
            aux_loss = jnp.sum(
                aux_error * legal_weights[:, :, None]
            ) / jnp.maximum(legal_weights.sum() * aux_error.shape[-1], 1.0)
            weighted_win = jnp.sum(win_loss * sample_weights) / jnp.maximum(
                sample_weights.sum(), 1e-8)
            return (value_loss
                    + args.win_coef * weighted_win
                    + args.q_coef * q_loss + args.pair_coef * pair_loss
                    + args.aux_coef * aux_loss)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    best_params = params
    best_rmse = float("inf")
    rng = np.random.default_rng(args.seed + 1)
    for epoch in range(args.epochs):
        shuffled = rng.permutation(train_indices)
        for batch, valid_mask in padded_index_batches(shuffled, args.batch_size):
            params, opt_state, _ = step(
                params, opt_state, jnp.asarray(batch), jnp.asarray(valid_mask))
        if epoch % args.eval_every == 0 or epoch + 1 == args.epochs:
            prediction = np.asarray(independent_critic_forward(params, inputs)[0])
            rmse = regression_metrics(
                np.asarray(data["value_target"])[validation_mask], prediction[validation_mask])["rmse"]
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = jax.device_get(params)
    value, _, win_logit, q = independent_critic_forward(best_params, inputs)
    aux = jax.nn.sigmoid(independent_critic_aux_forward(best_params, inputs))
    return (best_params, np.asarray(value), np.asarray(jax.nn.sigmoid(win_logit)),
            np.asarray(q), np.asarray(aux))


def critic_micro_update_smoke(params, inputs, data, train_mask, validation_mask, args):
    indices = np.flatnonzero(train_mask)[:min(args.batch_size, int(train_mask.sum()))]
    weights = jnp.asarray(make_weights(np.asarray(data["bucket_id"]), train_mask))
    legal = jnp.asarray(data["legal"], jnp.float32)
    aux_targets = jnp.asarray(action_aux_targets(data), jnp.float32)
    optimizer = optax.adam(args.lr * 0.1)
    opt_state = optimizer.init(params)

    def loss_fn(current):
        _, logits, win_logit, q = independent_critic_forward(current, inputs[indices])
        aux_logits = independent_critic_aux_forward(current, inputs[indices])
        sample_weights = weights[indices]
        value_loss = weighted_hl_gauss_loss(
            logits, data["value_target"][indices], sample_weights)
        win_loss = optax.sigmoid_binary_cross_entropy(
            win_logit, data["win_target"][indices])
        q_error = optax.huber_loss(q, data["q"][indices], delta=2.0)
        q_loss = jnp.sum(q_error * legal[indices]) / jnp.maximum(legal[indices].sum(), 1.0)
        aux_error = optax.sigmoid_binary_cross_entropy(
            aux_logits, aux_targets[indices])
        aux_loss = jnp.sum(aux_error * legal[indices, :, None]) / jnp.maximum(
            legal[indices].sum() * aux_error.shape[-1], 1.0)
        return (value_loss + args.win_coef * jnp.average(
            win_loss, weights=sample_weights) + args.q_coef * q_loss
            + args.aux_coef * aux_loss)

    before_value = np.asarray(independent_critic_forward(params, inputs)[0])
    loss_before, gradients = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    updated = optax.apply_updates(params, updates)
    loss_after = loss_fn(updated)
    after_value = np.asarray(independent_critic_forward(updated, inputs)[0])
    before_rmse = regression_metrics(
        np.asarray(data["value_target"])[validation_mask],
        before_value[validation_mask])["rmse"]
    after_rmse = regression_metrics(
        np.asarray(data["value_target"])[validation_mask],
        after_value[validation_mask])["rmse"]
    finite = all(bool(np.isfinite(np.asarray(leaf)).all())
                 for leaf in jax.tree.leaves(updated))
    return {
        "kind": "fixed_actor_critic_only_one_update",
        "samples": int(len(indices)),
        "learning_rate": args.lr * 0.1,
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "validation_rmse_before": before_rmse,
        "validation_rmse_after": after_rmse,
        "parameters_finite": finite,
        "actor_parameters_touched": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--actor", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-critic")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--win-coef", type=float, default=0.25)
    parser.add_argument("--q-coef", type=float, default=0.25)
    parser.add_argument("--pair-coef", type=float, default=0.20)
    parser.add_argument("--aux-coef", type=float, default=0.20)
    parser.add_argument("--jax-cache-dir")
    args = parser.parse_args(argv)

    if args.jax_cache_dir:
        from qqt_rl.training.jax_cache import configure_persistent_cache
        configure_persistent_cache(explicit=args.jax_cache_dir)

    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = [digest(path) for path in args.actor]
    actor_params = [load_params(path) for path in args.actor]
    loaded = np.load(args.data)
    data_np = {key: loaded[key] for key in loaded.files}
    data = {key: jnp.asarray(value) if value.dtype.kind not in "USO" else value
            for key, value in data_np.items()}
    train_mask = data_np["split_name"] == "train"
    validation_mask = data_np["split_name"] == "validation"
    test_mask = data_np["split_name"] == "test"

    features = np.zeros((len(train_mask), 392), np.float32)
    next_features = np.zeros((len(train_mask), 15, 392), np.float32)
    baseline_value = np.zeros((len(train_mask),), np.float32)
    baseline_next = np.zeros((len(train_mask), 15), np.float32)
    for actor_id, params in enumerate(actor_params):
        indices = np.flatnonzero(data_np["actor_id"] == actor_id)
        if len(indices) == 0:
            continue
        features[indices] = actor_features(
            params, data_np["obs"][indices], data_np["global"][indices])
        flattened_obs = data_np["next_obs"][indices].reshape(
            -1, *data_np["next_obs"].shape[2:])
        flattened_global = data_np["next_global"][indices].reshape(-1, 24)
        next_features[indices] = actor_features(
            params, flattened_obs, flattened_global).reshape(len(indices), 15, -1)
        baseline_value[indices] = actor_baseline_values(
            [params], np.zeros(len(indices), np.int32), data_np["obs"][indices],
            data_np["global"][indices])
        baseline_next[indices] = actor_baseline_values(
            [params], np.zeros(len(flattened_obs), np.int32), flattened_obs,
            flattened_global).reshape(len(indices), 15)

    head_inputs = np.concatenate([features, data_np["context"]], axis=1)
    head_next_inputs = np.concatenate([next_features, data_np["next_context"]], axis=2)
    flat_obs = data_np["obs"].astype(np.float32).reshape(len(train_mask), -1) / 255.0
    raw_inputs = np.concatenate([flat_obs, data_np["global"], data_np["context"]], axis=1)
    next_flat_obs = data_np["next_obs"].astype(np.float32).reshape(len(train_mask), 15, -1) / 255.0
    raw_next = np.concatenate(
        [next_flat_obs, data_np["next_global"], data_np["next_context"]], axis=2)
    raw_mean = np.zeros((1, raw_inputs.shape[1]), np.float32)
    raw_std = np.ones((1, raw_inputs.shape[1]), np.float32)

    head_params = []
    head_value = np.zeros((len(train_mask),), np.float32)
    head_win = np.zeros((len(train_mask),), np.float32)
    head_q = np.zeros((len(train_mask), 15), np.float32)
    for actor_id, actor_params_item in enumerate(actor_params):
        actor_indices = np.flatnonzero(data_np["actor_id"] == actor_id)
        if len(actor_indices) == 0:
            continue
        actor_data = {
            key: value[actor_indices] if hasattr(value, "shape")
            and value.shape[:1] == (len(train_mask),) else value
            for key, value in data.items()
        }
        actor_train = train_mask[actor_indices]
        actor_validation = validation_mask[actor_indices]
        params_item, value_item, win_item, q_item = train_head(
            jnp.asarray(head_inputs[actor_indices]),
            jnp.asarray(head_next_inputs[actor_indices]), actor_data,
            actor_train, actor_validation, args,
            initial_value_head=actor_params_item["heads"]["wv"])
        head_params.append(params_item)
        head_value[actor_indices] = value_item
        head_win[actor_indices] = win_item
        head_q[actor_indices] = q_item
    independent_params, independent_value, independent_win, independent_q, independent_aux = train_independent(
        jnp.asarray(raw_inputs), jnp.asarray(raw_next), data,
        train_mask, validation_mask, args)
    micro_smoke = critic_micro_update_smoke(
        independent_params, jnp.asarray(raw_inputs), data, train_mask,
        validation_mask, args)

    baseline_win = 1.0 / (1.0 + np.exp(-baseline_value / 5.0))
    baseline_q = data_np["immediate_reward"] + args.gamma * baseline_next * (
        1.0 - data_np["immediate_done"].astype(np.float32))
    models = {
        "frozen_actor_value_baseline": (baseline_value, baseline_win, baseline_q),
        "frozen_feature_value_head": (head_value, head_win, head_q),
        "independent_critic": (independent_value, independent_win, independent_q),
    }
    aux_targets = action_aux_targets(data_np)
    results = {}
    for split, mask in (("validation", validation_mask), ("test", test_mask)):
        results[split] = {}
        for name, predictions in models.items():
            aux = independent_aux if name == "independent_critic" else None
            results[split][name] = evaluate(
                name, *predictions, data_np, mask,
                aux_probability=aux,
                aux_targets=aux_targets if aux is not None else None)
    hashes_after = [digest(path) for path in args.actor]
    actor_unchanged = hashes_before == hashes_after
    summary = {
        "schema": "bun_critic_pretrain_eval_v1",
        "created_date": "2026-09-26",
        "data": args.data,
        "actors": [{"path": path, "sha256_before": before, "sha256_after": after}
                   for path, before, after in zip(args.actor, hashes_before, hashes_after)],
        "actor_hash_unchanged": actor_unchanged,
        "initial_critic": ({"path": args.initial_critic,
                            "sha256": digest(args.initial_critic)}
                           if args.initial_critic else None),
        "config": vars(args),
        "results": results,
        "critic_only_micro_update_smoke": micro_smoke,
        "wall_seconds": time.time() - started,
    }
    with (output_dir / "value_head.pkl").open("wb") as file:
        pickle.dump(jax.device_get(head_params), file)
    with (output_dir / "independent_critic.pkl").open("wb") as file:
        pickle.dump(jax.device_get(independent_params), file)
    np.savez_compressed(output_dir / "normalization.npz", mean=raw_mean, std=raw_std)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not actor_unchanged:
        raise SystemExit("actor checkpoint hash changed")


if __name__ == "__main__":
    main()
