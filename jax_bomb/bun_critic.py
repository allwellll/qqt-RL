"""Frozen-actor critic pretraining utilities for Bun experiments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from .jax_net import BIN_CENTERS, NUM_VALUE_BINS, V_MAX, V_MIN, _tf_block

AUX_TARGET_NAMES = (
    "self_survive_4s", "nontrade_kill", "own_bomb_death_risk",
    "safe_escape_exists", "forced_kill_state", "bomb_creates_forced_kill",
    "bomb_creates_trade", "minimum_escape_time_score",
)


def frozen_transformer_features(actor_params, obs, global_state):
    """Return pooled actor features without exposing them to critic gradients."""
    n = obs.shape[0]
    channels, height, width = obs.shape[1:]
    bf16 = jnp.bfloat16
    patch_dim = actor_params["tok"][0].shape[0]
    patch = int(round((patch_dim / channels) ** 0.5))
    grid_height = -(-height // patch)
    grid_width = -(-width // patch)
    token_count = grid_height * grid_width
    padded_height = grid_height * patch
    padded_width = grid_width * patch
    tokens = obs.astype(bf16)
    if padded_height != height or padded_width != width:
        tokens = jnp.pad(tokens, [
            (0, 0), (0, 0), (0, padded_height - height),
            (0, padded_width - width)])
    tokens = tokens.reshape(
        n, channels, grid_height, patch, grid_width, patch)
    tokens = tokens.transpose(0, 2, 4, 1, 3, 5).reshape(
        n, token_count, channels * patch * patch)
    token_weight, token_bias = actor_params["tok"]
    position = actor_params["pos"].astype(bf16)
    tokens = (tokens @ token_weight.astype(bf16)
              + token_bias.astype(bf16) + position[:, :token_count])
    state_token = (global_state.astype(bf16)
                   @ actor_params["state_w"].astype(bf16)
                   + actor_params["state_b"].astype(bf16))
    tokens = jnp.concatenate([
        tokens, state_token[:, None] + position[:, -1:]], axis=1)
    for block in actor_params["blocks"]:
        tokens = _tf_block(tokens, block, 4)
    return jax.lax.stop_gradient(
        tokens[:, :token_count].mean(axis=1).astype(jnp.float32))


def value_from_logits(logits):
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * BIN_CENTERS, axis=-1)


def init_value_head(key, feature_dim):
    weight = random.normal(key, (feature_dim, NUM_VALUE_BINS)) / jnp.sqrt(feature_dim)
    return {"weight": weight, "bias": jnp.zeros((NUM_VALUE_BINS,))}


def init_frozen_feature_critic(key, feature_dim):
    value_key, win_key = random.split(key)
    value = init_value_head(value_key, feature_dim)
    win_weight = random.normal(win_key, (feature_dim,)) / jnp.sqrt(feature_dim)
    return {"value": value, "win_weight": win_weight,
            "win_bias": jnp.asarray(0.0, jnp.float32)}


def value_head_forward(params, features):
    logits = features @ params["weight"] + params["bias"]
    return value_from_logits(logits), logits


def frozen_feature_critic_forward(params, features):
    value, logits = value_head_forward(params["value"], features)
    win_logit = features @ params["win_weight"] + params["win_bias"]
    return value, logits, win_logit


def hl_gauss_targets(targets, sigma=1.5):
    half_width = (V_MAX - V_MIN) / (NUM_VALUE_BINS - 1) / 2.0
    upper = (BIN_CENTERS[None] + half_width - targets[:, None]) / sigma
    lower = (BIN_CENTERS[None] - half_width - targets[:, None]) / sigma
    probabilities = jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower)
    return probabilities / jnp.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-8)


def weighted_hl_gauss_loss(logits, targets, weights):
    target_distribution = hl_gauss_targets(targets)
    losses = -jnp.sum(
        target_distribution * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1e-8)


def init_independent_critic(key, input_dim, hidden=256, action_count=15):
    keys = random.split(key, 6)

    def linear(linear_key, fan_in, fan_out, scale=1.0):
        weight = random.normal(linear_key, (fan_in, fan_out)) * jnp.sqrt(scale / fan_in)
        return weight, jnp.zeros((fan_out,))

    return {
        "layer1": linear(keys[0], input_dim, hidden),
        "layer2": linear(keys[1], hidden, hidden),
        "value": linear(keys[2], hidden, NUM_VALUE_BINS),
        "win": linear(keys[3], hidden, 1),
        "q": linear(keys[4], hidden, action_count),
        "aux": linear(keys[5], hidden, action_count * len(AUX_TARGET_NAMES)),
    }


def independent_critic_forward(params, inputs):
    layer1_weight, layer1_bias = params["layer1"]
    layer2_weight, layer2_bias = params["layer2"]
    hidden = jax.nn.gelu(inputs @ layer1_weight + layer1_bias)
    hidden = jax.nn.gelu(hidden @ layer2_weight + layer2_bias)
    value_weight, value_bias = params["value"]
    win_weight, win_bias = params["win"]
    q_weight, q_bias = params["q"]
    value_logits = hidden @ value_weight + value_bias
    return (
        value_from_logits(value_logits),
        value_logits,
        (hidden @ win_weight + win_bias)[:, 0],
        20.0 * jnp.tanh((hidden @ q_weight + q_bias) / 20.0),
    )


def independent_critic_aux_forward(params, inputs, action_count=15):
    layer1_weight, layer1_bias = params["layer1"]
    layer2_weight, layer2_bias = params["layer2"]
    hidden = jax.nn.gelu(inputs @ layer1_weight + layer1_bias)
    hidden = jax.nn.gelu(hidden @ layer2_weight + layer2_bias)
    if "aux" not in params:
        return jnp.zeros((len(inputs), action_count, len(AUX_TARGET_NAMES)), jnp.float32)
    aux_weight, aux_bias = params["aux"]
    target_count = aux_bias.shape[0] // action_count
    return (hidden @ aux_weight + aux_bias).reshape(len(inputs), action_count, target_count)
