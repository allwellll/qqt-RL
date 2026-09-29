#!/usr/bin/env python3
"""Evaluate a frozen actor against the host-batched 40-tick Bun rule bot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import math
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_env as env
from jax_bomb import bun_safety
from jax_bomb import jax_train
from jax_bomb.bun_frozen_opponents import (
    FrozenTacticalOpponent,
    clear_destructible_bricks,
    tactical_bot_provenance,
)
from jax_bomb.bun_rule_bot import state_from_bun_state
from jax_bomb.bun_tactical_labels import label_batch
from jax_bomb.jax_net import transformer_forward


def load_params(path: str):
    with open(path, "rb") as file:
        value = pickle.load(file)
    return jax.tree.map(jnp.asarray, value.get("params", value))


def digest(path: str) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    env.prepare()
    env.configure_training(
        "danger_arena=1", 1, reward_profile="danger_arena",
        tactical_bomb_placement_reward=1.0,
        tactical_bomb_resolution_reward=1.0,
        base_bomb_reward=1.0,
        forced_kill_reward=1.0)
    params = load_params(args.checkpoint)
    opponent = FrozenTacticalOpponent(player_id=1)
    key = jax.random.PRNGKey(np.uint32(args.seed & 0xFFFFFFFF))
    states = clear_destructible_bricks(env.init_batch(key, args.games))
    active = np.ones((args.games,), np.bool_)
    first_contact = np.full((args.games,), -1, np.int32)
    visited = np.zeros((args.games, env.H, env.W), np.bool_)
    counters = {
        name: np.zeros((args.games,), np.int32)
        for name in (
            "bombs", "safe_detonations", "own_detonations", "surviving_causal",
            "surviving_physical", "trade", "own_bomb", "opponent_physical",
            "opponent_causal", "danger_exposure", "escapable_danger",
            "safe_resolution", "danger_death", "avoidable_danger_death",
            "steal", "capture", "bomb_opportunities",
            "placements_on_opportunity", "safe_tactical_placements",
            "tactical_safe_resolutions", "forced_kill_opportunities",
            "forced_kill_created", "placements_on_forced_kill",
            "rule_safe_attack_available", "rule_placements_on_safe_attack")
    }
    funnel = {name: np.zeros((args.games,), np.bool_) for name in (
        "encounter", "safe_attack_opportunity", "attack", "restricted_escape",
        "forced_kill_created", "self_safe_escape_exists", "escape_executed",
        "enemy_dies", "self_survives")}
    entropy_sum = np.zeros((args.games,), np.float64)
    decisions = np.zeros((args.games,), np.int32)
    winners = np.full((args.games,), -2, np.int8)
    done_tick = np.full((args.games,), args.max_steps, np.int32)
    policy_seconds = 0.0

    @jax.jit
    def policy(current_states):
        obs = jax_train.both_perspectives(current_states)[:args.games]
        global_state = jax_train.both_states(current_states)[:args.games]
        move_mask, ability_mask = jax_train.both_masks(current_states)
        move_mask = move_mask[:args.games]
        ability_mask = ability_mask[:args.games]
        move_logits, ability_logits, _, _ = transformer_forward(
            params, obs, global_state)
        joint = jax_train.adjusted_joint_logits(
            move_logits, ability_logits, move_mask, ability_mask,
            jnp.zeros((args.games, env.N_MOVES, env.N_BOMB), jnp.bool_),
            "off", 0.0)
        actions = jnp.argmax(joint, axis=-1)
        probabilities = jax.nn.softmax(joint)
        entropy = -(probabilities * jax.nn.log_softmax(joint)).sum(axis=-1)
        return jnp.stack([actions // env.N_BOMB, actions % env.N_BOMB], axis=-1), entropy

    @jax.jit
    def step_batch(current_states, actions, step_key):
        keys = jax.random.split(step_key, args.games)
        return jax.vmap(lambda state, action, rng: env.step(
            state, action, rng, auto_reset=False, return_info=True))(
                current_states, actions, keys)

    probe_actions = jnp.asarray([[4, 1], [4, 0]], jnp.int32)

    @jax.jit
    def tactical_probe(current_states):
        """Env-native opportunity probe: would a player0 bomb here be
        tactical / a forced kill? Same analyzer the training reward uses."""
        def one(state):
            analysis = bun_safety.analyze_tactical_bomb_placements(
                state, probe_actions)
            return analysis.tactical[0], analysis.forces_kill[0]
        return jax.vmap(one)(current_states)

    warm_actions, _ = policy(states)
    jax.block_until_ready(warm_actions)
    started = time.time()
    for tick in range(args.max_steps):
        if not active.any():
            break
        key, step_key = jax.random.split(key)
        policy_started = time.perf_counter()
        actor_action, entropy = policy(states)
        jax.block_until_ready(actor_action)
        policy_seconds += time.perf_counter() - policy_started
        bot_action = jnp.asarray(opponent.decide_batch(states), jnp.int32)
        actions = jnp.stack([actor_action, bot_action], axis=1)
        candidate, done, info = step_batch(states, actions, step_key)
        jax.block_until_ready(done)
        host_info = jax.device_get(info)
        env_tactical, env_forced = tactical_probe(states)
        env_tactical = np.asarray(env_tactical)
        env_forced = np.asarray(env_forced)
        host_actions = np.asarray(actions)
        host_positions = np.asarray(states.core.pos)
        old_distance = np.abs(host_positions[:, 0] - host_positions[:, 1]).sum(axis=-1)
        mappings = [state_from_bun_state(jax.tree.map(lambda value, i=i: value[i], jax.device_get(states))) for i in range(args.games)]
        labels = label_batch(mappings, np.zeros(args.games, np.int32))
        safe_available = np.asarray([label.safe_attack_available for label in labels])
        enemy_escape_count = np.asarray([label.enemy_escape_count for label in labels])
        self_escape = np.asarray([label.safe_escape_exists for label in labels])
        forced_kill_created = np.asarray(host_info["forced_kill_created"])[:, 0]
        missing = first_contact < 0
        first_contact[active & missing & (old_distance <= 3.0)] = tick
        funnel["encounter"] |= active & (old_distance <= 3.0)
        funnel["safe_attack_opportunity"] |= active & env_forced
        placed = active & (host_actions[:, 0, 1] == 1)
        funnel["attack"] |= placed
        funnel["restricted_escape"] |= placed & (enemy_escape_count <= 1)
        funnel["forced_kill_created"] |= active & forced_kill_created
        funnel["self_safe_escape_exists"] |= placed & self_escape
        cells = np.asarray(host_info["cell"])
        active_indices = np.flatnonzero(active)
        visited[active_indices, cells[active_indices, 0, 0], cells[active_indices, 0, 1]] = True
        counters["bombs"] += active * (host_actions[:, 0, 1] == 1)
        counters["bomb_opportunities"] += active * env_tactical
        counters["placements_on_opportunity"] += active * placed * env_tactical
        counters["forced_kill_opportunities"] += active * env_forced
        counters["placements_on_forced_kill"] += active * placed * env_forced
        counters["forced_kill_created"] += active * forced_kill_created
        counters["rule_safe_attack_available"] += active * safe_available
        counters["rule_placements_on_safe_attack"] += active * placed * safe_available
        counters["safe_tactical_placements"] += active * np.asarray(
            host_info["safe_tactical_bomb_placed"])[:, 0]
        counters["tactical_safe_resolutions"] += active * np.asarray(
            host_info["tactical_bomb_safe_resolution"])[:, 0]
        counters["safe_detonations"] += active * np.asarray(host_info["safe_bomb_escape"])[:, 0]
        counters["own_detonations"] += active * np.asarray(host_info["own_detonation"])[:, 0]
        counters["surviving_causal"] += active * np.asarray(host_info["surviving_causal_kill"])[:, 0]
        counters["surviving_physical"] += active * np.asarray(host_info["surviving_physical_kill"])[:, 0]
        counters["trade"] += active * np.asarray(host_info["mutual_death"])
        counters["own_bomb"] += active * np.asarray(host_info["own_bomb_defeat"])[:, 0]
        counters["opponent_physical"] += active * np.asarray(host_info["opponent_physical_defeat"])[:, 0]
        counters["opponent_causal"] += active * np.asarray(host_info["opponent_causal_defeat"])[:, 0]
        counters["danger_exposure"] += active * np.asarray(host_info["danger_exposure"])[:, 0]
        counters["escapable_danger"] += active * np.asarray(host_info["escapable_danger"])[:, 0]
        counters["safe_resolution"] += active * np.asarray(host_info["danger_safe_resolution"])[:, 0]
        funnel["escape_executed"] |= active & np.asarray(host_info["danger_safe_resolution"])[:, 0]
        killed = active & (np.asarray(host_info["surviving_causal_kill"])[:, 0] | np.asarray(host_info["surviving_physical_kill"])[:, 0])
        funnel["enemy_dies"] |= killed
        funnel["self_survives"] |= killed & np.asarray(host_info["alive"])[:, 0]
        counters["danger_death"] += active * np.asarray(host_info["danger_to_death"])[:, 0]
        counters["avoidable_danger_death"] += active * np.asarray(host_info["avoidable_danger_death"])[:, 0]
        counters["steal"] += active * np.asarray(host_info["steal"])[:, 0]
        counters["capture"] += active * np.asarray(host_info["capture"])[:, 0]
        entropy_sum += active * np.asarray(entropy)
        decisions += active
        host_done = np.asarray(done)
        newly_done = active & host_done
        winners[newly_done] = np.asarray(host_info["winner"])[newly_done]
        done_tick[newly_done] = tick + 1
        active &= ~host_done
        states = jax.tree.map(
            lambda old, new: jnp.where(
                jnp.asarray(active).reshape((-1,) + (1,) * (old.ndim - 1)), new, old),
            states, candidate)
    elapsed = time.time() - started

    own_detonations = counters["own_detonations"]
    def wilson(rate_count):
        n=args.games; p=rate_count/n; z=1.96; den=1+z*z/n; center=(p+z*z/(2*n))/den; radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return [max(0.0,center-radius),min(1.0,center+radius)]
    summary = {
        "games": args.games,
        "completed": int(np.sum(winners != -2)),
        "p0_win_rate": float(np.mean(winners == 0)),
        "p0_surviving_causal_kill_rate": float(np.mean(counters["surviving_causal"] > 0)),
        "p0_surviving_physical_kill_rate": float(np.mean(counters["surviving_physical"] > 0)),
        "mutual_death_rate": float(np.mean(counters["trade"] > 0)),
        "p0_own_bomb_defeat_rate": float(np.mean(counters["own_bomb"] > 0)),
        "p0_opponent_physical_defeat_rate": float(np.mean(counters["opponent_physical"] > 0)),
        "p0_opponent_causal_defeat_rate": float(np.mean(counters["opponent_causal"] > 0)),
        "p0_avg_bombs": float(np.mean(counters["bombs"])),
        "p0_bomb_placement_opportunities": int(
            counters["bomb_opportunities"].sum()),
        "p0_conditional_tactical_placement_rate": float(
            counters["placements_on_opportunity"].sum()
            / max(counters["bomb_opportunities"].sum(), 1)),
        "p0_forced_kill_opportunities": int(
            counters["forced_kill_opportunities"].sum()),
        "p0_conditional_forced_kill_placement_rate": float(
            counters["placements_on_forced_kill"].sum()
            / max(counters["forced_kill_opportunities"].sum(), 1)),
        "p0_forced_kill_created_count": int(counters["forced_kill_created"].sum()),
        "p0_forced_kill_created_rate": float(
            np.mean(counters["forced_kill_created"] > 0)),
        "secondary_rule_bot_labels": {
            "note": ("rule-bot 40-tick lookahead label, kept for cross-check "
                     "only; headline metrics use env-native signals"),
            "safe_attack_available_ticks": int(
                counters["rule_safe_attack_available"].sum()),
            "conditional_placement_rate": float(
                counters["rule_placements_on_safe_attack"].sum()
                / max(counters["rule_safe_attack_available"].sum(), 1)),
            "env_vs_rule_opportunity_gap": int(
                counters["forced_kill_opportunities"].sum()
                - counters["rule_safe_attack_available"].sum()),
        },
        "p0_safe_tactical_placement_count": int(
            counters["safe_tactical_placements"].sum()),
        "p0_safe_tactical_placements_per_game": float(
            np.mean(counters["safe_tactical_placements"])),
        "p0_safe_tactical_resolutions": int(
            counters["tactical_safe_resolutions"].sum()),
        "p0_owned_resolutions": int(own_detonations.sum()),
        "p0_tactical_resolution_ratio": float(
            counters["tactical_safe_resolutions"].sum()
            / max(counters["safe_tactical_placements"].sum(), 1)),
        "p0_safe_detonation_ratio": float(
            counters["safe_detonations"].sum() / max(own_detonations.sum(), 1)),
        "p0_contact_rate": float(np.mean(first_contact >= 0)),
        "p0_mean_first_contact_tick": (
            float(first_contact[first_contact >= 0].mean())
            if np.any(first_contact >= 0) else None),
        "p0_avg_policy_entropy": float(
            entropy_sum.sum() / max(decisions.sum(), 1)),
        "p0_avg_unique_cells": float(np.mean(visited.sum(axis=(1, 2)))),
        "p0_danger_exposure_episode_rate": float(np.mean(counters["danger_exposure"] > 0)),
        "p0_escapable_danger_episode_rate": float(np.mean(counters["escapable_danger"] > 0)),
        "p0_safe_resolution_episode_rate": float(np.mean(counters["safe_resolution"] > 0)),
        "p0_safe_resolution_ratio": float(
            counters["safe_resolution"].sum() / max(counters["escapable_danger"].sum(), 1)),
        "p0_danger_to_death_rate": float(np.mean(counters["danger_death"] > 0)),
        "p0_avoidable_danger_death_rate": float(np.mean(counters["avoidable_danger_death"] > 0)),
        "p0_steal_rate": float(np.mean(counters["steal"] > 0)),
        "p0_capture_rate": float(np.mean(counters["capture"] > 0)),
        "avg_ticks": float(done_tick.mean()),
        "actor_inference_ms_per_game_tick": float(
            1000.0 * policy_seconds / max(decisions.sum(), 1)),
        "paired_episode_seeds": [int(args.seed+i) for i in range(args.games)],
        "wilson_95": {name: wilson(int(values.sum())) for name,values in {
            "win": winners==0,"surviving_causal":counters["surviving_causal"]>0,
            "surviving_physical":counters["surviving_physical"]>0,
            "trade":counters["trade"]>0,"own_bomb":counters["own_bomb"]>0,
        }.items()},
        "funnel": {name:{"count":int(values.sum()),"rate":float(values.mean()),"wilson_95":wilson(int(values.sum()))} for name,values in funnel.items()},
        "delayed_escape_audit": {
            "reward_events": int(counters["safe_resolution"].sum()),
            "episodes_with_reward": int((counters["safe_resolution"]>0).sum()),
            "duplicate_reward_rate": float(np.mean(counters["safe_resolution"]>1)),
            "reward_per_episode": float(counters["safe_resolution"].sum()/args.games),
            "reward_per_explosion": float(counters["safe_resolution"].sum()/max(counters["own_detonations"].sum(),1)),
        },
    }
    result = {
        "schema": "bun_tactical_opponent_eval_v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": digest(args.checkpoint),
        "seed": args.seed,
        "curriculum": "danger_arena=1",
        "destructible_bricks_cleared": True,
        "opponent": tactical_bot_provenance(),
        "elapsed_seconds": elapsed,
        "summary": summary,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
