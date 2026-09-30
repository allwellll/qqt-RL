"""Frozen actor (player 0) vs host-batched Bun Tactical v2 (player 1) in danger_arena.

Reusable evaluator: JIT programs take params as arguments, so one process can
evaluate many checkpoints with a single compile; rule-bot decisions and funnel
labels run in a JAX-free host worker pool. Metrics are identical to the
original single-checkpoint loop (bun_tactical_opponent_eval_v1).
"""
from __future__ import annotations

import hashlib
from functools import partial
import math
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from . import bun_env as env
from . import jax_train
from .bun_frozen_opponents import clear_destructible_bricks, tactical_bot_provenance
from .bun_rule_bot import state_from_bun_state
from .jax_net import transformer_forward

SCHEMA = "bun_tactical_opponent_eval_v1"
ACTOR_PLAYER = 0
BOT_PLAYER = 1


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


def _wilson(count: int, n: int) -> list[float]:
    p = count / n
    z = 1.96
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _make_policy(params, games):
    # Params are closed over (compile-time constants), exactly like the original loop:
    # passing them as jit arguments changes bf16 fusion numerics and flips argmax ties.
    @jax.jit
    def policy(states):
        obs = jax_train.both_perspectives(states)[:games]
        global_state = jax_train.both_states(states)[:games]
        move_mask, ability_mask = jax_train.both_masks(states)
        move_logits, ability_logits, _, _ = transformer_forward(params, obs, global_state)
        joint = jax_train.adjusted_joint_logits(
            move_logits, ability_logits, move_mask[:games], ability_mask[:games],
            jnp.zeros((games, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
        actions = jnp.argmax(joint, axis=-1)
        probabilities = jax.nn.softmax(joint)
        entropy = -(probabilities * jax.nn.log_softmax(joint)).sum(axis=-1)
        return jnp.stack([actions // env.N_BOMB, actions % env.N_BOMB], axis=-1), entropy
    return policy


@jax.jit
def _advance(states, actions, key, active):
    """One tick; finished games (and games finishing now) keep their previous state."""
    n = states.core.pos.shape[0]
    key, step_key = jax.random.split(key)
    keys = jax.random.split(step_key, n)
    candidate, done, info = jax.vmap(lambda state, action, rng: env.step(
        state, action, rng, auto_reset=False, return_info=True))(states, actions, keys)
    keep = active & ~done
    merged = jax.tree.map(
        lambda old, new: jnp.where(keep.reshape((-1,) + (1,) * (old.ndim - 1)), new, old),
        states, candidate)
    return merged, key, done, info


class TacticalOpponentEvaluator:
    def __init__(self, host_workers: int = 0):
        from qqt_rl.bots.host_batch import HostBatcher
        env.prepare()
        env.configure_training(
            "danger_arena=1", 1, reward_profile="danger_arena",
            tactical_bomb_placement_reward=1.0,
            tactical_bomb_resolution_reward=1.0)
        self.provenance = tactical_bot_provenance()
        if not self.provenance["hash_verified"]:
            raise RuntimeError(
                f"frozen tactical bot hash changed: {self.provenance['module_sha256']}")
        self.host_workers = host_workers
        self.host = HostBatcher(host_workers, bot_player=BOT_PLAYER, label_player=ACTOR_PLAYER)

    def close(self) -> None:
        self.host.close()

    def _host_step(self, host_states, active):
        """Bot actions + labels for active games only (inactive games are discarded by _advance)."""
        games = active.shape[0]
        index = np.flatnonzero(active)
        mappings = [state_from_bun_state(jax.tree.map(lambda v, i=i: v[i], host_states))
                    for i in index]
        bot_action = np.zeros((games, 2), np.int32)
        labels = np.zeros((games, 3), np.int32)
        if len(index):
            bot_action[index], labels[index] = self.host(mappings)
        return bot_action, labels[:, 0].astype(bool), labels[:, 1], labels[:, 2].astype(bool)

    def run(self, checkpoint: str, games: int, seed: int, max_steps: int = 300,
            per_episode: bool = False) -> dict:
        params = load_params(checkpoint)
        key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
        states = clear_destructible_bricks(env.init_batch(key, games))
        spawn_cells = np.floor(np.asarray(states.core.pos)).astype(np.int32)
        active = np.ones((games,), np.bool_)
        first_contact = np.full((games,), -1, np.int32)
        visited = np.zeros((games, env.H, env.W), np.bool_)
        counters = {
            name: np.zeros((games,), np.int32)
            for name in (
                "bombs", "safe_detonations", "own_detonations", "surviving_causal",
                "surviving_physical", "trade", "own_bomb", "opponent_physical",
                "opponent_causal", "danger_exposure", "escapable_danger",
                "safe_resolution", "danger_death", "avoidable_danger_death",
                "steal", "capture", "bomb_opportunities",
                "placements_on_opportunity", "safe_tactical_placements",
                "tactical_safe_resolutions")
        }
        funnel = {name: np.zeros((games,), np.bool_) for name in (
            "encounter", "safe_attack_opportunity", "attack", "restricted_escape",
            "forced_kill_created", "self_safe_escape_exists", "escape_executed",
            "enemy_dies", "self_survives")}
        entropy_sum = np.zeros((games,), np.float64)
        decisions = np.zeros((games,), np.int32)
        winners = np.full((games,), -2, np.int8)
        done_tick = np.full((games,), max_steps, np.int32)
        policy_seconds = host_seconds = step_seconds = 0.0

        policy = _make_policy(params, games)
        jax.block_until_ready(policy(states))
        jax.block_until_ready(_advance(
            states, jnp.zeros((games, 2, 2), jnp.int32), key, jnp.ones((games,), jnp.bool_)))
        started = time.time()
        for tick in range(max_steps):
            if not active.any():
                break
            policy_started = time.perf_counter()
            actor_action, entropy = policy(states)
            jax.block_until_ready(actor_action)
            policy_seconds += time.perf_counter() - policy_started
            host_states = jax.device_get(states)
            host_started = time.perf_counter()
            bot_action, safe_available, enemy_escape_count, self_escape = self._host_step(
                host_states, active)
            host_seconds += time.perf_counter() - host_started
            actions = jnp.stack([actor_action, jnp.asarray(bot_action)], axis=1)
            step_started = time.perf_counter()
            states, key, done, info = _advance(states, actions, key, jnp.asarray(active))
            host_info = jax.device_get(info)
            step_seconds += time.perf_counter() - step_started
            host_actions = np.asarray(actions)
            host_positions = np.asarray(host_states.core.pos)
            old_distance = np.abs(host_positions[:, 0] - host_positions[:, 1]).sum(axis=-1)
            missing = first_contact < 0
            first_contact[active & missing & (old_distance <= 3.0)] = tick
            funnel["encounter"] |= active & (old_distance <= 3.0)
            funnel["safe_attack_opportunity"] |= active & safe_available
            placed = active & (host_actions[:, 0, 1] == 1)
            funnel["attack"] |= placed
            funnel["restricted_escape"] |= placed & (enemy_escape_count <= 1)
            funnel["forced_kill_created"] |= placed & safe_available
            funnel["self_safe_escape_exists"] |= placed & self_escape
            cells = np.asarray(host_info["cell"])
            active_indices = np.flatnonzero(active)
            visited[active_indices, cells[active_indices, 0, 0], cells[active_indices, 0, 1]] = True
            p0 = lambda name: np.asarray(host_info[name])[:, 0]
            counters["bombs"] += active * (host_actions[:, 0, 1] == 1)
            counters["bomb_opportunities"] += active * safe_available
            counters["placements_on_opportunity"] += active * placed * safe_available
            counters["safe_tactical_placements"] += active * p0("safe_tactical_bomb_placed")
            counters["tactical_safe_resolutions"] += active * p0("tactical_bomb_safe_resolution")
            counters["safe_detonations"] += active * p0("safe_bomb_escape")
            counters["own_detonations"] += active * p0("own_detonation")
            counters["surviving_causal"] += active * p0("surviving_causal_kill")
            counters["surviving_physical"] += active * p0("surviving_physical_kill")
            counters["trade"] += active * np.asarray(host_info["mutual_death"])
            counters["own_bomb"] += active * p0("own_bomb_defeat")
            counters["opponent_physical"] += active * p0("opponent_physical_defeat")
            counters["opponent_causal"] += active * p0("opponent_causal_defeat")
            counters["danger_exposure"] += active * p0("danger_exposure")
            counters["escapable_danger"] += active * p0("escapable_danger")
            counters["safe_resolution"] += active * p0("danger_safe_resolution")
            funnel["escape_executed"] |= active & p0("danger_safe_resolution")
            killed = active & (p0("surviving_causal_kill") | p0("surviving_physical_kill"))
            funnel["enemy_dies"] |= killed
            funnel["self_survives"] |= killed & p0("alive")
            counters["danger_death"] += active * p0("danger_to_death")
            counters["avoidable_danger_death"] += active * p0("avoidable_danger_death")
            counters["steal"] += active * p0("steal")
            counters["capture"] += active * p0("capture")
            entropy_sum += active * np.asarray(entropy)
            decisions += active
            host_done = np.asarray(done)
            newly_done = active & host_done
            winners[newly_done] = np.asarray(host_info["winner"])[newly_done]
            done_tick[newly_done] = tick + 1
            active &= ~host_done
        elapsed = time.time() - started

        result = {
            "schema": SCHEMA,
            "checkpoint": checkpoint,
            "checkpoint_sha256": digest(checkpoint),
            "seed": seed,
            "max_steps": max_steps,
            "curriculum": "danger_arena=1",
            "destructible_bricks_cleared": True,
            "opponent": self.provenance,
            "elapsed_seconds": elapsed,
            "runtime": {"host_workers": self.host_workers,
                        "host_seconds": host_seconds, "policy_seconds": policy_seconds,
                        "step_seconds": step_seconds,
                        "backend": jax.default_backend()},
            "summary": _summary(games, seed, counters, funnel, winners, first_contact,
                                entropy_sum, decisions, visited, done_tick, policy_seconds),
        }
        if per_episode:
            result["per_episode"] = {
                "actor_player": ACTOR_PLAYER,
                "spawn_cells": spawn_cells.tolist(),
                "surviving_causal_kill": (counters["surviving_causal"] > 0).tolist(),
                "surviving_physical_kill": (counters["surviving_physical"] > 0).tolist(),
                "own_bomb_defeat": (counters["own_bomb"] > 0).tolist(),
                "opponent_physical_defeat": (counters["opponent_physical"] > 0).tolist(),
                "opponent_causal_defeat": (counters["opponent_causal"] > 0).tolist(),
                "mutual_death": (counters["trade"] > 0).tolist(),
                "avoidable_danger_death": (counters["avoidable_danger_death"] > 0).tolist(),
                "danger_to_death": (counters["danger_death"] > 0).tolist(),
                "bombs": counters["bombs"].tolist(),
            }
        return result


def _summary(games, seed, counters, funnel, winners, first_contact, entropy_sum,
             decisions, visited, done_tick, policy_seconds) -> dict:
    own_detonations = counters["own_detonations"]
    return {
        "games": games,
        "completed": int(np.sum(winners != -2)),
        "p0_win_rate": float(np.mean(winners == 0)),
        "p0_surviving_causal_kill_rate": float(np.mean(counters["surviving_causal"] > 0)),
        "p0_surviving_physical_kill_rate": float(np.mean(counters["surviving_physical"] > 0)),
        "mutual_death_rate": float(np.mean(counters["trade"] > 0)),
        "p0_own_bomb_defeat_rate": float(np.mean(counters["own_bomb"] > 0)),
        "p0_opponent_physical_defeat_rate": float(np.mean(counters["opponent_physical"] > 0)),
        "p0_opponent_causal_defeat_rate": float(np.mean(counters["opponent_causal"] > 0)),
        "p0_opponent_defeat_rate": float(np.mean(
            (counters["opponent_physical"] > 0) | (counters["opponent_causal"] > 0))),
        "p0_avg_bombs": float(np.mean(counters["bombs"])),
        "p0_bomb_placement_opportunities": int(counters["bomb_opportunities"].sum()),
        "p0_conditional_tactical_placement_rate": float(
            counters["placements_on_opportunity"].sum()
            / max(counters["bomb_opportunities"].sum(), 1)),
        "p0_safe_tactical_placement_count": int(counters["safe_tactical_placements"].sum()),
        "p0_safe_tactical_placements_per_game": float(
            np.mean(counters["safe_tactical_placements"])),
        "p0_safe_tactical_resolutions": int(counters["tactical_safe_resolutions"].sum()),
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
        "p0_avg_policy_entropy": float(entropy_sum.sum() / max(decisions.sum(), 1)),
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
        "paired_episode_seeds": [int(seed + i) for i in range(games)],
        "wilson_95": {name: _wilson(int(values.sum()), games) for name, values in {
            "win": winners == 0, "surviving_causal": counters["surviving_causal"] > 0,
            "surviving_physical": counters["surviving_physical"] > 0,
            "trade": counters["trade"] > 0, "own_bomb": counters["own_bomb"] > 0,
        }.items()},
        "funnel": {name: {"count": int(values.sum()), "rate": float(values.mean()),
                          "wilson_95": _wilson(int(values.sum()), games)}
                   for name, values in funnel.items()},
        "delayed_escape_audit": {
            "reward_events": int(counters["safe_resolution"].sum()),
            "episodes_with_reward": int((counters["safe_resolution"] > 0).sum()),
            "duplicate_reward_rate": float(np.mean(counters["safe_resolution"] > 1)),
            "reward_per_episode": float(counters["safe_resolution"].sum() / games),
            "reward_per_explosion": float(
                counters["safe_resolution"].sum() / max(counters["own_detonations"].sum(), 1)),
        },
    }


FLEE_PROBE_SCHEMA = "bun_flee_bot_probe_v1"
_FLEE_COUNTERS = ("surviving_kill", "own_bomb", "opponent_defeat", "mutual_death",
                  "avoidable_danger_death", "danger_death", "bombs")


def _flee_policy_actions(params, states):
    n = states.core.pos.shape[0]
    obs = jax_train.both_perspectives(states)[:n]
    global_state = jax_train.both_states(states)[:n]
    move_mask, ability_mask = jax_train.both_masks(states)
    move_logits, ability_logits, _, _ = transformer_forward(params, obs, global_state)
    joint = jax_train.adjusted_joint_logits(
        move_logits, ability_logits, move_mask[:n], ability_mask[:n],
        jnp.zeros((n, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
    actions = jnp.argmax(joint, axis=-1)
    actor = jnp.stack([actions // env.N_BOMB, actions % env.N_BOMB], axis=-1)
    return actor, move_mask[n:], ability_mask[n:]


@partial(jax.jit, static_argnums=(3,))
def _flee_episode(params, states, key, max_steps):
    """Whole episode on device: argmax actor (player 0) vs the training-time JAX flee-bot mixture."""
    n = states.core.pos.shape[0]
    counters = {name: jnp.zeros((n,), jnp.int32) for name in _FLEE_COUNTERS}
    carry = (states, key, jnp.ones((n,), jnp.bool_), counters, jnp.full((n,), max_steps, jnp.int32))

    def body(carry, tick):
        states, key, active, counters, done_tick = carry
        key, bot_key, step_key = jax.random.split(key, 3)
        actor, bot_mm, bot_bm = _flee_policy_actions(params, states)
        bot = jax_train.flee_bot_actions(
            states.core.pos[:, 1], states.core.pos[:, 0], bot_mm, bot_bm, bot_key)
        actions = jnp.stack([actor, bot], axis=1)
        keys = jax.random.split(step_key, n)
        candidate, done, info = jax.vmap(lambda state, action, rng: env.step(
            state, action, rng, auto_reset=False, return_info=True))(states, actions, keys)
        a = active.astype(jnp.int32)
        p0 = lambda name: info[name][:, 0].astype(jnp.int32)
        events = {
            "surviving_kill": p0("surviving_causal_kill"),
            "own_bomb": p0("own_bomb_defeat"),
            "opponent_defeat": (info["opponent_physical_defeat"][:, 0]
                                | info["opponent_causal_defeat"][:, 0]).astype(jnp.int32),
            "mutual_death": info["mutual_death"].astype(jnp.int32),
            "avoidable_danger_death": p0("avoidable_danger_death"),
            "danger_death": p0("danger_to_death"),
            "bombs": (actor[:, 1] == 1).astype(jnp.int32),
        }
        counters = {name: counters[name] + a * events[name] for name in _FLEE_COUNTERS}
        done_tick = jnp.where(active & done, tick + 1, done_tick)
        keep = active & ~done
        states = jax.tree.map(
            lambda old, new: jnp.where(keep.reshape((-1,) + (1,) * (old.ndim - 1)), new, old),
            states, candidate)
        return (states, key, keep, counters, done_tick), None

    (_, _, _, counters, done_tick), _ = jax.lax.scan(body, carry, jnp.arange(max_steps))
    return counters, done_tick


def run_flee_bot_probe(checkpoint: str, games: int, seed: int, max_steps: int = 300) -> dict:
    """Supplementary probe vs the phase-1 JAX flee bot (mostly non-attacking). Not the main protocol.

    Params are passed as jit arguments (one compile for all checkpoints), so bf16
    numerics differ slightly from the Tactical v2 evaluator; results are only
    comparable within this probe.
    """
    env.prepare()
    env.configure_training(
        "danger_arena=1", 1, reward_profile="danger_arena",
        tactical_bomb_placement_reward=1.0, tactical_bomb_resolution_reward=1.0)
    params = load_params(checkpoint)
    key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
    states = clear_destructible_bricks(env.init_batch(key, games))
    run_key = jax.random.fold_in(key, 1)
    started = time.time()
    counters, done_tick = jax.device_get(_flee_episode(params, states, run_key, max_steps))
    elapsed = time.time() - started
    rate = lambda name: int((counters[name] > 0).sum())
    rates = {
        "surviving_kill": rate("surviving_kill"), "own_bomb_death": rate("own_bomb"),
        "killed_by_opponent": rate("opponent_defeat"), "mutual_death": rate("mutual_death"),
        "avoidable_danger_death": rate("avoidable_danger_death"),
        "danger_to_death": rate("danger_death"),
    }
    return {
        "schema": FLEE_PROBE_SCHEMA,
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest(checkpoint),
        "seed": seed, "games": games, "max_steps": max_steps,
        "curriculum": "danger_arena=1", "destructible_bricks_cleared": True,
        "opponent": {"name": "jax_train.flee_bot_actions", "mixture": {
            "idle": 0.25, "roam": 0.05, "pure_flee": 0.50, "smart_kite": 0.20}},
        "elapsed_seconds": elapsed,
        "summary": {
            "games": games,
            **{f"{name}_rate": count / games for name, count in rates.items()},
            "counts": rates,
            "wilson_95": {name: _wilson(count, games) for name, count in rates.items()},
            "avg_bombs": float(np.mean(counters["bombs"])),
            "avg_ticks": float(np.mean(done_tick)),
        },
    }
