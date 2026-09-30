#!/usr/bin/env python3
"""Rule-bot self-play baseline: quantify the teacher's OWN self-death rate.

The user reported (from real web replays) that the rule bot itself gets killed
by its own bombs, does not flee when standing in danger, and lacks visible
offense. This harness drives the frozen tactical rule bot as BOTH players in the
danger_arena and reports, per player, the same capability counters the actor
eval uses. It is deterministic under a fixed seed and produces the teacher
baseline the capability gate is measured against. No learned weights involved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from jax_bomb.bun_frozen_opponents import (
    FrozenTacticalOpponent, clear_destructible_bricks, tactical_bot_provenance)

METRICS = ("bombs", "own_bomb", "avoidable_danger_death", "danger_death",
           "surviving_causal", "surviving_physical", "own_detonations",
           "safe_detonations", "danger_exposure", "escapable_danger",
           "safe_resolution", "trade")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    env.prepare()
    env.configure_training(
        "danger_arena=1", 1, reward_profile="danger_arena",
        tactical_bomb_placement_reward=1.0,
        tactical_bomb_resolution_reward=1.0)
    bots = (FrozenTacticalOpponent(player_id=0),
            FrozenTacticalOpponent(player_id=1))
    key = jax.random.PRNGKey(np.uint32(args.seed & 0xFFFFFFFF))
    states = clear_destructible_bricks(env.init_batch(key, args.games))
    active = np.ones((args.games,), np.bool_)
    # counters[metric] shape (games, 2players)
    counters = {name: np.zeros((args.games, 2), np.int32) for name in METRICS}
    winners = np.full((args.games,), -2, np.int8)
    done_tick = np.full((args.games,), args.max_steps, np.int32)

    @jax.jit
    def step_batch(current_states, actions, step_key):
        keys = jax.random.split(step_key, args.games)
        return jax.vmap(lambda state, action, rng: env.step(
            state, action, rng, auto_reset=False, return_info=True))(
                current_states, actions, keys)

    started = time.time()
    for tick in range(args.max_steps):
        if not active.any():
            break
        key, step_key = jax.random.split(key)
        a0 = np.asarray(bots[0].decide_batch(states), np.int32)
        a1 = np.asarray(bots[1].decide_batch(states), np.int32)
        actions = jnp.asarray(np.stack([a0, a1], axis=1), jnp.int32)
        candidate, done, info = step_batch(states, actions, step_key)
        jax.block_until_ready(done)
        host = jax.device_get(info)
        host_actions = np.asarray(actions)
        act = active[:, None]
        counters["bombs"] += act * (host_actions[:, :, 1] == 1)
        counters["own_bomb"] += act * np.asarray(host["own_bomb_defeat"])
        counters["avoidable_danger_death"] += act * np.asarray(host["avoidable_danger_death"])
        counters["danger_death"] += act * np.asarray(host["danger_to_death"])
        counters["surviving_causal"] += act * np.asarray(host["surviving_causal_kill"])
        counters["surviving_physical"] += act * np.asarray(host["surviving_physical_kill"])
        counters["own_detonations"] += act * np.asarray(host["own_detonation"])
        counters["safe_detonations"] += act * np.asarray(host["safe_bomb_escape"])
        counters["danger_exposure"] += act * np.asarray(host["danger_exposure"])
        counters["escapable_danger"] += act * np.asarray(host["escapable_danger"])
        counters["safe_resolution"] += act * np.asarray(host["danger_safe_resolution"])
        counters["trade"] += (act[:, 0] * np.asarray(host["mutual_death"]))[:, None]
        host_done = np.asarray(done)
        newly = active & host_done
        winners[newly] = np.asarray(host["winner"])[newly]
        done_tick[newly] = tick + 1
        active &= ~host_done
        states = jax.tree.map(
            lambda old, new: jnp.where(
                jnp.asarray(active).reshape((-1,) + (1,) * (old.ndim - 1)), new, old),
            states, candidate)
    elapsed = time.time() - started

    def both(name, reducer):
        return [float(reducer(counters[name][:, p])) for p in range(2)]

    per_player = {
        name: {
            "episode_rate": both(name, lambda c: np.mean(c > 0)),
            "per_game_mean": both(name, np.mean),
            "total": [int(counters[name][:, p].sum()) for p in range(2)],
        } for name in METRICS
    }
    # pooled teacher self-death: either player self-bombing, averaged over 2N
    pooled = {
        "self_bomb_defeat_player_rate": float(
            np.mean(counters["own_bomb"] > 0)),
        "avoidable_danger_death_player_rate": float(
            np.mean(counters["avoidable_danger_death"] > 0)),
        "avg_bombs_per_player": float(np.mean(counters["bombs"])),
        "surviving_kill_player_rate": float(np.mean(
            (counters["surviving_causal"] + counters["surviving_physical"]) > 0)),
        "safe_detonation_ratio": float(
            counters["safe_detonations"].sum()
            / max(counters["own_detonations"].sum(), 1)),
        "safe_resolution_ratio": float(
            counters["safe_resolution"].sum()
            / max(counters["escapable_danger"].sum(), 1)),
        "trade_rate": float(np.mean(counters["trade"][:, 0] > 0)),
    }
    result = {
        "schema": "bun_rulebot_selfplay_baseline_v1",
        "seed": args.seed,
        "games": args.games,
        "curriculum": "danger_arena=1",
        "opponent": tactical_bot_provenance(),
        "elapsed_seconds": elapsed,
        "avg_ticks": float(done_tick.mean()),
        "pooled_teacher": pooled,
        "per_player": per_player,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["pooled_teacher"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
