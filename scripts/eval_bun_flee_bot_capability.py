#!/usr/bin/env python3
"""Same-protocol capability acceptance eval: learned actor (P0) vs device-side
JIT flee-bot mixture (P1).

Purpose (user acceptance requirement): quantify whether the 4-GPU combat/kill
run actually raised *kill ability* — NOT throughput. Every checkpoint is scored
under one PRE-DECLARED FIXED SEED with INDEPENDENT eval runs and the SAME
protocol (同口径): identical curriculum / HP / reward profile as training, and
the opponent is the pure-JIT `flee_bot_actions` mixture from jax_train.py (idle
25% / roam 5% / pure_flee 50% / smart_kite 20%). The Python rule bot is NOT
used here.

Reported per checkpoint (actor = player 0):
  - bomb placement            (放泡, per-game mean + episode rate)
  - surviving non-trade kill  (存活非trade击杀 = surviving causal|physical kill)
  - self-kill                 (自炸 = own_bomb_defeat)
  - trade                     (同归于尽 = mutual_death)
  - win / loss / draw / ongoing  (胜/负/平/未终局, from winner)
  - episode length            (done tick; unfinished = capped at max-steps)

forced-kill has NO env metric under the legacy reward profile — the env never
labels a "forced kill" event, so it is intentionally absent from this table
(documented, not silently dropped).

The actor policy is sampled with a fixed PRNG (categorical over masked logits,
same as training), so the whole eval is deterministic given --seed.
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

from jax_bomb import jax_train as T
from jax_bomb.bun_env import (H, W, N_OBS_CH, configure_training, init_batch,
                              prepare, step)


def build_actor(spec: str, arch: str, hidden: int, key):
    """spec == "random" -> fresh init_net baseline; else load a checkpoint."""
    if spec == "random":
        return T.init_net(key, arch, N_OBS_CH, H, W, hidden=hidden)
    return T.load_params(spec)


def evaluate(params, arch, seed: int, games: int, max_steps: int):
    key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
    key, init_key = jax.random.split(key)
    states = init_batch(init_key, games)

    @jax.jit
    def rollout_step(states, key):
        key, k_act, k_bot, k_step = jax.random.split(key, 4)
        n = games
        obs = T.both_perspectives(states)              # (2N, C, H, W)
        mm, bm = T.both_masks(states)                  # (2N,5),(2N,2)
        gv = T.both_states(states)                     # (2N, G)
        # P0 = learned actor
        acts, _, _ = T.sample_actions(
            params, arch, obs[:n], (mm[:n], bm[:n]), k_act, state=gv[:n])
        a0 = acts
        # P1 = device-side JIT flee-bot mixture (default training ratios)
        a1 = T.flee_bot_actions(
            states.pos[:, 1], states.pos[:, 0], mm[n:], bm[n:], k_bot)
        env_acts = jnp.stack([a0, a1], axis=1)         # (N, 2, 2)
        keys = jax.random.split(k_step, n)
        new_states, done, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, auto_reset=False,
                                  return_info=True))(states, env_acts, keys)
        return new_states, done, info, key

    active = np.ones((games,), np.bool_)
    bombs = np.zeros((games,), np.int32)          # P0 bombs placed
    self_kill = np.zeros((games,), np.int32)      # P0 own_bomb_defeat
    surviving_kill = np.zeros((games,), np.int32)  # P0 surviving causal|physical
    trade = np.zeros((games,), np.int32)          # mutual_death (shared)
    winners = np.full((games,), -2, np.int8)
    done_tick = np.full((games,), max_steps, np.int32)

    started = time.time()
    for tick in range(max_steps):
        if not active.any():
            break
        states, done, info, key = rollout_step(states, key)
        jax.block_until_ready(done)
        host = jax.device_get(info)
        act = active.astype(np.int32)
        bombs += act * np.asarray(host["bomb_placed"])[:, 0]
        self_kill += act * np.asarray(host["own_bomb_defeat"])[:, 0]
        surviving_kill += act * (
            np.asarray(host["surviving_causal_kill"])[:, 0]
            + np.asarray(host["surviving_physical_kill"])[:, 0])
        trade += act * np.asarray(host["mutual_death"])
        host_done = np.asarray(done)
        newly = active & host_done
        winners[newly] = np.asarray(host["winner"])[newly]
        done_tick[newly] = tick + 1
        active &= ~host_done
    elapsed = time.time() - started

    n = float(games)
    return {
        "games": games,
        "max_steps": max_steps,
        "elapsed_seconds": round(elapsed, 2),
        "bomb_placement": {
            "per_game_mean": float(np.mean(bombs)),
            "episode_rate": float(np.mean(bombs > 0)),
            "total": int(bombs.sum()),
        },
        "surviving_non_trade_kill": {
            "per_game_mean": float(np.mean(surviving_kill)),
            "episode_rate": float(np.mean(surviving_kill > 0)),
            "total": int(surviving_kill.sum()),
        },
        "self_kill": {
            "per_game_mean": float(np.mean(self_kill)),
            "episode_rate": float(np.mean(self_kill > 0)),
            "total": int(self_kill.sum()),
        },
        "trade": {
            "episode_rate": float(np.mean(trade > 0)),
            "total": int(trade.sum()),
        },
        "outcome": {
            "win_rate": float(np.mean(winners == 0)),
            "loss_rate": float(np.mean(winners == 1)),
            "draw_rate": float(np.mean(winners == -1)),
            "ongoing_rate": float(np.mean(winners == -2)),
        },
        "episode_length": {
            "mean": float(np.mean(done_tick)),
            "median": float(np.median(done_tick)),
            "max": int(done_tick.max()),
        },
        "forced_kill": "no env metric under legacy reward profile",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True,
                        help="pre-declared fixed eval seed (same across all)")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--arch", default="mlp4")
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--curriculum", required=True,
                        help="training-identical curriculum for same protocol")
    parser.add_argument("--hp", type=int, default=1)
    parser.add_argument("--reward-profile", default="legacy")
    parser.add_argument("--actors", nargs="+", required=True,
                        help="ordered list: 'random' or checkpoint paths")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="display label per actor (same length as --actors)")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if len(args.actors) != len(args.labels):
        raise SystemExit("--actors and --labels must have equal length")

    prepare()
    configure_training(args.curriculum, args.hp, False, args.reward_profile)

    init_key = jax.random.PRNGKey(np.uint32((args.seed ^ 0x5DEECE66) & 0xFFFFFFFF))
    results = []
    for label, spec in zip(args.labels, args.actors):
        init_key, sub = jax.random.split(init_key)
        params = build_actor(spec, args.arch, args.hidden, sub)
        metrics = evaluate(params, args.arch, args.seed, args.games,
                           args.max_steps)
        metrics["label"] = label
        metrics["actor"] = spec
        results.append(metrics)
        print(f"[{label}] {spec}", flush=True)
        print(json.dumps({k: metrics[k] for k in (
            "bomb_placement", "surviving_non_trade_kill", "self_kill",
            "trade", "outcome", "episode_length")},
            ensure_ascii=False, indent=2), flush=True)

    report = {
        "schema": "bun_flee_bot_capability_eval_v1",
        "seed": args.seed,
        "games": args.games,
        "max_steps": args.max_steps,
        "protocol": {
            "opponent": "device-side JIT flee_bot_actions mixture "
                        "(idle25/roam5/pure_flee50/smart_kite20)",
            "actor_player": 0,
            "arch": args.arch,
            "hidden": args.hidden,
            "curriculum": args.curriculum,
            "hp": args.hp,
            "reward_profile": args.reward_profile,
            "note": "SPS/throughput is NOT a capability metric and is excluded",
        },
        "results": results,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\nreport -> {out}", flush=True)


if __name__ == "__main__":
    main()
