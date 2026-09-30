#!/usr/bin/env python3
"""Difficulty ladder: each learned checkpoint (P0) vs every bun_jax_bots tier (P1).

Same device-side path as training (rule_bot_actions / flee_bot_actions, pure
JIT). danger_arena respawns on death and only ends on timeout, so outcomes are
reported as per-episode event rates (1 episode = --max-steps ticks):
kill (learner surviving kill), death, self (own-bomb), trade, bot_self.
p = kill/(kill+death) is the signal the adaptive curriculum uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_jax_bots as B
from jax_bomb import jax_train as T
from jax_bomb.bun_env import (H, W, N_OBS_CH, configure_training, init_batch,
                              prepare, step)

EVENTS = ("kill", "death", "self", "trade", "bot_self", "bombs")


def make_rollout(arch, games, max_steps):
    legacy = len(T.JAX_BOT_NAMES) - 1

    def one_tick(carry, _):
        states, key, params, kind = carry
        key, k_act, k_bot, k_flee, k_step = jax.random.split(key, 5)
        obs = T.both_perspectives(states)
        mm, bm = T.both_masks(states)
        gv = T.both_states(states)
        a0, _, _ = T.sample_actions(params, arch, obs[:games],
                                    (mm[:games], bm[:games]), k_act,
                                    state=gv[:games])
        kinds = jnp.full((games,), jnp.minimum(kind, legacy - 1), jnp.int32)
        rule = B.rule_bot_actions(states, jnp.ones((games,), jnp.int32),
                                  mm[games:], bm[games:], k_bot, kinds)
        flee = T.flee_bot_actions(states.pos[:, 1], states.pos[:, 0],
                                  mm[games:], bm[games:], k_flee)
        a1 = jnp.where(kind == legacy, flee, rule)
        acts = jnp.stack([a0, a1], axis=1)
        new_states, _, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, auto_reset=False, return_info=True)
        )(states, acts, jax.random.split(k_step, games))
        f = lambda name: info[name].astype(jnp.float32)
        kill = jnp.maximum(f("surviving_causal_kill")[:, 0],
                           f("surviving_physical_kill")[:, 0])
        ev = jnp.stack([kill, f("death")[:, 0], f("own_bomb_defeat")[:, 0],
                        f("mutual_death"), f("own_bomb_defeat")[:, 1],
                        f("bomb_placed")[:, 0]], axis=-1).sum(0)
        return (new_states, key, params, kind), ev

    @jax.jit
    def rollout(params, kind, key):
        key, ik = jax.random.split(key)
        states = init_batch(ik, games)
        _, ev = jax.lax.scan(one_tick, (states, key, params, kind), None,
                             length=max_steps)
        return ev.sum(0)

    return rollout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actors", nargs="+", required=True,
                    help="checkpoint paths, or 'random' for fresh init")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--arch", default="transformer")
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--tiers", default=",".join(T.JAX_BOT_NAMES))
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--curriculum", default="danger_arena=1")
    ap.add_argument("--reward-profile", default="legacy")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    labels = args.labels or [Path(a).stem for a in args.actors]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    prepare()
    configure_training(args.curriculum, 1, False, args.reward_profile)
    rollout = make_rollout(args.arch, args.games, args.max_steps)

    results = {}
    header = f"{'actor':>16} {'tier':>12} " + " ".join(f"{e:>8}" for e in EVENTS) + "        p"
    print(header, flush=True)
    for label, spec in zip(labels, args.actors):
        if spec == "random":
            params = T.init_net(jax.random.PRNGKey(0), args.arch, N_OBS_CH, H, W,
                                **({"hidden": args.hidden}
                                   if args.arch != "transformer" else {}))
        else:
            params = T.load_params(spec)
        results[label] = {}
        for tier in tiers:
            kind = jnp.int32(T.JAX_BOT_NAMES.index(tier))
            ev = np.asarray(rollout(params, kind, jax.random.PRNGKey(args.seed)))
            rate = {e: float(v) / args.games for e, v in zip(EVENTS, ev)}
            denom = rate["kill"] + rate["death"]
            rate["p"] = rate["kill"] / denom if denom > 0 else float("nan")
            results[label][tier] = rate
            print(f"{label:>16} {tier:>12} "
                  + " ".join(f"{rate[e]:8.3f}" for e in EVENTS)
                  + f" {rate['p']:8.3f}", flush=True)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "jax_bot_ladder_v1", "seed": args.seed,
            "games": args.games, "max_steps": args.max_steps,
            "curriculum": args.curriculum, "arch": args.arch,
            "unit": "events per learner episode (P0)", "results": results,
        }, indent=2) + "\n")
        print(f"report -> {out}")


if __name__ == "__main__":
    main()
