#!/usr/bin/env python3
"""Probe critic value/win calibration on forced (must-win / must-lose) states.

We build a handful of board states whose outcome for player 0 is essentially
決定的 (决定的): either player 0 dies within a couple of ticks no matter what, or
the opponent dies while player 0 sits safely out of every blast. The forced
label is *verified* by simulating a few ticks with both players idle (no actor
policy needed), then we ask the critic for its value estimate (discounted
return in [-20, 20]), win probability (sigmoid of the win head) and its best
legal Q. A well-calibrated critic should score must-win states near the top of
the range with win_prob≈1 and must-lose states near the bottom with win_prob≈0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("QQT_ALLOW_CPU", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_env as env
from jax_bomb.bun_critic import independent_critic_forward
from scripts import generate_bun_critic_data as g
from scripts.train_bun_separate_ac import load_checkpoint

IDLE = jnp.asarray([4, 0], jnp.int32)  # move=stay, no bomb
CRITIC_INPUT_DIM = env.N_OBS_CH * env.H * env.W + 24 + 44


def mirror_doomed(seed):
    """Opponent (player 1) is boxed in by four fuse=1 blasts; player 0 is far
    away and untouched. Player 0 should win."""
    base = g.open_state(seed)
    state = base._replace(core=base.core._replace(
        pos=jnp.asarray([[1.5, 1.5], [6.5, 6.5]], jnp.float32)))
    for row, column in ((6, 4), (6, 8), (4, 6), (8, 6)):
        state = g.with_bomb(state, row, column, 1, 0, 3)
    return state


def boxed_self(seed):
    """Player 0 sits on the intersection of two fuse=1 blasts with walls on the
    escape squares; opponent is far. Player 0 should lose."""
    base = g.open_state(seed)
    core = base.core._replace(
        pos=jnp.asarray([[6.5, 6.5], [1.5, 12.5]], jnp.float32))
    # wall off the four orthogonal escape cells so idle == only option == death
    wall = core.wall
    for row, column in ((5, 6), (7, 6), (6, 5), (6, 7)):
        wall = wall.at[row, column].set(True)
    state = base._replace(core=core._replace(wall=wall))
    state = g.with_bomb(state, 6, 6, 1, 1, 2)
    return state


# label, bucket for context encoding, builder
SCENARIOS = [
    ("must_lose", "doomed", g.scenario_state, "doomed"),
    ("must_lose", "doomed", boxed_self, "boxed_self_on_bomb"),
    ("must_win", "direct_owner_kill", g.scenario_state, "direct_owner_kill"),
    ("must_win", "doomed", mirror_doomed, "mirror_doomed"),
]


def build_states(seed_base):
    states = []
    meta = []
    for index, (label, bucket, builder, name) in enumerate(SCENARIOS):
        seed = seed_base + index
        if builder is g.scenario_state:
            state = g.scenario_state(bucket, seed)
        else:
            state = builder(seed)
        states.append(state)
        meta.append({"label": label, "bucket": bucket, "name": name})
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *states)
    return stacked, meta


def verify_forced_outcome(stacked, ticks=8):
    """Advance every state with both players idle and report survival/winner."""
    batch = stacked.core.pos.shape[0]
    idle_actions = jnp.broadcast_to(IDLE, (batch, 2, 2))

    @jax.jit
    def step_once(states, key):
        keys = jax.random.split(key, batch)
        candidate, done, info = jax.vmap(
            lambda s, a, r: env.step(s, a, r, auto_reset=False, return_info=True)
        )(states, idle_actions, keys)
        return candidate, done, info

    states = stacked
    ever_done = np.zeros(batch, bool)
    winner = np.full(batch, -1, np.int32)
    key = jax.random.PRNGKey(0)
    for _ in range(ticks):
        key, step_key = jax.random.split(key)
        states, done, info = step_once(states, step_key)
        done = np.asarray(done)
        step_winner = np.asarray(info["winner"])
        newly = done & ~ever_done
        winner[newly] = step_winner[newly]
        ever_done = ever_done | done
    alive = np.asarray(states.core.alive)
    return alive, winner, ever_done


def critic_inputs(stacked, meta):
    obs0, _, global0, _, _, _ = g.observe(stacked)
    obs0 = np.asarray(obs0, np.float32)
    global0 = np.asarray(global0, np.float32)
    rows = []
    for index in range(obs0.shape[0]):
        state = jax.tree.map(lambda value: value[index], stacked)
        bucket_id = g.BUCKETS.index(meta[index]["bucket"])
        # neural self-play opponent context (opponent_id=1, league "recent")
        context = g.context_vector(
            state, actor_id=0, opponent_id=1, bucket_id=bucket_id,
            league_opponent_id=g.OPPONENT_NAMES.index("recent"))
        rows.append(np.concatenate(
            [obs0[index].reshape(-1), global0[index], context]))
    return np.stack(rows).astype(np.float32)


def legal_best_q(q, move_mask, ability_mask):
    """Return the best critic Q among actions legal in each source state."""
    legal = (jnp.asarray(move_mask)[:, 0, :, None]
             & jnp.asarray(ability_mask)[:, 0, None, :]).reshape(q.shape)
    return jnp.max(jnp.where(legal, q, -jnp.inf), axis=1)


def evaluate(critic_path, inputs, move_mask, ability_mask):
    params, _ = load_checkpoint(critic_path)
    params = jax.tree.map(jnp.asarray, params)
    got = inputs.shape[1]
    if params["layer1"][0].shape[0] != got:
        raise SystemExit(
            f"critic expects input_dim={params['layer1'][0].shape[0]} but built "
            f"{got}; checkpoint incompatible with current obs/context layout")
    value, _, win_logit, q = independent_critic_forward(params, jnp.asarray(inputs))
    return (np.asarray(value), 1.0 / (1.0 + np.exp(-np.asarray(win_logit))),
            np.asarray(legal_best_q(q, move_mask, ability_mask)))


def monte_carlo_truth(actor_path, stacked, samples, horizon, seed_base):
    """Ground-truth policy return + win rate per state via self-play rollouts.

    Respects respawn / long-horizon game semantics — a single kill is not
    treated as a terminal win. Mirrors how value_target was generated
    (gamma=0.995, stochastic policy, no hard-safety), so it is the honest target
    the value head is trained toward.
    """
    actor = jax.tree.map(jnp.asarray, load_checkpoint(actor_path)[0])
    policy_prob = g.build_policy_probabilities(actor)
    rollout = g.build_rollout(actor, actor, samples, horizon, False, False)
    batch = stacked.core.pos.shape[0]
    values = np.zeros(batch, np.float32)
    win_rates = np.zeros(batch, np.float32)
    rng = np.random.default_rng(seed_base)
    for index in range(batch):
        state = jax.tree.map(lambda value: value[index], stacked)
        probs = np.asarray(policy_prob(state)[0], np.float64)
        probs = probs / max(probs.sum(), 1e-8)
        sampled = rng.choice(len(probs), size=samples, p=probs)
        first_actions = jnp.asarray(np.stack(
            [sampled // env.N_BOMB, sampled % env.N_BOMB], axis=-1), jnp.int32)
        returns, wins, _, _, _ = rollout(
            state, first_actions, jnp.asarray(1, jnp.int32),
            g.prng_seed(seed_base + index))
        values[index] = float(np.asarray(returns).mean())
        win_rates[index] = float(np.asarray(wins).mean())
    return values, win_rates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--critic", action="append", default=[],
        help="critic .pkl checkpoint(s); repeat to compare arms")
    parser.add_argument(
        "--run-dir",
        default="checkpoints/bun_high_throughput_20260929",
        help="if no --critic given, evaluate latest cycle critic of every arm here")
    parser.add_argument("--seed-base", type=int, default=20260929)
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--mc-horizon", type=int, default=128)
    parser.add_argument("--no-mc", action="store_true",
                        help="skip Monte-Carlo ground truth (faster)")
    args = parser.parse_args(argv)

    env.prepare()
    stacked, meta = build_states(args.seed_base)
    alive, winner, done = verify_forced_outcome(stacked, args.ticks)
    inputs = critic_inputs(stacked, meta)
    _, _, _, _, move_mask, ability_mask = g.observe(stacked)

    critics = list(args.critic)
    if not critics:
        run = ROOT / args.run_dir
        for arm in sorted(p for p in run.iterdir() if p.is_dir()):
            cycles = sorted(p for p in arm.glob("cycle_*_critic.pkl")
                            if "target" not in p.name)
            if cycles:
                critics.append(str(cycles[-1]))

    # verification of the forced labels
    print("=== forced-outcome verification (both players idle, no policy) ===")
    for index, info in enumerate(meta):
        p0, p1 = bool(alive[index, 0]), bool(alive[index, 1])
        expect_win = info["label"] == "must_win"
        ok = (p0 and not p1) if expect_win else (p1 and not p0)
        print(f"  [{info['name']:22s}] label={info['label']:9s} "
              f"p0_alive={p0} p1_alive={p1} winner={int(winner[index]):+d} "
              f"-> {'OK' if ok else 'MISLABELED'}")
    print("  note: 抢包子 has respawn; instant death != terminal loss, so the "
          "MC discounted return below is the real accuracy yardstick.")

    for critic_path in critics:
        value, win_prob, legal_best_q_value = evaluate(
            critic_path, inputs, move_mask, ability_mask)
        mc_value = mc_win = None
        if not args.no_mc:
            actor_path = critic_path.replace("_critic.pkl", "_actor.pt")
            if Path(actor_path).exists():
                mc_value, mc_win = monte_carlo_truth(
                    actor_path, stacked, args.mc_samples, args.mc_horizon,
                    args.seed_base)
        print(f"\n=== critic: {critic_path} ===")
        header = (f"  {'scenario':22s} {'label':9s} {'critic_V':>9s} "
                  f"{'win_prob':>9s} {'best_Q':>8s}")
        if mc_value is not None:
            header += f" {'MC_V(true)':>11s} {'MC_win':>7s} {'V_err':>8s}"
        print(header)
        v_errs = []
        for index, info in enumerate(meta):
            line = (f"  {info['name']:22s} {info['label']:9s} "
                    f"{value[index]:9.3f} {win_prob[index]:9.3f} "
                    f"{legal_best_q_value[index]:8.3f}")
            if mc_value is not None:
                err = value[index] - mc_value[index]
                v_errs.append(abs(err))
                line += (f" {mc_value[index]:11.3f} {mc_win[index]:7.2f} "
                         f"{err:+8.3f}")
            print(line)
        wins = [value[i] for i, m in enumerate(meta) if m["label"] == "must_win"]
        loses = [value[i] for i, m in enumerate(meta) if m["label"] == "must_lose"]
        summary = (f"  -- critic value separation win-vs-lose = "
                   f"{np.mean(wins) - np.mean(loses):+.3f} "
                   f"(win={np.mean(wins):+.3f}, lose={np.mean(loses):+.3f})")
        if v_errs:
            summary += f"; mean |critic_V - MC_V| = {np.mean(v_errs):.3f}"
        print(summary)


if __name__ == "__main__":
    main()
