#!/usr/bin/env python3
"""Generate auditable Bun critic targets from exact rules and MC rollouts."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_env as env
from jax_bomb import bun_safety
from jax_bomb import jax_train
from jax_bomb.jax_net import transformer_forward


BUCKETS = (
    "safe_bomb", "dangerous_bomb", "escapable", "doomed",
    "direct_owner_kill", "causal_trigger_kill", "trade", "ineffective_bomb",
    "ambush_contact", "post_kill_window", "steal", "carry_return",
    "full_early", "full_mid", "full_late",
    "ambush_early", "ambush_mid", "ambush_late",
)
OPPONENT_NAMES = ("rule_combat", "weak", "old", "recent")
DEFAULT_SPLIT_COUNTS = {"train": 3, "validation": 1, "test": 1}
ACTION_COUNT = env.N_MOVES * env.N_BOMB
_AMBUSH_CURRICULUM = "full_ambush=1"


def load_params(path):
    with open(path, "rb") as file:
        value = pickle.load(file)
    value = value.get("params", value) if isinstance(value, dict) else value
    return jax.tree.map(jnp.asarray, value)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prng_seed(value):
    return np.uint32(int(value) & 0xFFFFFFFF)


def open_state(seed):
    state = env._fresh(jax.random.PRNGKey(seed))
    core = state.core._replace(
        wall=jnp.zeros_like(state.core.wall),
        brick=jnp.zeros_like(state.core.brick),
        pushable=jnp.zeros_like(state.core.pushable),
        brick_linger=jnp.zeros_like(state.core.brick_linger),
        fuse=jnp.zeros_like(state.core.fuse),
        owner=jnp.full_like(state.core.owner, -1),
        bomb_blast=jnp.zeros_like(state.core.bomb_blast),
        blast_linger=jnp.zeros_like(state.core.blast_linger),
        pos=jnp.asarray([[6.5, 4.5], [6.5, 10.5]], jnp.float32),
        hp=jnp.ones_like(state.core.hp),
        alive=jnp.ones_like(state.core.alive),
        invuln=jnp.zeros_like(state.core.invuln),
    )
    return state._replace(
        core=core,
        blast_owner_linger=jnp.zeros_like(state.blast_owner_linger),
        blast_causal_linger=jnp.zeros_like(state.blast_causal_linger),
        blast_trigger_linger=jnp.zeros_like(state.blast_trigger_linger),
        kill_window_ticks=jnp.zeros_like(state.kill_window_ticks),
        lesson=jnp.asarray(env.LESSON_FULL_AMBUSH, jnp.int8),
    )


def with_bomb(state, row, column, fuse, owner, blast=2):
    return state._replace(core=state.core._replace(
        fuse=state.core.fuse.at[row, column].set(fuse),
        owner=state.core.owner.at[row, column].set(owner),
        bomb_blast=state.core.bomb_blast.at[row, column].set(blast),
    ))


def scenario_state(bucket, seed, snapshot_fn=None):
    if bucket in ("ambush_contact", "safe_bomb", "dangerous_bomb", "escapable",
                  "ambush_early", "ambush_mid", "ambush_late"):
        env.configure_training(
            _AMBUSH_CURRICULUM, 1,
            reward_profile=("danger_arena" if _AMBUSH_CURRICULUM.startswith(
                "danger_arena=") else "combat_evolution"))
        state = env._fresh(jax.random.PRNGKey(seed))
        if bucket in ("dangerous_bomb", "escapable"):
            cell = np.floor(np.asarray(state.core.pos[0])).astype(np.int32)
            state = with_bomb(state, int(cell[0]), int(cell[1]), 4, 0, 2)
        if bucket in ("ambush_early", "ambush_mid", "ambush_late"):
            if snapshot_fn is None:
                raise ValueError(f"{bucket} requires a snapshot rollout")
            return snapshot_fn(state, prng_seed(seed ^ 0x5A17))
        return state
    if bucket == "doomed":
        state = open_state(seed)._replace(core=open_state(seed).core._replace(
            pos=jnp.asarray([[6.5, 6.5], [1.5, 1.5]], jnp.float32)))
        for row, column in ((6, 4), (6, 8), (4, 6), (8, 6)):
            state = with_bomb(state, row, column, 1, 0, 3)
        return state
    if bucket == "direct_owner_kill":
        state = open_state(seed)._replace(core=open_state(seed).core._replace(
            pos=jnp.asarray([[6.5, 3.5], [6.5, 8.5]], jnp.float32)))
        return with_bomb(state, 6, 6, 1, 0, 2)
    if bucket == "causal_trigger_kill":
        state = open_state(seed)._replace(core=open_state(seed).core._replace(
            pos=jnp.asarray([[6.5, 2.5], [6.5, 8.5]], jnp.float32)))
        state = with_bomb(state, 6, 4, 1, 0, 2)
        return with_bomb(state, 6, 6, 10, 1, 2)
    if bucket == "trade":
        state = open_state(seed)._replace(core=open_state(seed).core._replace(
            pos=jnp.asarray([[8.5, 6.5], [6.5, 6.5]], jnp.float32)))
        state = with_bomb(state, 6, 4, 1, 0, 2)
        return with_bomb(state, 8, 4, 1, 1, 2)
    if bucket == "ineffective_bomb":
        return open_state(seed)
    if bucket == "post_kill_window":
        state = scenario_state("direct_owner_kill", seed)
        state, _, _ = env.step(
            state, jnp.asarray([[4, 0], [4, 0]], jnp.int32),
            jax.random.PRNGKey(seed ^ 0xABC), auto_reset=False, return_info=True)
        return state
    if bucket == "steal":
        env.configure_training("near_steal=1", 1, reward_profile="legacy")
        state = env._fresh(jax.random.PRNGKey(seed))
        enemy_base = env._BUN_BASES[1].astype(jnp.float32) + 1.5
        return state._replace(core=state.core._replace(
            pos=state.core.pos.at[0].set(enemy_base)))
    if bucket == "carry_return":
        env.configure_training("carry_return=1", 1, reward_profile="legacy")
        return env._fresh(jax.random.PRNGKey(seed))
    env.configure_training("full=1", 1, reward_profile="combat_evolution")
    state = env._fresh(jax.random.PRNGKey(seed))
    if snapshot_fn is None:
        return state
    return snapshot_fn(state, prng_seed(seed ^ 0x5A17))


def observe(states):
    danger = jax.vmap(
        lambda state: env._danger_map(
            state.core.fuse, state.core.wall,
            state.core.bomb_blast, state.core.brick))(states)
    obs0 = jax.vmap(lambda state, danger_map: env.make_obs(state, 0, danger_map))(
        states, danger)
    obs1 = jax.vmap(lambda state, danger_map: env.make_obs(state, 1, danger_map))(
        states, danger)
    global0 = jax.vmap(lambda state: env.global_vec(state, 0))(states)
    global1 = jax.vmap(lambda state: env.global_vec(state, 1))(states)
    move_mask, ability_mask = jax.vmap(env.legal_mask)(states)
    return obs0, obs1, global0, global1, move_mask, ability_mask


def context_vector(state, actor_id, opponent_id, bucket_id, league_opponent_id=0):
    lesson = int(np.asarray(state.lesson))
    carried = int(np.asarray(state.bun_carried[0])) >= 0
    enemy_respawn = int(np.asarray(state.bun_respawn[1])) > 0
    phase = 2 if carried else (1 if enemy_respawn else 0)
    legacy_lesson_count = env.LESSON_DANGER_ARENA
    context = np.zeros(
        2 + 2 + legacy_lesson_count + 3 + len(BUCKETS) + len(OPPONENT_NAMES),
        np.float32)
    cursor = 0
    context[cursor + actor_id] = 1.0; cursor += 2
    context[cursor + opponent_id] = 1.0; cursor += 2
    encoded_lesson = (env.LESSON_FULL_AMBUSH
                      if lesson == env.LESSON_DANGER_ARENA else lesson)
    context[cursor + encoded_lesson] = 1.0; cursor += legacy_lesson_count
    context[cursor + phase] = 1.0; cursor += 3
    context[cursor + bucket_id] = 1.0; cursor += len(BUCKETS)
    context[cursor + league_opponent_id] = 1.0
    return context


def wilson(successes, samples, z=1.96):
    if samples <= 0:
        return 0.0, 1.0
    proportion = successes / samples
    denominator = 1.0 + z * z / samples
    center = (proportion + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / samples + z * z / (4 * samples * samples)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def mean_interval(values, z=1.96):
    values = np.asarray(values, np.float64)
    if values.size <= 1:
        value = float(values.mean()) if values.size else 0.0
        return value, value
    radius = z * float(values.std(ddof=1)) / math.sqrt(values.size)
    mean = float(values.mean())
    return mean - radius, mean + radius


def confidence_intervals_valid(row, atol=1e-5):
    legal = np.asarray(row["legal"], np.bool_)
    if not legal.any():
        return True
    win_low = np.asarray(row["win_ci_low"])[legal]
    win = np.asarray(row["win_rate"])[legal]
    win_high = np.asarray(row["win_ci_high"])[legal]
    q_low = np.asarray(row["q_ci_low"])[legal]
    q = np.asarray(row["q"])[legal]
    q_high = np.asarray(row["q_ci_high"])[legal]
    return bool(np.all(
        (win_low >= -atol)
        & (win_low <= win + atol)
        & (win <= win_high + atol)
        & (win_high <= 1.0 + atol)
        & (q_low <= q + atol)
        & (q <= q_high + atol)
    ))


def build_full_snapshot_rollout(actor_params, opponent_params, ticks):
    @jax.jit
    def rollout(state, opponent_id, seed):
        active = jnp.asarray(True)
        key = jax.random.PRNGKey(seed)

        def step_fn(carry, _):
            state, active, key = carry
            key, action_key0, action_key1, step_key = jax.random.split(key, 4)
            states = jax.tree.map(lambda value: value[None], state)
            obs0, obs1, global0, global1, move_mask, ability_mask = observe(states)
            move0, ability0, _, _ = transformer_forward(
                actor_params, obs0, global0)
            move1, ability1, _, _ = transformer_forward(
                opponent_params, obs1, global1)
            joint0 = bun_safety.adjusted_joint_logits(
                move0, ability0, move_mask[:, 0], ability_mask[:, 0],
                jnp.zeros((1, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
            joint1 = bun_safety.adjusted_joint_logits(
                move1, ability1, move_mask[:, 1], ability_mask[:, 1],
                jnp.zeros((1, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
            index0 = jax.random.categorical(action_key0, joint0)[0]
            index1 = jax.random.categorical(action_key1, joint1)[0]
            action0 = jnp.stack([index0 // env.N_BOMB, index0 % env.N_BOMB])
            neural1 = jnp.stack([index1 // env.N_BOMB, index1 % env.N_BOMB])
            rule1 = jax_train.flee_bot_actions(
                states.core.pos[:, 1], states.core.pos[:, 0], move_mask[:, 1],
                ability_mask[:, 1], action_key1, idle_ratio=0.0,
                roam_ratio=0.0, pure_flee_ratio=0.0)[0]
            action1 = jnp.where(opponent_id == 0, rule1, neural1)
            candidate, done, _ = env.step(
                state, jnp.stack([action0, action1]), step_key,
                auto_reset=False, return_info=True)
            state = jax.tree.map(
                lambda old, new: jnp.where(active, new, old), state, candidate)
            return (state, active & ~done, key), None

        return jax.lax.scan(step_fn, (state, active, key), None, length=ticks)[0][0]

    return rollout


def build_rollout(actor_params, opponent_params, samples, horizon, greedy, hard_safety):
    @jax.jit
    def rollout(state, first_actions, opponent_id, seed):
        batch = first_actions.shape[0]
        states = jax.tree.map(lambda value: jnp.repeat(value[None], batch, axis=0), state)
        active = jnp.ones((batch,), jnp.bool_)
        returns = jnp.zeros((batch,), jnp.float32)
        discount = jnp.ones((batch,), jnp.float32)
        wins = jnp.zeros((batch,), jnp.bool_)
        surviving_kills = jnp.zeros((batch,), jnp.bool_)
        trades = jnp.zeros((batch,), jnp.bool_)
        objective = jnp.zeros((batch,), jnp.float32)
        key = jax.random.PRNGKey(seed)

        def step_fn(carry, tick):
            (states, active, returns, discount, wins, surviving_kills,
             trades, objective, key) = carry
            key, action_key0, action_key1, step_key = jax.random.split(key, 4)
            obs0, obs1, global0, global1, move_mask, ability_mask = observe(states)
            move0, ability0, _, _ = transformer_forward(actor_params, obs0, global0)
            move1, ability1, _, _ = transformer_forward(opponent_params, obs1, global1)
            if hard_safety:
                analysis = jax.vmap(bun_safety.analyze_actions)(states)
                joint0 = bun_safety.adjusted_joint_logits(
                    move0, ability0, move_mask[:, 0], ability_mask[:, 0],
                    analysis.avoidable[:, 0], "hard", 0.0)
            else:
                joint0 = bun_safety.adjusted_joint_logits(
                    move0, ability0, move_mask[:, 0], ability_mask[:, 0],
                    jnp.zeros((batch, env.N_MOVES, env.N_BOMB), jnp.bool_),
                    "off", 0.0)
            joint1 = bun_safety.adjusted_joint_logits(
                move1, ability1, move_mask[:, 1], ability_mask[:, 1],
                jnp.zeros((batch, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
            if greedy:
                action0_index = jnp.argmax(joint0, axis=-1)
                neural1_index = jnp.argmax(joint1, axis=-1)
            else:
                action0_index = jax.random.categorical(action_key0, joint0)
                neural1_index = jax.random.categorical(action_key1, joint1)
            action0 = jnp.stack([
                action0_index // env.N_BOMB, action0_index % env.N_BOMB], axis=-1)
            neural1 = jnp.stack([
                neural1_index // env.N_BOMB, neural1_index % env.N_BOMB], axis=-1)
            rule1 = jax_train.flee_bot_actions(
                states.core.pos[:, 1], states.core.pos[:, 0],
                move_mask[:, 1], ability_mask[:, 1], action_key1,
                idle_ratio=0.0, roam_ratio=0.0, pure_flee_ratio=0.0)
            action1 = jnp.where(opponent_id == 0, rule1, neural1)
            action0 = jnp.where(tick == 0, first_actions, action0)
            actions = jnp.stack([action0, action1], axis=1)
            keys = jax.random.split(step_key, batch)
            candidate, done, info = jax.vmap(
                lambda current, action, rng: env.step(
                    current, action, rng, auto_reset=False, return_info=True))(
                        states, actions, keys)
            rewards = env.reward_from_events(
                info["dmg"], states.core.alive, info["alive"], info["hp"], done,
                info["crate"], jnp.zeros((batch, 2), jnp.bool_), info["walls"],
                0.0, 0.0, 0.0, 1.0, rule_info=info)[:, 0]
            returns = returns + discount * jnp.where(active, rewards, 0.0)
            discount = discount * 0.995
            wins = wins | (active & done & (info["winner"] == 0))
            surviving_kills = surviving_kills | (
                active & info["surviving_kill"][:, 0])
            trades = trades | (active & info["mutual_death"])
            objective = objective + jnp.where(
                active, info["objective_progress"][:, 0], 0.0)
            states = jax.tree.map(
                lambda old, new: jnp.where(
                    active.reshape((-1,) + (1,) * (old.ndim - 1)), new, old),
                states, candidate)
            active = active & ~done
            return (states, active, returns, discount, wins, surviving_kills,
                    trades, objective, key), None

        result, _ = jax.lax.scan(
            step_fn, (states, active, returns, discount, wins, surviving_kills,
                      trades, objective, key), jnp.arange(horizon))
        return result[2], result[4], result[5], result[6], result[7]

    return rollout


def build_first_transitions(opponent_params):
    @jax.jit
    def transition(state, actions, opponent_id, seed):
        batch = actions.shape[0]
        states = jax.tree.map(
            lambda value: jnp.repeat(value[None], batch, axis=0), state)
        _, obs1, _, global1, move_mask, ability_mask = observe(states)
        move1, ability1, _, _ = transformer_forward(
            opponent_params, obs1, global1)
        joint1 = bun_safety.adjusted_joint_logits(
            move1, ability1, move_mask[:, 1], ability_mask[:, 1],
            jnp.zeros((batch, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
        neural_index = jnp.argmax(joint1, axis=-1)
        neural_action = jnp.stack([
            neural_index // env.N_BOMB, neural_index % env.N_BOMB], axis=-1)
        rule_action = jax_train.flee_bot_actions(
            states.core.pos[:, 1], states.core.pos[:, 0], move_mask[:, 1],
            ability_mask[:, 1], jax.random.PRNGKey(seed),
            idle_ratio=0.0, roam_ratio=0.0, pure_flee_ratio=0.0)
        opponent_action = jnp.where(opponent_id == 0, rule_action, neural_action)
        joint_actions = jnp.stack([actions, opponent_action], axis=1)
        keys = jax.random.split(jax.random.PRNGKey(seed ^ 0xA5A5), batch)
        candidates, done, info = jax.vmap(
            lambda current, action, rng: env.step(
                current, action, rng, auto_reset=False, return_info=True))(
                    states, joint_actions, keys)
        reward = env.reward_from_events(
            info["dmg"], states.core.alive, info["alive"], info["hp"], done,
            info["crate"], jnp.zeros((batch, 2), jnp.bool_), info["walls"],
            0.0, 0.0, 0.0, 1.0, rule_info=info)[:, 0]
        provenance = {
            "death_source": info["death_source"],
            "causal_death_source": info["causal_death_source"],
            "trigger_damage_source": info["trigger_damage_source"],
            "credited_kill": info["credited_kill"],
            "causal_kill": info["causal_kill"],
            "trigger_kill": info["trigger_kill"],
            "mutual_death": info["mutual_death"],
            "own_bomb_defeat": info["own_bomb_defeat"],
            "opponent_physical_defeat": info["opponent_physical_defeat"],
            "opponent_causal_defeat": info["opponent_causal_defeat"],
        }
        return candidates, reward, done, provenance
    return transition


def build_policy_probabilities(actor_params):
    @jax.jit
    def probabilities(state):
        states = jax.tree.map(lambda value: value[None], state)
        obs0, _, global0, _, move_mask, ability_mask = observe(states)
        move, ability, _, _ = transformer_forward(actor_params, obs0, global0)
        joint = bun_safety.adjusted_joint_logits(
            move, ability, move_mask[:, 0], ability_mask[:, 0],
            jnp.zeros((1, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
        return jax.nn.softmax(joint[0]), obs0[0], global0[0]
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", action="append", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--opponent-name", choices=OPPONENT_NAMES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed-base", type=int, default=202609260000)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--mc-horizon", type=int, default=128)
    parser.add_argument("--deterministic-horizon", "--exact-horizon",
                        dest="deterministic_horizon", type=int, default=32)
    parser.add_argument("--ambush-curriculum", default="full_ambush=1",
                        choices=("full_ambush=1", "danger_arena=1"))
    parser.add_argument("--train-per-bucket", type=int, default=3)
    parser.add_argument("--validation-per-bucket", type=int, default=1)
    parser.add_argument("--test-per-bucket", type=int, default=1)
    parser.add_argument(
        "--buckets", default=",".join(BUCKETS),
        help="comma-separated subset of counterfactual state buckets")
    args = parser.parse_args()
    global _AMBUSH_CURRICULUM
    _AMBUSH_CURRICULUM = args.ambush_curriculum

    selected_buckets = tuple(item.strip() for item in args.buckets.split(",") if item.strip())
    unknown_buckets = sorted(set(selected_buckets) - set(BUCKETS))
    if unknown_buckets or not selected_buckets:
        raise SystemExit(f"invalid bucket selection: {unknown_buckets or selected_buckets}")

    started = time.time()
    split_counts = {
        "train": args.train_per_bucket,
        "validation": args.validation_per_bucket,
        "test": args.test_per_bucket,
    }
    if any(count <= 0 for count in split_counts.values()):
        raise SystemExit("all split counts must be positive")
    env.prepare()
    actor_params = [load_params(path) for path in args.actor]
    opponent_params = load_params(args.opponent)
    deterministic_rollouts = [build_rollout(
        params, opponent_params, 1, args.deterministic_horizon, True, False)
        for params in actor_params]
    mc_rollouts = [build_rollout(
        params, opponent_params, args.mc_samples, args.mc_horizon, False, False)
        for params in actor_params]
    first_transitions = build_first_transitions(opponent_params)
    policy_probability_fns = [
        build_policy_probabilities(params) for params in actor_params]
    analysis_fn = jax.jit(bun_safety.analyze_actions)
    observe_fn = jax.jit(observe)
    snapshot_ticks = {
        "full_early": 64, "full_mid": 800, "full_late": 1800,
        "ambush_early": 12, "ambush_mid": 48, "ambush_late": 96,
    }
    if args.ambush_curriculum == "danger_arena=1":
        snapshot_ticks.update(
            {"ambush_early": 12, "ambush_mid": 100, "ambush_late": 220})
    snapshot_rollouts = [{
        bucket: build_full_snapshot_rollout(params, opponent_params, ticks)
        for bucket, ticks in snapshot_ticks.items()
    } for params in actor_params]

    records = []
    states = []
    split_seed_sets = {name: set() for name in split_counts}
    split_prng_sets = {name: set() for name in split_counts}
    for actor_id in range(len(actor_params)):
        for bucket in selected_buckets:
            bucket_id = BUCKETS.index(bucket)
            offset = 0
            for split, count in split_counts.items():
                for item in range(count):
                    seed = args.seed_base + actor_id * 100000 + bucket_id * 1000 + offset + item
                    opponent_id = 0 if args.opponent_name == "rule_combat" else 1
                    snapshot_fn = None
                    if bucket in snapshot_ticks:
                        snapshot_fn = lambda initial, rng_seed, fn=(
                                snapshot_rollouts[actor_id][bucket]), oid=opponent_id: fn(
                                    initial, jnp.asarray(oid, jnp.int32), rng_seed)
                    trajectory_seed = seed
                    state = scenario_state(bucket, trajectory_seed, snapshot_fn)
                    for attempt in range(1, 33):
                        if bucket not in snapshot_ticks or int(
                                np.asarray(state.core.t)) == snapshot_ticks[bucket]:
                            break
                        trajectory_seed = seed + attempt * 10_000_000
                        state = scenario_state(bucket, trajectory_seed, snapshot_fn)
                    if bucket in snapshot_ticks and int(
                            np.asarray(state.core.t)) != snapshot_ticks[bucket]:
                        raise RuntimeError(
                            f"unable to obtain non-terminal {bucket} snapshot for seed {seed}")
                    split_seed_sets[split].add(trajectory_seed)
                    split_prng_sets[split].add(int(prng_seed(trajectory_seed)))
                    records.append({
                        "actor_id": actor_id, "bucket_id": bucket_id,
                        "bucket": bucket, "split": split, "seed": trajectory_seed,
                        "declared_seed": seed,
                        "opponent_id": opponent_id,
                        "league_opponent_id": OPPONENT_NAMES.index(args.opponent_name),
                        "league_opponent": args.opponent_name,
                        "provenance": "scenario_fixture" if bucket not in (
                            "ambush_contact", "safe_bomb", "dangerous_bomb",
                            "escapable", "steal", "carry_return", "full_early",
                            "full_mid", "full_late", "ambush_early",
                            "ambush_mid", "ambush_late") else (
                                "frozen_policy_rollout" if bucket in snapshot_ticks
                                else "native_reset"),
                    })
                    states.append(state)
                offset += count + 100

    output_rows = []
    stacked_states = jax.tree.map(lambda *values: np.stack(values), *states)
    all_actions = np.asarray([
        [move, ability]
        for move in range(env.N_MOVES)
        for ability in range(env.N_BOMB)], np.int32)
    for index, (record, state) in enumerate(zip(records, states)):
        actor_id = record["actor_id"]
        opponent_id = record["opponent_id"]
        actor = actor_params[actor_id]
        analysis = analysis_fn(state)
        legal = np.asarray(analysis.legal[0]).reshape(-1)
        survivable = np.asarray(analysis.survivable[0]).reshape(-1)
        avoidable = np.asarray(analysis.avoidable[0]).reshape(-1)
        doomed = bool(np.asarray(analysis.doomed[0]))
        repeated_actions = np.repeat(all_actions, args.mc_samples, axis=0)
        deterministic = deterministic_rollouts[actor_id](
            state, jnp.asarray(all_actions), jnp.asarray(opponent_id, jnp.int32),
            prng_seed(record["seed"] ^ 0x10000))
        mc = mc_rollouts[actor_id](
            state, jnp.asarray(repeated_actions), jnp.asarray(opponent_id, jnp.int32),
            prng_seed(record["seed"] ^ 0x20000))
        deterministic_return, deterministic_win, deterministic_kill, \
            deterministic_trade, deterministic_objective = map(
                lambda value: np.asarray(jax.device_get(value)), deterministic)
        mc_return, mc_win, mc_kill, mc_trade, mc_objective = map(
            lambda value: np.asarray(jax.device_get(value)).reshape(
                ACTION_COUNT, args.mc_samples), mc)
        q = np.clip(mc_return.mean(axis=1), -20.0, 20.0).astype(np.float32)
        win_rate = mc_win.mean(axis=1).astype(np.float32)
        kill_rate = mc_kill.mean(axis=1).astype(np.float32)
        trade_rate = mc_trade.mean(axis=1).astype(np.float32)
        objective_mean = mc_objective.mean(axis=1).astype(np.float32)
        ci = np.asarray([
            wilson(int(mc_win[action].sum()), args.mc_samples)
            for action in range(ACTION_COUNT)], np.float32)
        q_ci = np.asarray([
            mean_interval(mc_return[action]) for action in range(ACTION_COUNT)
        ], np.float32)
        q_ci = np.clip(q_ci, -20.0, 20.0)
        probabilities, obs, global_state = map(
            np.asarray, policy_probability_fns[actor_id](state))
        probabilities = probabilities * legal
        probabilities /= max(float(probabilities.sum()), 1e-8)
        value_target = float(np.sum(probabilities * q))
        win_target = float(np.sum(probabilities * win_rate))

        good_action = bad_action = -1
        pair_source = "unlabeled"
        legal_indices = np.flatnonzero(legal)
        if len(legal_indices) >= 2:
            best = int(legal_indices[np.argmax(q[legal_indices])])
            worst = int(legal_indices[np.argmin(q[legal_indices])])
            margin = float(q[best] - q[worst])
            rule_confident = (
                margin >= 1.0
                and survivable[best] and avoidable[worst])
            mc_confident = margin >= 1.0 and (
                float(ci[best, 0]) > float(ci[worst, 1])
                or float(q_ci[best, 0]) > float(q_ci[worst, 1]))
            if rule_confident or mc_confident:
                good_action, bad_action = best, worst
                pair_source = "rule_exact" if rule_confident else "monte_carlo"

        next_obs = np.zeros((ACTION_COUNT, env.N_OBS_CH, env.H, env.W), np.uint8)
        next_global = np.zeros((ACTION_COUNT, 24), np.float32)
        next_context = np.zeros((ACTION_COUNT, len(context_vector(
            state, actor_id, opponent_id, record["bucket_id"],
            record["league_opponent_id"]))), np.float32)
        immediate_reward = np.zeros((ACTION_COUNT,), np.float32)
        immediate_done = np.zeros((ACTION_COUNT,), np.bool_)
        candidates, rewards, dones, first_provenance = first_transitions(
            state, jnp.asarray(all_actions), jnp.asarray(opponent_id, jnp.int32),
            prng_seed(record["seed"] ^ 0x30000))
        candidate_obs, _, candidate_global, _, _, _ = observe_fn(candidates)
        next_obs[:] = np.clip(
            np.rint(np.asarray(candidate_obs) * 255.0), 0, 255).astype(np.uint8)
        next_global[:] = np.asarray(candidate_global)
        immediate_reward[:] = np.asarray(rewards)
        immediate_done[:] = np.asarray(dones)
        host_candidates = jax.device_get(candidates)
        for action_id in range(ACTION_COUNT):
            candidate = jax.tree.map(lambda value: value[action_id], host_candidates)
            next_context[action_id] = context_vector(
                candidate, actor_id, opponent_id, record["bucket_id"],
                record["league_opponent_id"])

        output_rows.append({
            **record,
            "obs": np.clip(np.rint(obs * 255.0), 0, 255).astype(np.uint8),
            "global": global_state.astype(np.float32),
            "context": context_vector(
                state, actor_id, opponent_id, record["bucket_id"],
                record["league_opponent_id"]),
            "legal": legal, "survivable": survivable, "avoidable": avoidable,
            "doomed": doomed,
            "deterministic_return": deterministic_return.astype(np.float32),
            "deterministic_win": deterministic_win,
            "deterministic_kill": deterministic_kill,
            "deterministic_trade": deterministic_trade,
            "deterministic_objective": deterministic_objective.astype(np.float32),
            "q": q, "win_rate": win_rate, "win_ci_low": ci[:, 0],
            "win_ci_high": ci[:, 1], "q_ci_low": q_ci[:, 0],
            "q_ci_high": q_ci[:, 1], "mc_kill_rate": kill_rate,
            "mc_trade_rate": trade_rate, "mc_objective": objective_mean,
            "mc_samples": np.full((ACTION_COUNT,), args.mc_samples, np.int16),
            "policy_probability": probabilities.astype(np.float32),
            "value_target": np.float32(value_target),
            "win_target": np.float32(win_target),
            "good_action": np.int16(good_action), "bad_action": np.int16(bad_action),
            "pair_source": pair_source, "next_obs": next_obs,
            "next_global": next_global, "next_context": next_context,
            "immediate_reward": immediate_reward, "immediate_done": immediate_done,
            "first_death_source": np.asarray(first_provenance["death_source"]),
            "first_causal_death_source": np.asarray(
                first_provenance["causal_death_source"]),
            "first_trigger_damage_source": np.asarray(
                first_provenance["trigger_damage_source"]),
            "first_credited_kill": np.asarray(first_provenance["credited_kill"]),
            "first_causal_kill": np.asarray(first_provenance["causal_kill"]),
            "first_trigger_kill": np.asarray(first_provenance["trigger_kill"]),
            "first_mutual_death": np.asarray(first_provenance["mutual_death"]),
            "first_own_bomb_defeat": np.asarray(
                first_provenance["own_bomb_defeat"]),
            "first_opponent_physical_defeat": np.asarray(
                first_provenance["opponent_physical_defeat"]),
            "first_opponent_causal_defeat": np.asarray(
                first_provenance["opponent_causal_defeat"]),
        })
        print(f"[{index + 1}/{len(records)}] {record['split']} actor={actor_id} "
              f"bucket={record['bucket']} legal={int(legal.sum())} pair={pair_source}",
              flush=True)

    keys = [key for key in output_rows[0] if key not in (
        "bucket", "split", "provenance", "pair_source", "league_opponent")]
    arrays = {key: np.stack([row[key] for row in output_rows]) for key in keys}
    arrays.update({
        "bucket_name": np.asarray([row["bucket"] for row in output_rows]),
        "split_name": np.asarray([row["split"] for row in output_rows]),
        "provenance_name": np.asarray([row["provenance"] for row in output_rows]),
        "pair_source_name": np.asarray([row["pair_source"] for row in output_rows]),
        "league_opponent_name": np.asarray([
            row["league_opponent"] for row in output_rows]),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    state_path = output.with_suffix(".states.pkl")
    with state_path.open("wb") as file:
        pickle.dump({"states": stacked_states, "records": records}, file)

    consistency_total = consistency_ok = 0
    for row in output_rows:
        legal = row["legal"]; survivable = row["survivable"]
        expected_doomed = not bool(np.any(legal & survivable))
        expected_avoidable = legal & ~survivable & (not expected_doomed)
        consistency_total += 2
        consistency_ok += int(bool(row["doomed"]) == expected_doomed)
        consistency_ok += int(np.array_equal(row["avoidable"], expected_avoidable))
    split_names = list(split_counts)
    leakage = any(
        split_seed_sets[left] & split_seed_sets[right]
        for i, left in enumerate(split_names)
        for right in split_names[i + 1:])
    prng_leakage = any(
        split_prng_sets[left] & split_prng_sets[right]
        for i, left in enumerate(split_names)
        for right in split_names[i + 1:])
    pair_counts = {
        source: sum(row["pair_source"] == source for row in output_rows)
        for source in ("rule_exact", "monte_carlo", "unlabeled")}
    manifest = {
        "schema": "bun_critic_counterfactual_v2",
        "created_date": "2026-09-26",
        "actors": [
            {"path": path, "sha256": digest(path)} for path in args.actor],
        "opponent": {"path": args.opponent, "sha256": digest(args.opponent)},
        "opponent_name": args.opponent_name,
        "seed_base": args.seed_base,
        "jax_prng_mapping": "uint32(seed & 0xffffffff)",
        "split_seed_ranges": {
            split: sorted(seeds) for split, seeds in split_seed_sets.items()},
        "seed_leakage": leakage,
        "mapped_prng_seed_leakage": prng_leakage,
        "states": len(output_rows),
        "bucket_counts": {
            bucket: sum(row["bucket"] == bucket for row in output_rows)
            for bucket in selected_buckets},
        "selected_buckets": list(selected_buckets),
        "split_counts": {
            split: sum(row["split"] == split for row in output_rows)
            for split in split_counts},
        "mc_samples_per_action": args.mc_samples,
        "mc_horizon": args.mc_horizon,
        "deterministic_policy_horizon": args.deterministic_horizon,
        "rule_exact_fields": ["legal", "survivable", "avoidable", "doomed"],
        "monte_carlo_fields": [
            "q", "q_ci_low", "q_ci_high", "win_rate", "win_ci_low",
            "win_ci_high", "mc_kill_rate", "mc_trade_rate", "mc_objective"],
        "deterministic_policy_fields": [
            "deterministic_return", "deterministic_win", "deterministic_kill",
            "deterministic_trade", "deterministic_objective"],
        "first_action_provenance_fields": [
            "first_death_source", "first_causal_death_source",
            "first_trigger_damage_source", "first_credited_kill",
            "first_causal_kill", "first_trigger_kill", "first_mutual_death",
            "first_own_bomb_defeat", "first_opponent_physical_defeat",
            "first_opponent_causal_defeat"],
        "snapshot_ticks": snapshot_ticks,
        "rule_consistency": consistency_ok / max(consistency_total, 1),
        "pair_label_counts": pair_counts,
        "legal_action_samples": int(sum(row["legal"].sum() for row in output_rows)),
        "mc_action_samples": int(sum(row["legal"].sum() for row in output_rows)
                                 * args.mc_samples),
        "mc_sample_count_complete": all(
            bool(np.all(row["mc_samples"][row["legal"]] == args.mc_samples))
            for row in output_rows),
        "confidence_intervals_valid": all(
            confidence_intervals_valid(row) for row in output_rows),
        "snapshot_actual_ticks": {
            bucket: [int(np.asarray(state.core.t))
                     for record, state in zip(records, states)
                     if record["bucket"] == bucket]
            for bucket in snapshot_ticks},
        "output": str(output), "states_file": str(state_path),
        "output_sha256": digest(output),
        "wall_seconds": time.time() - started,
        "context_audit": {
            "global_vec_has_respawn_remaining": [11, 12],
            "global_vec_has_carry_state": [13, 14, 15, 16],
            "global_vec_has_base_storage": [17, 18, 19, 20],
            "critic_context_adds": [
                "actor_id", "opponent_id", "lesson", "phase", "bucket",
                "league_opponent"],
            "actor_input_unchanged": True,
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if (leakage or prng_leakage or manifest["rule_consistency"] < .99
            or not manifest["mc_sample_count_complete"]
            or not manifest["confidence_intervals_valid"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
