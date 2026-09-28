#!/usr/bin/env python3
"""Generate counterfactual Critic replay against one frozen tactical-v2 bot."""
from __future__ import annotations

import argparse
import hashlib
import json
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

from jax_bomb import bun_env as env
from jax_bomb import bun_safety, jax_train
from jax_bomb.bun_frozen_opponents import FrozenTacticalOpponent, clear_destructible_bricks, tactical_bot_provenance
from jax_bomb.jax_net import transformer_forward
from jax_bomb.bun_tactical_labels import derive_action_labels
from scripts.generate_bun_critic_data import BUCKETS, context_vector, mean_interval, wilson

ACTION_COUNT = env.N_MOVES * env.N_BOMB
ALL_ACTIONS = np.asarray([[m, a] for m in range(env.N_MOVES) for a in range(env.N_BOMB)], np.int32)
BUCKET = "ambush_contact"
BUCKET_ID = BUCKETS.index(BUCKET)


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_params(path: str):
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    value = value.get("params", value) if isinstance(value, dict) else value
    return jax.tree.map(jnp.asarray, value)


def repeat_state(state, count: int):
    return jax.tree.map(lambda value: jnp.repeat(value[None], count, axis=0), state)


def select_state(old, new, active):
    return jax.tree.map(lambda a, b: jnp.where(
        active.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), old, new)


def make_kernels(actor):
    @jax.jit
    def observe(states):
        obs = jax_train.both_perspectives(states)
        glob = jax_train.both_states(states)
        masks = jax_train.both_masks(states)
        n = states.core.pos.shape[0]
        return obs[:n], glob[:n], masks[0][:n], masks[1][:n]

    @jax.jit
    def policy(states, key, greedy):
        obs, glob, mm, am = observe(states)
        move, ability, _, _ = transformer_forward(actor, obs, glob)
        joint = bun_safety.adjusted_joint_logits(
            move, ability, mm, am,
            jnp.zeros((len(obs), env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
        sampled = jax.random.categorical(key, joint)
        chosen = jnp.where(greedy, jnp.argmax(joint, axis=-1), sampled)
        return jnp.stack([chosen // env.N_BOMB, chosen % env.N_BOMB], axis=-1), jax.nn.softmax(joint), obs, glob

    @jax.jit
    def step(states, p0_actions, p1_actions, key):
        count = states.core.pos.shape[0]
        keys = jax.random.split(key, count)
        actions = jnp.stack([p0_actions, p1_actions], axis=1)
        candidates, done, info = jax.vmap(lambda state, action, rng: env.step(
            state, action, rng, auto_reset=False, return_info=True))(states, actions, keys)
        candidates = clear_destructible_bricks(candidates)
        reward = env.reward_from_events(
            info["dmg"], states.core.alive, info["alive"], info["hp"], done,
            info["crate"], jnp.zeros((count, 2), jnp.bool_), info["walls"],
            0.0, 0.0, 0.0, 1.0, moves=actions[:, :, 0], bombs=actions[:, :, 1],
            rule_info=info)[:, 0]
        return candidates, done, reward, info
    return observe, policy, step


def collect_state(seed: int, ticks: int, actor, bot, kernels):
    observe, policy, step = kernels
    state = clear_destructible_bricks(env._fresh(
        jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))))
    state = state._replace(lesson=jnp.asarray(env.LESSON_DANGER_ARENA, jnp.int8))
    current = repeat_state(state, 1)
    key = jax.random.PRNGKey(np.uint32((seed ^ 0x13579) & 0xFFFFFFFF))
    for _ in range(ticks):
        key, actor_key, step_key = jax.random.split(key, 3)
        action0, _, _, _ = policy(current, actor_key, jnp.asarray(False))
        action1 = jnp.asarray(bot.decide_batch(current), jnp.int32)
        candidate, done, _, _ = step(current, action0, action1, step_key)
        current = select_state(current, candidate, ~done)
        if bool(np.asarray(done)[0]):
            break
    return jax.tree.map(lambda value: value[0], current)


def rollout(first_states, actor, bot, kernels, seed: int, horizon: int, greedy: bool):
    _, policy, step = kernels
    states = first_states
    count = int(states.core.pos.shape[0])
    active = np.ones(count, np.bool_)
    returns = np.zeros(count, np.float32)
    wins = np.zeros(count, np.bool_)
    kills = np.zeros(count, np.bool_)
    trades = np.zeros(count, np.bool_)
    objective = np.zeros(count, np.float32)
    causal_onset = np.full(count, -1, np.int16)
    discount = np.ones(count, np.float32)
    key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
    for tick in range(horizon):
        key, actor_key, step_key = jax.random.split(key, 3)
        action0, _, _, _ = policy(states, actor_key, jnp.asarray(greedy))
        action1 = jnp.asarray(bot.decide_batch(states), jnp.int32)
        candidate, done, reward, info = step(states, action0, action1, step_key)
        host_done = np.asarray(done)
        host_reward = np.asarray(reward)
        returns += discount * np.where(active, host_reward, 0.0)
        discount *= 0.995
        wins |= active & host_done & (np.asarray(info["winner"]) == 0)
        kills |= active & np.asarray(info["surviving_kill"])[:, 0]
        causal_now = active & np.asarray(info["causal_kill"])[:, 0]
        causal_onset[(causal_onset < 0) & causal_now] = tick + 1
        trades |= active & np.asarray(info["mutual_death"])
        objective += np.where(active, np.asarray(info["objective_progress"])[:, 0], 0.0)
        states = select_state(states, candidate, jnp.asarray(active & ~host_done))
        active &= ~host_done
        if not active.any():
            break
    alive = np.asarray(states.core.alive)
    return returns, wins, kills, trades, objective, alive[:, 0], alive[:, 1], causal_onset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--train-states", type=int, default=3)
    parser.add_argument("--validation-states", type=int, default=1)
    parser.add_argument("--test-states", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=2)
    parser.add_argument("--mc-horizon", type=int, default=24)
    parser.add_argument("--split-stride", type=int, default=100000)
    args = parser.parse_args()
    started = time.time()
    env.prepare()
    env.configure_training("danger_arena=1", 1, reward_profile="danger_arena")
    actor = load_params(args.actor)
    bot = FrozenTacticalOpponent(1)
    provenance = tactical_bot_provenance()
    kernels = make_kernels(actor)
    analyze = jax.jit(bun_safety.analyze_actions)
    rows = []
    split_counts = {"train": args.train_states, "validation": args.validation_states, "test": args.test_states}
    split_seeds = {key: [] for key in split_counts}
    for split_index, (split, count) in enumerate(split_counts.items()):
        for item in range(count):
            seed = args.seed_base + split_index * args.split_stride + item
            split_seeds[split].append(seed)
            state = collect_state(seed, (8, 24, 48, 72)[item % 4], actor, bot, kernels)
            analysis = analyze(state)
            legal = np.asarray(analysis.legal[0]).reshape(-1)
            survivable = np.asarray(analysis.survivable[0]).reshape(-1)
            avoidable = np.asarray(analysis.avoidable[0]).reshape(-1)
            doomed = bool(np.asarray(analysis.doomed[0]))
            states = repeat_state(state, ACTION_COUNT)
            p1 = np.asarray(bot.decide_batch(states), np.int32)
            _, policy_probability, obs_batch, global_batch = kernels[1](states, jax.random.PRNGKey(np.uint32((seed ^ 1) & 0xFFFFFFFF)), jnp.asarray(True))
            p0 = jnp.asarray(ALL_ACTIONS)
            candidate, immediate_done, immediate_reward, first = kernels[2](
                states, p0, jnp.asarray(p1), jax.random.PRNGKey(np.uint32((seed ^ 2) & 0xFFFFFFFF)))
            repeated = jax.tree.map(lambda value: jnp.repeat(value, args.mc_samples, axis=0), candidate)
            det = rollout(candidate, actor, bot, kernels, seed ^ 3, args.mc_horizon, True)
            mc = rollout(repeated, actor, bot, kernels, seed ^ 4, args.mc_horizon, False)
            det_return, det_win, det_kill, det_trade, det_obj, det_self_alive, det_enemy_alive, det_causal_onset = det
            mc_return, mc_win, mc_kill, mc_trade, mc_obj, mc_self_alive, mc_enemy_alive, mc_causal_onset = [np.asarray(x).reshape(ACTION_COUNT, args.mc_samples) for x in mc]
            q = np.clip(mc_return.mean(1), -20, 20).astype(np.float32)
            win_rate = mc_win.mean(1).astype(np.float32)
            kill_rate = mc_kill.mean(1).astype(np.float32)
            trade_rate = mc_trade.mean(1).astype(np.float32)
            objective = mc_obj.mean(1).astype(np.float32)
            q_ci = np.asarray([mean_interval(x) for x in mc_return], np.float32)
            win_ci = np.asarray([wilson(int(x.sum()), args.mc_samples) for x in mc_win], np.float32)
            probability = np.asarray(policy_probability[0]).reshape(-1) * legal
            probability /= max(float(probability.sum()), 1e-8)
            good = bad = -1; pair_source = "unlabeled"
            legal_ids = np.flatnonzero(legal)
            if len(legal_ids) >= 2:
                best = int(legal_ids[np.argmax(q[legal_ids])]); worst = int(legal_ids[np.argmin(q[legal_ids])])
                if q[best] - q[worst] >= 1.0 and (survivable[best] or avoidable[worst]):
                    good, bad, pair_source = best, worst, "rule_or_mc"
            next_obs, next_global, _, _ = kernels[0](candidate)
            candidate_safety = jax.vmap(bun_safety.analyze_actions)(candidate)
            enemy_escape_count = np.asarray(candidate_safety.survivable)[:, 1].reshape(ACTION_COUNT, -1).sum(axis=1).astype(np.int16)
            self_escape_count = np.asarray(candidate_safety.survivable)[:, 0].reshape(ACTION_COUNT, -1).sum(axis=1).astype(np.int16)
            selected_safety = jax.vmap(lambda action: bun_safety.analyze_selected_actions(state, action))(
                jnp.stack([jnp.asarray(ALL_ACTIONS), jnp.asarray(p1)], axis=1))
            minimum_escape_time = np.asarray(selected_safety.resolution_ticks)[:, 0].astype(np.float32)
            first_causal_actor = np.asarray(first["causal_kill"])[:, 0]
            causal_kill_onset = np.where(first_causal_actor, 0, np.asarray(det_causal_onset)).astype(np.int16)
            initial_forced = bool(np.asarray(analysis.doomed[1])) and not bool(np.asarray(analysis.doomed[0]))
            abilities = ALL_ACTIONS[:, 1]
            actor_columns = np.zeros((ACTION_COUNT,), np.int32)
            own_bomb_first = np.asarray(first["own_bomb_defeat"])[np.arange(ACTION_COUNT), actor_columns]
            action_labels=derive_action_labels(legal=legal,survivable=survivable,abilities=abilities,self_alive=det_self_alive,enemy_alive=det_enemy_alive,nontrade_kill=det_kill,trade=det_trade,own_bomb_first=own_bomb_first,enemy_escape_count=enemy_escape_count,minimum_escape_time=minimum_escape_time,causal_kill_onset=causal_kill_onset,initial_forced=initial_forced)
            context = context_vector(state, 0, 0, BUCKET_ID, 0)
            next_context = np.stack([context_vector(jax.tree.map(lambda value, i=i: value[i], candidate), 0, 0, BUCKET_ID, 0) for i in range(ACTION_COUNT)])
            rows.append({
                "actor_id":np.int64(0),"bucket_id":np.int64(BUCKET_ID),"seed":np.int64(seed),"declared_seed":np.int64(seed),"opponent_id":np.int64(0),"league_opponent_id":np.int64(0),
                "obs":np.clip(np.rint(np.asarray(obs_batch[0])*255),0,255).astype(np.uint8),"global":np.asarray(global_batch[0],np.float32),"context":context,
                "legal":legal,"survivable":survivable,"avoidable":avoidable,"doomed":np.bool_(doomed),
                "deterministic_return":np.asarray(det_return,np.float32),"deterministic_win":np.asarray(det_win),"deterministic_kill":np.asarray(det_kill),"deterministic_trade":np.asarray(det_trade),"deterministic_objective":np.asarray(det_obj,np.float32),
                "q":q,"win_rate":win_rate,"win_ci_low":win_ci[:,0],"win_ci_high":win_ci[:,1],"q_ci_low":q_ci[:,0],"q_ci_high":q_ci[:,1],"mc_kill_rate":kill_rate,"mc_trade_rate":trade_rate,"mc_objective":objective,"mc_samples":np.full(ACTION_COUNT,args.mc_samples,np.int16),
                "policy_probability":probability.astype(np.float32),"value_target":np.float32((probability*q).sum()),"win_target":np.float32((probability*win_rate).sum()),"good_action":np.int16(good),"bad_action":np.int16(bad),
                "next_obs":np.clip(np.rint(np.asarray(next_obs)*255),0,255).astype(np.uint8),"next_global":np.asarray(next_global,np.float32),"next_context":next_context.astype(np.float32),
                "immediate_reward":np.asarray(immediate_reward,np.float32),"immediate_done":np.asarray(immediate_done),
                **action_labels,
                "first_death_source":np.asarray(first["death_source"]),"first_causal_death_source":np.asarray(first["causal_death_source"]),"first_trigger_damage_source":np.asarray(first["trigger_damage_source"]),"first_credited_kill":np.asarray(first["credited_kill"]),"first_causal_kill":np.asarray(first["causal_kill"]),"first_trigger_kill":np.asarray(first["trigger_kill"]),"first_mutual_death":np.asarray(first["mutual_death"]),"first_own_bomb_defeat":np.asarray(first["own_bomb_defeat"]),"first_opponent_physical_defeat":np.asarray(first["opponent_physical_defeat"]),"first_opponent_causal_defeat":np.asarray(first["opponent_causal_defeat"]),
                "bucket_name":BUCKET,"split_name":split,"provenance_name":"tactical_v2_commit_mc","pair_source_name":pair_source,"league_opponent_name":"tactical_v2",
            })
            print(json.dumps({"split":split,"seed":seed,"legal":int(legal.sum()),"pair":pair_source}),flush=True)
    arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,**arrays)
    overlap=bool(set(split_seeds['train'])&set(split_seeds['validation']) or set(split_seeds['train'])&set(split_seeds['test']) or set(split_seeds['validation'])&set(split_seeds['test']))
    identity=provenance['opponent_identity_hash']
    manifest={"schema":"bun_tactical_v2_counterfactual_v3","actor":{"path":args.actor,"sha256":digest(args.actor)},"opponent":provenance,"rollout_opponent_hash":identity,"critic_data_opponent_hash":identity,"eval_opponent_hash":identity,"hash_alignment":len({identity})==1,"first_action_committed":True,"subsequent_replanning":True,"labels":["safe_attack_available","self_survive_4s","enemy_survive_4s","safe_escape_exists","enemy_escape_count","forced_kill_state","bomb_creates_forced_kill","bomb_creates_trade","own_bomb_death_risk","causal_kill_onset","forced_kill_onset","minimum_escape_time"],"labels_are_actor_observation":False,"states":len(rows),"seed_base":args.seed_base,"split_stride":args.split_stride,"split_seeds":split_seeds,"seed_leakage":overlap,"mc_samples":args.mc_samples,"mc_horizon":args.mc_horizon,"output":str(output),"output_sha256":digest(output),"wall_seconds":time.time()-started}
    Path(args.manifest).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    if overlap or not manifest['hash_alignment'] or provenance['module_sha256'] != provenance['expected_sha256']:
        raise SystemExit(2)

if __name__ == '__main__': main()
