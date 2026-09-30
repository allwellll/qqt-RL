#!/usr/bin/env python3
"""One bounded Bun PPO run with a frozen-distribution independent critic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_bomb import bun_env as env
from jax_bomb import bun_supervision
from jax_bomb import jax_net
from jax_bomb import jax_train
from jax_bomb.bun_frozen_opponents import (
    FrozenTacticalOpponent,
    clear_destructible_bricks,
    tactical_bot_provenance,
)
from jax_bomb.bun_critic import (
    AUX_TARGET_NAMES,
    independent_critic_aux_forward,
    independent_critic_forward,
    weighted_hl_gauss_loss,
)
from jax_bomb.jax_net import transformer_aux_forward, transformer_forward
from scripts.adapt_bun_critic_onpolicy import diagnostics, migrate_critic
from scripts.train_bun_bc import _expand, _load

OPPONENT_INDEX = {name: index for index, name in enumerate(jax_train.BUN_OPPONENT_NAMES)}
OPPONENT_CHOICES = tuple(OPPONENT_INDEX) + ("tactical",)
BUCKET_BY_LESSON = {
    env.LESSON_FULL: 12,
    env.LESSON_FULL_AMBUSH: 8,
    env.LESSON_COMBAT_KILL: 4,
    env.LESSON_KILL_RUSH: 9,
    env.LESSON_NEAR_STEAL: 10,
    env.LESSON_CARRY_RETURN: 11,
    env.LESSON_DANGER_ARENA: 8,
}


def ensure_actor_aux_heads(actor, key):
    """Graft safe-action/escape/margin heads onto a policy backbone in place.

    The audit found the danger/escape supervision only ever reached the critic
    or a sidecar NPZ, never the actor's shared transformer backbone. Adding these
    heads (reading the same pooled feature as the policy head) is what lets the
    BCE gradient train the backbone. Grafting starts a NEW actor lineage: the
    obs/backbone are unchanged, only three consequence-prediction heads are added.
    """
    heads = actor.get("heads")
    if not isinstance(heads, dict) or "wsafe" in heads:
        return actor
    embed = heads["wm"][0].shape[0]
    ks, ke, kg = jax.random.split(key, 3)
    grafted = dict(heads)
    grafted["wsafe"] = jax_net._linear_init(
        ks, embed, env.N_MOVES * env.N_BOMB, scale=0.01)
    grafted["wesc"] = jax_net._linear_init(ke, embed, 1, scale=0.01)
    grafted["wmargin"] = jax_net._linear_init(kg, embed, 1, scale=0.01)
    return {**actor, "heads": grafted}


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def load_checkpoint(path):
    with open(path, "rb") as file:
        value = pickle.load(file)
    if isinstance(value, dict) and "params" in value:
        return value["params"], value.get("opt_state")
    return value, None


def save_checkpoint(path, params, opt_state, metadata):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # Actor、Critic 与 Target Critic 分别持久化；Target Critic 只通过 Polyak
    # 更新跟随在线 Critic，不能与 Actor 参数树混用。先写临时文件再原子发布，
    # 让 crash recovery 永远回到上一份完整 checkpoint。
    with temporary.open("wb") as file:
        pickle.dump({"params": jax.device_get(params),
                     "opt_state": jax.device_get(opt_state),
                     "metadata": metadata}, file)
    os.replace(temporary, destination)


def critic_context(global_state, actor_id, opponent_name, lessons):
    count = global_state.shape[0]
    context = np.zeros((count, 44), np.float32)
    context[:, actor_id] = 1.0
    coarse_rule = opponent_name in ("rule_combat", "tactical")
    context[:, 2 + (0 if coarse_rule else 1)] = 1.0
    lessons = np.asarray(lessons, np.int32)
    legacy_lesson_count = env.LESSON_DANGER_ARENA
    encoded_lessons = np.where(
        lessons == env.LESSON_DANGER_ARENA, env.LESSON_FULL_AMBUSH, lessons)
    context[np.arange(count), 4 + encoded_lessons] = 1.0
    phase_start = 4 + legacy_lesson_count
    phase = np.where(global_state[:, 13] >= 0.5, 2,
                     np.where(global_state[:, 12] > 0, 1, 0))
    context[np.arange(count), phase_start + phase] = 1.0
    bucket_start = phase_start + 3
    buckets = np.asarray([BUCKET_BY_LESSON.get(int(x), 12) for x in lessons])
    context[np.arange(count), bucket_start + buckets] = 1.0
    league_start = 37
    league_names = ("rule_combat", "weak", "old", "recent")
    encoded_opponent = "rule_combat" if opponent_name == "tactical" else opponent_name
    context[:, league_start + league_names.index(encoded_opponent)] = 1.0
    return context


def collect_tactical_rollout(current_actor, states, key, num_steps):
    """Collect one PPO rollout against the frozen Python tactical bot.

    ``obs`` 的主 shape 为 ``[time, env, channel, height, width]``；
    ``global_state`` 为 Critic 额外使用的集中式状态。学习者策略和环境 kernel
    保留在 GPU，P1 决策每个 batch tick 只传回 host 一次。The learner policy and environment kernels remain on GPU.  P1 decisions are
    transferred once per batched tick to the host Python bot; the whole batch is
    decided together.  No bot action or search result enters learner gradients.
    """
    opponent = FrozenTacticalOpponent(player_id=1)
    states = clear_destructible_bricks(states)
    num_envs = int(states.core.pos.shape[0])
    observations = []
    global_states = []
    actions_out = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    move_masks = []
    ability_masks = []
    unsafe_masks = []
    audit_rows = []
    safe_labels = []
    escape_labels = []
    margin_labels = []

    @jax.jit
    def policy_step(params, current_states, sample_key):
        obs = jax_train.both_perspectives(current_states)
        masks = jax_train.both_masks(current_states)
        global_state = jax_train.both_states(current_states)
        learner_actions, learner_logp, learner_value = jax_train.sample_actions(
            params, "transformer", obs[:num_envs],
            (masks[0][:num_envs], masks[1][:num_envs]), sample_key,
            state=global_state[:num_envs])
        return obs[:num_envs], global_state[:num_envs], masks, learner_actions, learner_logp, learner_value

    @jax.jit
    def supervision_step(current_states):
        """Learner (player 0) deterministic safety labels for the current board.

        Pure function of the public state via the safety oracle; no opponent
        policy or future rollout is consulted, so it is an admissible gradient
        target for the actor's shared backbone.
        """
        labels = jax.vmap(bun_supervision.actor_supervision_labels)(current_states)
        return (labels["safe_action"][:, 0], labels["escape"][:, 0],
                labels["margin"][:, 0])

    @jax.jit
    def environment_step(current_states, env_actions, step_key):
        step_keys = jax.random.split(step_key, num_envs)
        candidate, done, info = jax.vmap(
            lambda state, action, rng: env.step(
                state, action, rng, return_info=True)
        )(current_states, env_actions, step_keys)
        zeros = jnp.zeros((num_envs, 2), jnp.bool_)
        reward = jax_train.reward_from_events(
            info["dmg"], current_states.core.alive, info["alive"], info["hp"], done,
            info["crate"], zeros, info["walls"], 0.0, 0.0, 0.0, 1.0,
            moves=env_actions[:, :, 0], bombs=env_actions[:, :, 1],
            idle_penalty=0.015, rule_info=info)
        audit = jnp.stack([
            info["bomb_placed"][:, 0].astype(jnp.float32),
            info["safe_tactical_bomb_placed"][:, 0].astype(jnp.float32),
            info["tactical_bomb_safe_resolution"][:, 0].astype(jnp.float32),
            info["tactical_bomb_placement_reward"][:, 0],
            info["tactical_bomb_resolution_reward"][:, 0],
            info["own_bomb_defeat"][:, 0].astype(jnp.float32),
            info["mutual_death"].astype(jnp.float32),
        ], axis=-1)
        return clear_destructible_bricks(candidate), done, reward, audit

    for _ in range(num_steps):
        key, policy_key, step_key = jax.random.split(key, 3)
        obs, global_state, masks, learner_action, learner_logp, learner_value = (
            policy_step(current_actor, states, policy_key))
        safe_label, escape_label, margin_label = supervision_step(states)
        tactical_action = jnp.asarray(opponent.decide_batch(states), jnp.int32)
        env_actions = jnp.stack([learner_action, tactical_action], axis=1)
        states, done, reward, audit = environment_step(
            states, env_actions, step_key)
        observations.append(obs)
        global_states.append(global_state)
        actions_out.append(learner_action)
        log_probs.append(learner_logp)
        values.append(learner_value)
        rewards.append(reward[:, 0])
        dones.append(done)
        move_masks.append(masks[0][:num_envs])
        ability_masks.append(masks[1][:num_envs])
        unsafe_masks.append(jnp.zeros(
            (num_envs, env.N_MOVES, env.N_BOMB), jnp.bool_))
        audit_rows.append(audit)
        safe_labels.append(safe_label)
        escape_labels.append(escape_label)
        margin_labels.append(margin_label)

    batch = (
        jnp.stack(observations), jnp.stack(global_states),
        jnp.stack(actions_out), jnp.stack(log_probs), jnp.stack(values),
        jnp.stack(rewards), jnp.stack(dones),
        (jnp.stack(move_masks), jnp.stack(ability_masks),
         jnp.stack(unsafe_masks)),
    )
    supervision = (jnp.stack(safe_labels), jnp.stack(escape_labels),
                   jnp.stack(margin_labels))
    audit = jnp.stack(audit_rows).sum(axis=(0, 1))
    return states, batch, audit, jnp.zeros((num_envs,), jnp.float32), supervision


def make_inputs(obs, global_state, lessons, opponent_name):
    time_steps, num_envs = obs.shape[:2]
    repeated_lessons = np.broadcast_to(
        np.asarray(lessons)[None, :], (time_steps, num_envs)).reshape(-1)
    flat_global = global_state.reshape(-1, global_state.shape[-1])
    context = critic_context(flat_global, 0, opponent_name, repeated_lessons)
    return np.concatenate([
        obs.reshape(time_steps * num_envs, -1), flat_global, context], axis=1
    ).astype(np.float32).reshape(time_steps, num_envs, -1)


def _counterfactual_aux(data):
    actor_ids = data["actor_id"].astype(np.int32)
    action_count = data["legal"].shape[1]
    rows = np.broadcast_to(np.arange(len(actor_ids))[:, None], (len(actor_ids), action_count))
    actions = np.broadcast_to(np.arange(action_count)[None, :], rows.shape)
    actor_columns = np.broadcast_to(actor_ids[:, None], rows.shape)
    credited = data["first_credited_kill"][rows, actions, actor_columns]
    causal = data["first_causal_kill"][rows, actions, actor_columns]
    own_bomb = data["first_own_bomb_defeat"][rows, actions, actor_columns]
    kill = (credited | causal) & ~data["first_mutual_death"]
    survival = data["survivable"]
    safe_escape = data["safe_escape_exists"] if "safe_escape_exists" in data else survival
    forced = data["forced_kill_state"] if "forced_kill_state" in data else kill
    bomb_forced = data["bomb_creates_forced_kill"] if "bomb_creates_forced_kill" in data else forced
    bomb_trade = data["bomb_creates_trade"] if "bomb_creates_trade" in data else data["first_mutual_death"]
    own_risk = data["own_bomb_death_risk"] if "own_bomb_death_risk" in data else own_bomb
    minimum = data["minimum_escape_time"] if "minimum_escape_time" in data else np.where(safe_escape, 1.0, 40.0)
    return np.stack([
        np.asarray(data.get("self_survive_4s", survival), np.bool_),
        kill, own_risk, safe_escape, forced, bomb_forced, bomb_trade,
        np.where(safe_escape, np.exp(-minimum / 40.0), 0.0),
    ], axis=-1).astype(np.float32)


def load_counterfactual(paths, context_dim=44):
    rows = []
    for path in paths:
        data = np.load(path)
        if len(data["obs"]) == 0:
            continue
        obs = data["obs"].astype(np.float32).reshape(len(data["obs"]), -1) / 255.0
        if data["context"].shape[1] > context_dim:
            raise ValueError(f"counterfactual context too wide: {path}")
        context = np.pad(data["context"].astype(np.float32),
                         ((0, 0), (0, context_dim - data["context"].shape[1])))
        rows.append({
            "inputs": np.concatenate([obs, data["global"], context], axis=1),
            "value": data["value_target"].astype(np.float32),
            "q": data["q"].astype(np.float32),
            "legal": data["legal"].astype(np.float32),
            "good": data["good_action"].astype(np.int32),
            "bad": data["bad_action"].astype(np.int32),
            "aux": _counterfactual_aux(data),
            "split": data["split_name"],
            "opponent": data["league_opponent_name"],
        })
    if not rows:
        raise ValueError("counterfactual replay contains no states")
    return {key: np.concatenate([row[key] for row in rows], axis=0)
            for key in rows[0]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--target-critic")
    parser.add_argument("--reference-actor", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--recent", required=True)
    parser.add_argument("--counterfactual", nargs="+", required=True)
    parser.add_argument("--bc-data", nargs="+", required=True)
    parser.add_argument("--save-actor", required=True)
    parser.add_argument("--save-critic", required=True)
    parser.add_argument("--save-target-critic")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--opponent", choices=OPPONENT_CHOICES, required=True)
    parser.add_argument("--opponent-schedule", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--entropy", type=float, default=0.003)
    parser.add_argument("--kl", type=float, default=0.02)
    parser.add_argument("--bc-coef", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--target-tau", type=float, default=0.05)
    parser.add_argument("--counterfactual-coef", type=float, default=0.30)
    parser.add_argument("--counterfactual-q-coef", type=float, default=0.40)
    parser.add_argument("--counterfactual-pair-coef", type=float, default=0.15)
    parser.add_argument("--counterfactual-aux-coef", type=float, default=0.20)
    parser.add_argument("--actor-aux-coef", type=float, default=0.0,
                        help="Weight for the actor's danger/escape/margin/"
                             "safe-action BCE supervision on the shared "
                             "backbone. >0 grafts aux heads and starts a new "
                             "actor lineage; 0 keeps the legacy actor untouched.")
    parser.add_argument("--reward-profile", default="combat_evolution",
                        choices=("legacy", "auto_sparse", "combat_evolution",
                                 "danger_arena"))
    parser.add_argument("--danger-escape-reward", type=float, default=0.75)
    parser.add_argument("--avoidable-danger-death-penalty", type=float,
                        default=4.0)
    parser.add_argument("--tactical-bomb-placement-reward", type=float,
                        default=0.0)
    parser.add_argument("--tactical-bomb-resolution-reward", type=float,
                        default=0.0)
    parser.add_argument("--carry-rollout-state", action="store_true")
    parser.add_argument("--jax-cache-dir")
    args = parser.parse_args(argv)

    if args.jax_cache_dir:
        from qqt_rl.training.jax_cache import configure_persistent_cache
        configure_persistent_cache(explicit=args.jax_cache_dir)

    started = time.time()
    env.prepare()
    env.configure_training(
        args.curriculum, 1, reward_profile=args.reward_profile,
        danger_escape_reward=args.danger_escape_reward,
        avoidable_danger_death_penalty=args.avoidable_danger_death_penalty,
        tactical_bomb_placement_reward=args.tactical_bomb_placement_reward,
        tactical_bomb_resolution_reward=args.tactical_bomb_resolution_reward)
    actor, actor_opt_state = load_checkpoint(args.actor)
    reference, _ = load_checkpoint(args.reference_actor)
    critic, critic_opt_state = load_checkpoint(args.critic)
    critic_had_aux = isinstance(critic, dict) and "aux" in critic
    actor = jax.tree.map(jnp.asarray, actor)
    reference = jax.tree.map(jnp.asarray, reference)
    actor_grafted_aux = False
    if args.actor_aux_coef > 0.0:
        before = "wsafe" in actor.get("heads", {})
        actor = ensure_actor_aux_heads(
            actor, jax.random.PRNGKey(np.uint32((args.seed ^ 0xA11) & 0xFFFFFFFF)))
        actor_grafted_aux = not before and "wsafe" in actor["heads"]
    has_actor_aux = isinstance(actor.get("heads"), dict) and "wsafe" in actor["heads"]
    weak = jax.tree.map(jnp.asarray, load_checkpoint(args.weak)[0])
    old = jax.tree.map(jnp.asarray, load_checkpoint(args.old)[0])
    recent = jax.tree.map(jnp.asarray, load_checkpoint(args.recent)[0])
    bc = _load(_expand(args.bc_data))
    cf = load_counterfactual(args.counterfactual)
    critic = migrate_critic(critic, bc["obs"].shape[1] * env.H * env.W + 24 + 44)
    if args.target_critic:
        target_critic = migrate_critic(
            load_checkpoint(args.target_critic)[0],
            bc["obs"].shape[1] * env.H * env.W + 24 + 44)
    else:
        target_critic = jax.tree.map(lambda value: jnp.array(value), critic)
    actor_optimizer = optax.adam(args.actor_lr)
    critic_optimizer = optax.adam(args.critic_lr)
    if actor_opt_state is None or actor_grafted_aux:
        # grafting new heads changes the param tree → optimizer moments must be
        # re-initialised so apply_updates sees a matching structure.
        actor_opt_state = actor_optimizer.init(actor)
    if critic_opt_state is None or not critic_had_aux:
        critic_opt_state = critic_optimizer.init(critic)
    rng = np.random.default_rng(args.seed)

    def tactical_collect(current_actor, states, key):
        return collect_tactical_rollout(
            current_actor, states, key, args.num_steps)

    jax_collectors = {}
    for opponent_name in OPPONENT_INDEX:
        weights = np.zeros((len(jax_train.BUN_OPPONENT_NAMES),), np.float32)
        weights[OPPONENT_INDEX[opponent_name]] = 1.0

        def build_collect(frozen_weights):
            @jax.jit
            def collect(current_actor, states, key):
                states = clear_destructible_bricks(states)
                return jax_train.collect_rollout(
                    current_actor, "transformer", states, key, args.num_steps,
                    flee_bot_ratio=0.0, safety_mode="off",
                    bun_opponent_weights=jnp.asarray(frozen_weights),
                    bun_opponent_weak_params=weak,
                    bun_opponent_old_params=old,
                    bun_opponent_recent_params=recent)

            def collect_with_supervision(current_actor, states, key):
                # jax-opponent rollouts don't retain per-transition boards, so no
                # deterministic safety labels are produced; aux supervision is a
                # tactical-danger_arena-only signal (supervision=None here).
                final_states, batch, nov, kills = collect(
                    current_actor, states, key)
                return final_states, batch, nov, kills, None
            return collect_with_supervision
        jax_collectors[opponent_name] = build_collect(weights)

    def collect_for(opponent_name, current_actor, states, key):
        return (tactical_collect(current_actor, states, key)
                if opponent_name == "tactical"
                else jax_collectors[opponent_name](current_actor, states, key))

    opponent_schedule = [
        item.strip() for item in args.opponent_schedule.split(",") if item.strip()]
    if not opponent_schedule:
        opponent_schedule = [args.opponent]
    unknown = sorted(set(opponent_schedule) - set(OPPONENT_CHOICES))
    if unknown:
        raise ValueError(f"unknown opponent schedule entries: {unknown}")

    @jax.jit
    def actor_update(params, opt_state, obs, global_state, actions, old_logp,
                     advantages, move_mask, ability_mask, bc_idx,
                     safe_label, escape_label, margin_label, aux_scale):
        def loss_fn(current):
            move_logits, ability_logits, _, _, aux = transformer_aux_forward(
                current, obs, global_state)
            joint = jax_train.adjusted_joint_logits(
                move_logits, ability_logits, move_mask, ability_mask,
                jnp.zeros((len(obs), env.N_MOVES, env.N_BOMB), jnp.bool_),
                "off", 0.0)
            log_probs = jax.nn.log_softmax(joint)
            action_index = actions[:, 0] * env.N_BOMB + actions[:, 1]
            selected = log_probs[jnp.arange(len(obs)), action_index]
            normalized_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = jnp.exp(selected - old_logp)
            policy_loss = jnp.maximum(
                -normalized_adv * ratio,
                -normalized_adv * jnp.clip(ratio, 0.8, 1.2)).mean()
            probabilities = jnp.exp(log_probs)
            entropy = -(probabilities * log_probs).sum(axis=-1).mean()
            ref_move, ref_ability, _, _ = transformer_forward(
                reference, obs, global_state)
            ref_joint = jax_train.adjusted_joint_logits(
                ref_move, ref_ability, move_mask, ability_mask,
                jnp.zeros((len(obs), env.N_MOVES, env.N_BOMB), jnp.bool_),
                "off", 0.0)
            ref_log_probs = jax.nn.log_softmax(ref_joint)
            kl = (probabilities * (log_probs - ref_log_probs)).sum(axis=-1).mean()
            bc_obs = jnp.asarray(bc["obs"])[bc_idx].astype(jnp.float32) / 255.0
            bc_state = jnp.asarray(bc["state"])[bc_idx].astype(jnp.float32) / 255.0
            bc_move, bc_ability, _, _ = transformer_forward(current, bc_obs, bc_state)
            bc_move = jnp.where(jnp.asarray(bc["move_mask"])[bc_idx], bc_move, -1e9)
            bc_ability = jnp.where(jnp.asarray(bc["ability_mask"])[bc_idx], bc_ability, -1e9)
            rows = jnp.arange(len(bc_idx))
            bc_loss = (-jax.nn.log_softmax(bc_move)[rows, jnp.asarray(bc["move_action"])[bc_idx]].mean()
                       -jax.nn.log_softmax(bc_ability)[rows, jnp.asarray(bc["ability_action"])[bc_idx]].mean())
            # 审计根因修复：安全/逃生/margin 监督经共享 backbone 回流 actor 梯度
            # （旧 pipeline 只训 critic/sidecar）。aux_scale 为 0 时（非 tactical
            # rollout 或未开启）梯度贡献恒为 0，不改变原策略行为。
            if "wsafe" in current["heads"]:
                safe_bce = optax.sigmoid_binary_cross_entropy(
                    aux["safe_action"], safe_label).mean()
                escape_bce = optax.sigmoid_binary_cross_entropy(
                    aux["escape"], escape_label).mean()
                margin_mse = ((jax.nn.sigmoid(aux["margin"])
                               - margin_label) ** 2).mean()
                aux_loss = safe_bce + escape_bce + margin_mse
            else:
                aux_loss = jnp.asarray(0.0, jnp.float32)
            total = (policy_loss - args.entropy * entropy + args.kl * kl
                     + args.bc_coef * bc_loss + aux_scale * aux_loss)
            return total, (policy_loss, entropy, kl, bc_loss, aux_loss)
        (loss, aux), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = actor_optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, (loss,) + aux

    cf_train = np.flatnonzero(cf["split"] == "train")

    @jax.jit
    def critic_update(params, opt_state, inputs, targets, cf_idx):
        def loss_fn(current):
            _, logits, _, _ = independent_critic_forward(current, inputs)
            on_loss = weighted_hl_gauss_loss(
                logits, targets, jnp.ones_like(targets))
            _, cf_logits, _, cf_q = independent_critic_forward(
                current, jnp.asarray(cf["inputs"])[cf_idx])
            cf_loss = weighted_hl_gauss_loss(
                cf_logits, jnp.asarray(cf["value"])[cf_idx],
                jnp.ones((len(cf_idx),), jnp.float32))
            cf_legal = jnp.asarray(cf["legal"])[cf_idx]
            q_error = optax.huber_loss(
                cf_q, jnp.asarray(cf["q"])[cf_idx], delta=2.0)
            q_loss = jnp.sum(q_error * cf_legal) / jnp.maximum(cf_legal.sum(), 1.0)
            good = jnp.asarray(cf["good"])[cf_idx]
            bad = jnp.asarray(cf["bad"])[cf_idx]
            pair_mask = (good >= 0) & (bad >= 0)
            rows = jnp.arange(len(cf_idx))
            pair_loss = jnp.sum(jax.nn.softplus(-(
                cf_q[rows, jnp.maximum(good, 0)]
                - cf_q[rows, jnp.maximum(bad, 0)])) * pair_mask
            ) / jnp.maximum(pair_mask.sum(), 1)
            aux_logits = independent_critic_aux_forward(
                current, jnp.asarray(cf["inputs"])[cf_idx])
            aux_error = optax.sigmoid_binary_cross_entropy(
                aux_logits, jnp.asarray(cf["aux"])[cf_idx])
            aux_loss = jnp.sum(aux_error * cf_legal[:, :, None]) / jnp.maximum(
                cf_legal.sum() * aux_error.shape[-1], 1.0)
            replay_loss = (cf_loss
                           + args.counterfactual_q_coef * q_loss
                           + args.counterfactual_pair_coef * pair_loss
                           + args.counterfactual_aux_coef * aux_loss)
            total = ((1.0 - args.counterfactual_coef) * on_loss
                     + args.counterfactual_coef * replay_loss)
            return total, (on_loss, cf_loss, q_loss, pair_loss, aux_loss)
        (loss, aux), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = critic_optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, (loss,) + aux

    history = []
    carry_key = jax.random.PRNGKey(np.uint32(args.seed & 0xFFFFFFFF))
    states = env.init_batch(carry_key, args.num_envs)
    for update_index in range(args.updates):
        current_opponent = opponent_schedule[update_index % len(opponent_schedule)]
        seed = args.seed + update_index * 1009
        key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
        if update_index > 0 and not args.carry_rollout_state:
            states = env.init_batch(key, args.num_envs)
        lessons = np.asarray(states.lesson)
        collect_started = time.time()
        final_states, batch, rollout_audit, _, supervision = collect_for(
            current_opponent, actor, states, key)
        jax.block_until_ready(batch[5])
        obs, global_state, actions, old_logp, _, reward, done, masks = batch
        obs = np.asarray(obs[:, :args.num_envs], np.float32)
        global_state = np.asarray(global_state[:, :args.num_envs], np.float32)
        actions = np.asarray(actions[:, :args.num_envs], np.int32)
        old_logp = np.asarray(old_logp[:, :args.num_envs], np.float32)
        reward = np.asarray(reward[:, :args.num_envs], np.float32)
        done = np.asarray(done[:, :args.num_envs], np.bool_)
        move_mask = np.asarray(masks[0][:, :args.num_envs])
        ability_mask = np.asarray(masks[1][:, :args.num_envs])
        inputs = make_inputs(obs, global_state, lessons, current_opponent)
        values, _, _, _ = independent_critic_forward(
            target_critic, jnp.asarray(inputs.reshape(-1, inputs.shape[-1])))
        values = np.asarray(values).reshape(reward.shape)
        final_obs = np.asarray(jax_train.both_perspectives(final_states)[:args.num_envs])
        final_global = np.asarray(jax_train.both_states(final_states)[:args.num_envs])
        final_context = critic_context(
            final_global, 0, current_opponent, lessons)
        final_inputs = np.concatenate([
            final_obs.reshape(args.num_envs, -1), final_global, final_context], axis=1)
        bootstrap, _, _, _ = independent_critic_forward(
            target_critic, jnp.asarray(final_inputs))
        bootstrap = np.asarray(bootstrap)
        next_values = np.concatenate([values[1:], bootstrap[None]], axis=0)
        advantages = np.asarray(jax_train.compute_gae(
            jnp.asarray(reward), jnp.asarray(values), jnp.asarray(next_values),
            jnp.asarray(done), args.gamma, args.lam))
        returns = advantages + values
        flat_count = args.num_steps * args.num_envs
        bc_idx = jnp.asarray(rng.choice(len(bc["obs"]), min(512, flat_count), replace=True))
        # Learner safety labels align 1:1 with the rollout obs rows. Absent them
        # (jax-opponent rollout or aux disabled) feed zeros with aux_scale=0 so
        # the traced multiplier zeroes the gradient without recompilation.
        if supervision is not None and has_actor_aux:
            safe_label = jnp.asarray(
                np.asarray(supervision[0][:, :args.num_envs], np.float32)
                .reshape(flat_count, -1))
            escape_label = jnp.asarray(
                np.asarray(supervision[1][:, :args.num_envs], np.float32)
                .reshape(flat_count))
            margin_label = jnp.asarray(
                np.asarray(supervision[2][:, :args.num_envs], np.float32)
                .reshape(flat_count))
            aux_scale = jnp.asarray(args.actor_aux_coef, jnp.float32)
        else:
            safe_label = jnp.zeros((flat_count, env.N_MOVES * env.N_BOMB),
                                   jnp.float32)
            escape_label = jnp.zeros((flat_count,), jnp.float32)
            margin_label = jnp.zeros((flat_count,), jnp.float32)
            aux_scale = jnp.asarray(0.0, jnp.float32)
        actor, actor_opt_state, actor_metrics = actor_update(
            actor, actor_opt_state,
            jnp.asarray(obs.reshape(flat_count, *obs.shape[2:])),
            jnp.asarray(global_state.reshape(flat_count, -1)),
            jnp.asarray(actions.reshape(flat_count, 2)),
            jnp.asarray(old_logp.reshape(flat_count)),
            jnp.asarray(advantages.reshape(flat_count)),
            jnp.asarray(move_mask.reshape(flat_count, -1)),
            jnp.asarray(ability_mask.reshape(flat_count, -1)), bc_idx,
            safe_label, escape_label, margin_label, aux_scale)
        cf_idx = jnp.asarray(rng.choice(cf_train, min(256, len(cf_train)), replace=True))
        critic, critic_opt_state, critic_metrics = critic_update(
            critic, critic_opt_state,
            jnp.asarray(inputs.reshape(flat_count, -1)),
            jnp.asarray(returns.reshape(flat_count)), cf_idx)
        target_critic = jax.tree.map(
            lambda target, online: ((1.0 - args.target_tau) * target
                                    + args.target_tau * online),
            target_critic, critic)
        updated_values, _, _, _ = independent_critic_forward(
            critic, jnp.asarray(inputs.reshape(flat_count, -1)))
        updated_values = np.asarray(updated_values).reshape(reward.shape)
        diag = diagnostics(reward, done, returns, updated_values, bootstrap,
                           args.gamma, args.lam)
        actor_values = [float(x) for x in jax.device_get(actor_metrics)]
        critic_values = [float(x) for x in jax.device_get(critic_metrics)]
        record = {
            "update": update_index + 1, "seed": seed,
            "opponent": current_opponent,
            "lesson_counts": {env.LESSON_NAMES[index]: int((lessons == index).sum())
                              for index in np.unique(lessons)},
            "actor": dict(zip(("loss", "policy_loss", "entropy", "kl",
                                "bc_loss", "aux_loss"), actor_values)),
            "critic": dict(zip((
                "loss", "onpolicy_loss", "counterfactual_value_loss",
                "counterfactual_q_loss", "counterfactual_pair_loss",
                "counterfactual_aux_loss"), critic_values)),
            "diagnostics": diag,
            "mean_reward": float(reward.mean()),
            "episode_resets": int(done.sum()),
            "equivalent_full_300_tick_games": float(flat_count / 300.0),
            "transitions_per_second": float(flat_count / max(time.time() - collect_started, 1e-6)),
            "target_tau": args.target_tau,
        }
        if current_opponent == "tactical":
            audit_values = np.asarray(jax.device_get(rollout_audit), np.float64)
            record["reward_audit"] = dict(zip((
                "bomb_placed", "safe_tactical_bomb_placed",
                "tactical_bomb_safe_resolution",
                "tactical_bomb_placement_reward",
                "tactical_bomb_resolution_reward", "own_bomb_defeat",
                "mutual_death"), map(float, audit_values)))
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.carry_rollout_state:
            states = final_states

    heldout_seed = args.seed + 900_000
    heldout_key = jax.random.PRNGKey(np.uint32(heldout_seed & 0xFFFFFFFF))
    heldout_states = env.init_batch(heldout_key, args.num_envs)
    heldout_lessons = np.asarray(heldout_states.lesson)
    actor_hash_before_heldout = hashlib.sha256(pickle.dumps(
        jax.device_get(actor), protocol=4)).hexdigest()
    heldout_final, heldout_batch, _, _, _ = collect_for(
        "tactical", actor, heldout_states, heldout_key)
    h_obs, h_global, _, _, _, h_reward, h_done, _ = heldout_batch
    h_obs = np.asarray(h_obs[:, :args.num_envs], np.float32)
    h_global = np.asarray(h_global[:, :args.num_envs], np.float32)
    h_reward = np.asarray(h_reward[:, :args.num_envs], np.float32)
    h_done = np.asarray(h_done[:, :args.num_envs], np.bool_)
    h_inputs = make_inputs(h_obs, h_global, heldout_lessons, "tactical")
    h_value, _, _, _ = independent_critic_forward(
        critic, jnp.asarray(h_inputs.reshape(-1, h_inputs.shape[-1])))
    h_value = np.asarray(h_value).reshape(h_reward.shape)
    h_final_obs = np.asarray(
        jax_train.both_perspectives(heldout_final)[:args.num_envs])
    h_final_global = np.asarray(
        jax_train.both_states(heldout_final)[:args.num_envs])
    h_final_context = critic_context(
        h_final_global, 0, "tactical", heldout_lessons)
    h_final_inputs = np.concatenate([
        h_final_obs.reshape(args.num_envs, -1), h_final_global,
        h_final_context], axis=1)
    h_bootstrap, _, _, _ = independent_critic_forward(
        critic, jnp.asarray(h_final_inputs))
    h_bootstrap = np.asarray(h_bootstrap)
    h_returns = np.zeros_like(h_reward)
    running = h_bootstrap.copy()
    for tick in range(args.num_steps - 1, -1, -1):
        running = h_reward[tick] + args.gamma * running * (~h_done[tick])
        h_returns[tick] = running
    heldout = diagnostics(
        h_reward, h_done, h_returns, h_value, h_bootstrap,
        args.gamma, args.lam)
    heldout_by_lesson = {}
    for lesson_id in np.unique(heldout_lessons):
        columns = np.flatnonzero(heldout_lessons == lesson_id)
        heldout_by_lesson[env.LESSON_NAMES[int(lesson_id)]] = diagnostics(
            h_reward[:, columns], h_done[:, columns], h_returns[:, columns],
            h_value[:, columns], h_bootstrap[columns], args.gamma, args.lam)
    actor_hash_after_heldout = hashlib.sha256(pickle.dumps(
        jax.device_get(actor), protocol=4)).hexdigest()

    finite_actor = all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actor))
    finite_critic = all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(critic))
    finite_target_critic = all(
        np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(target_critic))
    metadata = {"seed": args.seed, "updates": args.updates,
                "curriculum": args.curriculum, "opponent": args.opponent,
                "reward_profile": args.reward_profile,
                "danger_escape_reward": args.danger_escape_reward,
                "avoidable_danger_death_penalty": args.avoidable_danger_death_penalty,
                "tactical_bomb_placement_reward": args.tactical_bomb_placement_reward,
                "tactical_bomb_resolution_reward": args.tactical_bomb_resolution_reward,
                "carry_rollout_state": args.carry_rollout_state}
    save_checkpoint(args.save_actor, actor, actor_opt_state, metadata)
    save_checkpoint(args.save_critic, critic, critic_opt_state, metadata)
    if args.save_target_critic:
        save_checkpoint(args.save_target_critic, target_critic, None, metadata)
    result = {"schema": "bun_separate_ac_micro_v1", "config": vars(args),
              "frozen_code_opponent": (
                  tactical_bot_provenance() if args.opponent == "tactical" else None),
              "actor_input_sha256": file_hash(args.actor),
              "critic_input_sha256": file_hash(args.critic),
              "actor_output_sha256": file_hash(args.save_actor),
              "critic_output_sha256": file_hash(args.save_critic),
              "target_critic_input_sha256": (
                  file_hash(args.target_critic) if args.target_critic else None),
              "target_critic_output_sha256": (
                  file_hash(args.save_target_critic)
                  if args.save_target_critic else None),
              "finite_actor": finite_actor, "finite_critic": finite_critic,
              "finite_target_critic": finite_target_critic,
              "heldout_seed": heldout_seed, "heldout": heldout,
              "heldout_by_lesson": heldout_by_lesson,
              "heldout_actor_hash_unchanged": (
                  actor_hash_before_heldout == actor_hash_after_heldout),
              "history": history, "wall_seconds": time.time() - started}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not finite_actor or not finite_critic or not finite_target_critic:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
