"""Self-play PPO for the naive JAX bomberman + end-to-end SPS bench.

Structure mirrors Average Joe (rollout via lax.scan, 2N batch, scan-based
minibatch PPO) but single-device and dependency-light (optax only).
--distill-data 提供离线蒸馏阶段：teacher（torch collect_distill 收集的
obs7/logits/masks）KL 蒸馏，可接 --distill-then-ppo 继续自对弈 PPO。
"""

import argparse
import glob
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

RULE_NAME = os.environ.get("JAXBOMB_RULE", "battle")
IS_BUN = RULE_NAME == "bun"
if IS_BUN:
    from .bun_env import (H, W, MAX_HP, MAX_STEPS, N_BOMB, N_MOVES,
                          N_OBS_CH, _danger_map, configure_start_state_curriculum,
                          configure_spawn_buckets, configure_training,
                          global_vec, init_batch,
                          legal_mask, make_obs, prepare as prepare_environment,
                          reward_from_events as _bun_reward_from_events, step)
    from .bun_safety import adjusted_joint_logits, both_avoidable_masks
    from . import bun_jax_bots
    JAX_BOT_NAMES = bun_jax_bots.TIER_NAMES + ("legacy_flee",)
else:
    from .jax_env import (H, W, MAX_HP, MAX_STEPS, N_BOMB, N_MOVES,
                          N_OBS_CH, _danger_map, global_vec, init_batch,
                          legal_mask, make_obs, step)
    _bun_reward_from_events = None
from .jax_net import (BIN_CENTERS, NUM_VALUE_BINS, V_MAX, V_MIN,
                      count_params, init_net, net_forward)
from .platform import device_summary, setup_platform

# 旧 ViTModel 兼容模式：旧 checkpoint 使用 13 个空间观测通道，当前环境
# 在末尾新增 pushable(ch13)。默认关闭；由 multicard_train 的
# --legacy-obs13 在导入本模块前显式设置，避免改变现有 14 通道训练。
LEGACY_OBS13 = os.environ.get("JAXBOMB_LEGACY_OBS13", "0") == "1"

# ---------------- 稠密奖励（对齐 torch config.py 生产值） ----------------
# 前三项恒生效（真信号，不退火）：掉血/造成伤害 ±hit_reward、每 tick 步罚、
# 终局击杀固定 win_bonus（超时血多者胜 × 退火系数，见 collect_rollout）。
# danger_penalty/brick_reward/combo/place_bonus 暂缓（行为塑形，若补须乘
# _explore_coef 退火到 0 —— 论文消融：塑形早期慢速提升、后期退化不稳定）。
STEP_PENALTY = 0.004
HIT_REWARD = 1.5
WIN_BONUS = 10.0
TRADE_WIN_BONUS = 3.5          # 同归于尽/换血险胜降级奖励（正奖励，不赶尽杀绝，但明显低于纯胜）
LOSE_BONUS_START = 6.0         # 击杀败者前期惩罚（正数，实际奖励为负）
LOSE_BONUS_FLOOR = 3.0         # 后期保留的最低失败惩罚
TIMEOUT_LEAD_BONUS = 2.0       # 超时血量领先方固定奖励
TIMEOUT_MAX_BONUS = 2.0        # 兼容旧测试/超参数别名
TIMEOUT_TRAIL_PENALTY = 1.0    # 超时血量落后方固定惩罚
TIMEOUT_DRAW_BONUS = 0.0       # 平血超时双方奖励
MUTUAL_HIT_PENALTY = 0.0       # 双方同 tick 互损额外惩罚（如 1.0 表示互换血扣 1 分）
DOUBLE_DEATH_PENALTY = 5.0     # 双方同 tick 全部阵亡（真同归双亡 0:0）惩罚（对齐战败 5.0）
WIN_HP_BONUS = 0.0             # 纯净获胜残余血量奖励（每剩 1 滴血奖励分，如 0.5）

BUN_OPPONENT_NAMES = (
    "idle", "roam", "rule_combat", "weak", "old", "recent", "history")


def parse_bun_opponent_weights(spec: str) -> np.ndarray:
    """Parse a fixed Bun opponent-pool distribution in canonical order."""
    values = {name: 0.0 for name in BUN_OPPONENT_NAMES}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, raw = item.split("=", 1)
        if name not in values:
            raise ValueError(f"unknown Bun opponent type: {name}")
        values[name] = float(raw)
    weights = np.asarray([values[name] for name in BUN_OPPONENT_NAMES], np.float32)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Bun opponent weights must be non-negative and sum > 0")
    return weights / weights.sum()


def sample_bun_opponent_kinds(key, count: int, weights):
    """Sample per-environment frozen opponent identities reproducibly."""
    logits = jnp.log(jnp.maximum(jnp.asarray(weights, jnp.float32), 1e-20))
    return jrandom.categorical(key, logits, shape=(count,)).astype(jnp.int8)


def select_bun_opponent_actions(kinds, candidates):
    """Select `(N,2)` actions from `(K,N,2)` candidates by opponent kind."""
    candidates = jnp.asarray(candidates)
    rows = jnp.arange(kinds.shape[0])
    return candidates[kinds.astype(jnp.int32), rows]


def hl_gauss_value_loss(v_logits, targets, v_min=V_MIN, v_max=V_MAX,
                        num_bins=NUM_VALUE_BINS, sigma=1.5):
    """HL-Gauss categorical cross-entropy calibrated to current reward range."""
    centers = jnp.linspace(v_min, v_max, num_bins)
    half_width = (v_max - v_min) / (num_bins - 1) / 2.0
    upper = (centers + half_width - targets[:, None]) / sigma
    lower = (centers - half_width - targets[:, None]) / sigma
    probs = jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower)
    probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-8)
    return -jnp.mean(jnp.sum(probs * jax.nn.log_softmax(v_logits, axis=-1), axis=-1))


def novelty_transition(visited, cells, done):
    """Return first-visit events and the next shared per-episode visit map.

    `cells` is `(N, 2, 2)` in `[row, col]` order and must be from the physical
    post-step state before auto-reset. The map is shared by both players: a
    simultaneous arrival at the same fresh cell earns exactly one credit, with
    P0 as the stable tie-breaker.
    """
    was_visited = jax.vmap(
        lambda v, rc: v[rc[:, 0], rc[:, 1]])(visited, cells)
    newly = ~was_visited
    same_cell = ((cells[:, 0, 0] == cells[:, 1, 0])
                 & (cells[:, 0, 1] == cells[:, 1, 1]))
    newly = newly.at[:, 1].set(newly[:, 1] & ~same_cell)
    next_visited = jax.vmap(
        lambda v, rc: v.at[rc[:, 0], rc[:, 1]].set(True))(visited, cells)
    next_visited = jnp.where(done[:, None, None],
                             jnp.zeros_like(next_visited), next_visited)
    return newly, next_visited


def reward_from_events(dmg, alive_before, alive_after, hp_after, done,
                       crate_grew, newly, walls_destroyed, crate_coef,
                       explore_coef, brick_coef, timeout_alpha,
                       win_bonus=WIN_BONUS, lose_bonus=LOSE_BONUS_START,
                       timeout_lead_bonus=TIMEOUT_LEAD_BONUS,
                       timeout_trail_penalty=TIMEOUT_TRAIL_PENALTY,
                       timeout_draw_bonus=TIMEOUT_DRAW_BONUS,
                       mutual_hit_penalty=MUTUAL_HIT_PENALTY,
                       double_death_penalty=DOUBLE_DEATH_PENALTY,
                       win_hp_bonus=WIN_HP_BONUS,
                       trade_win_bonus=TRADE_WIN_BONUS,
                       moves=None, bombs=None, idle_penalty=0.015,
                       rule_info=None):
    """Compute JAX PPO rewards from post-step events without reset-state leakage.

    奖励只归因于 auto-reset 前的物理事件：击杀、死亡、抢包、运回与安全放泡
    均由环境 info 明确标记，不能从 reset 后的新局状态反推，否则会把下一局事件
    错记到上一条 transition。
    """
    if IS_BUN:
        return _bun_reward_from_events(
            dmg, alive_before, alive_after, hp_after, done,
            crate_grew, newly, walls_destroyed, crate_coef, explore_coef,
            brick_coef, timeout_alpha, win_bonus, lose_bonus,
            timeout_lead_bonus, timeout_trail_penalty, timeout_draw_bonus,
            mutual_hit_penalty, double_death_penalty, win_hp_bonus,
            trade_win_bonus, moves=moves, bombs=bombs,
            idle_penalty=idle_penalty, rule_info=rule_info)
    dmg = dmg.astype(jnp.float32)
    dealt = dmg.sum(axis=-1, keepdims=True) - dmg
    rew = ((dealt - dmg) * HIT_REWARD
           - STEP_PENALTY * alive_before.astype(jnp.float32))
    # 专属发呆惩罚：仅当存活且选择 move=4(IDLE) 且未放雷时扣除，杜绝原地对峙挂机
    if moves is not None and bombs is not None:
        is_idle = (moves == 4) & (bombs == 0) & alive_before
        rew = rew - idle_penalty * is_idle.astype(jnp.float32)

    # 同 tick 双方互损换血惩罚（减弱无意义肉搏互损）
    mutual_hit = (dealt > 0.0) & (dmg > 0.0)
    rew = rew - mutual_hit_penalty * mutual_hit.astype(jnp.float32)

    rew = rew + crate_coef * crate_grew.astype(jnp.float32)
    rew = rew + explore_coef * newly.astype(jnp.float32)
    rew = rew + brick_coef * walls_destroyed.astype(jnp.float32)[:, None] / 2.0

    n_alive = alive_after.sum(axis=-1)
    death_done = done & (n_alive == 1)

    # 纯净击杀（胜者该 tick 未受爆炸伤害）：获胜大奖 10 分 + 残血加成
    clean_win = death_done[:, None] & alive_after & (dmg == 0.0)
    # 换血同归获胜（胜者该 tick 虽受波及掉血但成功终结比赛）：
    # 获得收敛的正向小奖（trade_win_bonus，如 3.5 分，扣除互损 1 分后净收益约 +2.5 分）
    # 绝不搞二极管重罚！依然鼓励杀死比赛，但与优雅纯胜拉开分差，形成对战道德与对战风格分级
    trade_win = death_done[:, None] & alive_after & (dmg > 0.0)
    lose = death_done[:, None] & ~alive_after

    rew = rew + win_bonus * clean_win.astype(jnp.float32)
    rew = rew + trade_win_bonus * trade_win.astype(jnp.float32)
    rew = rew - lose_bonus * lose.astype(jnp.float32)

    # 获胜残余血量激励（仅纯胜享有残血大奖，鼓励高血量高容错的优雅击杀）
    hp_f = hp_after.astype(jnp.float32)
    rew = rew + (win_hp_bonus * hp_f) * clean_win.astype(jnp.float32)

    # 双方同 tick 全部阵亡（真·双亡，n_alive == 0，双输 0:0 暴毙惩罚）
    double_death = done & (n_alive == 0)
    rew = rew - double_death_penalty * double_death[:, None].astype(jnp.float32)

    all_alive = done & (n_alive == 2)
    p0_lead = all_alive & (hp_f[:, 0] > hp_f[:, 1])
    p1_lead = all_alive & (hp_f[:, 1] > hp_f[:, 0])
    draw = all_alive & (hp_f[:, 0] == hp_f[:, 1])
    timeout_rew = jnp.stack([
        timeout_lead_bonus * p0_lead.astype(jnp.float32)
        - timeout_trail_penalty * p1_lead.astype(jnp.float32)
        + timeout_draw_bonus * draw.astype(jnp.float32),
        timeout_lead_bonus * p1_lead.astype(jnp.float32)
        - timeout_trail_penalty * p0_lead.astype(jnp.float32)
        + timeout_draw_bonus * draw.astype(jnp.float32),
    ], axis=-1)
    # timeout_alpha 保留为兼容参数；新的固定超时计分不再乘退火系数。
    del timeout_alpha
    return rew + timeout_rew


# ---------------- policy head ----------------


def sample_actions(params, arch, obs, masks, key, state=None,
                   joint_unsafe=None, safety_mode="off", safety_penalty=0.0):
    """obs (N,C,H,W)，masks = (move_mask (N,5), bomb_mask (N,2)) bool。

    返回 (act (N,2), logp (N,), val (N,))。非法动作 logits 置 -inf 后采样
    （与 torch masked_dist 同语义：softmax 概率归零）。state (N,G) 为全局
    状态向量（transformer 的 state token），None=无（旧路径/对拍）。"""
    mv, bm, v, _ = net_forward(params, arch, obs, state)
    mm, bmb = masks
    n = obs.shape[0]
    if safety_mode == "off" or joint_unsafe is None:
        mv_m = jnp.where(mm, mv, jnp.full_like(mv, -jnp.inf))
        bm_m = jnp.where(bmb, bm, jnp.full_like(bm, -jnp.inf))
        k1, k2 = jrandom.split(key)
        a_m = jrandom.categorical(k1, mv_m)
        a_b = jrandom.categorical(k2, bm_m)
        lp = (jax.nn.log_softmax(mv_m)[jnp.arange(n), a_m]
              + jax.nn.log_softmax(bm_m)[jnp.arange(n), a_b])
    else:
        joint = adjusted_joint_logits(
            mv, bm, mm, bmb, joint_unsafe, safety_mode, safety_penalty)
        action = jrandom.categorical(key, joint)
        a_m = action // N_BOMB
        a_b = action % N_BOMB
        lp = jax.nn.log_softmax(joint)[jnp.arange(n), action]
    return jnp.stack([a_m, a_b], axis=-1), lp, v


def both_perspectives(states):
    """返回 (2N, C, H, W)：p0 视角 + p1 视角拼接。

    危险图与视角无关 → 两个视角共享同一份（make_obs 传预计算 danger），
    每 tick 的 danger_map 计算减半（collect_rollout 热点，实测 2 遍版本
    每 iter 4.76s → 共享后 ≈2.4s）。
    """
    danger = jax.vmap(lambda s: _danger_map(s.fuse, s.wall, s.bomb_blast,
                                            s.brick))(states)
    obs0 = jax.vmap(lambda s, d: make_obs(s, 0, d))(states, danger)
    obs1 = jax.vmap(lambda s, d: make_obs(s, 1, d))(states, danger)
    if LEGACY_OBS13:
        obs0 = obs0[:, :13]
        obs1 = obs1[:, :13]
    return jnp.concatenate([obs0, obs1], axis=0)


def both_masks(states):
    """返回 ((2N,5), (2N,2))：与 both_perspectives 的 obs 行序对齐。

    legal_mask 的 vmap 结果是 (N,2,5)/(N,2,2)（每 state 两玩家的 mask）；
    obs 行序 = [N 个 p0 视角, N 个 p1 视角]，所以 p0 视角帧配玩家 0 的
    mask、p1 视角帧配玩家 1 的 —— 按玩家拆开拼接，不是整块 concat。
    """
    m, b = jax.vmap(legal_mask)(states)          # (N,2,5), (N,2,2)
    m0, m1 = m[:, 0], m[:, 1]
    b0, b1 = b[:, 0], b[:, 1]
    return (jnp.concatenate([m0, m1]), jnp.concatenate([b0, b1]))


def both_states(states):
    """返回 (2N, G) 全局状态向量，行序与 both_perspectives 对齐（p0 视角配
    玩家 0 的全局量、p1 配玩家 1 的）。血量/成长属性/存活/进度 —— 论文式
    双序列输入的第二路（transformer 的 state token）。"""
    v0 = jax.vmap(lambda s: global_vec(s, 0))(states)
    v1 = jax.vmap(lambda s: global_vec(s, 1))(states)
    return jnp.concatenate([v0, v1], axis=0)


# ---------------- flee bot & rollout ----------------


def flee_bot_actions(p_bot, p_opp, mm, bm, key, idle_ratio=0.25, roam_ratio=0.05, pure_flee_ratio=0.50):
    """Vectorized Mixture Rule Bot for JAX.
    p_bot: (K, 2) float32 [y, x]
    p_opp: (K, 2) float32 [y, x]
    mm: (K, 5) bool  [0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: IDLE]
    bm: (K, 2) bool  [0: NOOP, 1: BOMB]
    key: PRNGKey

    混合策略（按比例严格配比）：
    1. 静态死靶 (Idle Bot, 占 25% 规则池 = 15% 全局): 完全静止不放炮 (move=4, bomb=0)，保持见不动靶即诛杀的反射；
    2. 漫游走位 (Roam Bot, 占 5% 规则池 = 3% 全局): 纯合法随机走位不放炮；
    3. 纯逃跑追逐靶 (Pure Flee / Runner Bot, 占 50% 规则池 = 30% 全局): 极速远离对手、绝不放雷 (move=kite, bomb=0)，专门解决被纯逃跑怪遛死的追杀短板！
    4. 智能拉扯反击 (Smart Kite Bot, 占 20% 规则池 = 12% 全局): 远离对手并在贴身时落雷，具备防自灭机制。
    """
    k_type, k_mv, k_bm, k_rand = jrandom.split(key, 4)
    K = p_bot.shape[0]
    bot_type = jrandom.uniform(k_type, (K,))

    offsets = jnp.array([
        [-1.0, 0.0],  # 0: UP
        [ 1.0, 0.0],  # 1: DOWN
        [ 0.0,-1.0],  # 2: LEFT
        [ 0.0, 1.0],  # 3: RIGHT
        [ 0.0, 0.0],  # 4: IDLE
    ], jnp.float32)
    cand_pos = p_bot[:, None, :] + offsets[None, :, :]
    opp_dist = jnp.linalg.norm(p_bot - p_opp, axis=-1)
    cand_dists = jnp.linalg.norm(cand_pos - p_opp[:, None, :], axis=-1)

    # 1. 智能拉扯走位：
    # 敌近 (<=4.0 格) 优先远离对手拉开距离；敌远 (>4.0 格) 则合法方向均匀漫游，杜绝死卡墙角
    rand_scores = jrandom.uniform(k_rand, (K, 5)) * 2.0
    evade_scores = cand_dists
    kite_scores = jnp.where(opp_dist[:, None] <= 4.0, evade_scores, rand_scores)
    kite_scores = kite_scores + jnp.where(mm, 0.0, -1e6)
    noise = jrandom.uniform(k_mv, (K, 5), minval=-0.05, maxval=0.05)
    kite_move = jnp.argmax(kite_scores + noise, axis=-1)

    # 2. 漫游走位 (合法移动中随机选取)
    roam_scores = rand_scores + jnp.where(mm, 0.0, -1e6)
    roam_move = jnp.argmax(roam_scores, axis=-1)

    # 3. 静止走位 (动作 4: MOVE_IDLE)
    idle_move = jnp.full((K,), 4, dtype=jnp.int32)

    # 阈值切分
    t_idle = idle_ratio
    t_roam = idle_ratio + roam_ratio
    t_flee = idle_ratio + roam_ratio + pure_flee_ratio

    # 动作分配: idle -> roam -> kite (Pure Flee 和 Smart Kite 均使用 kite_move)
    move_act = jnp.where(
        bot_type < t_idle,
        idle_move,
        jnp.where(bot_type < t_roam, roam_move, kite_move)
    )

    # 炸弹动作：
    # 仅 Smart Kite 版 (bot_type >= t_flee) 在近身且安全时放雷！
    # 前面的 Idle、Roam 和 Pure Flee (占 70%) 全部 0 放雷！
    can_escape = mm[:, :4].sum(axis=-1) >= 2
    near = opp_dist <= 2.8
    rand_drop = jrandom.uniform(k_bm, (K,)) < 0.45
    want_bomb = (bot_type >= t_flee) & near & can_escape & rand_drop & bm[:, 1]
    bomb_act = want_bomb.astype(jnp.int32)

    return jnp.stack([move_act, bomb_act], axis=-1)


def jax_bot_seat_actions(states, key, kinds, n_bot, enabled):
    """Graded device-side rule bots for the first `n_bot` envs.

    Seat layout matches the flee-bot split (and `mask_bot_advantages`): envs
    `[:n_bot//2]` have the bot on P1, envs `[n_bot//2:n_bot]` on P0, so the
    learner practises both seats. `kinds` (n_bot,) index `JAX_BOT_NAMES`;
    `enabled` is a static numpy bool mask used only to prune unused branches.
    Returns `(n_bot, 2)` actions for the bot seats.
    """
    weights = np.asarray(enabled, np.float32)
    half = n_bot // 2
    sub = jax.tree_util.tree_map(lambda x: x[:n_bot], states)
    rows = jnp.arange(n_bot)
    seats = (rows < half).astype(jnp.int32)
    move_mask, ability_mask = jax.vmap(legal_mask)(sub)
    bot_mm, bot_bm = move_mask[rows, seats], ability_mask[rows, seats]
    k_rule, k_flee = jrandom.split(key)
    legacy = len(JAX_BOT_NAMES) - 1
    use_rule = bool(np.asarray(weights)[:legacy].sum() > 0)
    use_flee = bool(np.asarray(weights)[legacy] > 0)
    acts = None
    if use_rule:
        acts = bun_jax_bots.rule_bot_actions(
            sub, seats, bot_mm, bot_bm, k_rule, jnp.minimum(kinds, legacy - 1))
    if use_flee:
        flee = flee_bot_actions(
            sub.pos[rows, seats], sub.pos[rows, 1 - seats], bot_mm, bot_bm, k_flee)
        acts = flee if acts is None else jnp.where(
            (kinds == legacy)[:, None], flee, acts)
    return acts


def apply_jax_bot_actions(a0, a1, bot_acts, n_bot):
    half = n_bot // 2
    return (a0.at[half:n_bot].set(bot_acts[half:]),
            a1.at[:half].set(bot_acts[:half]))


def bun_opponent_actions(states, obs, gv, masks, arch, key, kinds,
                         weak_params, old_params, recent_params):
    """Return P1 actions for the real asymmetric Bun opponent pool."""
    n = states.pos.shape[0]
    mm, bm = masks[0][n:], masks[1][n:]
    key, roam_key, rule_key, weak_key, old_key, recent_key = jrandom.split(key, 6)
    idle = jnp.tile(jnp.asarray([[4, 0]], jnp.int32), (n, 1))
    roam_scores = jrandom.uniform(roam_key, (n, N_MOVES))
    roam_move = jnp.argmax(jnp.where(mm, roam_scores, -1.0), axis=-1)
    roam = jnp.stack([roam_move, jnp.zeros((n,), jnp.int32)], axis=-1)
    rule = flee_bot_actions(
        states.pos[:, 1], states.pos[:, 0], mm, bm, rule_key,
        idle_ratio=0.0, roam_ratio=0.0, pure_flee_ratio=0.0)
    weak, _, _ = sample_actions(
        weak_params, arch, obs[n:], (mm, bm), weak_key, state=gv[n:])
    old, _, _ = sample_actions(
        old_params, arch, obs[n:], (mm, bm), old_key, state=gv[n:])
    recent, _, _ = sample_actions(
        recent_params, arch, obs[n:], (mm, bm), recent_key, state=gv[n:])
    return select_bun_opponent_actions(
        kinds, jnp.stack(
            [idle, roam, rule, weak, old, recent, recent], axis=0))


def collect_rollout(params, arch, states, key, num_steps, no_mask=False,
                    obs_quant=False, checkpoint=False, crate_coef=0.0,
                    explore_coef=0.0, brick_coef=0.0, timeout_alpha=1.0,
                    lose_bonus=LOSE_BONUS_START, win_bonus=WIN_BONUS,
                    timeout_lead_bonus=TIMEOUT_LEAD_BONUS,
                    timeout_trail_penalty=TIMEOUT_TRAIL_PENALTY,
                    timeout_draw_bonus=TIMEOUT_DRAW_BONUS,
                    mutual_hit_penalty=MUTUAL_HIT_PENALTY,
                    double_death_penalty=DOUBLE_DEATH_PENALTY,
                    win_hp_bonus=WIN_HP_BONUS,
                    trade_win_bonus=TRADE_WIN_BONUS,
                    flee_bot_ratio=0.20,
                    idle_penalty=0.015,
                    action_repeat=1,
                    safety_mode="off",
                    safety_penalty=0.0,
                    safety_margin=0,
                    bun_opponent_weights=None,
                    bun_opponent_weak_params=None,
                    bun_opponent_old_params=None,
                    bun_opponent_recent_params=None,
                    jax_bot_pool=None,
                    jax_bot_enabled=None,
                    return_bot_stats=False):
    """自对弈：同一网络打两边，可选混入部分逃跑对手环境。states (N, ...)。返回 (new_states, batch, nov, kills)。

    jax_bot_pool：`JAX_BOT_NAMES` 顺序的权重（可为 traced jnp，便于自适应课程
    每 iter 改权重不重编译）；jax_bot_enabled：静态 bool 掩码（numpy，裁剪未用
    分支，默认 = pool>0）。给定时前 flee_bot_ratio 比例的 env 由分级规则 bot
    （bun_jax_bots）替代 flee-bot，每局结束按权重重采样难度；
    return_bot_stats=True 时额外返回 (K,4) 每档
    [局数, 学习者胜, 学习者负, 学习者自炸] 计数。

    nov：每 env/玩家的 novelty 计数（未加权，与 batch.rew 同口径窗口累计）。
    训练侧除以 num_steps × coef 即得"探索分/帧"，与 rew 均值直接对比——
    探索分单局天然封顶 coef×可达格数（195 格地图 ≈ coef×195），coef=0.01 时
    全图逛完 1.95 分，远低于单次伤害 1.5×N 与击杀 10，不会压过胜负信号。

    step 在终局后**就地重置**（对齐正式版 auto_reset），所以胜负判定用
    step 前的 alive 快照（重置后 alive 恒全 True，无法区分谁死）。
    batch 含每 tick 的 (move_mask, bomb_mask)——PPO loss 用它屏蔽非法动作。
    no_mask=True：mask 全放开（性能 A/B 用，行为=无 mask 旧版）。
    obs_quant=True：obs buffer 存 uint8（×255 量化，PPO 反量化后进网络）。
    obs 8 通道都是低精度值（二值/离散/0-1），uint8 精度 1/255 ≈ 0.4% 优于
    bf16 尾数，buffer 从 fp32 45GB 降到 11GB（8192×512 OOM 的解法）。
    checkpoint=True：scan body 用 jax.checkpoint 包裹——反向重算中间量，
    不保留每 tick 的 obs/激活（8192×512 下 scan 中间量 44.8GB 是量化后
    剩余瓶颈，checkpoint 可进一步降到 buffer 本身大小）。
    crate_coef：开箱成长奖励系数（0=关）。关卡模式多数地图出生点被砖隔开，
    前期无交战通道，正信号只有破砖吃箱——bootstrap 奖励（长退火）加速
    前期学习；退火后只剩真胜负（参考实现 stage0 composite_reward 同款思路）。
    explore_coef：探索 novelty 奖励系数（0=关）。每 tick 玩家中心格若是
    **本局首次到达**（共享 visited 掩码，done 清零）→ +explore_coef。这是
    "整局只走几格就重罚"的稠密版：走过的格不再给分，坐桩/困在出生点几乎
    零探索分，破砖开路才拿分。与 crate 同款长退火。掩码在 scan carry 里，
    不进 BombState/ckpt（断点接续零兼容问题）。传入 jnp 标量（随 iter 变化
    不触发重编译）。
    brick_coef：炸墙奖励系数（0=关）。每炸毁一块砖（含灌木）双方各
    +brick_coef/2。治"出生点 3 格死锁"：crate 奖励的链路（炸→掷爆率→
    吃到）太长太弱学不会，给"炸墙"本身即时正反馈，破墙开路才有后续探索/
    吃箱/交手。同上乘统一退火（headless 实测 500 iter 模型在隔离图仍
    放炮少，炸墙是冷启动关键）。
    timeout_alpha：超时血多者奖励退火系数（动态退火，早期 0.0，后期 1.0）。
    lose_bonus：击杀败者惩罚（动态退火，早期 LOSE_BONUS_START，后期 LOSE_BONUS_FLOOR）。
    win_bonus：击杀胜者固定奖励（默认 10.0）。
    trade_win_bonus：同归于尽/换血险胜奖励（默认 3.5）。
    timeout_lead_bonus：超时血量领先方奖励（默认 2.0）。
    timeout_trail_penalty：超时血量落后方惩罚（默认 1.0）。
    timeout_draw_bonus：超时平局双方奖励（默认 0.0）。
    mutual_hit_penalty：同 tick 双方互损换血惩罚（默认 0.0）。
    double_death_penalty：双方同时暴毙双亡重罚（默认 0.0）。
    win_hp_bonus：获胜残余血量加成（默认 0.0）。
    flee_bot_ratio：混入逃跑对手环境比例（默认 0.20）。
    返回 (final_states, batch, nov, kills)：nov 每 env/玩家 novelty 累计；
    kills 每 env 窗口内击杀局数（death_done 累计）——动态退火 α=1-tanh(k·x)
    的 x 来源（每局击杀率 = mean(kills)/n_episodes）。
    """
    n = states.pos.shape[0]
    use_bun_opponents = bun_opponent_weights is not None
    n_flee = int(n * flee_bot_ratio)
    ones_m = jnp.ones((2 * n, N_MOVES), jnp.bool_)
    ones_b = jnp.ones((2 * n, N_BOMB), jnp.bool_)
    visited0 = jnp.zeros((n, H, W), jnp.bool_)      # 探索掩码（scan carry）
    nov0 = jnp.zeros((n, 2), jnp.float32)           # 每 env/玩家 未加权 novelty 累计
    kills0 = jnp.zeros((n,), jnp.float32)           # 每 env 击杀局数（动态退火 x 来源）
    if use_bun_opponents:
        key, opponent_key = jrandom.split(key)
        opponent_kinds0 = sample_bun_opponent_kinds(
            opponent_key, n, bun_opponent_weights)
    else:
        opponent_kinds0 = jnp.zeros((n,), jnp.int8)
    use_jax_bots = jax_bot_pool is not None and n_flee > 0 and not use_bun_opponents
    n_kinds = len(JAX_BOT_NAMES) if IS_BUN else 1
    if use_jax_bots and jax_bot_enabled is None:
        jax_bot_enabled = np.asarray(jax_bot_pool) > 0
    if use_jax_bots:
        key, bot_key = jrandom.split(key)
        bot_kinds0 = jnp.zeros((n,), jnp.int8).at[:n_flee].set(
            sample_bun_opponent_kinds(bot_key, n_flee, jax_bot_pool))
    else:
        bot_kinds0 = jnp.zeros((n,), jnp.int8)
    bot_stats0 = jnp.zeros((n_kinds, 4), jnp.float32)
    # Learner seat in bot envs: P0 where the bot is P1 (first half), else P1.
    learner_seat = jnp.where(jnp.arange(n_flee) < n_flee // 2, 0, 1)

    def one_step(carry, _):
        (states, key, visited, nov, kills, opponent_kinds,
         bot_kinds, bot_stats) = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        obs = both_perspectives(states)               # (2N, C, H, W)
        masks = (ones_m, ones_b) if no_mask else both_masks(states)
        joint_unsafe = (both_avoidable_masks(states, safety_margin)
                        if IS_BUN and safety_mode != "off"
                        else jnp.zeros((2 * n, N_MOVES, N_BOMB), jnp.bool_))
        gv = both_states(states)                      # (2N, G) 全局状态向量
        acts, lps, vals = sample_actions(params, arch, obs, masks, key,
                                         state=gv, joint_unsafe=joint_unsafe,
                                         safety_mode=safety_mode,
                                         safety_penalty=safety_penalty)
        a0, a1 = acts[:n], acts[n:]
        if use_bun_opponents:
            key, opponent_action_key = jrandom.split(key)
            a1 = bun_opponent_actions(
                states, obs, gv, masks, arch, opponent_action_key,
                opponent_kinds, bun_opponent_weak_params,
                bun_opponent_old_params, bun_opponent_recent_params)
        elif use_jax_bots:
            key, k_bot = jrandom.split(key)
            bot_acts = jax_bot_seat_actions(
                states, k_bot, bot_kinds[:n_flee], n_flee, jax_bot_enabled)
            a0, a1 = apply_jax_bot_actions(a0, a1, bot_acts, n_flee)
        elif n_flee > 0:
            key, k_bot1, k_bot0 = jrandom.split(key, 3)
            mm_all, bm_all = masks
            n_flee_half = n_flee // 2
            if n_flee_half > 0:
                bot1_acts = flee_bot_actions(
                    states.pos[:n_flee_half, 1], states.pos[:n_flee_half, 0],
                    mm_all[n:n + n_flee_half], bm_all[n:n + n_flee_half], k_bot1
                )
                a1 = a1.at[:n_flee_half].set(bot1_acts)
            if n_flee > n_flee_half:
                bot0_acts = flee_bot_actions(
                    states.pos[n_flee_half:n_flee, 0], states.pos[n_flee_half:n_flee, 1],
                    mm_all[n_flee_half:n_flee], bm_all[n_flee_half:n_flee], k_bot0
                )
                a0 = a0.at[n_flee_half:n_flee].set(bot0_acts)

        env_acts = jnp.stack([a0, a1], axis=1)        # (N, 2, 2)
        keys = jrandom.split(kstep, n)                # 每 env 一步的 RNG（地图/宝箱）
        new_states, done, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts,
                                                               keys)
        # Use the physical post-step cells captured before auto-reset. On a
        # terminal tick `new_states` is already the next episode's spawn map.
        newly, new_visited = novelty_transition(visited, info["cell"], done)
        # 稠密奖励（对齐 torch step 的 hit/step/win 段）：
        #   - 掉 1 血 -HIT_REWARD / 造成 1 伤害 +HIT_REWARD（info.dmg 结算后快照，
        #     auto_reset 前取值 —— 1v1 里对方掉血 = 我的泡干的）；
        #   - 每 tick -STEP_PENALTY（防磨洋工；**用 step 前 alive0**，对齐 torch
        #     死亡 tick 死者也扣步罚 —— info.alive 是结算后，死者已 False）；
        #   - 终局：死亡（n_alive==1）击杀方 +win_bonus / 败者 -lose_bonus；
        #     超时全员存活（n_alive==2）按血量领先 +timeout_lead_bonus、
        #     落后 -timeout_trail_penalty，平血 timeout_draw_bonus。
        #     info.alive/hp 是结算后值，不受 auto_reset 重置污染。
        rew = reward_from_events(
            info["dmg"], states.alive, info["alive"], info["hp"], done,
            info["crate"], newly, info["walls"], crate_coef, explore_coef,
            brick_coef, timeout_alpha, win_bonus, lose_bonus,
            timeout_lead_bonus, timeout_trail_penalty, timeout_draw_bonus,
            mutual_hit_penalty, double_death_penalty, win_hp_bonus,
            trade_win_bonus,
            moves=env_acts[:, :, 0], bombs=env_acts[:, :, 1],
            idle_penalty=idle_penalty, rule_info=info)
        nov = nov + newly.astype(jnp.float32)       # 统计用：探索分/帧可监控
        n_alive = info["alive"].sum(axis=-1)          # (N,)
        death_done = done & (n_alive == 1)
        kill_event = info["death"].any(axis=-1) if IS_BUN else death_done
        kills = kills + kill_event.astype(jnp.float32)

        if action_repeat == 2:
            key, kstep2, k_bot1_2, k_bot0_2 = jrandom.split(key, 4)
            # AI 维持上一帧移动惯性，放炮指令仅首帧触发脉冲 (bomb=0)
            a0_2 = jnp.stack([a0[:, 0], jnp.zeros_like(a0[:, 1])], axis=-1)
            a1_2 = jnp.stack([a1[:, 0], jnp.zeros_like(a1[:, 1])], axis=-1)
            # 规则怪 (Flee Bot) 在第 2 物理 tick 执行 100ms 敏捷避险刷新
            if use_bun_opponents:
                obs2 = both_perspectives(new_states)
                gv2 = both_states(new_states)
                masks2 = both_masks(new_states)
                a1_2 = bun_opponent_actions(
                    new_states, obs2, gv2, masks2, arch, k_bot1_2,
                    opponent_kinds, bun_opponent_weak_params,
                    bun_opponent_old_params, bun_opponent_recent_params)
                a1_2 = a1_2.at[:, 1].set(0)
            elif use_jax_bots:
                bot_acts2 = jax_bot_seat_actions(
                    new_states, k_bot1_2, bot_kinds[:n_flee], n_flee,
                    jax_bot_enabled)
                a0_2, a1_2 = apply_jax_bot_actions(a0_2, a1_2, bot_acts2, n_flee)
            elif n_flee > 0:
                mm2_all, bm2_all = both_masks(new_states)
                if n_flee_half > 0:
                    bot1_acts2 = flee_bot_actions(
                        new_states.pos[:n_flee_half, 1], new_states.pos[:n_flee_half, 0],
                        mm2_all[n:n + n_flee_half], bm2_all[n:n + n_flee_half], k_bot1_2
                    )
                    a1_2 = a1_2.at[:n_flee_half].set(bot1_acts2)
                if n_flee > n_flee_half:
                    bot0_acts2 = flee_bot_actions(
                        new_states.pos[n_flee_half:n_flee, 0], new_states.pos[n_flee_half:n_flee, 1],
                        mm2_all[n_flee_half:n_flee], bm2_all[n_flee_half:n_flee], k_bot0_2
                    )
                    a0_2 = a0_2.at[n_flee_half:n_flee].set(bot0_acts2)
            env_acts2 = jnp.stack([a0_2, a1_2], axis=1)
            keys2 = jrandom.split(kstep2, n)
            st2, done2, info2 = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(new_states, env_acts2, keys2)
            newly2, new_visited = novelty_transition(new_visited, info2["cell"], done2)
            rew2 = reward_from_events(
                info2["dmg"], new_states.alive, info2["alive"], info2["hp"], done2,
                info2["crate"], newly2, info2["walls"], crate_coef, explore_coef,
                brick_coef, timeout_alpha, win_bonus, lose_bonus,
                timeout_lead_bonus, timeout_trail_penalty, timeout_draw_bonus,
                mutual_hit_penalty, double_death_penalty, win_hp_bonus,
                trade_win_bonus,
                moves=env_acts2[:, :, 0], bombs=env_acts2[:, :, 1],
                idle_penalty=idle_penalty, rule_info=info2)
            rew = rew + jnp.where(done[:, None], 0.0, rew2)
            nov = nov + jnp.where(done[:, None], 0.0, newly2.astype(jnp.float32))
            n_alive2 = info2["alive"].sum(axis=-1)
            death_done2 = done2 & (n_alive2 == 1) & (~done)
            kills = kills + death_done2.astype(jnp.float32)
            new_states = jax.tree_util.tree_map(
                lambda s1, s2: jnp.where(done.reshape((-1,) + (1,) * (s1.ndim - 1)), s1, s2),
                new_states, st2)
            done = done | done2
        if use_bun_opponents:
            key, next_opponent_key = jrandom.split(key)
            sampled_kinds = sample_bun_opponent_kinds(
                next_opponent_key, n, bun_opponent_weights)
            opponent_kinds = jnp.where(done, sampled_kinds, opponent_kinds)
        if use_jax_bots:
            if return_bot_stats and IS_BUN:
                # 按 tick 事件统计（danger_arena 死亡复活、仅超时终局，winner 无信息）
                rows = jnp.arange(n_flee)

                def seat(name):
                    return info[name][:n_flee][rows, learner_seat].astype(jnp.float32)

                ended = jnp.ones((n_flee,), jnp.float32)
                kill = jnp.maximum(seat("surviving_causal_kill"),
                                   seat("surviving_physical_kill"))
                death = seat("death")
                self_kill = seat("own_bomb_defeat")
                upd = jnp.stack([ended, kill, death, self_kill], axis=-1)  # ticks,kill,death,self
                bot_stats = bot_stats.at[bot_kinds[:n_flee].astype(jnp.int32)].add(upd)
            key, next_bot_key = jrandom.split(key)
            resampled = sample_bun_opponent_kinds(next_bot_key, n_flee, jax_bot_pool)
            bot_kinds = bot_kinds.at[:n_flee].set(
                jnp.where(done[:n_flee], resampled, bot_kinds[:n_flee]))
        d = jnp.concatenate([done, done])
        rew = jnp.concatenate([rew[:, 0], rew[:, 1]])
        obs_s = (jnp.round(obs * 255.0).astype(jnp.uint8)
                 if obs_quant else obs)
        state_s = (jnp.round(gv * 255.0).astype(jnp.uint8)
                   if obs_quant else gv)
        acts_exec = jnp.concatenate([a0, a1], axis=0)
        stored_masks = (masks[0], masks[1], joint_unsafe)
        data = (obs_s, state_s, acts_exec, lps, vals, rew, d, stored_masks)
        return (new_states, key, new_visited, nov, kills, opponent_kinds,
                bot_kinds, bot_stats), data
    body = (jax.checkpoint(one_step) if checkpoint else one_step)
    (final_states, _, _, nov, kills, _, _, bot_stats), data = jax.lax.scan(
        body, (states, key, visited0, nov0, kills0, opponent_kinds0,
               bot_kinds0, bot_stats0), None,
        length=num_steps)
    obs, state, acts, lps, vals, rew, done, masks = data
    batch = (obs, state, acts, lps, vals, rew, done, masks)
    if return_bot_stats:
        return final_states, batch, nov, kills, bot_stats
    return final_states, batch, nov, kills


def collect_rollout_two(params_a, params_b, arch, states, key, num_steps,
                        no_mask=False, obs_quant=False):
    """两策略自对弈 rollout（评估用）：p0 视角用 params_a、p1 用 params_b。

    与 collect_rollout 同一环境语义（auto_reset/稠密奖励），但**不存 obs
    buffer**（评估不训练），只返回胜率计数：
      win_stats = (p0_wins, p0_losses) —— 终局击杀（死亡 tick 存活者胜）与
    超时（血高者胜）各计一局；episode 就地重置，每个终局 tick 计一次。
    p0 胜率 = p0_wins / (p0_wins + p0_losses)。"""
    n = states.pos.shape[0]
    ones_m = jnp.ones((2 * n, N_MOVES), jnp.bool_)
    ones_b = jnp.ones((2 * n, N_BOMB), jnp.bool_)

    def one_step(carry, _):
        states, key = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        obs = both_perspectives(states)
        masks = (ones_m, ones_b) if no_mask else both_masks(states)
        gv = both_states(states)
        # p0 帧（obs[:n]）用 params_a，p1 帧（obs[n:]）用 params_b
        a0, _, _ = sample_actions(params_a, arch, obs[:n],
                                  (masks[0][:n], masks[1][:n]), k0,
                                  state=gv[:n])
        a1, _, _ = sample_actions(params_b, arch, obs[n:],
                                  (masks[0][n:], masks[1][n:]), k1,
                                  state=gv[n:])
        env_acts = jnp.stack([a0, a1], axis=1)
        keys = jrandom.split(kstep, n)
        new_states, done, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts,
                                                               keys)
        # 胜率计数（与当前规则终局口径一致）
        if IS_BUN:
            p0_win = done & (info["winner"] == 0)
            p0_lose = done & (info["winner"] == 1)
        else:
            n_alive = info["alive"].sum(axis=-1)
            death_done = done & (n_alive == 1)
            p0_win = death_done & info["alive"][:, 0]
            p0_lose = death_done & ~info["alive"][:, 0]
            all_alive = done & (n_alive == 2)
            hp_f = info["hp"]
            p0_win = p0_win | (all_alive & (hp_f[:, 0] > hp_f[:, 1]))
            p0_lose = p0_lose | (all_alive & (hp_f[:, 0] < hp_f[:, 1]))
        return (new_states, key), (p0_win.sum(), p0_lose.sum())

    (final_states, _), (w, l) = jax.lax.scan(
        one_step, (states, key), None, length=num_steps)
    return final_states, (w.sum(), l.sum())

# ---------------- GAE ----------------

def compute_gae(rew, val, next_val, done, gamma, lam):
    # GAE 从 rollout 尾部反向累计 TD residual；done 截断跨局 bootstrap，
    # 防止 reset 后新一局的 value 泄漏到上一局。
    def scan_fn(adv_prev, inputs):
        r, v, nv, d = inputs
        bootstrap = jnp.where(d, 0.0, nv)
        delta = r + gamma * bootstrap - v
        adv = delta + gamma * lam * (1.0 - d) * adv_prev
        return adv, adv

    _, advs = jax.lax.scan(scan_fn, jnp.zeros_like(val[0]),
                           (rew[::-1], val[::-1], next_val[::-1], done[::-1]))
    return advs[::-1]


def mask_bot_advantages(advs, rets, vals, n, flee_bot_ratio):
    """屏蔽规则 Bot 所在席位的 PPO 优势度与回报梯度。
    Bot 席位并非由神经网络策略生成，将其优势度置 0、回报对齐当前估值，
    确保神经网络仅从与 Bot 交手（击杀/追逐）的视角中学习，彻底避免 Bot 样本污染。
    """
    n_flee = int(n * flee_bot_ratio)
    if n_flee <= 0:
        return advs, rets
    n_flee_half = n_flee // 2
    if n_flee_half > 0:
        advs = advs.at[:, n : n + n_flee_half].set(0.0)
        rets = rets.at[:, n : n + n_flee_half].set(vals[:, n : n + n_flee_half])
    if n_flee > n_flee_half:
        advs = advs.at[:, n_flee_half : n_flee].set(0.0)
        rets = rets.at[:, n_flee_half : n_flee].set(vals[:, n_flee_half : n_flee])
    return advs, rets


# ---------------- PPO update ----------------


def ppo_update(params, opt, opt_state, arch, batch, key, minibatch,
               clip_eps, vf_coef, ent_coef, epochs, axis_name=None,
               return_loss=False, adv_top_frac=0.25,
               safety_mode="off", safety_penalty=0.0,
               reference_params=None, reference_kl_coef=0.0):
    """PPO 更新。axis_name 非 None 时（数据并行 pmap）对梯度做 pmean
    allreduce：每卡 minibatch 减半（等效全局 minibatch 不变 → 梯度步数
    不变），梯度跨卡平均后各卡用相同更新量，参数保持逐卡一致。

    return_loss=True 时返回 (params, opt_state, last_loss)，last_loss 为
    最后 epoch 的平均 PPO loss（value_and_grad 顺带得到，零额外计算）。

    clipped ratio 控制策略漂移，value loss 拟合 GAE return，entropy 保留探索；
    Actor 与独立 Critic/Target Critic 的参数树不会在这里混合。
    """
    obs, state, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    adv_f = advs.reshape(-1)
    adv_norm = jnp.where(adv_f == 0.0, 0.0, (adv_f - jnp.mean(adv_f)) / (jnp.std(adv_f) + 1e-8))
    mb_half = max(minibatch // 2, 1)
    n_keep = max((int(total * adv_top_frac) // mb_half) * mb_half,
                 mb_half) if 0.0 < adv_top_frac < 1.0 else total
    n_keep = min(n_keep, total)
    adv_for_topk = jnp.where(adv_f == 0.0, -1e9, jnp.abs(adv_norm))
    _, actor_idx = jax.lax.top_k(adv_for_topk, n_keep) \
        if 0.0 < adv_top_frac < 1.0 else (jnp.zeros((0,)), jnp.arange(total))
    obs_q = (obs.dtype == jnp.uint8)
    obs_f = obs.reshape(total, *obs.shape[2:])   # 保持 uint8，body 内延迟反量化
    st_f = state.reshape(total, -1)              # 全局状态向量（同量化）
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks[:2]
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)
    unsafe_f = (masks[2].reshape(total, N_MOVES, N_BOMB)
                if len(masks) > 2 else
                jnp.zeros((total, N_MOVES, N_BOMB), jnp.bool_))

    def one_epoch(params, opt_state, key):
        ka, kc = jrandom.split(key)
        actor = actor_idx[jrandom.permutation(ka, n_keep)]
        critic = jrandom.permutation(kc, total)[:n_keep]
        idx = jnp.concatenate([actor.reshape(-1, mb_half),
                               critic.reshape(-1, mb_half)], axis=1)

        def body(carry, mb):
            params, opt_state = carry
            o, a, ol, ad, rt, mm, bm, unsafe = (
                obs_f[mb], acts_f[mb], old_f[mb], adv_norm[mb], ret_f[mb],
                mm_f[mb], bm_f[mb], unsafe_f[mb])
            st = st_f[mb]
            if obs_q:
                o = o.astype(jnp.float32) / 255.0   # 延迟反量化（minibatch 级）
                st = st.astype(jnp.float32) / 255.0

            def loss_fn(p):
                return _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt,
                                 clip_eps, vf_coef, ent_coef,
                                 unsafe, safety_mode, safety_penalty,
                                 reference_params, reference_kl_coef)

            # value_and_grad 与 grad 等价开销，顺带得到 loss 供监控日志
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            if axis_name is not None:
                grads = jax.lax.pmean(grads, axis_name)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_val

        (params, opt_state), losses = jax.lax.scan(
            body, (params, opt_state), idx)
        return params, opt_state, jnp.mean(losses)

    last_loss = None
    for _ in range(epochs):
        key, ek = jrandom.split(key)
        params, opt_state, last_loss = one_epoch(params, opt_state, ek)
    if return_loss:
        return params, opt_state, last_loss
    return params, opt_state


def _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt, clip_eps, vf_coef,
              ent_coef, joint_unsafe=None, safety_mode="off",
              safety_penalty=0.0, reference_params=None,
              reference_kl_coef=0.0):
    """PPO loss（minibatch 级）——ppo_update / ppo_update_lsgd /
    ppo_update_gradsync 共用同一实现，三种同步模式的数值路径一致。"""
    mv, bm_, _v, v_logits = net_forward(p, arch, o, st)
    n = o.shape[0]
    nh = n // 2
    if safety_mode == "off" or joint_unsafe is None:
        mv_a = jnp.where(mm[:nh], mv[:nh], jnp.full_like(mv[:nh], -jnp.inf))
        bm_a = jnp.where(bm[:nh], bm_[:nh], jnp.full_like(bm_[:nh], -jnp.inf))
        lsm = jax.nn.log_softmax(mv_a)
        lsb = jax.nn.log_softmax(bm_a)
        lp = (lsm[jnp.arange(nh), a[:nh, 0]]
              + lsb[jnp.arange(nh), a[:nh, 1]])
        pm = jnp.exp(lsm)
        pb = jnp.exp(lsb)
        ent = (-(pm * jnp.where(pm > 0, lsm, 0.0)).sum(-1).mean()
               - (pb * jnp.where(pb > 0, lsb, 0.0)).sum(-1).mean())
        policy_log_probs = None
    else:
        joint = adjusted_joint_logits(
            mv[:nh], bm_[:nh], mm[:nh], bm[:nh], joint_unsafe[:nh],
            safety_mode, safety_penalty)
        policy_log_probs = jax.nn.log_softmax(joint)
        action_index = a[:nh, 0] * N_BOMB + a[:nh, 1]
        lp = policy_log_probs[jnp.arange(nh), action_index]
        probabilities = jnp.exp(policy_log_probs)
        ent = -(probabilities * jnp.where(
            probabilities > 0, policy_log_probs, 0.0)).sum(-1).mean()
    ratio = jnp.exp(lp - ol[:nh])
    pg1 = -ad[:nh] * ratio
    pg2 = -ad[:nh] * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pol = jnp.maximum(pg1, pg2).mean()
    val_l = hl_gauss_value_loss(v_logits[nh:], rt[nh:])
    kl = jnp.asarray(0.0, jnp.float32)
    if reference_params is not None and reference_kl_coef > 0:
        ref_mv, ref_bm, _, _ = net_forward(reference_params, arch, o[:nh], st[:nh])
        ref_joint = adjusted_joint_logits(
            ref_mv, ref_bm, mm[:nh], bm[:nh],
            jnp.zeros_like(joint_unsafe[:nh]), "off", 0.0)
        ref_log_probs = jax.nn.log_softmax(ref_joint)
        if policy_log_probs is None:
            current_joint = adjusted_joint_logits(
                mv[:nh], bm_[:nh], mm[:nh], bm[:nh],
                jnp.zeros_like(joint_unsafe[:nh]), "off", 0.0)
            policy_log_probs = jax.nn.log_softmax(current_joint)
        probabilities = jnp.exp(policy_log_probs)
        kl_terms = jnp.where(
            probabilities > 0,
            probabilities * (policy_log_probs - ref_log_probs), 0.0)
        kl = jnp.mean(jnp.sum(kl_terms, axis=-1))
    return pol + vf_coef * val_l - ent_coef * ent + reference_kl_coef * kl


def ppo_update_lsgd(params, opt, opt_state, arch, batch, key, minibatch,
                    clip_eps, vf_coef, ent_coef, epochs, axis_name=None,
                    sync_k=128, bf16_sync=False, sync_state=False,
                    return_loss=False, adv_top_frac=0.25):
    """Local SGD 版 PPO：minibatch 循环内**零通信**（每卡用本地梯度更新），
    每 sync_k 个 minibatch 做一次 pmean 全量同步（默认只平均参数，
    sync_state=True 时连 Adam 动量/方差一起平均，防本地漂移但流量×3）。

    通信量 = (总 minibatch 数 / sync_k) 次全模型同步。对比现状（每个
    minibatch 一次 pmean 梯度：1024 步/迭代 → 20 卡 ~50GB/迭代/卡），
    sync_k=256 降到 4 次/迭代 ~0.4GB、sync_k=128 8 次 ~0.7GB，
    bf16_sync 再减半 —— 跨机 RCCL/TCP 从分钟级降到秒级。

    与 ppo_update 的区别：后者每步平均"梯度再做优化步"（参数逐位一致）；
    本函数每步用本地梯度更新、定期平均"参数"（含可选动量）。sync_k=1 时
    两种语义也不等价，不保证与 ppo_update 逐位一致。axis_name=None 时
    退化为纯本地训练（sync 全是 no-op，等价 K=∞）。
    """
    obs, state, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    adv_f = advs.reshape(-1)
    adv_norm = jnp.where(adv_f == 0.0, 0.0, (adv_f - jnp.mean(adv_f)) / (jnp.std(adv_f) + 1e-8))
    mb_half = max(minibatch // 2, 1)
    n_keep = max((int(total * adv_top_frac) // mb_half) * mb_half,
                 mb_half) if 0.0 < adv_top_frac < 1.0 else total
    n_keep = min(n_keep, total)
    adv_for_topk = jnp.where(adv_f == 0.0, -1e9, jnp.abs(adv_norm))
    _, actor_idx = jax.lax.top_k(adv_for_topk, n_keep) \
        if 0.0 < adv_top_frac < 1.0 else (jnp.zeros((0,)), jnp.arange(total))
    obs_q = (obs.dtype == jnp.uint8)
    obs_f = obs.reshape(total, *obs.shape[2:])   # 保持 uint8，body 内延迟反量化
    st_f = state.reshape(total, -1)
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks[:2]
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)

    def local_step(carry, mb):
        """单个 minibatch 的本地更新（无任何通信）——与 ppo_update 的 body
        一致，只是去掉 pmean。"""
        params, opt_state = carry
        o, a, ol, ad, rt, mm, bm = (obs_f[mb], acts_f[mb], old_f[mb],
                                    adv_norm[mb], ret_f[mb], mm_f[mb], bm_f[mb])
        st = st_f[mb]
        if obs_q:
            o = o.astype(jnp.float32) / 255.0   # 延迟反量化（minibatch 级）
            st = st.astype(jnp.float32) / 255.0

        loss_val, grads = jax.value_and_grad(
            lambda p: _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt,
                                clip_eps, vf_coef, ent_coef))(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    def _sync(x):
        """pmean 全量同步（可选 bf16 半精度传输）。axis_name=None 时 no-op。
        bf16 只作用于 fp32 叶子；int 叶子（Adam 的 count）各副本本就相同，
        直接透传（pmean 的 1/axis_size 缩放会把 int 变 float）。"""
        if axis_name is None:
            return x
        if bf16_sync:
            x = jax.tree.map(
                lambda t: (t.astype(jnp.bfloat16)
                           if t.dtype == jnp.float32 else t), x)
        x = jax.tree.map(
            lambda t: (jax.lax.pmean(t, axis_name)
                       if t.dtype in (jnp.float32, jnp.bfloat16) else t), x)
        if bf16_sync:
            x = jax.tree.map(
                lambda t: (t.astype(jnp.float32)
                           if t.dtype == jnp.bfloat16 else t), x)
        return x

    last_loss = None
    for _ in range(epochs):
        key, ek = jrandom.split(key)
        ka, kc = jrandom.split(ek)
        actor = actor_idx[jrandom.permutation(ka, n_keep)]
        critic = jrandom.permutation(kc, total)[:n_keep]
        idx = jnp.concatenate([actor.reshape(-1, mb_half),
                               critic.reshape(-1, mb_half)], axis=1)
        n_mb = idx.shape[0]
        n_full, rem = divmod(n_mb, sync_k)
        chunk_means = []

        if n_full > 0:
            idx_c = idx[:n_full * sync_k].reshape(n_full, sync_k, mb_half * 2)

            def run_chunk(carry, idx_k):
                """sync_k 个本地 minibatch + 一次全量同步（scan 内嵌套 scan，
                同步在 chunk 边界执行 —— 不依赖 cond 内的 collective）。"""
                (params, opt_state), losses = jax.lax.scan(
                    local_step, carry, idx_k)
                params = _sync(params)
                if sync_state:
                    opt_state = _sync(opt_state)
                return (params, opt_state), jnp.mean(losses)

            (params, opt_state), cl = jax.lax.scan(
                run_chunk, (params, opt_state), idx_c)
            chunk_means.append(cl)
        if rem > 0:
            # 末尾不足 sync_k 个的余段：本地更新完也同步一次
            (params, opt_state), losses = jax.lax.scan(
                local_step, (params, opt_state), idx[n_full * sync_k:])
            params = _sync(params)
            if sync_state:
                opt_state = _sync(opt_state)
            chunk_means.append(jnp.mean(losses)[None])
        last_loss = jnp.mean(jnp.concatenate(chunk_means))
    if return_loss:
        return params, opt_state, last_loss
    return params, opt_state


def ppo_update_gradsync(params, opt, opt_state, arch, batch, key, minibatch,
                        clip_eps, vf_coef, ent_coef, epochs, axis_name=None,
                        sync_k=128, bf16_sync=False, return_loss=False,
                        adv_top_frac=0.25):
    """梯度累积 + 周期同步（大 batch 实现）——"平均梯度之和"线性性的精确版：

    每 sync_k 个 minibatch 拼成一个 sync_k× 大 minibatch，对其做一次
    value_and_grad（∇(1/K Σᵢ lossᵢ) = (1/K) Σᵢ ∇lossᵢ，与"冻结参数逐
    个累加梯度"数学同义）、一次 pmean 平均梯度、一次 opt.update。参数
    任何时刻逐位一致（更新只发生在同步点、从同一平均梯度出发），零
    漂移、零本地 Adam 分歧。代价：每迭代只有 n_mb/sync_k 次更新
    （大 batch 效应）——K=128 → 8 次/迭代，K=256 → 4 次/迭代。

    大 batch 实现比"scan 内逐 minibatch 累加梯度"快：后者在 DCU 上
    实测比 baseline 慢 2.4×（小 GEMM + 累加链编译差），前者单次大
    前向/反向（大 GEMM 效率更高）。激活内存随 batch 线性增长（64GB
    卡实测 131K 样本 OOM 78GB），故 big=sync_k×minibatch 超过
    GRAD_MAX_SAMPLES=65536 时直接报错：grad 模式限 K ≤ ~32-64
    （视 minibatch），更大 K 请用 param 模式（无内存限制）。

    sync_k=1 时与 ppo_update 逐位一致（big=minibatch，结构相同），
    本函数是现状无损路径的严格超集。
    """
    obs, state, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    adv_f = advs.reshape(-1)
    adv_norm = jnp.where(adv_f == 0.0, 0.0, (adv_f - jnp.mean(adv_f)) / (jnp.std(adv_f) + 1e-8))
    mb_half = max(minibatch // 2, 1)
    n_keep = max((int(total * adv_top_frac) // mb_half) * mb_half,
                 mb_half) if 0.0 < adv_top_frac < 1.0 else total
    n_keep = min(n_keep, total)
    adv_for_topk = jnp.where(adv_f == 0.0, -1e9, jnp.abs(adv_norm))
    _, actor_idx = jax.lax.top_k(adv_for_topk, n_keep) \
        if 0.0 < adv_top_frac < 1.0 else (jnp.zeros((0,)), jnp.arange(total))
    obs_q = (obs.dtype == jnp.uint8)
    obs_f = obs.reshape(total, *obs.shape[2:])
    st_f = state.reshape(total, -1)
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks[:2]
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)
    big = sync_k * minibatch
    # 大 batch 激活内存随样本数线性增长（64GB DCU 实测 131K 样本 OOM 78GB、
    # 32K 样本 ~20GB）。拆 sub-batch 无济于事（scan 累加版慢 2.4×、Python
    # 展开版不共享缓冲区），所以直接护栏：超限给清晰报错，大 K 请用 param
    # 模式（无此限制）。
    GRAD_MAX_SAMPLES = 65536
    if big > GRAD_MAX_SAMPLES:
        raise SystemExit(
            f"grad 模式内存护栏：sync_k×minibatch = {big} 样本 > "
            f"{GRAD_MAX_SAMPLES}（64GB DCU 实测 131K 样本 OOM）。"
            f"请减小 --lsgd-k（本配置 K ≤ {GRAD_MAX_SAMPLES // minibatch}）"
            f"或改用 param 模式（--lsgd-mode param，无此限制）")

    def update_big(carry, mb):
        """一个 sync_k×minibatch 的大批：单次 value_and_grad → pmean → update。
        mb 是 (big,) 的行下标（scan 内静态长度切片）。"""
        params, opt_state = carry
        o, a, ol, ad, rt, mm, bm = (obs_f[mb], acts_f[mb], old_f[mb],
                                    adv_norm[mb], ret_f[mb], mm_f[mb], bm_f[mb])
        st = st_f[mb]
        if obs_q:
            o = o.astype(jnp.float32) / 255.0
            st = st.astype(jnp.float32) / 255.0
        loss_val, g = jax.value_and_grad(
            lambda p: _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt,
                                clip_eps, vf_coef, ent_coef))(params)
        if axis_name is not None:
            if bf16_sync:
                g = jax.tree.map(
                    lambda t: (t.astype(jnp.bfloat16)
                               if t.dtype == jnp.float32 else t), g)
            g = jax.tree.map(
                lambda t: (jax.lax.pmean(t, axis_name)
                           if t.dtype in (jnp.float32, jnp.bfloat16) else t),
                g)
            if bf16_sync:
                g = jax.tree.map(
                    lambda t: (t.astype(jnp.float32)
                               if t.dtype == jnp.bfloat16 else t), g)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    last_loss = None
    for _ in range(epochs):
        key, ek = jrandom.split(key)
        ka, kc = jrandom.split(ek)
        actor = actor_idx[jrandom.permutation(ka, n_keep)]
        critic = jrandom.permutation(kc, total)[:n_keep]
        idx = jnp.concatenate([actor.reshape(-1, mb_half),
                               critic.reshape(-1, mb_half)], axis=1)
        n_mb = idx.shape[0]
        n_full, rem = divmod(n_mb, sync_k)
        chunk_means = []

        if n_full > 0:
            idx_big = idx[:n_full * sync_k].reshape(n_full, big)
            (params, opt_state), cl = jax.lax.scan(
                update_big, (params, opt_state), idx_big)
            chunk_means.append(cl)
        if rem > 0:
            # 末尾不足 sync_k 个 minibatch：剩余样本拼成一个大批更新一次
            idx_rem = idx[n_full * sync_k:].reshape(-1)
            (params, opt_state), lv = update_big((params, opt_state), idx_rem)
            chunk_means.append(lv[None])
        last_loss = jnp.mean(jnp.concatenate(chunk_means))
    if return_loss:
        return params, opt_state, last_loss
    return params, opt_state


# ---------------- 离线蒸馏（--distill-data） ----------------

NEG = jnp.finfo(jnp.float32).min   # 非法动作 logits 占位（与 torch masked_dist 同语义）


def load_distill_data(pattern: str, max_frames: int):
    """加载 collect_distill 的 npz（每文件 obs7 (T,2,7,13,13) uint8×255 /
    logits (T,2,7) fp32 / move_mask (T,2,5) / bomb_mask (T,2,2)）。

    展平成帧（env×view → F）：每局面 2 视角各 1 帧、各自 logits/masks。
    超过 max_frames 时按文件 stride 均匀抽稀（保住混合地图构成比例）。
    返回 (obs_u8 (F,7,H,W) uint8 host, teacher_probs (F,7) fp32 host,
          move_mask (F,5), bomb_mask (F,2))。
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"--distill-data {pattern} 无文件")
    total = 0
    parts = []
    for p in paths:
        d = np.load(p)
        o = d["obs7"]                      # (T,2,7,H,W)
        lg = d["logits"]                   # (T,2,7)
        mm = d["move_mask"]                # (T,2,5)
        bm = d["bomb_mask"]                # (T,2,2)
        F = o.shape[0] * o.shape[1]
        total += F
        parts.append((o, lg, mm, bm))
    print(f"distill: {len(paths)} 文件共 {total:,} 帧，上限 {max_frames:,}", flush=True)
    if total > max_frames:
        keep = max_frames / total
        for i, (o, lg, mm, bm) in enumerate(parts):
            stride = max(1, int(1.0 / keep))
            s = o.reshape(-1, *o.shape[2:])[::stride]
            parts[i] = (s, lg.reshape(-1, lg.shape[-1])[::stride],
                        mm.reshape(-1, mm.shape[-1])[::stride],
                        bm.reshape(-1, bm.shape[-1])[::stride])
    obs = np.concatenate([p[0] for p in parts]).astype(np.uint8)
    lg = np.concatenate([p[1] for p in parts]).astype(np.float32)
    mm = np.concatenate([p[2] for p in parts]).astype(np.bool_)
    bm = np.concatenate([p[3] for p in parts]).astype(np.bool_)
    # teacher 概率目标：logits 已 mask（非法 = -inf/finfo.min）→ softmax 只在合法上归一
    lg = np.where(lg <= -1e30, -np.inf, lg).astype(np.float32)
    ex = np.exp(lg - lg.max(-1, keepdims=True))
    p_t = (ex / ex.sum(-1, keepdims=True)).astype(np.float32)
    F = obs.shape[0]
    print(f"distill: 实际 {F:,} 帧 obs={obs.shape} probs={p_t.shape} "
          f"(≈{obs.nbytes/1e6:.0f}MB host)", flush=True)
    return obs, p_t, mm, bm


def build_distill_update(params, opt, opt_state, arch, batch, ent_coef):
    """离线蒸馏 one update（jitted）：从数据采样一批帧 → mask 后 logits →
    KL(student || teacher)（teacher 概率做目标）+ 熵奖励。返回 jitted fn。"""

    @jax.jit
    def upd(params, opt_state, obs, p_t, mm, bm, key):
        def loss_fn(p):
            mv, bmv, _v, _vl = net_forward(p, arch, obs)
            mv = jnp.where(mm, mv, jnp.full_like(mv, NEG))
            bmv = jnp.where(bm, bmv, jnp.full_like(bmv, NEG))
            lsm = jax.nn.log_softmax(mv)
            lsb = jax.nn.log_softmax(bmv)
            # KL = sum p_t(log p_t - log p_s)；H(teacher) 对参数恒量，最小化 CE 即可。
            # p_t 只在合法动作上归一，student log_softmax 也只在合法上 —— 一致。
            ce_m = -(p_t[:, :5] * lsm).sum(-1).mean()
            ce_b = -(p_t[:, 5:] * lsb).sum(-1).mean()
            pm = jnp.exp(lsm)
            pb = jnp.exp(lsb)
            ent = (-(pm * jnp.where(pm > 0, lsm, 0.0)).sum(-1).mean()
                   - (pb * jnp.where(pb > 0, lsb, 0.0)).sum(-1).mean())
            return ce_m + ce_b - ent_coef * ent

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return upd


def run_distill(params, opt, opt_state, args, key):
    """离线蒸馏阶段：--distill-iters 轮，每轮采样 --distill-batch 帧做一次
    KL 更新（host→device 分批搬，数据可远大于显存）。返回 (params, opt_state)。"""
    obs_u8, p_t, mm, bm = load_distill_data(args.distill_data,
                                            args.distill_max_frames)
    F = obs_u8.shape[0]
    upd = build_distill_update(params, opt, opt_state, args.arch,
                               args.distill_batch, args.ent_coef)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(args.distill_iters):
        idx = rng.integers(0, F, args.distill_batch)
        o = jnp.asarray(obs_u8[idx]).astype(jnp.float32) / 255.0
        pt = jnp.asarray(p_t[idx])
        m = jnp.asarray(mm[idx])
        b = jnp.asarray(bm[idx])
        key, k = jrandom.split(key)
        t1 = time.time()
        params, opt_state = upd(params, opt_state, o, pt, m, b, k)
        jax.block_until_ready(params)
        dt = time.time() - t1
        if it % 10 == 0 or it == args.distill_iters - 1:
            print(f"[distill {it}] {dt*1000:.0f}ms/批 "
                  f"({args.distill_batch*1000/max(dt,1e-6):,.0f} 帧/s)", flush=True)
    print(f"distill 完成：{args.distill_iters} 轮 × {args.distill_batch:,} 帧，"
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return params, opt_state


# ---------------- checkpoint（蒸馏产物 / PPO 续跑） ----------------

def save_params(params, path: str) -> None:
    """保存 params（嵌套 numpy 数组 pytree）到 pickle。device_get 先搬回 host。"""
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(f"params 已保存 -> {path}", flush=True)


def save_run_metadata(path: str, args) -> None:
    if not IS_BUN:
        return
    metadata_path = f"{os.path.splitext(path)[0]}.json"
    metadata = {
        "rule": "bun",
        "qqt_map_id": 806,
        "observation_channels": N_OBS_CH,
        "move_actions": N_MOVES,
        "ability_actions": N_BOMB,
        "ability_encoding": {"0": "none", "1": "bomb", "2": "use_item"},
        "max_steps": MAX_STEPS,
        "architecture": args.arch,
        "embed": args.embed,
        "depth": args.depth,
        "seed": getattr(args, "seed", 0),
        "bun_curriculum": getattr(args, "bun_curriculum", None),
        "bun_hp": getattr(args, "bun_hp", None),
        "bun_kill_window_reward": getattr(
            args, "bun_kill_window_reward", False),
        "bun_reward_profile": getattr(args, "bun_reward_profile", "legacy"),
        "bun_reward_v2": {
            name: getattr(args, f"bun_{name}", None) for name in (
                "base_bomb_reward", "enemy_threat_reward",
                "tactical_bomb_placement_reward", "forced_kill_reward",
                "tactical_bomb_resolution_reward", "danger_escape_reward",
                "avoidable_danger_death_penalty")},
        "flee_bot_ratio": getattr(args, "flee_bot_ratio", None),
        "load": getattr(args, "load", None),
        "bun_start_state_bank": getattr(args, "bun_start_state_bank", None),
        "bun_start_state_weights": getattr(
            args, "bun_start_state_weights", ""),
        "bun_spawn_buckets": getattr(args, "bun_spawn_buckets", ""),
        "ppo_iterations": getattr(args, "iters", None),
        "warmup_updates_committed": False,
        "safety_mode": getattr(args, "safety_mode", "off"),
        "safety_penalty": getattr(args, "safety_penalty", 0.0),
        "safety_margin": getattr(args, "safety_margin", 0),
        "safety_kl_coef": getattr(args, "safety_kl_coef", 0.0),
        "safety_reference": getattr(args, "safety_reference", None),
        "bun_opponent_pool": getattr(args, "bun_opponent_pool", ""),
        "bun_opponent_weak": getattr(args, "bun_opponent_weak", None),
        "bun_opponent_old": getattr(args, "bun_opponent_old", None),
        "bun_opponent_recent": getattr(args, "bun_opponent_recent", None),
        "bun_opponent_history": getattr(args, "bun_opponent_history", None),
        "flee_bot_ratio": getattr(args, "flee_bot_ratio", 0.0),
        "jax_bot_pool": getattr(args, "jax_bot_pool", ""),
        "jax_bot_adaptive": getattr(args, "jax_bot_adaptive", False),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_params(path: str):
    """从 pickle 加载 params 并放到设备（用于蒸馏初始权重 / PPO 续跑）。"""
    with open(path, "rb") as f:
        params = pickle.load(f)
    if isinstance(params, dict) and "params" in params:
        params = params["params"]
    return jax.device_put(params)


# ---------------- one_iter（可复用，probe 直接测训练主循环） ----------------


def _lsgd_updater(args):
    """按 --lsgd-k/--lsgd-mode 选有损同步函数；--lsgd-k 0 = 返回 None
    （现状无损路径：每个 minibatch pmean 梯度）。"""
    if getattr(args, "lsgd_k", 0) <= 0:
        return None
    if getattr(args, "lsgd_mode", "param") == "grad":
        return ppo_update_gradsync
    return ppo_update_lsgd


def build_one_iter(params, opt, opt_state, states, key, args,
                   reference_params=None, bun_opponent_pool=None,
                   jax_bot_enabled=None):
    """返回 jitted one_iter：(params, opt_state, states, key) -> 同形四元组。

    与训练主循环完全同构：collect_rollout → bootstrap → GAE → PPO update。
    jax_bot_enabled（静态 bool 掩码）给定时签名变为
    (params, opt_state, states, key, bot_weights) -> (..., bot_stats)：
    bot_weights 是 traced 输入，自适应课程改权重不触发重编译。
    """
    n = states.pos.shape[0]
    steps = args.num_steps
    use_pool = jax_bot_enabled is not None

    def one_iter(params, opt_state, states, key, bot_weights=None):
        rollout = collect_rollout(
            params, args.arch, states, key, steps,
            getattr(args, "no_mask", False),
            getattr(args, "obs_quant", False),
            getattr(args, "checkpoint", False),
            flee_bot_ratio=getattr(args, "flee_bot_ratio", 0.0),
            safety_mode=getattr(args, "safety_mode", "off"),
            safety_penalty=getattr(args, "safety_penalty", 0.0),
            safety_margin=getattr(args, "safety_margin", 0),
            bun_opponent_weights=(None if bun_opponent_pool is None
                                  else bun_opponent_pool[0]),
            bun_opponent_weak_params=(None if bun_opponent_pool is None
                                      else bun_opponent_pool[1]),
            bun_opponent_old_params=(None if bun_opponent_pool is None
                                     else bun_opponent_pool[2]),
            bun_opponent_recent_params=(None if bun_opponent_pool is None
                                        else bun_opponent_pool[3]),
            jax_bot_pool=bot_weights if use_pool else None,
            jax_bot_enabled=jax_bot_enabled,
            return_bot_stats=use_pool)
        states, batch = rollout[0], rollout[1]
        obs, state, acts, lps, vals, rew, done, masks = batch
        # bootstrap：rollout 尾部状态价值（全局状态向量同步传入）
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        fstate = both_states(states)
        fkey = jrandom.split(key)[0]
        if bun_opponent_pool is not None:
            obs, state, acts, lps, vals, rew, done = (
                obs[:, :n], state[:, :n], acts[:, :n], lps[:, :n],
                vals[:, :n], rew[:, :n], done[:, :n])
            masks = tuple(mask[:, :n] for mask in masks)
            _, _, fval = sample_actions(
                params, args.arch, fobs[:n],
                (fmasks[0][:n], fmasks[1][:n]), fkey, state=fstate[:n])
        else:
            _, _, fval = sample_actions(params, args.arch, fobs, fmasks, fkey,
                                        state=fstate)
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, args.gamma, args.lam)
        rets = advs + vals
        if bun_opponent_pool is None:
            advs, rets = mask_bot_advantages(
                advs, rets, vals, n, getattr(args, "flee_bot_ratio", 0.0))
        upd = _lsgd_updater(args)
        if upd is None:
            params, opt_state = ppo_update(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, args.minibatch, args.clip_eps, args.vf_coef,
                args.ent_coef, args.epochs,
                safety_mode=getattr(args, "safety_mode", "off"),
                safety_penalty=getattr(args, "safety_penalty", 0.0),
                reference_params=reference_params,
                reference_kl_coef=getattr(args, "safety_kl_coef", 0.0))
        else:
            kw = dict(sync_k=args.lsgd_k,
                      bf16_sync=getattr(args, "lsgd_bf16", False))
            if upd is ppo_update_lsgd:
                kw["sync_state"] = getattr(args, "lsgd_sync_state", False)
            params, opt_state = upd(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, args.minibatch, args.clip_eps, args.vf_coef,
                args.ent_coef, args.epochs, **kw)
        key = jrandom.split(key)[0]
        if use_pool:
            return params, opt_state, states, key, rollout[4]
        return params, opt_state, states, key

    return jax.jit(one_iter)


def compile_warmup(one_iter_j, params, opt_state, states, key, repeats=2,
                   extra=()):
    """Compile and exercise one iteration without committing training state."""
    warm_params = params
    warm_opt_state = opt_state
    warm_states = states
    warm_key = key
    for _ in range(repeats):
        out = one_iter_j(warm_params, warm_opt_state, warm_states, warm_key,
                         *extra)
        warm_params, warm_opt_state, warm_states, warm_key = out[:4]
    jax.block_until_ready(warm_params)


class JaxBotCurriculum:
    """Host-side per-tier outcome tracker + optional adaptive pool weights.

    stats rows = [ticks, learner surviving kills, learner deaths, self-kills].
    Adaptive mode keeps the learner where it still gets signal: a tier's
    weight is base * (floor + 4·p·(1-p)), p = EMA of kills/(kills+deaths),
    so mastered (p→1) and hopeless (p→0) tiers fade but never vanish.
    """

    PER = 300.0   # 汇报口径：每 300 tick（= danger_arena 一局）

    def __init__(self, base_weights, adaptive=False, ema=0.9, floor=0.1):
        self.base = np.asarray(base_weights, np.float64)
        self.adaptive = adaptive
        self.ema = ema
        self.floor = floor
        self.win = np.full(self.base.shape, 0.5)
        self.totals = np.zeros((self.base.shape[0], 4))

    def weights(self):
        if not self.adaptive:
            return (self.base / self.base.sum()).astype(np.float32)
        w = self.base * (self.floor + 4.0 * self.win * (1.0 - self.win))
        return (w / w.sum()).astype(np.float32)

    def update(self, stats):
        stats = np.asarray(stats, np.float64)
        self.totals += stats
        events = stats[:, 1] + stats[:, 2]
        seen = events > 0
        rate = np.where(seen, stats[:, 1] / np.maximum(events, 1), self.win)
        self.win = np.where(seen, self.ema * self.win + (1 - self.ema) * rate,
                            self.win)

    def summary(self, stats):
        stats = np.asarray(stats)
        parts = []
        for i, name in enumerate(JAX_BOT_NAMES):
            if self.base[i] <= 0:
                continue
            ep = stats[i, 0] / self.PER
            if ep > 0:
                parts.append(f"{name}:ep={ep:.0f} kill={stats[i,1]/ep:.2f} "
                             f"death={stats[i,2]/ep:.2f} "
                             f"self={stats[i,3]/ep:.2f} p={self.win[i]:.2f}")
            else:
                parts.append(f"{name}:ep=0")
        if self.adaptive:
            w = self.weights()
            parts.append("w=" + ",".join(
                f"{JAX_BOT_NAMES[i]}={w[i]:.2f}"
                for i in range(len(w)) if self.base[i] > 0))
        return " | ".join(parts)


def build_dp_one_iter(params, opt, opt_state, states, key, args, n_dev,
                      jax_bot_enabled=None):
    """数据并行（DP）one_iter：每卡 envs 切片独立 collect，更新时梯度
    pmean allreduce。n_dev 为卡数，states/key 首维须为 n_dev（pmap 切片）。

    与 build_one_iter 逐位一致的条件：
      - 每卡 minibatch = args.minibatch // n_dev（等效全局 minibatch 不变
        → 梯度步数不变，样本不重叠）
      - states 按卡切片后每卡独立 rollout（collect 侧零通信）
      - 梯度 pmean 后每卡应用相同更新 → 参数逐卡一致
    n_dev=1 时退化为 build_one_iter 语义（pmap 单设备）。
    """
    steps = args.num_steps
    mb_local = args.minibatch // n_dev
    assert mb_local >= 1, "minibatch 必须 >= 卡数"
    use_pool = jax_bot_enabled is not None

    def one_iter_shard(params, opt_state, states, key, bot_weights=None):
        rollout = collect_rollout(
            params, args.arch, states, key, steps,
            getattr(args, "no_mask", False),
            getattr(args, "obs_quant", False),
            getattr(args, "checkpoint", False),
            flee_bot_ratio=getattr(args, "flee_bot_ratio", 0.0),
            jax_bot_pool=bot_weights if use_pool else None,
            jax_bot_enabled=jax_bot_enabled,
            return_bot_stats=use_pool)
        states, batch = rollout[0], rollout[1]
        obs, state, acts, lps, vals, rew, done, masks = batch
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        fstate = both_states(states)
        fkey = jrandom.split(key)[0]
        _, _, fval = sample_actions(params, args.arch, fobs, fmasks, fkey,
                                    state=fstate)
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, args.gamma, args.lam)
        rets = advs + vals
        advs, rets = mask_bot_advantages(advs, rets, vals, states.pos.shape[0], getattr(args, "flee_bot_ratio", 0.0))
        upd = _lsgd_updater(args)
        if upd is None:
            params, opt_state = ppo_update(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, mb_local, args.clip_eps, args.vf_coef, args.ent_coef,
                args.epochs, axis_name="dev")
        else:
            kw = dict(sync_k=args.lsgd_k,
                      bf16_sync=getattr(args, "lsgd_bf16", False))
            if upd is ppo_update_lsgd:
                kw["sync_state"] = getattr(args, "lsgd_sync_state", False)
            params, opt_state = upd(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, mb_local, args.clip_eps, args.vf_coef, args.ent_coef,
                args.epochs, axis_name="dev", **kw)
        key = jrandom.split(key)[0]
        if use_pool:
            return params, opt_state, states, key, rollout[4]
        return params, opt_state, states, key

    if use_pool:
        return jax.pmap(one_iter_shard, axis_name="dev",
                        in_axes=(None, None, 0, 0, None))
    return jax.pmap(one_iter_shard, axis_name="dev",
                    in_axes=(None, None, 0, 0))


def _unreplicate(tree):
    """Drop the leading device axis (params identical across devices post-pmean)."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def run_dp_training(params, opt, opt_state, args, key, devs,
                    bun_opponent_pool, reference_params, bot_curriculum=None):
    """pmap 数据并行主循环：每卡 num_envs/devices 个 env，梯度 pmean 同步。

    build_dp_one_iter 的 pmap in_axes=(None,None,0,0)：params/opt_state 广播、
    states/key 按卡切片。每次 one_iter 返回值带 device 轴，参数经 pmean 逐卡
    一致，故下一轮前用 _unreplicate 取 [0] 复原为无设备轴（再次广播）。
    """
    n_dev = args.devices
    if len(devs) < n_dev:
        raise SystemExit(f"--devices {n_dev} 超过可用设备数 {len(devs)}")
    if args.num_envs % n_dev != 0:
        raise SystemExit(f"--num-envs {args.num_envs} 必须能被 --devices {n_dev} 整除")
    if args.minibatch % n_dev != 0:
        raise SystemExit(f"--minibatch {args.minibatch} 必须能被 --devices {n_dev} 整除")
    if bun_opponent_pool is not None:
        raise SystemExit("DP（--devices>1）暂不支持 --bun-opponent-pool（DP 路径仅 flee-bot 自对弈）")
    if getattr(args, "safety_mode", "off") != "off" or reference_params is not None:
        raise SystemExit("DP（--devices>1）暂不支持 safety / reference-KL")

    n_per = args.num_envs // n_dev
    steps = args.num_steps
    key, sk, ik = jrandom.split(key, 3)
    states = jax.vmap(lambda k: init_batch(k, n_per))(jrandom.split(sk, n_dev))
    iter_keys = jrandom.split(ik, n_dev)
    print(f"DP: devices={n_dev} envs/dev={n_per} (total={args.num_envs}) "
          f"minibatch/dev={args.minibatch // n_dev} "
          f"flee_bot_ratio={getattr(args, 'flee_bot_ratio', 0.0)}", flush=True)

    one_iter_dp = build_dp_one_iter(
        params, opt, opt_state, states, iter_keys, args, n_dev,
        jax_bot_enabled=(None if bot_curriculum is None
                         else bot_curriculum.base > 0))

    def extra():
        if bot_curriculum is None:
            return ()
        return (jnp.asarray(bot_curriculum.weights()),)

    # warmup（首次编译，不计入训练更新）
    t0 = time.time()
    wp = one_iter_dp(params, opt_state, states, iter_keys, *extra())[0]
    jax.block_until_ready(wp)
    print(f"warmup done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    rollout_agents = 2                       # 自对弈：两席都产出可训练样本
    offset = int(getattr(args, "iter_offset", 0))   # 续跑全局步偏移
    for it in range(args.iters):
        t1 = time.time()
        out = one_iter_dp(params, opt_state, states, iter_keys, *extra())
        params, opt_state, states, iter_keys = out[:4]
        if bot_curriculum is not None:
            bot_stats = np.asarray(jax.device_get(out[4])).sum(axis=0)
            bot_curriculum.update(bot_stats)
            print(f"  bots: {bot_curriculum.summary(bot_stats)}", flush=True)
        params = _unreplicate(params)
        opt_state = _unreplicate(opt_state)
        jax.block_until_ready(params)
        dt = time.time() - t1
        sps = rollout_agents * args.num_envs * steps / dt
        gstep = it + offset
        if args.save and args.save_every and it and gstep % args.save_every == 0:
            mid = f"{os.path.splitext(args.save)[0]}_it{gstep}.pt"
            save_params(params, mid)
            save_run_metadata(mid, args)
        print(f"[iter {gstep}] {dt:.2f}s  sps={sps:,.0f}", flush=True)
    tot = rollout_agents * args.num_envs * steps * args.iters / (time.time() - t0)
    print(f"FINAL end-to-end sps = {tot:,.0f} "
          f"({rollout_agents*args.num_envs*steps*args.iters:,} trainable-agent steps)",
          flush=True)
    if args.save:
        save_params(params, args.save)
        save_run_metadata(args.save, args)


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp4",
                    choices=["mlp", "mlp_bf16", "mlp4", "cnn", "transformer"])
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=None,
                    help="隐藏层宽。不传时按 arch 选：mlp=256 / mlp4=768 / cnn=256")
    ap.add_argument("--embed", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995,
                    help="生产对齐：一局最长 1800 tick，折扣要够长才看得到终局奖励")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safety-mode", choices=["off", "hard", "soft"],
                    default="off")
    ap.add_argument("--safety-penalty", type=float, default=6.0)
    ap.add_argument("--safety-margin", type=int, default=0)
    ap.add_argument("--safety-kl-coef", type=float, default=0.0)
    ap.add_argument("--safety-reference", default=None)
    ap.add_argument("--levels", default=None,
                    help="关卡 JSON；bun_train 默认使用 web/assets/maps/levels.json 并强制地图 806")
    ap.add_argument("--level-weights", default="",
                    help="普通模式关卡权重；bun_train 忽略并只训练地图 806")
    ap.add_argument("--flee-bot-ratio", type=float,
                    default=0.0 if IS_BUN else 0.20,
                    help="规则 Bot 环境比例；抢包子默认关闭，保持纯自博弈")
    ap.add_argument(
        "--jax-bot-pool", default="",
        help="分级 JAX 规则 bot 池（替代 flee-bot 席位，需 --flee-bot-ratio>0）："
             "dodge_easy/dodge/bomber_easy/hunter/hunter_hard/legacy_flee 权重，"
             "如 dodge=1,bomber_easy=1,hunter=2")
    ap.add_argument("--jax-bot-adaptive", action="store_true",
                    help="按学习者对各难度的 EMA 胜率自适应调权重 "
                         "(w∝base·(0.1+4p(1-p)))，聚焦仍有信号的难度")
    ap.add_argument(
        "--bun-opponent-pool", default="",
        help="真实非对称 P1 对手池：idle/roam/rule_combat/weak/old/recent 权重")
    ap.add_argument("--bun-opponent-weak", default=None,
                    help="冻结较弱历史 checkpoint，仅用于 opponent 前向")
    ap.add_argument("--bun-opponent-old", default=None,
                    help="冻结较旧 checkpoint，仅用于 opponent 前向")
    ap.add_argument("--bun-opponent-recent", default=None,
                    help="冻结最近 checkpoint，仅用于 opponent 前向")
    ap.add_argument("--bun-opponent-history", default=None,
                    help="兼容旧脚本；等价于 --bun-opponent-recent")
    ap.add_argument("--bun-start-state-bank", default=None)
    ap.add_argument("--bun-start-state-weights", default="")
    ap.add_argument(
        "--bun-spawn-buckets", default="",
        help="danger_arena 出生位置分桶权重（空 = 原分布）："
             "native/near/mid/far/below/upper_left/upper_right，"
             "如 native=0.3,near=0.1,mid=0.15,far=0.1,below=0.15,"
             "upper_left=0.1,upper_right=0.1")
    ap.add_argument("--bun-reward-profile",
                    choices=["legacy", "auto_sparse", "combat_evolution",
                             "danger_arena"],
                    default="legacy")
    # Reward V2 安全进攻塑形系数（仅 danger_arena profile 生效）。默认 0
    # 保持与旧 danger_arena 逐位一致；非 0 时在设备端 scan 内打开 tactical
    # bomb 追踪 / forces_kill 分析。
    ap.add_argument("--bun-danger-escape-reward", type=float, default=0.75)
    ap.add_argument("--bun-avoidable-danger-death-penalty", type=float,
                    default=4.0)
    ap.add_argument("--bun-tactical-bomb-placement-reward", type=float,
                    default=0.0)
    ap.add_argument("--bun-tactical-bomb-resolution-reward", type=float,
                    default=0.0)
    ap.add_argument("--bun-base-bomb-reward", type=float, default=0.0)
    ap.add_argument("--bun-forced-kill-reward", type=float, default=0.0)
    ap.add_argument("--bun-enemy-threat-reward", type=float, default=0.0)
    ap.add_argument(
        "--bun-curriculum",
        default="carry_home=0.4,near_steal=0.3,route_break=0.2,full=0.1",
        help="Bun reset 课程权重；支持 full/carry_home/near_steal/"
             "route_break/route_to_base/combat/combat_static/"
             "combat_moving/combat_kill/route_to_base_relaxed/route_capture/"
             "carry_return/full_contact/full_ambush")
    ap.add_argument("--bun-hp", type=int, default=1,
                    help="Bun 训练初始/复活 HP；默认与网页推理一致为 1")
    ap.add_argument(
        "--bun-kill-window-reward", action="store_true",
        help="Full 规则按唯一 owner 击杀→复活窗口突进→携包返程分阶段奖励")
    # ---- 离线蒸馏（--distill-data 提供则先跑蒸馏，再可选接 PPO）----
    ap.add_argument("--distill-data", default=None,
                    help="collect_distill 的 npz（支持 glob）。给定时先跑蒸馏")
    ap.add_argument("--distill-iters", type=int, default=200)
    ap.add_argument("--distill-batch", type=int, default=8192,
                    help="每轮采样的帧数（host→device 分批搬，可远小于总帧数）")
    ap.add_argument("--distill-max-frames", type=int, default=2_000_000,
                    help="总帧数上限（超出按文件 stride 均匀抽稀）")
    ap.add_argument("--distill-then-ppo", action="store_true",
                    help="蒸馏后继续自对弈 PPO 微调（不传则蒸馏完即停）")
    ap.add_argument("--no-mask", action="store_true",
                    help="性能 A/B：mask 全放开（行为=无 mask 旧版）")
    ap.add_argument("--obs-quant", action="store_true",
                    help="obs buffer 存 uint8（×255 量化，反量化进网络）："
                         "45GB→11GB，8192×512 OOM 的解法（精度 1/255 优于 "
                         "bf16 尾数，低精度 obs 通道无损）")
    ap.add_argument("--checkpoint", action="store_true",
                    help="scan body 用 jax.checkpoint：反向重算中间量，"
                         "配合 --obs-quant 让 8192×512 放下（省 scan "
                         "中间量 44.8GB），代价是 collect 变慢（重算）")
    # ---- Local SGD（有损同步，跨机降通信）----
    ap.add_argument("--lsgd-k", type=int, default=0,
                    help="Local SGD 同步周期：每 K 个 minibatch 同步一次参数"
                         "（0=现状：每个 minibatch pmean 梯度，逐位一致）。"
                         ">0 时 minibatch 循环内零通信、每 K 步 pmean 参数，"
                         "通信量降到 ~1/K。20 卡（10 机×2 卡）下 K=256 ≈ "
                         "4 次同步/迭代 ≈ 0.5-1.5s，K=128 ≈ 8 次 ≈ 1-3s")
    ap.add_argument("--lsgd-mode", default="param",
                    choices=["param", "grad"],
                    help="Local SGD 同步对象：param=每 K 步平均参数（保持 "
                         "1024 次更新/迭代，代价是 K 步本地漂移）；grad=冻结"
                         "参数上累加 K 个梯度、一次平均梯度同步、一次更新"
                         "（零漂移、参数始终逐位一致，代价是只有 1024/K 次"
                         "更新/迭代；K=1 时与现状逐位一致）")
    ap.add_argument("--lsgd-bf16", action="store_true",
                    help="Local SGD 同步时用 bf16 半精度传输（流量减半，"
                         "尾数损失可忽略）")
    ap.add_argument("--lsgd-sync-state", action="store_true",
                    help="Local SGD 同步时连 Adam 动量/方差一起平均"
                         "（防本地漂移，流量×3）")
    # ---- checkpoint ----
    ap.add_argument("--load", default=None,
                    help="初始权重 pickle（蒸馏出的 student / 续跑）")
    ap.add_argument("--save", default=None,
                    help="训练结束时保存 params 的路径")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 迭代存一次中间 ckpt（0=不存）。文件名 = "
                         "--save 去掉扩展名 + _it{N}，供中途评估/续跑")
    ap.add_argument("--devices", type=int, default=1,
                    help="数据并行卡数：>1 走 pmap DP（每卡 num-envs/devices 个 "
                         "env，梯度 pmean allreduce，参数逐卡一致）。仅支持 "
                         "flee-bot 自对弈（--flee-bot-ratio），与 opponent-pool/"
                         "safety/distill 互斥")
    ap.add_argument("--iter-offset", type=int, default=0,
                    help="DP 续跑用：中间 ckpt 命名/日志的全局步偏移。从 "
                         "actor_it{N}.pt 续跑时传 N，使新 ckpt 仍按全局步连号 "
                         "（it{N+200}...），不覆盖已存在的历史 ckpt")
    args = ap.parse_args()
    if args.hidden is None:
        args.hidden = 768 if args.arch == "mlp4" else 256

    key = jrandom.PRNGKey(args.seed)
    devs = setup_platform()          # 平台层：精度/设备探测（切 CUDA 只动这里）
    print(f"devices: {device_summary(devs)}", flush=True)
    if IS_BUN:
        active_levels = prepare_environment(args.levels)
        configure_training(
            args.bun_curriculum, args.bun_hp, args.bun_kill_window_reward,
            args.bun_reward_profile,
            danger_escape_reward=args.bun_danger_escape_reward,
            avoidable_danger_death_penalty=(
                args.bun_avoidable_danger_death_penalty),
            tactical_bomb_placement_reward=(
                args.bun_tactical_bomb_placement_reward),
            tactical_bomb_resolution_reward=(
                args.bun_tactical_bomb_resolution_reward),
            base_bomb_reward=args.bun_base_bomb_reward,
            forced_kill_reward=args.bun_forced_kill_reward,
            enemy_threat_reward=args.bun_enemy_threat_reward)
        configure_start_state_curriculum(
            args.bun_start_state_bank, args.bun_start_state_weights)
        spawn_audit = configure_spawn_buckets(args.bun_spawn_buckets)
        if spawn_audit is not None:
            print(f"spawn buckets={spawn_audit}", flush=True)
        if args.distill_data:
            raise ValueError("抢包子观测/动作维度独立，不能加载普通模式蒸馏数据")
        print(f"rule=bun map=806 levels={active_levels} obs={N_OBS_CH} ability={N_BOMB} "
              f"hp={args.bun_hp} curriculum={args.bun_curriculum} "
              f"kill_window_reward={args.bun_kill_window_reward} "
              f"reward_profile={args.bun_reward_profile} "
              f"v2_coeffs=(base_bomb={args.bun_base_bomb_reward},"
              f"forced_kill={args.bun_forced_kill_reward},"
              f"threat={args.bun_enemy_threat_reward},"
              f"placement={args.bun_tactical_bomb_placement_reward},"
              f"resolution={args.bun_tactical_bomb_resolution_reward},"
              f"escape={args.bun_danger_escape_reward},"
              f"avoidable={args.bun_avoidable_danger_death_penalty})",
              flush=True)
    elif args.levels:
        from . import levels as level_catalog
        level_catalog.set_active(args.levels, args.level_weights)
        print(f"rule=battle levels={args.levels}", flush=True)

    n, steps = args.num_envs, args.num_steps
    states = init_batch(key, n)
    key, net_key = jrandom.split(key)
    if args.arch in ("mlp", "mlp_bf16", "mlp4", "cnn"):
        kw = {"hidden": args.hidden}
    else:
        kw = {"embed": args.embed, "depth": args.depth}
    params = init_net(net_key, args.arch, N_OBS_CH, H, W, **kw)
    if args.load:
        params = load_params(args.load)
        print(f"已加载初始权重: {args.load} (params={count_params(params):,})",
              flush=True)
    else:
        print(f"arch={args.arch} params={count_params(params):,}", flush=True)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    # 离线蒸馏阶段（先于 PPO）：KL 到 teacher 分布，warm-start student。
    if args.distill_data:
        params, opt_state = run_distill(params, opt, opt_state, args, key)
        if not args.distill_then_ppo:
            if args.save:
                save_params(params, args.save)
            print("蒸馏阶段结束（未接 PPO，退出）", flush=True)
            return
        print("接自对弈 PPO 微调：", flush=True)

    reference_params = (load_params(args.safety_reference)
                        if args.safety_reference else None)
    if args.safety_kl_coef > 0 and reference_params is None:
        reference_params = params
    bun_opponent_pool = None
    if IS_BUN and args.bun_opponent_pool:
        if args.flee_bot_ratio:
            raise ValueError("--bun-opponent-pool 与 --flee-bot-ratio 不能同时启用")
        recent_path = args.bun_opponent_recent or args.bun_opponent_history
        old_path = args.bun_opponent_old or args.bun_opponent_weak
        if not args.bun_opponent_weak or not old_path or not recent_path:
            raise ValueError(
                "Bun opponent pool requires weak/old/recent checkpoints")
        weights = jnp.asarray(parse_bun_opponent_weights(args.bun_opponent_pool))
        bun_opponent_pool = (
            weights, load_params(args.bun_opponent_weak),
            load_params(old_path), load_params(recent_path))
        print("bun opponent pool="
              f"{dict(zip(BUN_OPPONENT_NAMES, map(float, weights)))} "
              f"weak={args.bun_opponent_weak} "
              f"old={old_path} recent={recent_path}", flush=True)

    bot_curriculum = None
    if args.jax_bot_pool:
        if not IS_BUN:
            raise ValueError("--jax-bot-pool 仅支持抢包子规则（JAXBOMB_RULE=bun）")
        if bun_opponent_pool is not None:
            raise ValueError("--jax-bot-pool 与 --bun-opponent-pool 不能同时启用")
        if not args.flee_bot_ratio or int(n * args.flee_bot_ratio) < 2:
            raise ValueError("--jax-bot-pool 需要 --flee-bot-ratio>0（bot 席位比例）")
        base = np.asarray(bun_jax_bots.parse_tier_names(
            args.jax_bot_pool, JAX_BOT_NAMES), np.float64)
        bot_curriculum = JaxBotCurriculum(base, adaptive=args.jax_bot_adaptive)
        print(f"jax bot pool={dict(zip(JAX_BOT_NAMES, map(float, base)))} "
              f"adaptive={args.jax_bot_adaptive} "
              f"bot_envs={int(n * args.flee_bot_ratio)}/{n}", flush=True)

    # ---- 数据并行（pmap DP）：--devices>1 时接管整个训练并直接返回 ----
    if args.devices > 1:
        run_dp_training(params, opt, opt_state, args, key, devs,
                        bun_opponent_pool, reference_params, bot_curriculum)
        return

    one_iter_j = build_one_iter(
        params, opt, opt_state, states, key, args, reference_params,
        bun_opponent_pool,
        jax_bot_enabled=(None if bot_curriculum is None
                         else bot_curriculum.base > 0))

    def extra():
        if bot_curriculum is None:
            return ()
        return (jnp.asarray(bot_curriculum.weights()),)

    # warmup（首次编译，不计入训练更新）
    t0 = time.time()
    compile_warmup(one_iter_j, params, opt_state, states, key, extra=extra())
    print(f"warmup done ({time.time()-t0:.1f}s)", flush=True)

    # 计时
    t0 = time.time()
    rollout_agents = 1 if bun_opponent_pool is not None else 2
    for it in range(args.iters):
        t1 = time.time()
        out = one_iter_j(params, opt_state, states, key, *extra())
        params, opt_state, states, key = out[:4]
        jax.block_until_ready(params)
        if bot_curriculum is not None:
            bot_stats = np.asarray(jax.device_get(out[4]))
            bot_curriculum.update(bot_stats)
            print(f"  bots: {bot_curriculum.summary(bot_stats)}", flush=True)
        dt = time.time() - t1
        sps = rollout_agents * n * steps / dt
        if args.save and args.save_every and it and it % args.save_every == 0:
            mid = f"{os.path.splitext(args.save)[0]}_it{it}.pt"
            save_params(params, mid)
            save_run_metadata(mid, args)
        print(f"[iter {it}] {dt:.2f}s  sps={sps:,.0f}", flush=True)
    tot = rollout_agents * n * steps * args.iters / (time.time() - t0)
    print(f"FINAL end-to-end sps = {tot:,.0f} "
          f"({rollout_agents*n*steps*args.iters:,} trainable-agent steps)",
          flush=True)
    if args.save:
        save_params(params, args.save)
        save_run_metadata(args.save, args)


if __name__ == "__main__":
    main()
