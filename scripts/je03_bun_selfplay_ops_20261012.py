"""Isolated Stage2 rollout/PPO snapshot. Never edits the live Stage1 sources.

Balanced learner seats and factual tick-count telemetry; PPO gradients remain
averaged across four replicas. Loss column0, preclip gradient norm column1.
"""
from jax_bomb.jax_train import *
from jax_bomb.jax_train import _ppo_loss

def balanced_indices(count):
    rows = jnp.arange(count)
    return rows + (rows % 2) * count, rows + (1 - rows % 2) * count

def advance_league(league, actor, generation, interval=128):
    return (actor, league[0]) if generation % interval == 0 else league

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
                    return_bot_stats=False,
                    return_components=False,
                    return_lesson_stats=False):
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
    opponent_ticks0 = jnp.zeros((7,), jnp.float32)
    lesson_ticks0 = jnp.zeros((16,), jnp.float32)
    component_sum0 = jnp.zeros((2, 8), jnp.float32)
    # Bun has 16 stable lesson ids; keep this optional telemetry independent
    # of the environment module's Python-only name tuple.
    lesson_counts0 = jnp.zeros((16,), jnp.float32)
    # Learner seat in bot envs: P0 where the bot is P1 (first half), else P1.
    learner_seat = jnp.where(jnp.arange(n_flee) < n_flee // 2, 0, 1)

    def one_step(carry, _):
        (states, key, visited, nov, kills, opponent_kinds,
         bot_kinds, bot_stats, component_sum, lesson_counts, opponent_ticks, lesson_ticks) = carry
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
            opponent_actions = bun_opponent_actions(
                states, obs, gv, masks, arch, opponent_action_key,
                opponent_kinds, bun_opponent_weak_params,
                bun_opponent_old_params, bun_opponent_recent_params)
            seats = jnp.arange(n) % 2
            a0 = jnp.where((seats == 0)[:, None], a0, opponent_actions)
            a1 = jnp.where((seats == 1)[:, None], a1, opponent_actions)
            opponent_ticks = opponent_ticks.at[opponent_kinds].add(1.0)
            lesson_ticks = lesson_ticks.at[states.lesson].add(1.0)
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
        reward_out = reward_from_events(
            info["dmg"], states.alive, info["alive"], info["hp"], done,
            info["crate"], newly, info["walls"], crate_coef, explore_coef,
            brick_coef, timeout_alpha, win_bonus, lose_bonus,
            timeout_lead_bonus, timeout_trail_penalty, timeout_draw_bonus,
            mutual_hit_penalty, double_death_penalty, win_hp_bonus,
            trade_win_bonus,
            moves=env_acts[:, :, 0], bombs=env_acts[:, :, 1],
            idle_penalty=idle_penalty, rule_info=info,
            return_components=return_components)
        if return_components:
            rew, components = reward_out
        else:
            rew = reward_out
            components = jnp.zeros((n, 2, 8), jnp.float32)
        component_sum = component_sum + components.sum(axis=0)
        lesson_counts = lesson_counts.at[states.lesson].add(
            done.astype(jnp.float32))
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
            reward_out2 = reward_from_events(
                info2["dmg"], new_states.alive, info2["alive"], info2["hp"], done2,
                info2["crate"], newly2, info2["walls"], crate_coef, explore_coef,
                brick_coef, timeout_alpha, win_bonus, lose_bonus,
                timeout_lead_bonus, timeout_trail_penalty, timeout_draw_bonus,
                mutual_hit_penalty, double_death_penalty, win_hp_bonus,
                trade_win_bonus,
                moves=env_acts2[:, :, 0], bombs=env_acts2[:, :, 1],
                idle_penalty=idle_penalty, rule_info=info2,
                return_components=return_components)
            if return_components:
                rew2, components2 = reward_out2
            else:
                rew2 = reward_out2
                components2 = jnp.zeros((n, 2, 8), jnp.float32)
            component_sum = component_sum + jnp.where(
                done[:, None, None], 0.0, components2).sum(axis=0)
            lesson_counts = lesson_counts.at[new_states.lesson].add(
                (done2 & ~done).astype(jnp.float32))
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
                bot_kinds, bot_stats, component_sum, lesson_counts, opponent_ticks, lesson_ticks), data
    body = (jax.checkpoint(one_step) if checkpoint else one_step)
    (final_states, _, _, nov, kills, _, _, bot_stats, component_sum,
     lesson_counts, opponent_ticks, lesson_ticks), data = jax.lax.scan(
        body, (states, key, visited0, nov0, kills0, opponent_kinds0,
               bot_kinds0, bot_stats0, component_sum0, lesson_counts0, opponent_ticks0, lesson_ticks0), None,
        length=num_steps)
    obs, state, acts, lps, vals, rew, done, masks = data
    batch = (obs, state, acts, lps, vals, rew, done, masks)
    extras = []
    if return_bot_stats:
        extras.append(bot_stats)
    if return_components:
        extras.append(component_sum)
    if return_lesson_stats:
        extras.append(lesson_counts)
    extras.extend([opponent_ticks, lesson_ticks])
    if extras:
        return (final_states, batch, nov, kills, *extras)
    return final_states, batch, nov, kills

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
            gradient_norm = optax.global_norm(grads)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss_val, gradient_norm)

        (params, opt_state), losses = jax.lax.scan(
            body, (params, opt_state), idx)
        return params, opt_state, jnp.array([jnp.mean(losses[0]), jnp.mean(losses[1])])

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
