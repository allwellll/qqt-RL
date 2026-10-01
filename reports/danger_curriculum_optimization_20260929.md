# 危险课程优化 — Bun Tactical v2 自炸/被动/进攻联合修复 (2026-09-29)

> 完成标准：代码 + 测试 + 真实 GPU 短 A/B + 固定评估。本报告只记录**已执行**的结果与证据，
> A/B 与长跑启动小节在对应实验结束后以真实数字回填。

## 0. 背景与目标

网页真实录像回放显示：学习策略与规则 Bot 都出现 (a) 被自己泡炸死、(b) 进攻不可见、
(c) 站在危险里却不逃/不防。此前的高吞吐 run `bun_high_throughput_20260929` **已废弃、不得
resume**（其 checkpoint 仅作只读参照）。目标是在 `qqt-RL` 内走完整闭环：**先审计 → 实现 →
短实验验收（真实 GPU A/B）→ 达标后才启动新长期 run**。

能力门槛（capability gate）：安全提升**不能靠完全不放泡**；要求放泡活跃、自炸显著下降、
危险逃生显著提升、非 trade 进攻指标不下降（最好转正）。

## 1. 根因审计结论 (#17)

1. **Actor 未接收危险监督（主因，已修）**：生产 actor 的 `loss_fn` 原本没有 aux 项，
   danger/escape/safe-action 监督从未进入 actor 的共享 backbone 梯度——只有 PPO 的
   "died→-1" 长信用链间接传导。

2. **训练分布中"逃生几乎总是可用"，自炸是策略问题而非标签/teacher 问题**（关键新证据）：
   在 danger_arena 训练分布上实测安全 oracle：

   | 指标（player0，4747 存活态，seed 424242） | margin=0 |
   |---|---|
   | 逃生可用率（存在可生存合法动作） | **100.0%** |
   | 平均可生存动作数 | **8.91** |
   | 存在安全放泡动作 | 92.7% |

   即逃生分支几乎总是存在，但 base actor 在固定评测里仍 **28.9% 自炸**——说明自炸是
   **策略没学会使用已存在的逃生**，正是 aux 监督 + 奖励重构（#20）要解决的目标，而非
   标签错误或 teacher 不安全。

3. **规则 Bot 本身在 sim 中已是安全主动 teacher**（推翻"规则 Bot 不安全"的前提）：
   见 §5。因此 28.9% 的自炸是**学习策略**的，不是规则 Bot 的。

## 2. 确定性多时间片 danger 特征进 obs（无泄漏）(#19)

`jax_bomb/bun_supervision.py`：
- `danger_slices`：对 (2,4,6,8,10) tick 阈值给出 "该 cell 是否在 k tick 内燃烧" 的布尔平面，
  继承 `_source_deadlines` 的墙/砖遮挡与链式传播，时间单调。
- `own_bomb_footprint`：仅取本方假想泡的 blast 源，回答"我将要制造的危险"。
- 全部是**当前公开状态**的纯函数：无对手策略、无未来 rollout、无隐藏 planner 状态；
  这些量一个完美规则推理者能直接从盘面读出，作为输入不泄漏任何智能体本可自算的信息。

## 3. auxiliary 真正进 Actor 梯度 + 奖励重构 (#20)

- `scripts/train_bun_separate_ac.py`：新增 `--actor-aux-coef`（默认 0.0，零风险；>0 时嫁接
  aux heads = 新架构 lineage，并把 `bun_supervision.actor_supervision_labels` 的
  safe_action/escape/margin **BCE 接入 actor 共享 backbone 梯度**）。逐 rollout step 用
  jitted `supervision_step` 生成 player0 标签，`actor_update` 累加 aux BCE。
- 标签来源是 `bun_safety.analyze_actions` 的 JAX oracle（对环境态），**不经过**规则 Bot 的
  乐观 `_survival_plan`，故 teacher 的任何乐观都不会泄漏进 actor 梯度。
- 奖励重构 `jax_bomb/bun_env.py:danger_arena_reward`：放泡本身**不奖励**；只有
  完成破箱/迫敌/**非 trade 击杀且自己安全解析**才奖励，逃离危险需 hazard identity one-shot，
  `own_bomb_defeat|trade` 屏蔽 tactical shaping，doomed 态豁免，mutual_death 覆盖为 -6.0。

验证：真实 GPU 上 aux_loss 有限且非零、backbone 权重相对 base 确实移动（L1 delta > 0）。

## 4. 安全 oracle margin 分析（决定不改动 A/B）

安全 oracle 的逃生判定 `_can_escape` 与规则 Bot 一样把玩家近似为 floor(pos) 的**单格**，
而环境 `_is_hit_by_explosion` 用 2×2 bbox（RADIUS 0.36）。理论上移动途中会有瞬态乐观。
oracle 已内置 `margin_ticks` 保守旋钮（默认 0）。实测 margin 0→1 的保守代价：

| 指标（player0） | margin=0 | margin=1 | 损失 |
|---|---|---|---|
| 逃生可用率 | 100.0% | 100.0% | 0.0% |
| 平均可生存动作数 | 8.91 | 8.88 | ~0 |
| 存在安全放泡动作 | 92.7% | 92.4% | **0.3%** |

结论：在训练分布里 margin=1 近似 **no-op**（既不会造成被动崩溃，也几乎不改变行为，因为
danger_arena 逃生空间宽）。因此**不改动正在运行的 A/B**（保持 A/B 对 #20 机制的代表性），
仅将 margin=1 记为长跑可选硬化项。瞬态 bbox 乐观主要在有砖/全图（网页自炸场景）显现。

## 5. 规则 Bot 安全战术 teacher + Python/JS 固定 fixture 回归 (#21)

**证据（sim 内，非猜测）**：
- 规则 Bot 自博弈自炸率：danger_arena（清砖）**0/128 = 0%**、danger_arena（带砖）
  **0/128 = 0%**，同时平均放泡 **2.61/2.58**（积极放泡）；全图默认（objective 相位）不放泡。
- **Python↔JS 逐字节一致**：把 1728 个真实战斗态（3 seed、双方）分别喂 Python `BunRuleBot`
  与 `web/bun_rule_bot.js`，action/reason/phase **0 处不一致**。
- 故"规则 Bot 不安全"在 sim 内不成立；网页自炸更可能来自 `web/sim.js` 物理与 JAX env 的
  分歧（`test_visual_parity.js` 只校验渲染资产、不校验物理，属已知未覆盖缺口），而非 Bot 逻辑。

因此 #21 的诚实交付是**把已安全的 teacher 锁进回归**，而不是给不存在的 bug 造"修复"：
- `tests/fixtures/bun_rule_bot_selfplay_parity_cases.json`：176 个按 reason 均衡的真实战斗态
  （escape_immediate / doomed_max_survival / safe_pressure_attack / wait_no_safe_attack /
  control_space / objective_route 各 ~30）。
- `tests/test_bun_rule_bot_selfplay_parity.py`：Python 复现精确 action/reason/phase。
- `web/test_bun_rule_bot_selfplay_parity.js`：JS 复现同一 fixture（176 例通过），锁定跨语言
  生存 planner parity（超出原有 10 条 FSM/routing case）。
- `tests/test_bun_rule_bot_safe_teacher.py`：真实环境端到端自博弈，断言 teacher **同时**
  安全（0 自炸）且主动（放泡>0），双向防回归（乐观自炸 / 被动崩溃）。

## 6. 固定能力评测 harness (#22)

`scripts/eval_bun_tactical_opponent.py`（learner p0 vs tactical p1，固定 seed/games）输出：
p0_win_rate、p0_surviving_causal/physical_kill_rate、mutual_death_rate、
p0_own_bomb_defeat_rate、p0_avg_bombs、p0_safe_resolution_ratio、
p0_conditional_tactical_placement_rate、p0_avoidable_danger_death_rate 及 funnel。
攻击活性（avg_bombs、tactical placement）与自炸（own_bomb_defeat）双指标同时可见，
正是 gate 判据所需。

## 7. 生产路径吞吐 smoke (#23)

host-bound（微网络 + horizon=40 规则 Bot host 循环），非 GPU 计算受限。热态反事实阶段
~90 s/cycle（冷 ~245 s，持久 XLA 缓存 2.7× 加速），训练 rollout ~50 tps；4×32×128 的一个
cycle ≈ 5–6 min。run 可行。详见 `reports/gpu_throughput_diagnosis_20260929.md`。

## 8. 基线：被动崩溃的经验证据（gate 的反例基准）

废弃 run 各 arm 最新 checkpoint 的固定评测（learner p0 vs tactical，games=128，seed 固定）：

| arm | win | causal_kill | own_bomb_defeat | avg_bombs | avoidable_death | safe_resolution | tactical_place |
|---|---|---|---|---|---|---|---|
| base（起点） | 0% | 5.5% | **28.9%** | **1.02** | 8.6% | 1.1% | 0.0% |
| control | 0% | 4.7% | 0.8% | **0.43** | **21.1%** | 1.4% | 0.0% |
| safe_medium | 0% | 3.9% | 0.8% | 0.31 | 12.5% | 1.6% | 0.0% |
| safe_high | 0% | 3.9% | 11.7% | 0.54 | 9.4% | 1.5% | 0.0% |

解读：control 相对 base 把自炸压到 0.8%，**但代价是放泡从 1.02 掉到 0.43、可避免死亡从
8.6% 升到 21.1%、击杀不升反降、tactical placement 全程 0**——它既不放泡也不逃，就是用户
废弃该 run 要避免的**被动崩溃**。这为"aux + 奖励重构"方向提供了必须超越的反例基准。

## 9. 短 A/B 四臂 + capability gate 判定 (#24)

方法：从 base actor 起，各跑 100 updates（32 envs × 128 steps，danger_arena，opponent
tactical，其余超参与生产一致：actor_lr 1e-5 / critic_lr 3e-5 / entropy 0.003 / kl 0.05 /
bc_coef 0.12 / target_tau 0.04 / counterfactual_coef 0.30 / counterfactual_aux_coef 0.20）。
两臂隔离 #20 机制（aux + placement/resolution shaping）：

- **treatment**（GPU0，seed 202609290001）：`actor_aux_coef=0.2`，
  `tactical_bomb_placement_reward=0.10`、`resolution=0.75`、`danger_escape=0.75`、
  `avoidable_penalty=4.0`。
- **control**（GPU1，seed 202609290002）：`actor_aux_coef=0.0`，placement=resolution=0.0
  （复现旧被动机制）。

> **[待回填]** 两臂结束后用 `eval_bun_tactical_opponent.py`（固定 seed、games=128）评测，
> 按 gate 判据比较：放泡是否保持活跃、own_bomb 是否显著下降、safe_resolution 是否上升、
> 非 trade 击杀是否不降。

## 10. 长期 run 启动 (#25)

> **[待回填，gate 通过后]** 经 `prepare_bun_safe_aggression_v7.py`（冻结 runtime/inputs、
> 预声明 seed、空 cache、per-arm warm-start）+ `launch_bun_safe_aggression_v7.py`
> （durable tmux supervisor、分段自动续跑、单 watchdog、early-best 与 latest 分离、
> 仅技术故障才停）。返回 PID / GPU / run 路径 / 首个有效 cycle。

## 11. 失败尝试与坑（供后续避免）

- 早期 self-play trace 用 seed 20260929 得 "0 自炸"，误判；换到出现自炸的种子 + 加大样本才
  定位到 **28.9% 属学习策略、规则 Bot 0%** 的真相——种子敏感，勿据单种子下结论。
- 端到端 safe-teacher 回归首版把 `@jax.jit` 定义在函数内，两个场景（清砖/带砖）pytree
  结构不同触发 "supplied 52 vs expected 147" 的 JIT 参数不匹配；把 step 提到模块级由 JAX 按
  输入结构分别缓存后解决。
- oracle margin 一度以为是自炸主因；实测在训练分布近似 no-op，证明自炸是策略问题，及时
  避免了给 A/B 引入无证据的改动。
