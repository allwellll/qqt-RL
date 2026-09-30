# Reward V2 实验报告（2026-09-30）

分支 `experiment/reward-v2`。所有训练在 `jax_bomb/jax_train.py` 设备端 JIT 路径上跑，对手是 `flee_bot_actions`（ratio 0.3，其余为共享参数自对弈）。评估用 main 已合入的 `scripts/eval_transformer_checkpoint_sweep.py`，对手 Tactical v2，协议与 overnight_tf_v2 it4000 完全一致。运行产物都在 `runs/reward_v2_20260930/`，没有覆盖或删除旧 run。

## 结论

- 新 reward 能稳住攻击性：同 seed、同 lr 1e-4、同 1500 iter 下，V2+threat 的放泡量和战术放泡率一直高于 Control（场均放泡 22–31 vs 12–14；conditional tactical placement 0.18–0.56 vs 0.09），Control 在 lr 3e-4 下攻击完全崩掉（it800 放泡 3 个/局、击杀 0.03–0.14，已中途停训）。
- 但没有任何 checkpoint 在 1024 局配对检验下显著超过 it4000 的存活击杀率（0.418）。最好的点是 V2+threat lr1e-4 it200（0.438，Δ+0.020±0.042）和 sweep thr08 it100（0.446，Δ+0.028±0.041），都在噪声内。
- 长训对所有配置都有害：V2+threat lr1e-4 到 it1500 击杀降到 0.250（Δ−0.168，显著）；lr 3e-4 到 it1500 击杀 0.115、自炸 0.257，属于明显退化。
- 所以目前 Reward V2 的收益是"防止攻击性塌缩 + 更多有效放泡"，不是"比 it4000 更强"。建议后续短训（≤300 iter，lr 1e-4）+ 按评估挑点，而不是跑长训；也建议补上对手池（冻结 it4000/Tactical 风格对手），见最后一节。

## 1. Reward 审计

审计对象是 412e13f 引入的 V2 项加上原有 danger_arena 奖励（`bun_env.reward_from_events` 的 danger_arena 分支和 `tactical_bomb_shaping_from_events`）。

| 项 | 系数 | 触发条件 | 结论 |
|---|---|---|---|
| death | −2 | 死亡 | 必要 |
| enemy_hit | +0.25 | 己方唯一来源命中对手 | 必要（小） |
| surviving_kill | +12 | 击杀且存活 | 必要，核心目标 |
| mutual_death | 整体覆盖为 −6 | 同 tick 双死 | 必要；覆盖写法确保换命不能拿任何正奖励 |
| self_kill | −8 | 自己的泡唯一致死 | 必要 |
| avoidable_danger_death | −4 | 有路可逃却死 | 必要；412e13f 的"去 sticky-OR"修复正确，保留 |
| danger_safe_resolution | +0.75 | 危险窗口过后存活且动过 | 必要 |
| base_bomb | 0.06 | 放泡 | 有漏洞，已修：原先只看同 tick 自炸清零，而自炸发生在放泡后约 30 tick，自杀泡照拿 base 奖励。改为只奖励"放泡时仍有逃生路径"的泡（`survivable_bomb_placed`） |
| tactical placement | 0.6 | 安全泡且压缩对手安全走位或新覆盖对手格 | 必要 |
| forced_kill | 2.0 | 安全泡使对手安全首步为 0 | 有漏洞，已修：对手本来已经 0 安全步（已被困死）时再补一个泡也算 forced kill。现在要求放泡前对手至少有 1 个安全步 |
| resolution | 1.0 | 标记过的战术泡结算且放泡者存活 | 必要；跨 tick 结算，结算 tick 若自炸/换命会清零 |
| **enemy_threat（新增）** | 0.4 | 安全泡新覆盖对手当前格 | 新增。按用户要求，"威胁对方"要明显高于单纯多放泡（0.4 vs 0.06）。与 placement 叠加：覆盖对手的安全泡共 0.06+0.6+0.4 |

其它审计结论：

- 没有双重计数问题：base / threat / placement / forced_kill 都在放泡 tick 各触发一次，resolution 在结算 tick 触发一次，设计上就是叠加。
- 刷泡风险有限：连续放泡受 bombs_cap 限制，每个泡最多拿 0.06；无威胁的刷泡收益低于一次 danger_safe_resolution。所有长训 replay 里没有出现"≥40 泡且 0 击杀"的局。
- 训练与评估口径：主评估仍用 Tactical v2 规则 bot 的 funnel 标签（`safe_attack_available`），和 env 内 `safe_tactical_bomb_placed` 定义不同。这次没有改评估（必须和 it4000 口径一致），报告里两套指标都列出（`condT` 为评估口径，`tact/g` 为 env 口径）。
- 仍有一处已知取舍：放泡格按移动前位置记 marker，与 base.step 的放泡格一致（base.step 先放泡再移动），不是 bug。

改动与测试（提交 89fa7b2）：`jax_bomb/bun_env.py`、`jax_bomb/bun_safety.py`、`jax_bomb/jax_train.py`（新 CLI `--bun-enemy-threat-reward`，ckpt 元数据记录全部 V2 系数、flee ratio、--load）。`tests/test_bun_reward_shaping.py` 新增 3 个用例（自杀泡无 base、threat 高于 base、已困死对手不算 forced kill），10/10 通过。全量 pytest（25 个其它测试文件）只剩 1 个失败 `test_training_modules.py::test_prepare_enables_repaired_v7_seed_namespace`，属于分支上既有的已知失败（与 reward 无关）；合入 main 后另外 6 个既有失败已不再出现。

## 2. JAX/JIT 与吞吐

- 核验：reward 全链路（`analyze_selected_actions`、`analyze_tactical_bomb_placements`、奖励计算）都在 `jax.lax.scan` 的 rollout 内，用 `lax.cond` 门控，没有 `pure_callback`/`io_callback`/`device_get`，没有用 FrozenTacticalOpponent、label_batch 或旧 counterfactual 链。
- 瓶颈：`bun_safety._can_escape` 每次用 H×W=195 步 `lax.scan` 做 BFS 松弛，每 tick 被调用几十次。有限 deadline ≤ FUSE+1，因此 FUSE+2 步后不动点必然收敛。改为 `_RELAX_STEPS = FUSE+2 = 32`。`tests/test_bun_safety_relax_equivalence.py` 在 512 个随机 rollout 状态上逐位比较短/长松弛下全部分析输出（selected / tactical / 全动作表），完全一致。

单卡 H200，transformer 192×4，1024 env × 128 step，flee 0.3，danger_arena=1，4 iter：

| 配置 | 优化前 sps | 优化后 sps |
|---|---|---|
| legacy, full=1（参考） | 211,208 | — |
| danger_arena, V2 系数全 0 | 128,497 | 167,566 |
| danger_arena, V2 on | 42,340 | 121,202（×2.86） |
| danger_arena, V2 on + threat 0.4 | — | 116,899 |

V2 系数全 0 的 danger_arena 也变快，是因为 danger tracking 本身也调用 `_can_escape`。剩余开销主要是 `analyze_tactical_bomb_placements` 里 2 玩家 × 5 步的 `_safe_move_count` 循环，可 vmap 继续优化。

## 3. 实验配置与墙钟

复现脚本：`scripts/reward_v2_20260930/{sweep.sh,long.sh,eval_arm.sh,replay_probe.py}`（`eval_arm.sh` 把 it4000 软链为 phase1、各 `final_itN.pt` 软链为 `phase2_itN.pt`，从而原样调用 main 的评估脚本）。

公共设置：warm start `qqt-RL/runs/overnight_tf_v2/phase2_it4000.pt`；`--arch transformer --embed 192 --depth 4 --num-envs 1024 --num-steps 128 --minibatch 2048 --epochs 2 --flee-bot-ratio 0.3 --bun-curriculum danger_arena=1 --bun-reward-profile danger_arena --save-every 100`。V2 = `base 0.06, placement 0.6, forced_kill 2.0, resolution 1.0`。每臂单卡，4 卡并行。

| 阶段 | 臂 | seed | lr | 额外系数 | iter | 训练墙钟 | sps |
|---|---|---|---|---|---|---|---|
| sweep | control | 20260930 | 3e-4 | — | 300 | 558 s | 166k |
| sweep | v2 | 20260930 | 3e-4 | V2 | 300 | 759 s | 122k |
| sweep | v2_threat04 | 20260930 | 3e-4 | V2 + threat 0.4 | 300 | 764 s | 121k |
| sweep | v2_threat08 | 20260930 | 3e-4 | V2（base 0.03）+ threat 0.8 | 300 | 769 s | 120k |
| long | control_lr3e4 | 20261002 | 3e-4 | — | 1354（中止） | 2230 s | 166k |
| long | thr04_lr3e4 | 20261002 | 3e-4 | V2 + threat 0.4 | 1500 | 3421 s | 119k |
| long | control_lr1e4 | 20261002 | 1e-4 | — | 1500 | 2456 s | 166k |
| long | thr04_lr1e4 | 20261002 | 1e-4 | V2 + threat 0.4 | 1500 | 3409 s | 119k |

门槛：sweep 里 thr04 it100 在 1024 局下击杀不降（Δ−0.005）、自炸不升（Δ−0.012）、放泡 +9/局，通过安全/攻击性门槛后才开长训。长训期间每 200 iter 做 64 局检查和 replay 探针；control_lr3e4 在 it800 攻击性塌缩（Tactical 击杀 0.125，flee 探针击杀 0.035，放泡 3 个/局），于 iter 1354 停训。评估合计约 64 分钟 GPU 时间。

## 4. 评估结果

主口径：Tactical v2，danger_arena，greedy，最大 300 步。先用 64 局（seed 20260930）扫全部 checkpoint 找候选，再用未参与挑点的 seed 20261001 跑 1024 局确认。出生格与 it4000 完全一致，Δ 为逐局配对差值 ±95% 区间，`*` 表示显著。

1024 局确认（seed 20261001）：

| ckpt | kill | own | mut | avoid | killed | bombs | tact/g | condT | safeDet | win | fk | Δkill | Δown | Δmutual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it4000 (ref) | 0.418 | 0.073 | 0.007 | 0.000 | 0.008 | 16.6 | 1.782 | 0.050 | 0.989 | 0.000 | 0.006 | — | — | — |
| sweep control it100 | 0.316 | 0.104 | 0.000 | 0.018 | 0.056 | 26.3 | 1.886 | 0.087 | 0.985 | 0.000 | 0.015 | -0.102±0.040* | +0.030±0.026* | -0.007±0.005* |
| sweep v2 it100 | 0.291 | 0.128 | 0.000 | 0.000 | 0.030 | 21.6 | 2.054 | 0.536 | 0.987 | 0.000 | 0.044 | -0.127±0.038* | +0.055±0.025* | -0.007±0.005* |
| sweep v2 it300 | 0.399 | 0.104 | 0.008 | 0.000 | 0.037 | 29.7 | 2.014 | 0.304 | 0.983 | 0.000 | 0.042 | -0.019±0.041 | +0.030±0.023* | +0.001±0.007 |
| sweep thr04 it100 | 0.413 | 0.062 | 0.017 | 0.000 | 0.036 | 25.8 | 2.152 | 0.333 | 0.991 | 0.000 | 0.036 | -0.005±0.039 | -0.012±0.022 | +0.010±0.006* |
| sweep thr04 it200 | 0.364 | 0.131 | 0.017 | 0.020 | 0.049 | 23.0 | 1.995 | 0.110 | 0.986 | 0.000 | 0.018 | -0.054±0.043* | +0.058±0.025* | +0.010±0.009* |
| sweep thr08 it100 | 0.446 | 0.116 | 0.016 | 0.000 | 0.085 | 31.7 | 1.863 | 0.438 | 0.984 | 0.000 | 0.048 | +0.028±0.041 | +0.043±0.027* | +0.009±0.009 |
| sweep thr08 it200 | 0.333 | 0.080 | 0.039 | 0.000 | 0.089 | 28.4 | 2.154 | 0.851 | 0.982 | 0.000 | 0.039 | -0.085±0.042* | +0.007±0.023 | +0.032±0.013* |
| long ctl lr1e4 it1000 | 0.443 | 0.059 | 0.014 | 0.000 | 0.040 | 13.6 | 1.656 | 0.088 | 0.989 | 0.000 | 0.020 | +0.025±0.039 | -0.015±0.021 | +0.007±0.009 |
| long ctl lr1e4 it1500 | 0.300 | 0.005 | 0.000 | 0.000 | 0.021 | 12.2 | 1.265 | 0.089 | 0.998 | 0.000 | 0.008 | -0.118±0.044* | -0.068±0.017* | -0.007±0.005* |
| long thr04 lr1e4 it200 | 0.438 | 0.062 | 0.017 | 0.000 | 0.034 | 30.0 | 2.260 | 0.557 | 0.989 | 0.000 | 0.048 | +0.020±0.042 | -0.011±0.022 | +0.010±0.009* |
| long thr04 lr1e4 it500 | 0.411 | 0.102 | 0.045 | 0.004 | 0.018 | 30.7 | 2.138 | 0.471 | 0.984 | 0.000 | 0.064 | -0.007±0.040 | +0.028±0.026* | +0.038±0.014* |
| long thr04 lr1e4 it1200 | 0.373 | 0.079 | 0.007 | 0.000 | 0.020 | 22.1 | 1.464 | 0.283 | 0.988 | 0.000 | 0.033 | -0.045±0.045* | +0.006±0.024 | +0.000±0.007 |
| long thr04 lr1e4 it1400 | 0.334 | 0.079 | 0.013 | 0.000 | 0.042 | 25.5 | 1.487 | 0.341 | 0.984 | 0.000 | 0.038 | -0.084±0.041* | +0.006±0.022 | +0.006±0.009 |
| long thr04 lr1e4 it1500 | 0.250 | 0.074 | 0.016 | 0.000 | 0.059 | 22.8 | 1.587 | 0.179 | 0.987 | 0.000 | 0.017 | -0.168±0.038* | +0.001±0.020 | +0.009±0.009 |
| long thr04 lr3e4 it400 | 0.420 | 0.117 | 0.033 | 0.007 | 0.020 | 21.5 | 1.872 | 0.172 | 0.987 | 0.000 | 0.028 | +0.002±0.041 | +0.044±0.026* | +0.026±0.012* |
| long thr04 lr3e4 it1500 | 0.115 | 0.257 | 0.012 | 0.000 | 0.150 | 22.8 | 1.254 | 0.071 | 0.956 | 0.000 | 0.007 | -0.303±0.038* | +0.184±0.030* | +0.005±0.008 |

列说明：kill = 存活因果击杀率，own = 自炸率，mut = 换命率，avoid = 可避免危险死亡率，killed = 被对手击杀率，bombs = 场均放泡，tact/g = env 口径安全战术放泡/局，condT = 评估口径有机会时放泡率，safeDet = 安全引爆比，fk = funnel 里 forced kill 创造率。win 在 danger_arena 协议下恒为 0（该课程没有 winner 终局），与 it4000 历史结果一致。

64 局全曲线（seed 20260930）在 `runs/reward_v2_20260930/eval/seed20260930_g64/<arm>/summary.csv` 与 `capability_curves.png`。

补充 replay 探针（`scripts/reward_v2_20260930/replay_probe.py`，对 JAX flee bot，256 局，设备端，不与主口径混比）：it4000 击杀 0.172 / 放泡 15.6 / 威胁泡 0.40；thr04_lr1e4 it200 击杀 0.230 / 19.1 / 0.65；thr04_lr1e4 it1500 0.203 / 17.4 / 0.62；control_lr1e4 it1500 0.215 / 13.7 / 0.40；thr04_lr3e4 it1500 0.293 但自炸 0.227。所有探针局都没有"刷泡不击杀"的退化，control_lr3e4 在 it800 的放泡量塌到 3 个/局。

## 5. 解读与下一步

- V2 + threat 的作用方向是对的：它持续提高放泡量、战术放泡和 forced kill 创造率（fk 0.036–0.064 vs it4000 0.006），而 Control 在同样训练下攻击性下降（lr 1e-4）或崩溃（lr 3e-4）。
- 攻击意图没有转化成更高的 Tactical v2 击杀率。原因更像是训练对手：70% 自对弈 + 30% 以逃跑/静止为主的 flee bot，与 Tactical v2 这种会反击的规则 bot 差别大；长训后策略对训练分布过拟合，击杀率下降，lr 3e-4 下自炸飙升。
- 换命率在 V2 各臂小幅上升（0.007 → 0.017–0.045），threat 系数越高越明显（thr08 it200 显著 +0.032），这是 threat 奖励的副作用，下一轮可以把 mutual 惩罚提高或把 threat 与 resolution 绑定（只在放泡者安全结算后兑现）。
- 建议：(1) 用 thr04 系数、lr 1e-4、≤300 iter 的短训，每 50 iter 存点并用 1024 局挑点；(2) 训练对手加入冻结 it4000 与会反击的 JAX bot，DP 路径目前不支持对手池，需要先补；(3) 若要继续提速，vmap `_safe_move_count`。
