# 运包抗拦截训练：冻结防守池（2026-11-21）

目标是让运包模型在更强的防守压力下仍能取包、脱身并交付。沿用 [11 月 15 日的两阶段方案](dcu_carry_interceptor_training_20261115.md)，本轮用 u244、u220、u153 分别在**有包子、砖块及完整交付规则的 map806** 上适应，再冻结通过筛选的拦截者，继续训练原运包模型 SP_web u23302。本轮运包产物使用 `runs/carry-*`，不继承纯对拼的 Critic、优化器或运行目录，也不自动上线。

| 阶段 | 更新谁 | 固定对手和奖励 |
|---|---|---|
| A：`interceptor` | 先用 u244 热启拦截 Actor；再分别用 u220、u153 热启 | 对手是能正常运包的 SP_web u23302。有效归因击杀 +5，死亡 −5，自己击杀额外 −2；没有偷包、运输、位置或终局奖励。 |
| B：`courier` | 从 SP_web u23302 热启运包 Actor | 默认 75% 冻结拦截者，25% 普通 SP_web。运包方保留交付、取包、回程和战斗辅助奖励。 |

拦截者在两阶段都不能偷笼内包子、拾地面包子、击杀接包或交付；可以在笼边防守、追击、放泡和使用其他道具。取消冻结对手的运包奖励本身不会改变策略，所以 A 必须实际更新 Actor，B 才接纳对应检查点。B 中防守权重不更新，策略身份按环境固定；正常 SP 对手与运包方都保留完整运包权限。

三个对拼源检查点在同一旧对手池的小样本分别 12 局全胜；u244 净击杀最高，因此先验证它。纯对拼成绩不证明谁在完整运包地图上拦截最好。原始 u930 保留作对照。各防守槽平分拦截预算：三选手默认每卡 512 环境，普通 SP 每卡 512；每槽 P0/P1 各 256。`--interceptor-fraction` 可调整总拦截比例，实际按座位对取整并写入日志。

### 权重、预检与启动

原始 Actor 来自 `runs/duel-pool-8card-e2048-mb2048-20261121/u000244`、`u000220`、`u000153` 的 `actor.pt`；入口固定并校验各自 SHA256。路径不同可用 `--interceptor-actor 路径` 配合 `--interceptor-start 版本`，哈希不变。使用持久盘上的原始训练权重；网页 Release 是投影后的推理权重。阶段 B 只读取可信本地 carry 阶段 A 检查点，核对源码、起始 Actor 身份、实际训练更新号、模型哈希和 ABI。A、B 首次运行都初始化新 Critic/Adam，默认前 10 次只更新 Critic。阶段内接续可用 `--checkpoint`；A 保持原 `--interceptor-start`，B 保持同一个冻结池。接续不恢复 Adam、环境和随机数状态。

不占 DCU、不载入模型的计划检查：

```bash
.venv/bin/python -m scripts.train_dcu_jax_carry plan \
  --stage interceptor --interceptor-start u244
```

`training_launch_policy.json` 当前关闭训练；本次准备代码和文档，**未启动 DCU 热启或运包续训**。后续实际开训时沿用原入口的启动门禁。用新目录先做一轮短测，确认 `INFERENCE_CHECK.json` 通过、Actor/Critic 已保存、没有非有限值或道具溢出，再扩大预算：

```bash
bash scripts/launch_dcu_jax_carry_8card.sh interceptor \
  runs/carry-defender-u244-smoke-20261121 --interceptor-start u244 \
  --max-updates 1 --critic-warmup 0 --hours 0.25
```

正式 A 按 u244 → u220 → u153 分开启动，只改 `--interceptor-start` 及输出目录。可先用每个 30 分钟上限，在编译后确认至少有 Actor 更新；预热不够、数值失败或没有交付抑制的候选先不进池。B 用实际选中的检查点替换下列 `$qqt_*_ckpt`；可重复 `--defender`，不自动使用 `LATEST`：

```bash
.venv/bin/python -m scripts.train_dcu_jax_carry plan --stage courier \
  --defender "$qqt_u244_ckpt" --defender "$qqt_u220_ckpt" --defender "$qqt_u153_ckpt"
bash scripts/launch_dcu_jax_carry_8card.sh courier runs/carry-sp-defender-league-20261121 \
  --defender "$qqt_u244_ckpt" --defender "$qqt_u220_ckpt" --defender "$qqt_u153_ckpt"
```

八卡保留每卡 2048 环境、64 tick、融合采样；首跑沿用完整运包已通过单卡验证的 512 policy/value/PPO 批形。每轮全局 1,048,576 环境步，新防守池的八卡速度尚无实测。如果有硬截止时间，两阶段统一传绝对 `--deadline-utc`，勿给每个任务重新计额。每个 run 最多 32 份检查点、2 GiB；三个 A 加一个 B 的上限约 8 GiB，另留 8 GiB 空闲。每份保存 Actor/Critic；每 5 次更新或距上次保存 120 秒后的完整更新保存一次，正常退出也保存。可在本轮输出目录创建 `USER_STOP`；入口不会接力启动其他阶段。

### 验收指标

```bash
.venv/bin/python -m scripts.summarize_dcu_carry runs/carry-sp-defender-league-20261121
```

`PLAN.json` / `CHECKPOINT.json` 记录池中每槽的来源、SHA 和席位预算；`updates.jsonl` 按槽输出交付、死亡、自死、携包者击杀和掉包。A 看同一冻结运包对手的交付率是否比面对原始对拼模型时下降，辅以携包者击杀和自死；拦截成功通常表现为平局。B 分别看对各防守槽的交付改善，同时检查普通 SP 对战是否退步。正式选型用固定新种子、双席及完整局重跑，比较父模型 SP、各候选，以及未入训练池的对手；同时看交付耗时和携包者死亡。训练累计值混有不同 Actor 版本，不能用来证明泛化提升。

### 已授权运行

用户于 2026-11-21 明确授权开训。先执行 `runs/carry-defender-u244-smoke-20261121` 单更新检查，通过后继续 u244 拦截训练；不自动接续 u220、u153 或运包阶段。训练门禁仅为本轮开启，GPU 诊断门禁仍关闭。

- smoke：1 次更新、1,048,576 环境步，`INFERENCE_CHECK.json` logp/value 误差均为 0；u000001 已保存，Adam 未保存。
- 正式 u244：`runs/carry-defender-u244-20261121`，8 卡、每卡 2048，45 分钟预算。首轮误差均为 0，u000001 已保存；首轮约 4,605 环境步/秒，包含首次 rollout 编译。
- 正式阶段已正常到时退出：104 次更新、109,051,904 环境步，`RESULT.json` 为 `UPDATES_COMPLETED`，`STOPPED.json` 标记 `healthy: true`，最终保存 `u000104`，Adam 未保存。结束时八卡空闲，所以当前 `rocm-smi` 利用率和显存使用均为 0。
- 末三轮端到端约 44.4k–44.6k 环境步/秒，每轮采样约 7.6 秒、PPO 约 16 秒。前 10 次 Critic warmup 后 Actor 才开始更新。此速度是当前短任务的稳定值，不是下一阶段多防守模型池吞吐保证。
- 暂不自动启动 u220、u153 或运包阶段；先按固定运包对手评测 u000000/u000104，确认这次适应是否提高拦截能力。

### 本次验证

隐藏 DCU，并用 `JAX_DISABLE_MOST_OPTIMIZATIONS=true` 和 `XLA_FLAGS=--xla_cpu_enable_fast_math=false` 跑 CPU 回归，共 39 项通过。覆盖座位平衡、禁运权限、融合采样、PPO 更新和冻结对手；真实 u244、u220、u153、u930 也通过了双席开局前向。真实 u244 的 CPU 短 rollout 未通过数值门槛：融合路径出现非有限值，分段路径虽有限，但重推理不一致。回执在 `runs/carry-defender-cpu-validation-20261121.log` 和 `runs/carry-defender-preflight-20261121/`。这不是 DCU 实测，不能据此认定八卡通过或失败；实机首轮必须通过现有 logp/value 门槛才能训练。
