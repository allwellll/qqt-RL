# CX-31 主动击杀机会审计

日期：2026-10-03

## 结论

CX-29/30 的真实网页 hard surviving non-trade kill 仍为零。本轮把“训练状态缺少安全攻击机会”和“actor 有机会但不选择”分开审计，没有训练新模型，也没有把 JAX 对手结果当作正式收益。结论是不启动静态 pressure、宽泛 threat 或逃路缩减奖励的长训。

## JAX/JIT 配对审计

历史 it4000 actor 在清砖、mixed 出生、danger-arena/1 HP 分布中，与 JAX `hunter_hard` 或当前随机自博弈 P1 rollout。静态 oracle 只标记 P0 自身可逃、放泡后敌方安全首步减少或降为零的状态；反事实只替换首步动作，观察 45 tick，覆盖 30-tick 引信。

| cohort | 事件 | 实际放泡 | surviving kill 对照→干预 | 死亡对照→干预 |
|---|---:|---:|---:|---:|
| JAX hard，seed 20266013 | 32 | 32 | 0→0 | 6→3 |
| JAX hard，seed 20266014 | 32 | 32 | 0→0 | 2→4 |
| JAX hard，seed 20266015 | 32 | 32 | 0→0 | 1→1 |

三 seed 共 96 次实际放泡，trade 均为 0，死亡合计 9→8，自炸 3→1；差异不足以解释为攻击收益。静态 forced 机会在三次 hard 审计合计 7 个状态 tick，actor 均未选择。逐 tick 计数高度相关，不能换算为独立局率；JAX hard 也不是网页 Hunter 等价物。

## 真实网页配对

使用历史 it4000 网页导出、正式 `web/sim.js` 与 `web/bun_hunter_bot.js`，固定 map 806、清砖、1 HP、CX-29 出生。近距合法放泡和按 Hunter 自身时间展开计算的逃路缩减事件都做同状态重放，干预支实际放泡，对照支逐 tick 与原轨迹恒等。

- hard/normal 各 12 个近距事件：surviving kill `0→0`、trade `0→0`；hard 死亡 `0→2`，normal `1→1`。
- hard/normal 各 8 个逃路缩减排序事件：surviving kill/trade 均 `0→0`；hard 死亡 `0→1`，normal `0→2`。
- 前 16 局全部合格 tick 中，hard/normal 分别有 297/595 个近距候选，逃路缩减 165/360 次，但 forced 均为零；最低剩余逃路为 22/33。事件来自同一批开局，不能当独立样本。

这些结果说明“近距离合法”“即时 threat”或“减少但仍有大量逃路”都不是安全攻击标签。CX-32 的人工 forced 格位证明机制存在，却不证明正常出生可达；见[CX-32 报告](cx32_geometry_windows_20261003.md)。

## 决策

不接入新奖励，不启动长训或 512 局确认。后续若找到网页时间展开 forced、P0 自身可逃且能由 JAX/JIT 表达的单一变量，必须先做 2,048-state 审计、3-update smoke 和配对 64 局快筛，再考虑 101/201-update gate。权威登记见[训练实验总账](../docs/training_experiment_ledger.md)。
