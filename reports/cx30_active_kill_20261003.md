# CX-30 对手混合主动击杀 pilot

日期：2026-10-03  
状态：完成 pilot，未晋升，未启动 401-update 长训或 512 局确认

## 问题与唯一变量

CX-29 的 trap16 综合安全收益已跨 seed 复现，但 it50 之后出现检查点退化，hard/normal surviving non-trade kill 没有改善。本轮只改变 P1 的 JAX 对手混合，固定 trap16、projected critic、surviving-kill reward 24、EMA 0.95、其余 PPO/reward/出生配置，并从同一 historical it4000 初始化：

| arm | P1 混合 | 用途 |
|---|---|---|
| reference | 50% JAX `hunter_hard` + 50% current stochastic self-play | 原配置延长的配对参照 |
| mixed-hard | 75% JAX `hunter_hard` + 25% current stochastic self-play | 提高接近 hard 的训练压力 |

两臂各 4 个固定 seed（20265001--20265004），201 updates，保存 it50/100/150/200/201。训练任务全部 `rc=0`、checkpoint 有限；64 局快筛使用固定 seed 20261001、相同出生、真实网页 hunter、300 tick 和 attack trace。

## 快筛结果（64 局）

`reference` 只有 1/16 保存点通过统一 gate；`mixed-hard` 有 8/16 通过，且 seed 20265002、20265003、20265004 各自至少有两个通过点。按保存点汇总如下：

| arm | checkpoint | 平均 C | Tactical S 总计/256 | hard+normal S 总计/512 | 平均 Tactical B | 平均 Tactical P | 通过点数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | it50 | 0.363 | 93 | 1 | 17.277 | 0.863 | 0/4 |
| reference | it100 | 0.313 | 93 | 1 | 16.574 | 0.832 | 1/4 |
| reference | it150 | 0.336 | 98 | 2 | 15.074 | 0.797 | 0/4 |
| reference | it201 | 0.328 | 103 | 0 | 15.078 | 0.824 | 0/4 |
| mixed-hard | it50 | 0.457 | 118 | 0 | 18.254 | 0.871 | 3/4 |
| mixed-hard | it100 | 0.340 | 95 | 1 | 17.707 | 0.836 | 1/4 |
| mixed-hard | it150 | 0.422 | 107 | 4 | 16.188 | 0.828 | 2/4 |
| mixed-hard | it201 | 0.395 | 82 | 3 | 14.520 | 0.746 | 2/4 |

快筛只说明 mixed-hard 比原配置更有希望进入下一轮候选筛选，不能证明主动击杀改善。所有 mixed-hard 保存点的 hard S 合计仍为 0；normal S 只有 it150 的 4 和 it201 的 3。it201 的 Tactical S、炸弹数和 pressure 下降，显示继续训练仍可能退化。reference 的 it50 C 中位数为 0.344，mixed-hard 各保存点的平均 C 高于该值，但这只是筛选条件，不是确认结果。

## 决策

不启动原配置或 mixed-hard 的 401-update 长训，也不把 pilot checkpoint 直接合入 main。原因是：

1. 原配置 reference 的跨 seed 通过率很低，不能把“延长同配置”解释为稳定提升。
2. mixed-hard 的通过点跨 seed 可复现，但收益主要仍在 Tactical 和 easy；hard/normal S 没有形成可确认的提升。
3. 201-update 末端 mixed-hard 的 Tactical S/B/P 下降，符合 CX-29 已观察到的后期退化风险。

后续若要继续，只应从预注册的 mixed-hard 候选中按 seed 聚合规则固定一个 checkpoint，再用新的训练 seed 做独立重训；不得挑单个 seed 的最佳点。确认设计为每 Bot 512 局、独立 eval seed、同出生、attack trace 和 10,000 次 paired bootstrap，同时报告 C、S/T/D/B 及 hard/normal 分项。晋升必须满足 Delta C 置信区间下界大于零、各 cell Delta S 置信区间下界不低于 -0.05、炸弹量不低于 baseline 的 75%，并且 hard/normal 不得以零计数或 easy 收益冒充改善。若独立重训仍无 hard/normal 转化，停止该方向。

权威索引见[训练实验总账](../docs/training_experiment_ledger.md)和[CX 实验经验](../docs/cx_experiment_lessons.md)；运行产物只保留在隔离 worktree 的 `runs/`，不纳入版本控制。
