# CX-33 JAX 对手梯度诊断

日期：2026-10-03

CX-33 只改变 JAX 训练对手池，作为 CX-32 control 的低成本诊断；它不是 CX-29 的晋升或长训依据。control 使用 `dodge=1, legacy_flee=1`，ladder 使用 `dodge=1, bomber_easy=2, hunter=2, hunter_hard=1, legacy_flee=1`。两臂各两个 seed、21 updates，均正常完成且 checkpoint finite；真实评估固定 seed、同出生、greedy、300 tick、每 Bot 64 局。

| 模型 | Tactical S | hard S | normal S | easy S |
|---|---:|---:|---:|---:|
| CX-32 control baseline | 24/64 | 0/64 | 0/64 | 5/64 |
| control seed 20266311 | 28/64 | 0/64 | 0/64 | 3/64 |
| control seed 20266312 | 30/64 | 0/64 | 0/64 | 2/64 |
| ladder seed 20266311 | 23/64 | 0/64 | 0/64 | 3/64 |
| ladder seed 20266312 | 22/64 | 0/64 | 0/64 | 6/64 |

ladder 在 JAX 训练侧出现 hunter/hunter_hard 的 kill/death 信号，但真实网页 hard/normal 仍为零，Tactical 也低于两个 control seed。该结果支持“代理对手暴露不等于网页攻击转化”，不支持把 ladder 或 trap16 扩展到长训。原始逐局矩阵留在隔离实验目录，不纳入 Git。

## 长训门控与预注册设计

当前建议：不启动 trap16 同配置长训。CX-29 的 it50 在 2/4 seed 通过、it101 为 0/4；两条通过轨迹 C 分别从 `.547→.422`、`.516→.250`。512 局确认虽显示 Tactical 安全收益跨 seed 复现，但 hard `4→0/0`、normal `0/0/0`，CX-30 的 201-update 对手混合 pilot 也没有 hard S，且末期 Tactical S/B/P 下滑。

只有在真实网页轨迹中发现可由 JAX/JIT 表达、P0 自身安全且将 Hunter 时间展开逃路降为零的单一机制变量后，才允许下一轮：

1. 先做 2,048-state 机会审计与 3-update CPU/GPU smoke，核对 finite、JIT、吞吐、checkpoint sidecar 和 action/safety 不变量。
2. 做四个配对训练 seed、control/treatment 两臂、101 updates；两臂只改该机制变量，保存 it50/100/101。所有保存点跑真实四 Bot 64 局、`--attack-trace`，单列 Tactical、hard、normal、easy 的 S/T/D/B。
3. 只有至少两个训练 seed 在多个保存点出现 hard 或 normal 非换命击杀，且 Tactical S、pressure 和 bombs/game 不退化，才预注册 8-run、201-update gate；保存 it50/100/150/200/201，reference 与 treatment 保持 seed 配对。
4. 任一臂若两条轨迹相对 it50 的 C 下降至少 6/64，或 it100 之后 hard+normal S 不增加且 Tactical S/pressure 下降，立即停止该臂。不得从 it50 checkpoint 续训冒充独立训练。
5. gate 通过后只选预登记聚合规则产生的一个 checkpoint，用新训练 seed 重训；换独立 eval seed 做每 Bot 512 局、同出生配对、10,000 次 bootstrap。Delta C 下界须大于零，每格 Delta S 下界不低于 `-0.05`，B 不低于基线 75%，并保留 hard/normal 限制。

该设计给出可审计的训练预算和停止条件，但在当前证据下不执行任何长训。
