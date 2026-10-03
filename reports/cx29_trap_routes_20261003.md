# CX-29 Trap16 路线实验报告

日期：2026-10-03

## 结论

trap16 的短程候选已通过独立复现和四 Bot 确认，但不支持直接按原配置延长训练。本文原先预注册的对手混合 201-update 门控试验已在 CX-30 完成：mixed-hard 提高了 64 局快筛通过点数，却仍没有 hard surviving-kill，且 it201 的 Tactical `S/B/P` 回落。因此不批准原配置、mixed-hard 或任意“同配置继续训练”的 401-update 长训。

原配置中 P1 为 50% JAX `hunter_hard` 与 50% 当前随机策略。trap16 只在 hunter 的目标场增加距离对手三格内、至多 16 个可达安全位置；其余训练配置固定。训练从 it4000 初始化，使用 projected critic、kill reward 24、reward shaping 0.6、EMA 0.95、512 env x 64 steps、top 25%、32 PPO epoch、学习率 3e-4，共 101 updates。

## 实施与验证

- 2048 个机制探针状态包含 11,675 个非射线候选，发生 16 次行为改变；合法性与安全预测均一致。
- 8 个正式训练任务（4 seed x control/trap16）均 `rc=0`，全部 checkpoint 有限。
- 64 局快筛使用真实 `web/bun_hunter_bot.js`，greedy、300 tick、eval seed 20261001；筛选和确认均使用四 Bot 与 attack trace。
- 预登记候选是 `trap16_s20264001_it50`，SHA256 为 `bd6adcb08cf527986b9469f6ad2bf3ead1c0c38feca1a54fbd01c84aea440975`。
- 独立重训候选 SHA256 为 `27954b38b238febdf9dba5981afea07e43813255659dc01491e64f6865514ef8`。
- 每模型每 Bot 512 局，eval seed 20264011，bootstrap seed 20264012，10,000 次配对 bootstrap。原候选的 Delta C 为 +0.142578，95% CI [0.060498, 0.226562]；独立重训为 +0.101562，95% CI [0.009766, 0.193359]；两臂均通过确认门槛。

原始证据只保存在未纳入版本控制的相对路径中：

- `runs/cx20261003_29_trap_routes/`
- `runs/cx20261003_29_screen/screen.json`
- `runs/cx20261003_29_confirm_eval/{status.json,completion_audit.json,behavior_audit.json,cells_audit.json,confirm.json}`

## 短程曲线与快筛

训练 loss 在四个 trap16 seed 的 101 updates 内约为 1.3 至 1.7，未给出可用于选择更长训练点的单调性能证据。真实四 Bot 快筛结果如下；`S` 为四 Bot surviving-kill 总数，`HN-S` 为 hard 与 normal 合计。

| arm | seed | it50 C / pass / S / HN-S | it101 C / pass / S / HN-S |
|---|---:|---|---|
| control | 20264001 | 0.422 / false / 30 / 0 | 0.453 / true / 35 / 0 |
| control | 20264002 | 0.531 / true / 39 / 0 | 0.359 / false / 29 / 0 |
| control | 20264003 | 0.281 / false / 26 / 0 | 0.375 / false / 28 / 0 |
| control | 20264004 | 0.547 / true / 37 / 1 | 0.484 / false / 36 / 0 |
| trap16 | 20264001 | 0.547 / true / 38 / 0 | 0.422 / false / 31 / 0 |
| trap16 | 20264002 | 0.438 / false / 30 / 2 | 0.250 / false / 26 / 1 |
| trap16 | 20264003 | 0.297 / false / 25 / 0 | 0.391 / false / 27 / 1 |
| trap16 | 20264004 | 0.516 / true / 42 / 0 | 0.250 / false / 23 / 0 |

trap16 在 it50 为 2/4 通过，在 it101 为 0/4。两条已通过轨迹在继续训练后降分：seed 20264001 为 0.547 到 0.422，seed 20264004 为 0.516 到 0.250。这个跨 seed 的检查点退化，连同候选选择集中在 it50，排除了“同配置已经表现出持续提升”的判断。

## 四 Bot 行为

512 局确认的 tactical surviving-kill 为基线 201、原候选 230、重训 229；tactical 炸弹数为 16.270、18.369、18.758；tactical 自炸为 28、8、1。收益主要来自 tactical 的安全性和产出，而不是难对手攻击转化。

hard surviving-kill 为基线 4、原候选 0、重训 0；normal 均为 0。easy 为 15、15、34。hard/normal 的样本杀伤率低，64 局筛选中零计数有较大噪声，但两次 512 局确认都未改善，且 hard 的差异 CI 为负，不能把该结果解释为解决了高难攻击。

## 已执行的延长门控与结论

为区分“延长同配置”与“改变训练压力”，CX-30 已执行 8 GPU、四 seed、两臂配对的 201-update pilot。所有 run 都从同一 it4000 初始化重新训练，不从 it50 checkpoint 续训；每个 seed 固定配对，保存 it50/100/150/200/201，训练预算为 `8 x 201` updates，批量和优化器不变。

两臂应保持 trap16，且只改变对手混合：

| arm | P1 对手混合 | 用途 |
|---|---|---|
| reference | 原 50% `hunter_hard` + 50% 当前策略 | 测量单纯延长的效应 |
| mixed-hard | 75% `hunter_hard` + 25% 当前策略 | 在不改奖励、网络或 trap16 的前提下增加接近 hard 的训练压力 |

启动前每臂均完成 3-update smoke 与 2048-state 机制探针，确认候选合法、安全预测和 action-diff 无回归。每个保存点均以固定 seed、真实 JS hunter 的 64 局四 Bot 快筛并启用 `--attack-trace`；选择规则预先固定为四 seed 平均 C，随后以 Tactical S 和 hard+normal S 打破并列，禁止挑单一最佳 seed。

CX-30 的完整结果见[CX-30 报告](cx30_active_kill_20261003.md)：reference 仅 `1/16` 保存点通过，mixed-hard 为 `8/16`，但所有 mixed-hard 保存点的 hard S 仍为 0，normal S 仅在 it150/it201 分别为 4/3，且 it201 Tactical S/B/P 下滑。它没有提供把任一臂扩至 401 updates 的证据。

若以后出现与真实网页猎手机制对齐的新单变量设计，先做 2048-state rollout 机会审计和 3-update smoke；随后才可做新的四 seed、两臂、101-update 快筛，保存 it50/100/101，以真实四 Bot 各 64 局、attack trace 观察 hard/normal 的实际转化。只有至少两个训练 seed 的多个保存点同时出现 hard 或 normal 的 surviving-kill 信号、无 Tactical S/B/P 退化，才可预注册 `8 x 201` 配对 gate（保存 it50/100/150/200/201）。任何两条轨迹相对 it50 的 C 下降至少 `6/64`、it100 与 it150 后 hard+normal S 未增加且 Tactical S 或 pressure 下降、或任一 Bot 触发 `score_eval_screen.py` 的 S/D/B/P no-collapse 门槛，立即停止该臂；目前没有满足该前提的候选。

即使未来通过 201-update gate，也只能把预注册聚合规则选出的单一 checkpoint 用新的训练 seed 重训，并做每 Bot 512 局真实四 Bot 确认（独立 eval seed、同出生、attack trace、10,000 次 paired bootstrap）。确认仍须同时满足 Delta C 置信区间下界大于零、每格 Delta S 下界不低于 `-0.05`、炸弹量至少为基线的 75%，并单列 hard/normal；不以 easy 或少死替代高难攻击。未达到这些条件前，不批准 401-update 长训。
