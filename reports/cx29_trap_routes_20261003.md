# CX-29 Trap16 路线实验报告

日期：2026-10-03

## 结论

trap16 的短程候选已通过独立复现和四 Bot 确认，但不支持直接按原配置延长训练。建议先修改对手混合，再以配对的 201-update 门控试验判断是否值得扩至 401 updates；本次未启动长程训练。

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

## 后续长程设计

先做一个 8 GPU 的四 seed、两臂配对 201-update pilot，所有 run 均从同一 it4000 初始化重新训练，不从 it50 checkpoint 续训。每个 seed 固定配对；save at 50、100、150、200 以及 final 201。训练预算为 8 x 201 updates，采用原批量与优化器配置。

两臂应保持 trap16，且只改变对手混合：

| arm | P1 对手混合 | 用途 |
|---|---|---|
| reference | 原 50% `hunter_hard` + 50% 当前策略 | 测量单纯延长的效应 |
| mixed-hard | 75% `hunter_hard` + 25% 当前策略 | 在不改奖励、网络或 trap16 的前提下增加接近 hard 的训练压力 |

在启动 pilot 前，先做每臂 3-update smoke 与 2048-state 机制探针，确认候选合法、安全预测和 action-diff 没有回归。每个保存点使用固定 seed、真实 JS hunter 的 64 局四 Bot 快筛，必须启用 `--attack-trace`。选择规则预先固定为：在四 seed 上先比较 checkpoint 的平均 C，再以 tactical S 和 hard+normal S 打破并列；不得按单一最佳 seed 选点。

仅当 mixed-hard 在至少 2/4 seed 的至少两个保存点通过快筛，且 hard+normal S 相对 reference 未下降，并且平均 C 不低于 it50 reference 中位数，才将两臂各扩至 401 updates。任一条件触发即停止该臂：两个或更多 seed 相对其 it50 checkpoint 的 C 下降至少 6/64；100 与 150 后 hard+normal S 未增加且 tactical S 或 pressure 指标下降；任一 Bot 的 S、D、B、P 触发现有 `score_eval_screen.py` 的 no-collapse 门槛。

扩展后仍在 50、100、150、200、300、400、401 保存并快筛。只有预注册规则选出的单一 checkpoint 才进入新的独立 seed、每 Bot 512 局真实四 Bot 确认与 10,000 次 paired bootstrap；确认须同时满足 Delta C CI 下界大于零、各 cell Delta S CI 下界不低于 -0.05、炸弹量不低于基线 75%。
