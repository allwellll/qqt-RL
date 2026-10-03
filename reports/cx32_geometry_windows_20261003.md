# CX-32 网页 Hunter 几何见证与短训 pilot

日期：2026-10-03

## 结论

map 806 清砖空图中确实存在单泡即可安全击杀真实网页 hard/normal Hunter 的人工起点，但把这些格位混入 JAX reset 的 21-update 短训没有产生正常出生评测的主动击杀。两条 window 模型 hard/normal surviving non-trade kill 均为 `0/64`，不做 512 局确认或长训。

## 机制见证

在 3,442 个 Manhattan 距离不超过 4 的开放格位对中，hard/normal 各有同一组 13 个相邻格位：P0 额外放一泡后，Hunter 时间展开的乐观逃生终点降为零，而 P0 仍可逃。每格 4 个模拟/Hunter seed 配对，规则 P0 的 surviving kill 为 `0→52`、死亡 `0→0`；历史 it4000 模型移动为 `0→44`，死亡 `4→8`、trade `0→8`。这是人工起点的 45-tick 见证，不是正常出生胜率。

正常出生的历史 it4000 在 hard/normal 各 16 局全部合格近距 tick 中没有 forced 状态，说明人工格位的可达性和时序仍未解决。

## 训练与限制

两臂均从 CX-29 trap16 配置出发，共同加入见证格位安全放泡 `+4` 奖励；control 的 reset 混入比例为 0，window 为 .25。两臂训练 seed 未配对，且相对于 CX-29 同时改变奖励和 reset 分布，不能作严格单变量因果试验。1-update smoke 与四条 21-update 训练均 `rc=0`、checkpoint finite。

固定 eval seed `20261001`、同出生、greedy、300 tick、真实网页三档和 Tactical v2，每格 64 局：

| 模型 | Tactical S | hard S | normal S | easy S |
|---|---:|---:|---:|---:|
| 历史 it4000 | 28 | 1 | 0 | 1 |
| control 20266111/12 | 29 / 24 | 0 / 0 | 0 / 0 | 1 / 5 |
| window 20266113/14 | 26 / 26 | 0 / 0 | 0 / 0 | 4 / 2 |

其中 `S` 是 surviving non-trade kill；64 局只用于否决或选点，不能估计小幅效果。JAX 静态标签把一个见证首步标成安全但 `forced_kill_created=false`、敌方安全首步 `5→5`，而网页时间展开标为 forced，说明两套标签尚未对齐。

## 决策

不追加训练、不做 512 局确认。先逐格对齐网页时间展开逃路、JAX 标签和模型 trade 行为；只有新的单一机制变量在至少两个训练 seed 的真实 hard/normal 64 局出现非换命击杀且 Tactical 不退化，才可进入 101/201-update gate。权威登记见[训练实验总账](../docs/training_experiment_ledger.md)；trap16 长训门控见[CX-29 报告](cx29_trap_routes_20261003.md)。
