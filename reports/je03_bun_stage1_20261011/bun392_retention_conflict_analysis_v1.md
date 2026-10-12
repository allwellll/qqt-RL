# Bun392 持续训练能力保持冲突分析

日期：2026-10-11。实验：`JE03-BUN831-NATIVE-TRACKS-20261011-v1`。本报告是只读根因审计，不改变配方、不停止 worker/dispatcher/evaluator，也不把诊断当作当前训练门禁。

## 结论先行

同一个原生 Bun392 g0 的能力没有在训练中被保持。base 在最早可见的 g32 就从 Hunter 安全击杀 `19/192` 降到 `1/192`，Tactical 从 `22/64` 降到 `16/64`；recipe g32 为 `7/192` 和 `14/64`。之后两臂都未恢复到 g0，base 在 g512 出现更严重的安全崩塌。与此同时 Bun 任务成功率上升，尤其 recipe 的 carry/steal 达到 `15--16/16`，但 Hunter safe kill 仍只有 `2--7/192`。因此最可信的现象是“训练任务局部学会，同时真实 Hunter 战斗/安全目标被遗忘或没有被优化”，不是简单的训练未收敛。

目前排序为：**A 奖励与 Hunter 目标错位 > B 课程/对手分布冲突 > C PPO 累计更新造成遗忘 > E 新输入/道具通道扰动旧表征 > F Critic/共享梯度作为放大器 > D Top25 捷径（只解释 recipe 的形状，不能解释共同退化）**。A/B/C/E 可能是组合根因；现有证据不足以把其中任一项单独定为充分原因。

## 证据来源与边界

审计读取了 authority `COMPLETE.json`、`COMPLETE_WITH_BUN_TASKS.json`、formal `events.jsonl`、`accounting_v1/events_*.jsonl` 和每个指定 checkpoint 的 `state.pkl`。计划 SHA 为 `3a620bf8cac6eedd627fd78d488504f9beaedb16725e7a4e3b90ea29059383ea`；原始训练源码、环境和 PPO 源码均按当前工作树读取。逐项机器结果和参数漂移在同目录 JSON 中，脚本为 `scripts/je03_bun392_retention_conflict_analysis_20261011.py`。

训练事件的 `stats` 是 `[mean reward, done rate, rollout kill sum]`，不是 Hunter 指标；`accounting_v1` 校正了真实 Adam 更新数：base `128/gen`，recipe `32/gen`（原始 formal 行的 128 是保留的遥测字段）。训练日志没有 entropy/KL，因此本报告明确标为缺失；不能用 loss 或 reward 代替它们。

## 权威能力曲线

表中 Hunter 为 `hard/normal/easy` 的 `safe_win/death/self`，每档 64 局；括号后三列均按该档计数。总 safe 是三档相加，不能把 Tactical 当 Hunter 替代。

| arm/g | Hunter safe (H/N/E) | Hunter death (H/N/E) | Hunter self (H/N/E) | Tactical safe/death/self | Bun tasks `carry/steal/route/full` |
|---|---:|---:|---:|---:|---:|
| g0（两臂相同） | 0/4/15 = **19** | 2/2/5 | 1/0/2 | 22/2/1 | 1/8/0/0 |
| base g32 | 0/0/1 = 1 | 4/6/6 | 1/3/5 | 16/2/0 | 10/10/0/0 |
| base g64 | 0/0/0 = 0 | 6/12/3 | 2/2/0 | 14/7/6 | 11/5/0/0 |
| base g128 | 0/0/3 = 3 | 10/15/12 | 6/2/8 | 13/6/4 | 11/1/0/0 |
| base g192 | 1/1/3 = 5 | 4/8/5 | 3/4/2 | 15/4/3 | 11/2/0/0 |
| base g256 | 0/1/2 = 3 | 2/5/14 | 0/1/0 | 9/1/1 | 11/9/0/0 |
| base g512 | 2/0/1 = 3 | 45/52/45 | 23/27/18 | 6/38/17 | 16/5/0/0 |
| base g768 | 0/0/1 = 1 | 16/28/26 | 5/5/7 | 4/10/1 | 16/5/0/0 |
| recipe g32 | 0/0/7 = 7 | 3/2/1 | 2/1/0 | 14/0/0 | 11/16/0/0 |
| recipe g64 | 0/0/1 = 1 | 4/2/2 | 0/0/1 | 13/2/0 | 13/16/0/0 |
| recipe g128 | 0/1/3 = 4 | 0/2/5 | 0/0/1 | 16/1/0 | 16/16/0/0 |
| recipe g192 | 0/3/1 = 4 | 0/8/3 | 0/1/1 | 11/0/0 | 15/16/0/0 |
| recipe g256 | 0/1/4 = 5 | 7/4/4 | 2/1/2 | 8/6/2 | 15/16/0/0 |
| recipe g512 | 0/2/4 = 6 | 3/5/4 | 2/1/4 | 6/2/0 | 16/16/0/1 |
| recipe g768 | 0/0/3 = 3 | 4/3/2 | 3/1/1 | 7/2/1 | 15/16/0/2 |
| recipe g1024 | 0/2/3 = 5 | 11/8/11 | 2/3/5 | 5/7/1 | 15/16/0/0 |
| recipe g1536 | 0/0/2 = 2 | 3/11/6 | 1/0/0 | 8/6/0 | 16/16/0/0 |

任务列是 `capture_episodes`（carry）、`steal_episodes`（near steal）、`route_break` 成功、`full` capture；full 的 death/self 单独在 JSON 中保留。route_break 在所有点没有有效成功。g0 full 为 death/self `2/2`；base g512 full 为 `0/0`，recipe g128/g192/g256/g1024/g1536 为 `0/0`，这正是任务指标变好不能推出 Hunter 变好的反例。

## 训练内证据与参数漂移

| arm/g | mean reward | done rate | rollout kill sum mean | 实际 Adam updates/gen | actor update L2 | critic update L2 | fresh ticks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| base g32 | .0444 | .00065 | 405.75 | 128 | 11.95 | 18.74 | 50.9k |
| base g64 | .0583 | .00178 | 414.50 | 128 | 13.29 | 18.86 | 51.2k |
| base g128 | .0633 | .00123 | 458.75 | 128 | 14.13 | 18.58 | 51.1k |
| base g192 | .0608 | .00093 | 499.50 | 128 | 13.76 | 19.72 | 51.7k |
| base g256 | .0665 | .00098 | 462.25 | 128 | 15.34 | 18.13 | 51.2k |
| base g512 | .0703 | .00110 | 474.25 | 128 | 14.19 | 20.61 | 51.4k |
| base g768 | .0583 | .00114 | 449.00 | 128 | 14.50 | 24.94 | 51.6k |
| recipe g32 | .0602 | .00103 | 218.50 | 32 | 3.71 | 4.99 | 104.9k |
| recipe g64 | .0974 | .00167 | 238.00 | 32 | 4.05 | 4.55 | 104.6k |
| recipe g128 | .1207 | .00204 | 249.25 | 32 | 3.99 | 4.90 | 104.0k |
| recipe g192 | .1245 | .00203 | 267.00 | 32 | 4.35 | 5.09 | 104.9k |
| recipe g256 | .1378 | .00225 | 274.75 | 32 | 4.42 | 5.58 | 104.6k |
| recipe g512 | .1286 | .00203 | 255.50 | 32 | 4.06 | 6.36 | 104.6k |
| recipe g768 | .1467 | .00247 | 266.75 | 32 | 5.29 | 6.50 | 104.7k |
| recipe g1024 | .1484 | .00253 | 283.50 | 32 | 4.76 | 6.82 | 104.4k |
| recipe g1536 | .1004 | .00109 | 165.75 | 32 | 4.37 | 5.77 | 103.6k |

最强的可证伪对照是 g0→g32：Hunter safe 大降而训练 reward 上升。recipe 的任务优势不能归因于更多有效更新：它真实每代更新只有 base 的四分之一，吞吐约两倍主要来自 Top25 只保留 25% 样本。base 的安全恶化在 g512 最明显（Hunter death `142/192`、self `68/192`），recipe 没有同样数量级的死亡爆发，但始终远低于 g0，且 g1536 reward/kill 信号回落。

相对 g0 的全树漂移（`relative_to_g0` 为差分 L2/ g0 参数 L2）显示：

| arm/g | actor 全树 | critic 全树 | actor Bun 新输入尾部 | critic Bun 新输入尾部 | actor heads | critic heads |
|---|---:|---:|---:|---:|---:|---:|
| base g32 | .156 | .253 | .163 | .355 | .013 | .109 |
| base g128 | .342 | .480 | .348 | .535 | .018 | .194 |
| base g256 | .512 | .698 | .520 | .692 | .020 | .244 |
| base g512 | .773 | 1.020 | .728 | .882 | .022 | .308 |
| base g768 | .992 | 1.243 | .856 | 1.020 | .022 | .344 |
| recipe g32 | .057 | .086 | .070 | .160 | .008 | .053 |
| recipe g128 | .113 | .166 | .118 | .283 | .014 | .080 |
| recipe g256 | .166 | .228 | .179 | .391 | .016 | .104 |
| recipe g768 | .299 | .408 | .366 | .577 | .017 | .159 |
| recipe g1536 | .431 | .575 | .524 | .698 | .020 | .216 |

新输入尾部包含 migrated token rows `126:216` 和 global state dims `11:24`，是诊断分区，不是独立语义证明。它和 Critic 的相对漂移都大于动作 heads，支持 E/F 作为放大器候选；但 g32 已退化而全树漂移仍较小，不能仅凭漂移断言 E 是首因。

训练源码中 `reward_from_events` 对任务给出较直接的稠密目标：near steal `+10`，carry crossing `+10`、capture `+35`，full capture `+30`、steal `+4`；同时安全项为 own hit `-1`、death `-4`、self kill `-4`，combat evolution 才额外把 surviving kill 提到 `+12` 并惩罚 mutual death `-6`、self `-8`。base 的课程权重是 danger arena `.35`、full ambush `.25`、full `.2`、carry `.2`；recipe 则 full `.3`、carry `.15`、near steal `.1`、route break `.1`、combat/full-contact/full-ambush 各 `.1`、danger `.05`。这提供了可检验的 A/B 机制：任务事件奖励更大、课程占比更高，而 Hunter 的安全击杀、死亡和自炸不受保护。

对手也共同存在潜在分布冲突：base 是 rule combat `.3` + frozen old `.7`，recipe 是 idle `.15`、roam `.03`、rule `.12`、old `.4`、recent `.3`；dynamic/refreshed 均 false，训练不会看到真实 Hunter 的 web 行为分布。两臂共同从相同 Bun392 g0 退化，使 B 比某一臂特有超参数更有解释力。

## 假设排序与判别

### A. Bun 任务奖励与 Hunter 目标错位（最可能）

**支持：** g32 reward 上升而 Hunter safe 由 19 降到 1（base）/7（recipe）；recipe 在 g128 已达到 carry/steal `16/16` 而 Hunter 仅 `4/192`；base g512 full 任务死亡清零但 Hunter death/self 爆炸。源码中的任务奖励直接奖励进入敌方区域、偷取、携带和 capture，Hunter 评估要求安全击杀和不自炸，两者并不等价。

**反证/边界：** reward 还含安全、击杀和自炸惩罚；g0 任务本身很弱，因此任务上升不必然是“错误”。仅凭 64/192 局不能量化真实期望差距。

**最小实验：** CPU 离线用同一 g0、g32、g128、g256 checkpoint 重放相同 Bun seed/spawn，逐 tick 分解 `capture/steal/progress/safety/death/self/kill` 的 reward 贡献，并对同一轨迹计算 Hunter funnel；只做反事实统计，不改训练。若任务 reward 增长集中在 progress/steal/capture，而 safety/kill contribution 不增且 Hunter 同步下降，A 得到直接支持。

### B. 课程/对手分布冲突

**支持：** 两臂的课程都不是 web Hunter 分布，且共同使用冻结 old/rule 类对手；recipe 更偏 full/carry/near，base 更偏 danger/full-ambush，但两者均在 g32 退化。训练内 reward/done 上升不能解释真实 Hunter 三档。

**反证/边界：** 这是离线配置取证，不是实际每个 lesson/opponent 的采样计数；固定对手可使分布稳定，不能直接证明因果。

**最小实验：** CPU 从 rollout reset key/lesson/opponent 记录生成 deterministic 统计，分别按 checkpoint 前缀计算 lesson 与 opponent 真实频率；将相同 g0 policy 在每类训练对手与 web Hunter 分层评估，观察是否只有某些训练对手能力提升。若固定 old/rule 上升而 Hunter 不变，B 得到支持。

### C. PPO 累计更新造成灾难性遗忘

**支持：** base actor/critic 全树漂移到 g512 `.773/1.020`、g768 `.992/1.243`，安全崩塌最重；recipe g1536 reward 下跌且 drift `.431/.575`。早期 g32 已退化，长期更新会进一步扩大损失。

**反证/边界：** recipe 真实更新只有 32/gen，却同样 g32 退化；g192/base 的 safe 有短暂回升，说明不是单调“步数阈值”。现有 state 没有旧 g0 KL/entropy 轨迹。

**最小实验：** 只读地按实际 Adam update count（base 128、recipe 32）重标横轴，比较每 1024 updates 的 Hunter/Tactical 变化；从已提交 state 做 CPU 参数距离和离线 logits KL（固定输入 manifest），寻找能力首次断点是否与漂移阈值同步。若同 update 数而非 generation 对齐，C 得到支持；否则 C 是放大器。

### D. Top25 优势筛选制造捷径

**支持：** recipe 唯一使用 `adv_top_frac=.25`，其任务成功更快、reward 更高、实际更新更少；Top25 按绝对标准化优势保留行，可能偏向容易获得高任务回报的轨迹。

**反证：** base `adv_top_frac=1.0` 也在 g32 退化，D 不能解释共同退化；recipe 的 Hunter 仍比 g0 差，且 Top25 不是两臂共同因素。

**最小实验：** 只对已保存 rollout/adv（若 marker 完整）统计全量行中 Top25 的 lesson、opponent、death/self、steal/capture 构成，与其余 75% 做 bootstrap；不重新训练。只有 Top25 对任务事件显著过采样且安全事件欠采样时，D 才升为 recipe 特有因素。

### E. 新 Bun 输入通道/道具头扰动旧表征

**支持：** Bun 新输入尾部的相对漂移高于动作 heads：base g32 actor/critic `.163/.355`，g768 `.856/1.020`；recipe g1536 `.524/.698`。Bun24/global24 中包含包子、道具、复活和能力语义，迁移只复制共享空间/global 前缀，其余是 seeded native initialization。g32 早期退化与新通道开始学习的时间一致。

**反证/边界：** recipe 的新输入尾部漂移小于 base 但仍退化；漂移与能力下降相关不等于方向错误；当前没有逐通道 logits/KL 或冻结新通道反事实。

**最小实验：** CPU 对固定 authority states 计算 g0→各点的 spatial shared、Bun-only token、global tail、ability/action head logits KL；用 checkpoint 参数做“零出力新通道”反事实 forward（不更新、不评估 web 之外的语义）。若只屏蔽新通道即可恢复 g0 Hunter logits/安全 funnel，E 得到支持。

### F. Critic 或共享梯度问题

**支持：** 两臂 Critic 漂移始终大于 Actor；base g768 Critic 相对漂移 1.243，recipe g1536 为 .575。Critic 的 value target 参与 GAE，错误价值会改变 Actor 的优势方向。

**反证/边界：** 两臂都声明独立 Actor/Critic，PPO 源码的 `_ppo_loss` 通过 `split_forward` 分别使用 actor logits 与 critic value，现有梯度隔离测试通过；独立 Critic 仍退化，因而不能沿用“共享 Critic 是唯一根因”。此外 Critic 漂移可能是 reward/分布冲突的结果。

**最小实验：** 对固定 batch 做 CPU 自动微分检查 `d(policy_loss)/d(critic)=0`、`d(value_loss)/d(actor)=0`，再以已保存 g0 batch 计算 value calibration/GAE sign flip；若隔离成立而 sign flip 随 Hunter 崩塌出现，F 是放大器而非首因。

## 解释与下一步（不阻塞当前 GPU）

现有证据回答了“Bun 任务学好了但 Hunter 坏了”：recipe 的任务曲线明确上升，Hunter 与 Tactical 安全没有同步上升；base 任务也改善但 Hunter 在 g512 发生死亡/自炸爆发。不能称作“两边都坏”，也不能称作“训练完全成功”。

建议仅在 CPU 异步开展上述四类离线反事实，并把结果作为监控附录，不设启动门禁、不等待 GPU。若后续需要新训练，优先设计能同时保留 Hunter safety/kill 的单变量目标约束或分层课程；在此根因报告完成前不热改当前配方、不把中间 checkpoint 选成 winner。第二阶段动态自对弈不因本报告提前开启。

## 运行与安全证据

```text
CUDA_VISIBLE_DEVICES='' /mnt/jpfs/afs/wangyaqi/code_room/qqt-gpu-sim/.venv/bin/python \
  scripts/je03_bun392_retention_conflict_analysis_20261011.py --self-test
CUDA_VISIBLE_DEVICES='' /mnt/jpfs/afs/wangyaqi/code_room/qqt-gpu-sim/.venv/bin/python \
  scripts/je03_bun392_retention_conflict_analysis_20261011.py --run
```

self-test 通过；full run 生成同目录 JSON，JAX 仅在 pickle 反序列化时解析，CUDA 被禁用。报告生成时现场进程仍为：dispatcher `370452`、CPU evaluator `370455`、base `380422`（GPU 0--3）、recipe `379993`（GPU 4--7）；`nvidia-smi` 显示约 30--32 GiB/card，未发送任何停止信号。
