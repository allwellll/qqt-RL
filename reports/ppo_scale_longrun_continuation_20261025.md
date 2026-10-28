# PPO scaled 长程续训 R2（20261025）

## 冻结合同

实验 `PPO-SCALE-LONGRUN-20261025-R2` 从 `runs/ppo_scale_20261025/r1_longrun` 三个 `ppo_scaled_batch/u064` 完整状态续训。每个源点的 state/policy SHA、源 update、Adam累计计数、双流环境/aux/RNG/history字段见 `runs/ppo_scale_20261025/r2_longrun/SOURCE_MANIFEST.json`。源点均已由 R1 GPU/CPU 完成收据确认，R1 不被修改。

唯一观察变量是经过 R1 技术验证的 scaled PPO batch 在长期环境步上的行为；奖励、输入、mask、网络、对手、学习率、optimizer、DAgger 和 KL 均固定。DAgger=0，combat KL=0。

| 项目 | 冻结值 |
|---|---:|
| GPU / env per device | 8 / 16 |
| rollout / global batch | 128 / 16,384 |
| global / local minibatch | 512 / 64 |
| optimizer steps / update | 32 |
| lr / epochs | 7.5e-5 / 1 |
| 新 update / seed | 2,048 |
| 新环境步 / seed | 33,554,432 |
| seed 数 | 3（2026102751/52/53） |
| 新环境步总预算 | 100,663,296 |

按已实测 R1 scaled 稳态约2622 env-step/s 估算约3.55小时/seed、10.66小时总计。短点结果仅观察指标，不作为健康训练的停止条件。

## 评估和接续

每64新 update（约1,048,576新增环境步）保存原子状态并排队16局轻量 JS natural dev；每512 update做完整自然64、Python Tactical v2 64、production JS Hunter难/中/易各64，最终点另行使用独立自然holdout64。categorical训练统计与生产greedy评估分开记录；机会、窗口、取包、交付、道具采样和战斗安全按实际分母记录。CPU/JS使用独立 owner，不等待评估才推进 GPU，也不以 holdout 调参。

## 当前执行收据

预注册和 source seal 已生成；R1 三源状态已实际读取并验证 finite、完整字段、keys=8×16×2、Adam计数和 SHA。R2 技术 smoke/新进程恢复和正式首更新由唯一 GPU owner 执行，状态持续写入同一 namespace；此处只记录已落盘收据，不把计划冒充训练结果。
