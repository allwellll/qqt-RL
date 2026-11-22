# 单卡 DCU 运包训练效率实测（2026-11-15）

本轮是用户明确要求的单卡性能诊断：使用真实 SP_web u23302 / u930，执行有界采样和
可丢弃的 Actor/Critic PPO 更新。没有持续训练、保存新权重或更新网页模型；原有停训策略保持关闭。

**本轮运包阶段最快可用结果：单卡 4096 环境，纯采样 31,889 环境步/秒，
连续采样 + PPO 实测 9,691 环境步/秒。** 三轮连续更新和重置检查通过。
2048 环境为 9,213 步/秒，约为峰值的 95%，显存分配池约一半。
这是本轮测试范围内的最好值，不是硬件理论上限，也不是八卡吞吐预测。
[机器可读结果汇总](../reports/dcu_single_card_summary_20261115.json)。

## 口径与环境

- 一张 BW DCU，65520 MiB 显存，现有 DTK 26.04 / JAX 0.6.0 环境。
- 使用 `.venv` 与 `scripts/dcu_env.sh`；没有安装、替换或升级系统训练依赖。
- 完整运包地图，真实双模型对战；B 阶段 75% 原始 u930 拦截对手、25% 普通 SP 对手。
- 一步指一个双人环境推进一个 tick，**不把两个玩家重复算两步**。
- 每轮 rollout 64 tick，PPO minibatch 512，一次完整扫描所有样本；没有减少 PPO 工作量换吞吐。
- 纯采样速度不含 PPO；“合计估算”用采样与 PPO 分别计时的中位数相加计算。
  连续采样、更新实测另列；都不含首次编译、模型加载和检查点写盘。
- 三次稳定采样取中位数；PPO 编译后两次计时。单一 seed 和短轮通过，不代表长期稳定性或模型质量。

## 环境数扫描

以下均为初版采样器、原有保守编译配置 `safe`：backend optimization level 0。
这些扫描中学习方按整个环境 batch 前向，尚未使用后续新增的 `--policy-chunk 512`。

| 采样器 / 单卡环境数 | 纯采样步/秒 | 采样 + PPO 合计估算步/秒 | 最大 logp 误差 | 一致性检查 |
|---|---:|---:|---:|---|
| 原分段采样 / 128 | 1,860 | 1,518 | 0.186162 | 拒绝，仅诊断计时 |
| fused / 512 | 6,066 | 3,509 | 0.00001359 | 通过 |
| fused / 1024 | 8,409 | 4,186 | 0.021429 | 仅通过首轮，后续验证见下文 |
| fused / 2048 | 11,580 | 4,848 | 0.031136 | 拒绝，仅诊断计时 |
| fused / 4096 | 13,507 | 5,159 | 0.095102 | 拒绝，仅诊断计时 |

这里“原配置”是本轮开始时的新运包入口，不是历史空笼击杀任务。
原配置也出现重推理偏差，因此 1,518 只是耗时参考，不能宣称它是可用训练吞吐。
初版 2048/4096 虽然采样更快，但没有通过原有门槛，**这些初版配置不作为正式训练推荐**。
下表分批推理修复后的新配置已重新验证通过。
没有放宽门槛：最大 logp 差 ≤ 0.03、value 差 ≤ 0.1，同时拒绝非有限值和道具队列溢出。

原始结果见 `reports/dcu_carry_bench_{baseline_e128,fused_e512,fused_e1024,fused_e2048,fused_e4096}_20261115.json`。
其中早期 baseline 报告的 `training_environment_steps_per_second` 有估算数值，
应结合 `inference_admitted:false` 解读；后续报告在拒绝时该字段为 null。

## 通过连续验证的优化配置

`fused + throughput + policy_chunk=512 + value_chunk=512`：

| 阶段 / 环境数 | 纯采样步/秒 | 连续采样 + PPO 实测步/秒 | 首轮及连续更新最大 logp/value 差 | 重置 |
|---|---:|---:|---|---|
| courier / 2048 | 26,514 | 9,213 | 全部 0 / 0 | 2048 局全部通过 |
| courier / 4096 | 31,889 | 9,691 | 全部 0 / 0 | 4096 局全部通过 |
| interceptor / 2048 | 26,903 | 9,337 | 全部 0 / 0 | 2048 局全部通过 |

[2048 实测报告](../reports/dcu_carry_bench_chunked_e2048_20261115.json)、
[4096 实测报告](../reports/dcu_carry_bench_chunked_e4096_20261115.json)。
2048 分开计时合计估算 9,043 步/秒，4096 为 9,607 步/秒，与连续测量接近；连续测量是本文优先采用的训练吞吐。
以上配置的重置验证均把全部环境放在 t=2399，rollout 后所有环境 t≤63，完整执行了超时结束和重置。
运包两档分别测了三轮连续更新；拦截阶段测了两轮，起点为 u930、对手为冻结 SP_web。
[拦截阶段实测报告](../reports/dcu_carry_bench_interceptor_e2048_20261115.json)。
该阶段稳定采样取两次中位数，其余阶段为三次。

## 后续验证发现

初版 fused / 1024 切到 `throughput` 后，采样约 21,139 步/秒，PPO 约 4.77 秒，
连续首轮计时约 8,501 步/秒。但连续验证时最大 logp 差变为 0.039150，
超过 0.03，因此此版本同样拒绝推荐；没有放宽阈值。
失败日志：`runs/carry-bench-throughput-e1024-20261115.log`。
此结果推动了固定学习方推理 batch 的改造，不能把 21,139 当成最终可用吞吐。

## 已实施的优化

1. 用设备端 `pmap + lax.scan` 执行完整 rollout，避免 Python 每 tick 多次调度模型、观测和环境。
2. Critic 不决定动作，改为收集后对同一份冻结 Critic 批量前向；默认每批 512，GAE 仍在设备上。
3. 完整地图 reset 使用静态模板，只按原生随机 key 规则重新选出生点。CPU 下对 64 个 seed
   逐叶比较，与原生 reset 完全一致；没有移除砖块、运输、复活、对手或奖励。
4. 保留旧采样器供诊断，提供 `--collector baseline|optimized|fused`，训练默认 fused；实测结束后按用户要求将通用多卡入口改为每卡 2048 环境。
5. 学习方推理新增 `--policy-chunk 512`，将并行环境数与网络前向 batch 分离，
   尽量与 PPO minibatch 保持相同计算形状；仍须通过重推理验证。
6. 提供进程内可选编译配置 `--compiler-profile throughput`，不覆盖用户显式指定的 `XLA_FLAGS`。

在初版 safe 扫描中，采样提升后 PPO 逐渐成为限制：1024 环境采样约 7.79 秒，PPO 约 7.86 秒。
4096 环境采样约 19.41 秒，PPO 约 31.40 秒；继续加环境并不能等比例提高完整训练吞吐。
显存也不是唯一限制：1024 的 JAX pool 约 8.50 GiB，4096 约 32.50 GiB，
这些是分配池统计，不是整卡显存实时占用。

## 单卡建议

当前 64 GiB 卡若以最高环境吞吐为目标，选择 courier / 4096；若希望留更多显存余量，
选择 2048。拦截阶段目前验证的是 2048，同样使用 throughput 和两个 chunk=512。
4096 比 2048 的完整训练吞吐只高约 5.2%，因此不建议仅为填满显存继续加环境。
2048 / 4096 的 JAX pool 分别约 16.50 / 32.50 GiB，峰值活跃分配约 8.28 / 15.88 GiB。
`rocm-smi` 在 4096 PPO 期间的一次快照为设备利用率 83.8%，不是整轮平均值。
首次 rollout 含编译约 157 / 178 秒，首次 PPO 含编译约 36 / 45 秒，均已从稳定吞吐排除。

不占用 DCU 的单卡配置检查：

```bash
.venv/bin/python -m scripts.train_dcu_jax_carry plan \
  --stage courier --devices 0 --envs 4096 --ticks 64 --minibatch 512 \
  --collector fused --policy-chunk 512 --value-chunk 512 --compiler-profile throughput
```

按用户在实测后的要求，通用训练入口和八卡脚本默认改为**每卡 2048 / throughput**，
两个 chunk 仍为 512。单卡追求本轮最佳速度时显式传入 4096。
默认值修改未重新启动训练；八卡吞吐和完整训练稳定性仍需在八卡机器上做首轮实机检查。
历史 JSON 里的源码 SHA 对应各次实测版本，不是后续修改默认值后的源码。
参数中的 `XLA_FLAGS` 若由调用方显式设置，会优先于 profile；对照 JSON 中的有效值复现。

## 复测

以下仅启动有界诊断，更新产生的参数全部丢弃，不产生检查点：

```bash
cd /root/private_data/qqt_rl
source scripts/dcu_env.sh
ulimit -c 0
python -u -m scripts.benchmark_dcu_carry \
  --device 0 --stage courier --collector fused \
  --envs 4096 --ticks 64 --minibatch 512 --value-chunk 512 --policy-chunk 512 \
  --compiler-profile throughput --repeats 3 --ppo \
  --pipeline-repeats 3 --validate-reset \
  --report reports/carry-benchmark-NEW_TIMESTAMP.json \
  --acknowledge-benchmark
```

报告路径必须不存在。`--diagnose-mismatch` 仅允许对未通过配置继续测耗时，
不能绕过训练入口检查。首次采样包含编译，单独记录；不要用编译期吞吐判断稳定速度。
`--pipeline-repeats` 连续更新临时学习方，并在计时区间外核对每轮采样与独立前向。
`--validate-reset` 把所有环境置于原生超时前一 tick，实际执行 reset 并核对重置后 tick。

正式训练仍使用 `scripts/train_dcu_jax_carry.py train --devices 0 ...`，
需根据用户之后的开训要求处理已有停训策略；本次性能测试不启动长期训练。
保存 Actor/Critic、不保存优化器，以及检查点预算详见
[两阶段训练文档](dcu_carry_interceptor_training_20261115.md)。

## 回归验证边界

[验证汇总](../reports/dcu_carry_performance_validation_20261115.json)：新增快速重置、参数门禁等
16 项 CPU 测试通过；原三组回归独立进程运行 36 项通过。但合并运行时为 50 通过、2 失败，
失败均为旧分段采样器的小型随机模型重推理检查。其具体编译/进程状态原因尚未定位，
不能宣称 CPU 回归全部通过，也没有删除断言或放宽容差。
当前可用配置的数值依据是上表真实 DCU 模型的连续逐轮重推理检查。
CPU 重跑显式隐藏 DCU，关闭编译缓存和 simplified JAXPR constants；日志保留在 runs。
