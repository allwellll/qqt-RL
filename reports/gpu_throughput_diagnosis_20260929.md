# GPU 吞吐诊断 — Bun Tactical v2 反事实训练 (2026-09-29)

## 结论 (TL;DR)

反事实 critic 数据生成阶段 **不是 GPU 计算受限**，而是 **冷编译主导 + 热态下 host-Python 规则机器人循环受限**。
在单张 H200 (143771 MiB) 上逐级真实测量：所有档 SM 均值 < 1.5%、中位 0%、显存峰值 ≤ 1.2 GB。
把 `state_batch` 从 1 调大既不提升单状态吞吐，又会破坏与 legacy 的位精确等价，因此**生产固定 `state_batch_size=1`**。

真正有效的吞吐杠杆是：**持久 XLA 编译缓存 (热态 ~2.7×) + 常驻 worker + 4-GPU 数据并行 (~4×)**，语义位精确不变。

## 证据

### 单卡逐级扫描 (runs/scan_20260929/scan_findings.json)

| 档 | 引擎 | state_batch | 冷/热 | 墙钟 (s) | SM 均值 | 显存峰值 |
|----|------|-------------|-------|----------|---------|----------|
| L4 | legacy | 1 | 冷 | 301.25 | 1.44% | 989 MiB |
| L7 | batched | 1 | 冷 | 238.87 | — | — |
| L0 | batched | 4 | 冷 | 235.38 | 0.59% | 969 MiB |
| L5 | batched | 4 | **热** | **70.67** | 1.39% | 839 MiB |

- `batched b1` vs `legacy`：冷态 238.87s vs 294s ≈ **1.2×**，且**位精确等价** (atol 1e-6)。
- `batched b4` 热 vs 冷：70.67s vs 232s ≈ **3.3×**，纯缓存差异。
- SM 恒 < 1.5%、显存 ≤ 1.2 GB → 计算/显存都不是瓶颈。

### 等价性矩阵 (关键契约锁)

- `legacy_gpu0` ≡ `batched_b1_gpu0`：**位精确等价**（保动作/标签/seed 契约）。
- `legacy` vs `batched_b4`：**不等价** — deterministic_objective 偏差达 4.0、policy_probability 1.31e-3。
- 根因：`state_batch>1` 改变 transformer matmul 分块/融合形状 → 前向 ~1.3e-3 舍入差 → 40-tick 贪婪 rollout 近似平手动作翻转 → objective 混沌放大。**非逻辑/seed/标签 bug**。
- 对应回归测试 (`tests/test_counterfactual_batch_equivalence.py`)：
  - `test_batched_batch1_matches_legacy_at_rollout_horizon` → 差异集合为空。
  - `test_batched_multistate_batch_diverges_from_legacy_at_rollout_horizon` → 锁定 batch>1 发散信号，防止有人误改默认并声称等价。

## 选定生产配置

```
counterfactual_engine            = batched
counterfactual_state_batch_size  = 1        # 与 legacy 位精确，且比 legacy 循环快 ~1.2×
persistent_xla_cache             = enabled  # candidate 级稳定目录，跨 cycle 复用
resident_worker                  = enabled  # 消除每 cycle 进程/JAX 启动，保持缓存热
critic_batch_size                = 64       # 契约锁定 (改动影响优化动力学=策略相关)
data_parallelism                 = 4 GPU 各跑 1 个独立候选
```

理由：`batch>1` 无吞吐收益且破坏等价，故回退到 1。SM 仍将偏低——工作负载本质 host-bound
(微网络 + horizon=40 规则 bot)，**不为利用率而添加无用计算**（符合"高有效吞吐而非假满载"目标）。

## 生产线上实测复现 (smoke, 2026-09-29)

在 4-GPU 全 pipeline 2-cycle smoke 中，逐档实测反事实阶段墙钟：

| 档 | cycle_001 (冷) | cycle_002 (热) | 加速 |
|----|----------------|----------------|------|
| control | 243.6 s | 88.9 s | 2.74× |
| safe_low | 246.0 s | 91.0 s | 2.70× |
| safe_medium | 247.8 s | 90.1 s | 2.75× |
| safe_high | 247.3 s | 90.5 s | 2.73× |

持久 XLA 缓存目录 ~283 MB/候选、candidate 级稳定路径，跨 cycle 复用被证实。这与扫描预测
(3.3×) 同量级；差异来自 smoke 中缓存尚未完全填满 (仅跑 2 cycle) 以及 host-bound 残差。
