# 高吞吐正式训练启动报告 — Bun Tactical v2 Safe-Aggression (2026-09-29)

## 1. 一句话摘要

新 lineage `bun_high_throughput_20260929`（360 cycle × 4 档，4-GPU 数据并行）已通过 2-cycle 全
pipeline 真实 smoke 验证并**已启动持久化正式训练**。config 引擎 = batched / `state_batch_size=1`
（与 legacy 位精确）、持久 XLA 缓存（热态 ~2.7×）、常驻 worker。语义（动作/标签/seed 契约）不变，
仅改吞吐工程与实验规模。**这是一个明确标记的新实验，不是旧 run 的等价延续。**

## 2. 新 run 契约与 lineage

| 项 | 值 |
|----|----|
| run_root | `runs/bun_high_throughput_20260929` |
| checkpoint_root | `checkpoints/bun_high_throughput_20260929` |
| manifest.json sha256 | `65353397534625e5b3ec8f1172e556bdd00083659f2fd63e0d2794bd13246cf4` |
| target_cycles / segment | 360 / 90（4 段，可分段续跑） |
| 引擎 / state_batch | batched / **1**（位精确 legacy） |
| critic_batch_size | 64（契约锁） |
| 数据并行 | 4 GPU 各 1 独立候选 (control/safe_low/safe_medium/safe_high) |
| 解释器 | `qqt-gpu-sim/.venv/bin/python`（共享 H200 CUDA 环境，只读引用） |

### 初始 checkpoint（warm-start，逐档续自同名 arm 的 latest）

`init_scheme = per_arm_latest_warm_start`，源 run
`qqt-gpu-sim/ckpt/bun_safe_aggression_v7_20260928_r6/production_360`（只读引用）。
manifest 的 bootstrap 段登记每个 warm-start 文件的 sha256，各档 cycle_000 checkpoint 与源逐字节一致：

| 档 | cycle_000_actor sha256 | seed |
|----|------------------------|------|
| control | `4a4b903d…7ea7` | 202609280001 |
| safe_low | `72a12a2f…98cb` | 202609280101 |
| safe_medium | `16e64d38…52d0` | 202609280201 |
| safe_high | `c3bcb475…4a870` | 202609280301 |

> 说明：warm-start 各档源为对应旧 arm 的 **latest**（非 eval-matched "best"）。因此本 run 起点
> 是"最新权重"，未声明为质量最优。这是明确记录的续训起点选择。

## 3. Smoke 验证（`runs/bun_high_throughput_smoke_20260929`，2 cycle × 4 档，全 GREEN）

- **≥2 完整 cycle**：4 档全部 `budget_complete`，写出 `COMPLETE.json`（target_cycles=2）。
- **checkpoint 可读 / 有限 / 权重变化**：cycle_002 全部 finite=True；actor 73/75 张量变化，
  max_abs_delta 由 cycle_1 的 ~5e-5 累积到 cycle_2 的 ~1e-4（真实学习信号）。
- **规则 bot 冻结**：跨 cycle eval 的 opponent `implementation_hash` 恒为 `e4e0dadb…64b5`。
- **无 seed 冲突**：replay_manifest `seed_leakage=False`（strict 检查开启）。
- **热缓存复用（candidate 级持久 XLA cache）**：反事实阶段冷→热 2.7×（下表）。
- **常驻 worker + 通用 Bot 契约**：worker_registry 记录常驻 PID 与 adopt 状态；registry 三 bot
  (tactical_v2 / random_roam / browser_model) 跨语言 fixtures 通过。
- **Web/JS 无回归**：`npm test` 11 个套件全通过。

| 档 | 反事实 cold (cycle_001) | 反事实 warm (cycle_002) | 加速 |
|----|-------------------------|--------------------------|------|
| control | 243.6 s | 88.9 s | 2.74× |
| safe_low | 246.0 s | 91.0 s | 2.70× |
| safe_medium | 247.8 s | 90.1 s | 2.75× |
| safe_high | 247.3 s | 90.5 s | 2.73× |

### smoke 过程中修复的两个真实阻塞（TDD/根因，非绕过）

1. **子进程解释器路径**：runner 假定 `REPO/.venv/bin/python`，新项目目录无 colocated venv →
   改为 `os.environ.get("BUN_V2_PYTHON") or sys.executable`（沿用启动候选的解释器）。
2. **冻结运行时缺 fixture**：`create_default_registry` 运行时按 frozen-runtime 根读取
   `tests/fixtures/bot_contract_cases.json` 计算 bot 指纹 → `copy_runtime` 补齐该 fixture 的冻结。

两处均为自洽正确性修复，未引入旧仓绝对路径；相关 pytest（per-arm init / supervisor-finalizer /
bot-contract）9 项通过。

## 4. 正式训练启动证据

| 项 | 值 |
|----|----|
| supervisor tmux session | `bun_prod_20260929`，PID **254696**（flock 单一所有者，可续跑adopt） |
| watchdog tmux session | `bun_prod_watch_20260929`，PID **254700**（30s 巡检，1800s 停滞告警，含有限性校验） |
| 候选 worker PID | control 254830 / safe_low 254831 / safe_medium 254832 / safe_high 254833（各绑一 GPU）|
| 启动首阶段 | 4 档均 `cycle0_fixed256`（cycle-0 基线 256-game eval）|

监控产物（watchdog 写出）：
- `runs/bun_high_throughput_20260929/watchdog_heartbeat.json` — 每 30s 刷新的整体健康 + 各档
  status/phase/cycle + 最新 checkpoint(号/mtime/有限性) + GPU 快照。
- `runs/bun_high_throughput_20260929/watchdog_notify.jsonl` — 仅状态跃迁（完成/失败/停滞）追加，
  单一所有者（flock）避免重复告警。

## 5. 吞吐预期

热态稳态每 cycle 墙钟（4 档并行）：反事实 ~90s + critic 校准 + joint(32env×128step×4) +（仅
milestone cycle 的 4-变体 eval）。EVAL_CYCLES 稀疏（0,1,2,4,8,16,…），多数 cycle 跳过昂贵 eval，
故稳态显著快于 smoke 观察的含-eval cycle（~8.9 min）。相对旧冷缓存串行 run，有效吞吐 ~10-13×，
语义位精确不变。SM 仍将偏低——host-bound 本质，不为利用率添加无用计算。

## 6. 安全与边界（本 run 遵守）

- 旧仓 `qqt-gpu-sim` 仅作只读引用 + checkpoint 源，未重启/修改其 run 运行时文件。
- 无 checkpoint 权重进入 Git；未 push/发布（待授权）。
- Git 用户保持 `allwellll`；未写入任何凭据/token。
- 遇 OOM/NaN/不等价立即回退——本 run 选 `state_batch=1` 即因 batch>1 破坏等价。
