# qqt-RL

纯 Bun（抢包子）强化学习仓库，覆盖 JAX 环境仿真、PPO/GAE 训练、独立 Critic/Target Critic、模型评估与浏览器真人测试。

当前代码基线来自 `qqt-gpu-sim` 的 safe-aggression r6 冻结 runtime，并合入 cycle 9 后的 seed namespace/runner 修复。旧仓库中正在运行的 360-cycle 四卡任务继续使用旧路径；新实验与后续调试才使用本仓库。

## 快速开始

```bash
cd /path/to/qqt-RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test]'
```

GPU 环境请按机器 CUDA/JAX 版本安装匹配的 JAX wheel；仓库不固定设备专属 wheel。正式训练默认拒绝静默回退到 CPU。

## 目录

- `jax_bomb/`：Bun 环境、网络、PPO/GAE、安全动作分析、规则 Bot 与 Critic 核心。
- `qqt_rl/training/`：配置、seed、持久 JAX cache、常驻 worker、进程日志和原子 checkpoint 发布辅助层。
- `qqt_rl/bots/`：Python/浏览器共享语义的 Bot schema、生命周期和显式注册表。
- `scripts/`：数据生成、Critic/Actor 训练、评估、单 candidate runner 与四卡 supervisor。
- `tests/`：环境、奖励归因、安全性、Critic、seed namespace 和模块化回归测试。
- `web/`：只含 Bun06 真人测试、规则 Bot、HTTP server 与 Node 测试。
- `runs/`、`data/`、`checkpoints/`：默认运行产物，全部被 `.gitignore` 排除。

## 验证与 Smoke

```bash
python -m compileall -q jax_bomb qqt_rl scripts tests
pytest -q
npm test
bash scripts/smoke_train_cpu.sh
```

CPU smoke 必须显式设置 `QQT_ALLOW_CPU=1`，脚本已自动处理。正式训练不设置该变量，若没有加速设备会直接失败，避免误用 CPU 长训。

## 准备 1..360 Cycle 训练

训练 bootstrap 不进入 Git。将以下文件放到任意外部目录，例如 `/path/to/bootstrap`：

```text
base_actor.pt
base_critic.pkl
base_target_critic.pkl
reference_actor.pt
bc_v2_escape_selective_bc.npz
generic_critic_replay_aux.npz
weak_actor.pt
old_actor.pt
```

准备单 cycle smoke：

```bash
python scripts/prepare_bun_safe_aggression_v7.py \
  --bootstrap-dir /path/to/bootstrap \
  --run-name smoke-1cycle \
  --cycles 1 \
  --segment-size 1 \
  --gpus 0,1,2,3
```

准备正式 360-cycle 运行：

```bash
python scripts/prepare_bun_safe_aggression_v7.py \
  --bootstrap-dir /path/to/bootstrap \
  --run-name safe-aggression-360 \
  --cycles 360 \
  --segment-size 90 \
  --gpus 0,1,2,3

python scripts/launch_bun_safe_aggression_v7.py \
  --manifest runs/safe-aggression-360/manifest.json
```

新 run 默认使用固定 shape 的 `--counterfactual-state-batch-size 4`、Critic batch `64`、每阶段常驻 worker，以及 candidate 级稳定 JAX 编译缓存。可用 `--jax-cache-dir /shared/path` 指定显式共享根目录；缓存目录始终不提交 Git。

runner 保留全局 cycle `1..360`、90-cycle 可恢复分段、固定 milestone 评估、strict train/validation/test seed 泄漏检查，以及 Actor/Critic/Target Critic 的有限值与哈希校验。checkpoint 使用同目录临时文件后 `os.replace` 原子发布。

## 模型评估

```bash
python scripts/eval_bun_tactical_opponent.py checkpoints/model.pt \
  --games 64 \
  --seed 202609280000 \
  --max-steps 300 \
  --json-out runs/eval/model-vs-tactical.json
```

评估结果写入 `runs/`。加 `--per-episode` 会额外保存逐局结果与出生格（用于配对比较）；`--host-workers N` 把规则 Bot 决策与漏斗标签分到 N 个进程，结果与串行逐位一致。如需导出浏览器模型，应将生成文件放入被忽略的 `web/models/`；本仓库不提交大模型权重。

## Transformer checkpoint 对战能力评估

目的：沿训练进程扫描 checkpoint，看战斗能力（安全击杀、自炸、被杀、危险处理）是在提升还是回退。这是 danger_arena 近身战斗探针，**不代表完整抢包子能力**（不含偷包/运包/回家）。

固定协议（与历史 `runs/eval_v2/*.json` 同口径，schema `bun_tactical_opponent_eval_v1`）：

- 课程 `danger_arena=1`，清空可破坏砖块，HP=1，最长 300 tick。
- Actor 固定为玩家 0、argmax 动作、无安全屏蔽；对手固定为玩家 1 的冻结 `bun.tactical_v2`（哈希校验）。
- 出生点由 seed 决定，环境内部随机交换两侧出生位置；同一 seed 下所有 checkpoint 的出生格完全相同（`spawns_identical_across_checkpoints` 核验），因此可做逐局配对比较。

```bash
# 64 局探索性全曲线（phase1 + 全部 phase2_itN + phase2.pt）
python scripts/eval_transformer_checkpoint_sweep.py \
  --run-dir runs/overnight_tf_v2 \
  --out-dir runs/transformer_sweep/overnight_tf_v2/seed20260930_g64 \
  --seed 20260930 --games 64 --device 0 \
  --teacher-json runs/eval_v2/rulebot_baseline.json

# 大样本确认：换 seed，只挑关键点
python scripts/eval_transformer_checkpoint_sweep.py \
  --run-dir runs/overnight_tf_v2 \
  --out-dir runs/transformer_sweep/overnight_tf_v2/confirm_seed20261001_g1024 \
  --checkpoints phase1,3500,4000,6000,final \
  --seed 20261001 --games 1024 --host-workers 48 --device 0
```

`--checkpoints` 支持 `all`、`phase1`、`final`、文件名、单个 iter（`4000`）和区间（`5500-8000`），可组合。所有 checkpoint 在同一进程串行评估，`env.step` 只编译一次；持久 JAX 编译缓存默认放在 `<out-dir>/../.jax_cache`，重跑时跳过编译。已存在且 checkpoint sha256、seed、局数、步数全匹配的 JSON 会被复用（`--reuse-dir` 可指向旧结果目录，`--force` 强制重跑），`--plot-only` 只重画图。参考耗时（单卡 H200）：64 局约 28 秒/点，1024 局约 60 秒/点；首个点多约 70 秒编译。

输出目录：

- `checkpoints/<name>.json`：逐 checkpoint 完整结果（含 `per_episode` 逐局结果与出生格、`runtime` 耗时拆分）。
- `summary.json` / `summary.csv`：汇总指标、Wilson 95% 区间，以及相对 `--reference`（默认 phase1）的逐局配对差值。
- `capability_curves.png`：9 宫格曲线，包括存活安全击杀、安全引爆比、自炸、被对手击杀（物理或因果，取并集）、可避免危险死亡、danger→death、场均放泡、战术解决率、策略熵。红色虚线是规则 Bot 自对弈基线。
- 可选 `--flee-probe`：另在 GPU 上整局跑对 JAX flee bot（阶段 1 训练对手，以逃跑/静止为主）的补充探针，输出 `flee_probe/`、`flee_probe_summary.json`、`flee_probe_curves.png`。它只测追击补刀，params 以 jit 参数传入，数值与主口径不同，不能与主曲线混比。

横轴：阶段 2 iteration。phase1 终点记为 0，`phase2_itN` 记为 N，`phase2.pt` 记为 `phase2.json` 中的 `ppo_iterations`（本次为 8000）。

解读：

- 误差线是局级二元比率的 Wilson 95% 区间。64 局时区间半宽约 ±0.10 到 0.12，相邻点差 0.1 以内基本是噪声，全曲线只用于找候选点和趋势。
- 安全引爆比、战术解决率按事件汇总，场均放泡和熵是均值，这几项不画区间。
- 结论以大样本确认为准：换一个未用于挑点的 seed，局数 ≥1024（区间半宽约 ±0.03）。比较两个 checkpoint 时看 `paired_vs_reference` 的配对差值区间，它扣除了出生格带来的方差，比两个独立区间是否重叠更灵敏。
- 如果某个点只在 64 局扫描里高，大样本下不再高，就是挑点偏差，不能称为提升。

导出一份可信的本地 Transformer checkpoint：

```bash
python scripts/export_bun_web_model.py checkpoints/model.pt \
  --output web/models/model.json \
  --name safe-aggression-model
```

checkpoint 使用 Python pickle 格式，只能加载本机训练并已核对哈希的可信文件，禁止导入来源不明的 pickle。

## Web 真人测试

```bash
npm run serve
```

浏览器访问终端打印的本地地址。蓝方使用 `W/A/S/D` 移动、`Space` 放泡、`R` 重开；红方可使用冻结安全战术 Bot，或在页面中加载上一步导出的模型 JSON。服务健康检查为 `/healthz`。

Bot 扩展契约和 Python/JavaScript 最小插件见 `docs/bot_plugins_zh.md`。

## v2 性能基准

```bash
python scripts/benchmark_v2_pipeline.py \
  --output-dir runs/benchmarks/v2-pipeline
```

基准输出 `benchmark.json`，包含持久缓存冷/热启动、legacy 与固定 shape 反事实 states/s、Critic batch 64/128/256 samples/s、真实 PPO smoke steps/s、逐字段 replay 等价报告和 GPU 利用率采样状态。默认使用 CPU，避免争抢仍在旧仓库运行的四卡任务。

## 训练语义

- 环境 observation 主布局为 `[batch, channel, height, width]`，rollout 再增加 time 维。
- 奖励从 reset 前事件归因，避免跨局污染；安全放泡分 placement 与 resolution 两阶段结算。
- PPO 使用 clipped policy objective、GAE return 与 entropy；独立 Critic 和 Target Critic 不复用 Actor 参数树。
- seed namespace 同时保证原始整数与 JAX `uint32` 表示在 candidate/cycle/split 间不碰撞。

详见 `docs/migration_report.md`。
