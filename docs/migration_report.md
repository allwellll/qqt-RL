# qqt-RL 迁移报告（20260928）

## 范围与基线

- 源目录：旧仓库 checkout
- 目标目录：当前仓库 checkout
- 正确基线：`logs/bun_safe_aggression_v7_20260928_r6/frozen_runtime_repair_r1_20260928_1515`
- 修复差异：相对原 r6 frozen runtime，采用修复后的 `jax_bomb/bun_seed_namespace.py` 与 `scripts/run_bun_tactical_v2_candidate.py`，保留严格 split 泄漏检查。

## 源文件映射

| 来源 | 目标 | 说明 |
|---|---|---|
| frozen repair `jax_bomb/bun_env.py` | `jax_bomb/bun_env.py` | Bun 状态机、课程、事件与奖励归因 |
| frozen repair `jax_bomb/bun_safety.py` | `jax_bomb/bun_safety.py` | 联合动作安全性与放泡逃生判定 |
| frozen repair `jax_bomb/bun_seed_namespace.py` | `jax_bomb/bun_seed_namespace.py` | cycle9 修复后的 v7 seed allocator |
| frozen repair `jax_bomb/bun_critic.py` | `jax_bomb/bun_critic.py` | 独立 Critic 与辅助头 |
| frozen repair `jax_bomb/bun_rule_bot.py` | `jax_bomb/bun_rule_bot.py` | 40-tick 安全战术规则对手 |
| frozen repair `jax_bomb/bun_curriculum.py` | `jax_bomb/bun_curriculum.py` | Bun 课程配置 |
| frozen repair `jax_bomb/bun_expert.py` | `jax_bomb/bun_expert.py` | 规则专家 |
| frozen repair `jax_bomb/bun_frozen_opponents.py` | `jax_bomb/bun_frozen_opponents.py` | 冻结对手适配，默认改为仓库内固定源码 |
| frozen repair `jax_bomb/bun_tactical_family.py` | `jax_bomb/bun_tactical_family.py` | 对手 family 参数 |
| frozen repair `jax_bomb/bun_tactical_labels.py` | `jax_bomb/bun_tactical_labels.py` | 战术辅助标签 |
| frozen repair `jax_bomb/jax_env.py` | `jax_bomb/jax_env.py` | Bun 复用的底层爆炸/碰撞内核 |
| frozen repair `jax_bomb/jax_net.py` | `jax_bomb/jax_net.py` | Actor 网络 |
| frozen repair `jax_bomb/jax_train.py` | `jax_bomb/jax_train.py` | PPO/GAE 与 rollout 核心 |
| frozen repair `jax_bomb/levels.py` | `jax_bomb/levels.py` | Web/训练同源地图加载 |
| frozen repair `jax_bomb/platform.py` | `jax_bomb/platform.py` | 加速器保护；新增显式 CPU smoke 开关 |
| frozen repair `jax_bomb/bun_train.py` | `jax_bomb/bun_train.py` | Bun PPO 兼容入口 |
| frozen repair `scripts/run_bun_tactical_v2_candidate.py` | 同路径 | v7 runner；抽取配置/I/O/进程模块并加强恢复校验 |
| frozen repair `scripts/launch_bun_safe_aggression_v7.py` | 同路径 | 四臂 supervisor；新增锁、重启收养、持久 retry、失败清理 |
| frozen repair `scripts/finalize_bun_safe_aggression_v7.py` | 同路径 | 完成汇总与 best/latest 发布 |
| frozen repair `scripts/generate_bun_tactical_v2_critic_data.py` | 同路径 | tactical-v2 counterfactual 数据 |
| frozen repair `scripts/generate_bun_critic_data.py` | 同路径 | Critic 数据生成公共逻辑 |
| frozen repair `scripts/concat_bun_critic_replay.py` | 同路径 | replay 合并；增强 raw/uint32 双重泄漏检查 |
| frozen repair `scripts/train_bun_critic.py` | 同路径 | 独立 Critic 训练 |
| frozen repair `scripts/train_bun_separate_ac.py` | 同路径 | Actor/Critic/Target Critic 联合微更新 |
| frozen repair `scripts/adapt_bun_critic_onpolicy.py` | 同路径 | on-policy Critic 适配 |
| frozen repair `scripts/train_bun_bc.py` | 同路径 | selective BC |
| frozen repair `scripts/eval_bun_tactical_opponent.py` | 同路径 | 固定种子评估 |
| frozen repair `scripts/augment_bun_critic_aux_labels.py` | 同路径 | Critic 辅助标签 |
| source `scripts/prepare_bun_safe_aggression_v7.py` | 同路径重写 | 外部 bootstrap、自包含目录、1..360 cycle |
| source `web/sim.js`、`web/bun_rule_bot.js` | 同路径 | Bun Web 仿真与规则 Bot |
| source Bun06 level/map/element assets | `web/assets/` | 仅保留 Bun06 必需地图和小型贴图 |
| 新增 | `qqt_rl/training/` | 配置、family、进程日志、hash、原子提交 |
| 新增 | `scripts/export_bun_web_model.py` | 可信 JAX Transformer checkpoint → 浏览器 JSON |
| 新增 | `web/index.html`、`web/app.js`、`web/server.js` | Bun-only 真人/模型测试页面与静态服务 |

## 明确删减

- 未复制 `logs/`、`ckpt/`、`data/`、`.venv/`、缓存、测试产物与历史实验。
- 未复制 ONNX/JSON 模型权重、WASM runtime、音乐及非必要大型二进制。
- 未复制非 Bun 地图入口、旧模式页面、legacy shell 实验矩阵、通用多卡入口和无关 benchmark。
- 未复制依赖旧实验绝对路径的 frozen binding 测试；以自包含 hash/seed/runner 测试替代。

## 安全与可恢复性

- `critic_seed_plan_v7` 限制 cycle `1..360`，candidate 和 split 使用独立 namespace，并测试 raw/`uint32` 全域无碰撞。
- replay 合并仍在 `scripts/concat_bun_critic_replay.py` 中执行严格 train/validation/test seed 交叉检查。
- Actor、Critic、Target Critic 与 JSON manifest 使用临时文件 + `os.replace` 原子发布。
- 正式运行默认要求加速器；仅 `scripts/smoke_train_cpu.sh` 显式允许 CPU。
- Git 忽略 `runs/`、`data/`、`checkpoints/`、`web/models/`、缓存和日志。

## 测试结果

- Python import/bytecode：`python -m compileall -q jax_bomb qqt_rl scripts tests` 通过。
- Python 测试：最终快照共收集 87 项，`pytest tests -q` 全部通过；仅保留 `jax_env.py` 中既有 NumPy/JAX bool 反转弃用警告，不影响结果。
- CPU 训练 smoke：`scripts/smoke_train_cpu.sh` 通过，完成 1 次真实 PPO 更新、16 个 trainable-agent steps，最终约 3,237 steps/s。
- 训练准备集成：临时副本中成功生成 1-cycle 四臂 manifest、v7 seed 配置、冻结 runtime 与四套 `cycle_000` checkpoint。
- Node Bun tests：Bun06 地图/碰撞/规则、规则 Bot fixtures/FSM/性能全部通过。
- Web HTTP smoke：主页、模型入口、`/healthz` 与 malformed URL 处理通过。
- Secret scan：未发现 GitHub token、AWS key、私钥或带凭据 URL；`.git/config` 中 origin 原始值为无凭据 URL，credential helper 仅复制配置项。
- 大文件检查：工作树无大于 5 MiB 的普通项目文件；未提交模型权重、WASM、日志或 checkpoint。
- 单 GPU 真实 smoke：**安全跳过**。2026-09-28 最终检查时，四个 GPU UUID 均存在 `qqt-gpu-sim` 旧正式 run 的计算进程（PID 110618、115336、421411、426173、447603、451085），目标仓库 GPU 进程数为 0。为遵守不中断/不争抢旧四卡训练的硬约束，没有在任一卡启动新 GPU 作业。

## 独立审查

- Spec review 与 quality review 由两个独立审查任务执行。
- 已修复的重要问题：强制 `critic_seed_namespace_version=v7`、raw 与 `uint32` 双重 seed 泄漏检查、run-name/manifest 路径边界、runtime/bootstrap/config 哈希验证、supervisor 单实例锁与 PID 身份校验、持久 retry、失败清理、finalizer 返回码及 checkpoint 校验、三件套恢复提交顺序、SciPy 依赖、Web malformed URL、模型第三动作与 value bin 范围。
- 新增 supervisor/finalizer 行为测试、GAE 边界测试、Web 导出 schema 测试与 prepare 集成 smoke。

## 运行切换边界

- 2026-09-28 审计时，旧仓库的四 GPU 360-cycle 正式任务仍从 `qqt-gpu-sim/logs/bun_safe_aggression_v7_20260928_r6/...` 及对应旧 `ckpt/data` 路径运行。
- 本迁移没有停止进程、修改旧运行文件、迁移 checkpoint 或改变旧路径。
- 当前旧 run 必须继续在旧仓库直到完成；从本仓库创建的全新实验、后续训练和调试才使用 `qqt-RL/runs`、`qqt-RL/data`、`qqt-RL/checkpoints`。

## Git 与远程

- 本地身份：`allwellll <1035628914@qq.com>`。
- 本地 credential helper 与旧仓库一致；报告和提交不记录 helper 内容或任何凭据。
- `.git/config` 的 `remote.origin.url` 为 `https://github.com/allwellll/qqt-RL.git`。
- 本地已创建清晰的 Bun-only 初始提交。
- 已复用旧仓库受保护的 credential helper，以非交互方式成功推送 `main`；仓库配置和提交内容均未写入凭据。远端 `main` 与本地 HEAD 已核对一致。
