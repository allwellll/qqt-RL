# qqt-RL

纯 Bun（抢包子）强化学习仓库，覆盖 JAX 环境仿真、PPO/GAE 训练、独立 Critic/Target Critic、模型评估与浏览器真人测试。

当前代码基线来自 `qqt-gpu-sim` 的 safe-aggression r6 冻结 runtime，并合入 cycle 9 后的 seed namespace/runner 修复。旧仓库中正在运行的 360-cycle 四卡任务继续使用旧路径；新实验与后续调试才使用本仓库。

## 快速开始

```bash
cd /mnt/jpfs/afs/wangyaqi/code_room/qqt-RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test]'
```

GPU 环境请按机器 CUDA/JAX 版本安装匹配的 JAX wheel；仓库不固定设备专属 wheel。正式训练默认拒绝静默回退到 CPU。

## 目录

- `jax_bomb/`：Bun 环境、网络、PPO/GAE、安全动作分析、规则 Bot 与 Critic 核心。
- `qqt_rl/training/`：配置、seed、进程日志、哈希和原子 checkpoint 发布辅助层。
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

runner 保留全局 cycle `1..360`、90-cycle 可恢复分段、固定 milestone 评估、strict train/validation/test seed 泄漏检查，以及 Actor/Critic/Target Critic 的有限值与哈希校验。checkpoint 使用同目录临时文件后 `os.replace` 原子发布。

## 模型评估

```bash
python scripts/eval_bun_tactical_opponent.py checkpoints/model.pt \
  --games 64 \
  --seed 202609280000 \
  --max-steps 300 \
  --json-out runs/eval/model-vs-tactical.json
```

评估结果写入 `runs/`。如需导出浏览器模型，应将生成文件放入被忽略的 `web/models/`；本仓库不提交大模型权重。

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

## 训练语义

- 环境 observation 主布局为 `[batch, channel, height, width]`，rollout 再增加 time 维。
- 奖励从 reset 前事件归因，避免跨局污染；安全放泡分 placement 与 resolution 两阶段结算。
- PPO 使用 clipped policy objective、GAE return 与 entropy；独立 Critic 和 Target Critic 不复用 Actor 参数树。
- seed namespace 同时保证原始整数与 JAX `uint32` 表示在 candidate/cycle/split 间不碰撞。

详见 `docs/migration_report.md`。
