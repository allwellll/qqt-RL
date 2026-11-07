# 海光 DCU 迁移

本次验证使用海光容器自带的 Python 3.11.9、DTK 26.04 和
JAX/JAXlib `0.6.0+das.opt1.dtk2604`，没有替换系统 Python、驱动或已有依赖。
项目 `.venv` 通过 `--system-site-packages` 复用海光编译的 JAX，另装
Optax 0.2.4、Chex 0.1.89。不要在这里安装 `jax[cuda12]`，也不要直接升级 JAX。

硬件探测：一张 `BW` DCU，物理显存 65520 MiB。JAX 默认可分配上限约
47.2 GiB；不能把原 H200 的 2048/3072 env 配置直接视作可用配置。

## 环境

在同一基础镜像中创建或重建项目环境：

```bash
cd /root/private_data/qqt_rl
python -m venv --system-site-packages .venv
mkdir -p .cache/tmp .cache/pip
TMPDIR="$PWD/.cache/tmp" PIP_CACHE_DIR="$PWD/.cache/pip" \
  .venv/bin/python -m pip install -r requirements-dcu.txt -e '.[test]'
source scripts/dcu_env.sh
python -m pip check
```

`source scripts/dcu_env.sh` 只改变当前 shell 及子进程。它设置 ROCm 后端、
关闭显存预分配，将编译/下载缓存留在项目目录，并预加载已有
`/opt/mpi/lib/libmpi.so`。这修复镜像中 `libhipfftMp.so.0: undefined symbol:
ompi_mpi_int`；不修改系统库。未正确加载 ROCm 时入口报错，不回退 CPU。

DCU 入口默认设置 `--xla_gpu_autotune_level=0`，并禁用
`constant_folding,multi_output_fusion` 编译 pass，同时使用
`--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true`
兼容模式，作为旧版海光 JAX 的可执行基线。
即使使用这些设置，DUEL 完整采样图首次编译仍约12分钟；这组参数尚未与
默认优化级别完成性能对照。不更改环境规则或 PPO 源码。显式设置的
`XLA_FLAGS` 优先。默认优化设置的诊断也出现过 XLA slow compile 告警，
相应本机日志位于 `runs/dcu-validation/`，不作为版本资产发布。

## 数据

使用 `training-bundle-20261107` 的全部19个真实运行文件及 manifest，约263 MiB。
原始下载命令为 `python scripts/fetch_training_bundle.py`。本容器访问
GitHub Release 下载域名 HTTPS 超时，可改用：

```bash
python -m scripts.fetch_dcu_bundle --mirror https://ghfast.top/
```

下载器从官方 GitHub API 的 HTTPS 元数据取得 manifest 和每个资产的
SHA256/大小，再校验镜像数据。所有文件验证后才安装，拒绝覆盖已有数据。
`.cache/bundle/` 保留已验证下载，便于失败重试。数据、缓存和权重均不入 Git。

## DCU 验证

先用 `rocm-smi --showuse --showmeminfo vram` 确认分配给自己的设备空闲。
以下每条命令独立执行，不同时占用一张卡：

```bash
source scripts/dcu_env.sh
python -u -m scripts.dcu_smoke --line DUEL --device 0 --update \
  --report runs/dcu-validation/DUEL.json
python -u -m scripts.dcu_smoke --line SP --device 0 --update \
  --report runs/dcu-validation/SP.json
```

省略 `--update` 仅验证真实模型前向。带 `--update` 时默认4 env、2 tick、
minibatch 8，执行一次真实 collector、GAE、PPO/Adam，检查有限数值及
Actor/Critic 参数实际变化。参数随后丢弃，不保存模型、不恢复旧队列。
该入口最多32 env、8 tick，仅用于明确请求的迁移诊断。
这与原128 tick、大批量长训在统计和性能上不同，不能作为训练效果证据。

## 正式入口

`python -m scripts.dcu_training` 保留原 portable 入口的参数、训练闸门、
数据/源码校验、固定更新上限、对手池逻辑、`USER_STOP` 及模型保存方式。
它先校验原入口的已知 SHA，再在内存中将 CUDA 初始化替换为 ROCm 初始化；
不修改封存源码或 manifest，不套用 H200 专用 CUDA 故障补丁及其确定性参数。
原入口变化时适配器会拒绝运行，要求重新审阅。

目前 `training_launch_policy.json` 仍关闭正式训练。未来明确决定开训并更新
本地策略后，小批量起步命令示例（此命令未作为正式长训执行）：

```bash
python -u -m scripts.dcu_training --line DUEL --train --gpu 0 \
  --updates 1 --envs 4 --minibatch 8 --out runs/dcu-DUEL-first-round
```

这里 `--gpu 0` 沿用原 CLI，实际选择的是 DCU。正式入口仍使用协议的128 tick；
须从小批量逐级测显存和速度，再定生产配置。输出目录必须新建。
此命令直接联合更新 Actor/Critic，尚无 Critic 独立预热。
Release 仅提供 Actor；Critic、Adam、环境和 RNG 重新初始化。
当前正式保存器仅保存 Actor，不保存 Critic 或完整断点状态。
详见 [原迁移说明](continue_training_on_new_machine.md)。

## 回归检查

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m pytest \
  tests/test_dcu_runtime.py tests/test_portable_training.py tests/test_training_stop_gate.py
```

包括 ROCm 选择、诊断规模限制、Release 解包安全和正式停训闸门。
当前22项针对性测试通过（另含3个subtest）。可版本管理的结果摘要见
[单卡验证记录](../reports/dcu_validation_20261108.json)，本机完整日志位于
`runs/dcu-validation/`。

## 已完成的验证（2026-11-08）

DUEL 已通过真实4 env × 2 tick、minibatch 8 的完整采样、GAE 和一次
PPO/Adam 更新，Actor/Critic 参数变化 L2 分别为0.16857/0.19361，
所有参数、优化器状态及 loss 均为有限值。没有保存模型或恢复旧任务。
结果见 `runs/dcu-validation/DUEL.json`。

SP 的真实环境初始化和 Actor/Critic 前向也已通过，结果见
`runs/dcu-validation/SP-forward.json`；SP 的 PPO 更新尚未验收。
当前单卡验收以已通过的 DUEL 完整训练链路为准。

首次采样含图构建、编译及执行合计865秒，其中 XLA 编译约735秒；
这不是每轮采样速度。编译结果保存在项目 `.jax_cache/dcu/`，同配置可复用。
网络结构、env/tick/minibatch、编译参数变化可能导致重新编译。
尚未验证128 tick、生产批量、长训稳定性或与 H200 的性能比值。

排查慢编译可用 `JAX_LOG_COMPILES=1`。本镜像的定时 `faulthandler` 栈转储
曾导致 Python 调试线程崩溃，已由 native backtrace 确认；正常验证入口不使用该机制。

## 后续8卡机器

当前入口只允许一张 DCU，是单卡兼容性验证。用户计划创建8卡机器后再继续
训练，当前机器不启动长训。届时需要单独实现并验证8卡数据并行及梯度聚合，
先检查设备、卡间通信和小规模多卡更新，再决定全局批量与正式训练配置；
不能直接把单卡入口的可见设备改成8张就视作完成多卡适配。
Release 仍是 Actor 热启动，无法精确恢复原 H200 的完整训练状态。
