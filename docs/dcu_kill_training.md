# 海光 DCU：空笼击杀对战训练

本入口针对用户本轮授权的“包子笼没有包子、专门强化对拼”训练。`dcu` 已合入
远程 `main` 的 `f5868656e067486147f1cef1aa7c8e3ab0288138`。历史训练入口的
`training_launch_policy.json` 继续保留；本文命令只启动这轮独立训练，不恢复旧任务。
当前机器只做短验证，约 5 小时正式训练留给后续 8 卡机器。

## 规则与实现

直接调用当前网页的 `web/sim-bun-native-v4-facing.js`，同时加载
`kill_opening_items.js`、`self_respawn_preview.js` 和 `r3_model_inputs.js`。
最后一个 respawn 适配器虽然名为 preview，但网页实战也加载它，会影响真实复活调度。
不使用旧 JAX v3 采样器来近似这套新规则。

完整评估发现 main 原有击杀计分只看死亡当 tick 的爆炸来源，而原生糖泡会延迟死亡，
导致真实击杀与自死漏计。此分支在同一网页引擎修复了归因：超时死亡继承最初困敌来源，
敌方接触碰爆归碰爆者，自困超时记自死；相互击杀双方计分，末 tick 击杀先入账再判胜负。
来源随回放快照保存，复活/解困清除；非 kill 模式保留原行为。训练和网页共用修复后的规则。

- 地图 806：保留硬墙，清空砖块和草丛，无包子、无运输得分，以击杀决胜。
- 一局 1800 tick；死亡后 30 tick 复活，复活保护 10 tick；死亡额外掉落 8 个成长道具。
- 开场 24 个镜像成长道具，三类各 8 个，排除出生点相邻位置。
- 与网页相同的 obs25/global37；继承模型额外的模式输入行固定为 0，与网页 mode-zero 导出一致。

每张卡对应一个常驻 Node CPU 模拟进程，DCU 批量推理、执行 PPO。
多卡通过 `pmap` 同步梯度：先平均，再裁剪、再 Adam；不是独立训练 8 份模型。
存活样本数量不同的分片按全局有效样本数归一化，死亡/被困的无效动作不参与策略损失，
Critic 仍学习这些状态及复活后的回报。死亡不切断 GAE，只有整局终止才切断。

## 对拼优化

从官方 `DUEL_latest` Actor 热启动；发布包没有可恢复的旧 Critic/Adam，故重新初始化。
先用 10 次 update 仅预热 Critic，Actor 及其 Adam 状态冻结，然后联合训练。

| 事件 | 奖励 |
|---|---:|
| 引擎确认的击杀 | +5 |
| 自己死亡 | -5 |
| 自己泡泡造成的败亡 | 额外 -2 |
| 唯一归因于己方的新困敌 | +0.25 |
| 自己新被困 | -0.25 |
| 整局胜 / 负 / 平 | +2 / -2 / 0 |

不按放泡数量、捡道具、接近敌人或运输给奖励，避免刷分替代真实击杀。
引擎归因负责区分他杀与自死；对比评估同时看击杀、死亡、自死与整局胜率。

默认 8 个对手分片中，4 个当前策略、2 个历史策略、1 个冻结 SP、1 个冻结 G。
对手在整局 1800 tick 内固定，只在新局切换；历史池最多保留 8 个整局边界快照。
双方座位各占一半。固定初始 Actor 的 KL 约束抑制短时间微调时的策略坍塌。

| 参数 | 默认值 |
|---|---:|
| DCU 数 / 每卡环境 | 8 / 8 |
| 全局环境 / rollout 长度 | 64 / 64 tick |
| 每轮全局样本 / 全局 minibatch | 4096 / 512 |
| 每轮优化遍历 | 1 次，8 个 minibatch |
| Actor / Critic 学习率 | 3e-5 / 1e-4 |
| PPO clip / 梯度范数上限 | 0.15 / 0.5 |
| gamma / GAE lambda | 0.997 / 0.95 |
| 熵系数 / 初始策略 KL 系数 | 0.005 / 0.01 |
| Critic | 128 桶分布，范围 [-20,20] |
| 保存 / 评估间隔 | 20 update / 1800 秒 |

训练策略温度为 1；评估双方温度为 0.3。基线、周期及收尾评估分别对战固定
DUEL/SP/G，每个对手都使用全部卡和均衡座位，使用固定评估种子及独立采样 RNG，
不消耗训练 RNG。完整评估每个对手 64 局，共 192 局；`full_game=true` 才可解读胜率。
短评估只验证流程，不能作为棋力结论。默认保留最后模型，脚本不会把奖励上升视为必然变强，
也不会自动替换网页模型；用各次完整评估比较基线与候选。

## 新机器准备

所有命令均在项目目录执行，模型、缓存和输出留在 `/root/private_data`。
使用与当前验证环境匹配的 DTK 26.04 / Python 3.11 镜像，保留平台海光 JAX/JAXlib
`0.6.0+das.opt1.dtk2604`。安装脚本检查版本，不匹配就退出，不升级驱动或系统训练库。
目前 Node 安装脚本面向 x86_64 镜像。

```bash
cd /root/private_data/qqt_rl
git switch dcu
git pull --ff-only origin dcu
bash scripts/setup_dcu_kill.sh
source scripts/dcu_env.sh
# 已存在的 bundle 会校验并复用；下载约 263 MiB
python -m scripts.fetch_dcu_bundle --mirror https://ghfast.top/
python -m scripts.train_dcu_kill plan
```

Node 22.16.0 安装到 `.tools/node/bin/node`，按官方 HTTPS SHASUMS 验证。
Python 新增依赖仅放项目 `.venv`，环境脚本预加载 DTK OpenMPI，避免 `ompi_mpi_int` 缺失。
本入口禁用持久编译缓存，避免跨后端/卡数复用海光编译产物；不清理旧缓存。
旧 v3 JAX 采样器约 12 分钟的编译耗时不适用于这个原生 JS 入口。

确认平台分配给自己的 8 张卡可用后，先运行设备通信与极短同步更新：

```bash
source scripts/dcu_env.sh
# 仅探测，不终止任何进程
rocm-smi
# 真实物理卡上的 psum / all_gather
python -u -m scripts.check_dcu_collectives --devices 0,1,2,3,4,5,6,7
# 仅两轮：先预热一次 Critic，再联合更新一次；输出必须是新目录
python -u -m scripts.train_dcu_kill train \
  --devices 0,1,2,3,4,5,6,7 --envs 2 --ticks 4 --minibatch 4 \
  --max-updates 2 --critic-warmup 1 --eval-every 0 --hours .1 \
  --out runs/kill-8card-preflight-001 --acknowledge-training
```

先确认通信检查 PASS，短训练 `RESULT.json` 为 `PASS_DCU_KILL_UPDATE`、
`cards=8`，Actor/Critic 都发生有限数值更新，再启动正式轮次。
本次当前机器只有一张 DCU，真实 8 卡 RCCL 通信、显存及吞吐仍需此步骤验收。

单卡复现命令：

```bash
python -u -m scripts.train_dcu_kill smoke \
  --devices 0 --envs 2 --ticks 4 --minibatch 4 \
  --max-updates 2 --critic-warmup 1 --eval-every 0 --hours .25 \
  --out runs/kill-single-smoke-001
```

## 启动约 5 小时的 8 卡训练

建议在平台作业或 tmux 中前台运行。新建输出目录，禁止覆盖已有结果。

```bash
cd /root/private_data/qqt_rl
mkdir -p runs
set -o pipefail
bash scripts/launch_dcu_kill_8card.sh runs/kill-8card-5h-001 \
  2>&1 | tee runs/kill-8card-5h-001.log
```

预算按进程启动起的墙钟计时，包括导入、编译、基线及周期评估。
Python 内部预算 4.95 小时（4 小时 57 分），按基线评估实测耗时的 1.5 倍加 30 秒预留
最终评估；到时停止采样、保存并在余量内评估。单次编译/优化操作不能从 Python 中间切断。
外层 `timeout` 在 5 小时发送 TERM，最多再给 90 秒清理，不启动任何自动后继任务。
用户停止或超时会允许保存已完成 update，未完成 rollout 不更新参数。

目前只测过单卡小批次，不能据此承诺 8 卡 5 小时的总步数或棋力提升。
Node 原生模拟器、CPU 核数、卡间通信都可能限制吞吐；保守配置是每卡 8 环境，
不建议未测吞吐就增加环境数。新机器上以 `updates.jsonl` 的稳定耗时估算实际可完成轮数。
阶段样本数等于 update 数 × 4096（默认配置；最后未完成 rollout 另计）。

监控与停止：

```bash
tail -f runs/kill-8card-5h-001/updates.jsonl
cat runs/kill-8card-5h-001/LATEST.json
# 仅停止这一轮，不影响其他任务
touch runs/kill-8card-5h-001/USER_STOP
```

## 产物和加载

- `PLAN.json`：设备、批量、依赖版本、规则与训练代码 SHA256。
- `updates.jsonl`：损失数组依次为总损失、策略项、Critic、熵、KL、clip 比例、价值 MSE、梯度范数。
  `metrics` 按己方/对方两行记录击杀、死亡、自死、放泡、被困 tick、整局积分。
- `uXXXXXX/actor.pt`、`critic.pt`、`CHECKPOINT.json`：已完成 update 的 Actor、Critic 和校验信息。
- `LATEST.json`：最后已保存的检查点；`EVAL-baseline.json`、`EVAL-uXXXXXX.json`、
  `EVAL-final.json`：各固定对手的原生击杀统计。整局积分为胜 1、平 0.5、负 0，除以完成局数得到胜率含平分。
- `RESULT.json`、`STOPPED.json`：最终更新数、参数变化与停止记录；异常请查看主日志和 worker 日志。

只加载自己训练且可信的 pickle 文件。Actor/Critic 哈希、代码/规则哈希、参数树、形状、类型、
有限性均检查；代码变更需要显式迁移。当前没有保存 Adam、环境和 RNG，所以
`--checkpoint` 是 Actor+Critic 热启动新一轮，不是逐位恢复；新轮次重新执行 Critic 预热。

```bash
# 替换为 LATEST.json 指向的具体目录；仍须使用一个新输出目录
bash scripts/launch_dcu_kill_8card.sh runs/kill-8card-next-001 \
  --checkpoint runs/kill-8card-5h-001/u000020
# 只评估，不更新；默认每个对手完整 1800 tick
python -u -m scripts.train_dcu_kill eval \
  --devices 0,1,2,3,4,5,6,7 --envs 8 \
  --checkpoint runs/kill-8card-5h-001/u000020 \
  --out runs/kill-eval-001 --hours .5
```

完整 K4 Actor 仍带固定零模式行，不能直接当成网页 global37 模型发布；网页导出时需通过
`r4_k4_observation.bun_mode_zero_actor` 移除该行并保留 R3 输入元数据。原始检查点应保留。

## 已验证范围

当前单 BW DCU 的真实网络采样、预热、PPO/Adam、保存/加载及完整原生击杀评估，
与 8 个虚拟 CPU 设备上的全局梯度/Adam 等价校验，详见
[验证记录](../reports/dcu_kill_validation_20261109.json)。虚拟设备数学测试不替代物理 8 卡验收。
