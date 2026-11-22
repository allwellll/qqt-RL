# 八卡 DCU：JAX 空笼击杀训练

这是当前推荐入口。2026-11-15 的前一轮 `train_dcu_kill` 使用 Node/CPU 模拟，
实测约 1300 环境步/秒，八卡利用率较低；已在 u000173 保存 Actor/Critic 后停止。
新入口 `train_dcu_jax_kill` 将环境、JAX 规则 bot、自对弈、GAE 与 PPO 放在 DCU 上。
历史 JS 入口仍保留用于网页规则对照；其吞吐和评估记录不能当作本入口的结果。

## 设备上的训练流程

训练入口和八卡启动器默认每卡 2048 个环境，8 卡共 16384 个；一次 rollout 64 tick，共 1048576 个环境步。
2026-11-21 起统一采用此默认值；历史 512 的实测记录保留，不代表新配置吞吐。
`pmap` 分配到八卡，`lax.scan` 在设备内步进，不逐 tick 调用 Node 或传回 CPU。
GAE 和训练批次保持在设备上。每轮仅将小型统计传回主机；保存时传回 Actor/Critic。
地图、模型文件读取与静态 schema 初始化在 CPU 上完成一次；XLA 首次编译也使用 CPU，
因此首次编译期间低 DCU 利用率不能代表稳定训练效率。

| 对手 | 环境占比 |
|---|---:|
| JAX hunter / hunter_hard 规则 bot | 合计 10% |
| 当前 Actor 的整局冻结副本 | 9% |
| 上一次整局边界的 Actor 副本 | 9% |
| 固定强历史 u930 / u840 模型 | 各 31.5% |
| 固定 SP / G 模型 | 各 4.5% |

双方座位均衡；模型对手只在整局边界刷新，整局内参数固定。
八卡梯度先平均再裁剪/Adam，策略项按全局有效动作数归一化。
Actor/Critic 学习率分别为 3e-5 / 1e-4。学习方前向和 PPO minibatch 均为每卡 2048，
全局 minibatch 为 16384，每轮遍历一次，共 64 个优化步。PPO clip=.15，gamma=.997，lambda=.95。
`--envs` 表示每卡环境数，`--minibatch` 表示每卡每次优化的样本数。

击杀 +5，死亡 -5，自死额外 -2，新困敌/被困 ±.25，整局胜负 ±2。
死亡后继续同局 GAE，只有 1800 tick 整局结束切断。
不奖励单纯放泡或捡道具。

## 环境与规则验证

复用项目已有 JAX 原生模拟器及私有接触/连锁修正，适配地图 806 的无包子击杀模式：
24 个对称成长道具、死亡额外 8 个掉落、30 tick 复活、10 tick 保护，初始/复活 HP=5。
基础输入对齐网页 kill 模式的 generic 编码：obs25/global37，额外模型模式行为零。
封存的 portable 数据和历史环境源文件未改动。

下列检查仅使用 CPU，可在训练时独立运行，不占用 DCU：

```bash
source scripts/dcu_env.sh
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_COMPILATION_CACHE=false \
  python -m tests.check_jax_kill_env
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_COMPILATION_CACHE=false \
  python -m tests.check_jax_kill_trajectory
```

已通过 32 个开局种子的对称性/数量检查、初始网页输入对照、延迟击杀、自死、
末 tick 结算、掉落与复活检查，以及 60 tick 固定动作的网页/JAX 物理轨迹对照。
这些检查不等于全部战局逐位一致；JAX 与 JS 的随机数不同，随机掉落/复活位置不逐位相同。

## 启动与续训

所有模型、依赖和输出放在本项目 `/root/private_data/qqt_rl`，
保留容器现有 DCU 驱动和海光 JAX；准备环境见 [迁移说明](dcu_migration.md)。
先在平台分配的空闲八卡上验证通信。以下通信命令会使用 DCU，不要与正在运行的训练叠加。

```bash
cd /root/private_data/qqt_rl
source scripts/dcu_env.sh
python -m scripts.train_dcu_jax_kill plan
python -u -m scripts.check_dcu_collectives --devices 0,1,2,3,4,5,6,7
# 新输出目录，默认从 u930 Actor 初始化，Critic 预热 10 次
bash scripts/launch_dcu_jax_kill_8card.sh runs/kill-jax-new-001 \
  2>&1 | tee runs/kill-jax-new-001.log
```

从已验证且可信的检查点热启动 Actor + Critic：

```bash
bash scripts/launch_dcu_jax_kill_8card.sh runs/kill-jax-next-001 \
  --checkpoint runs/kill-jax-old-001/u000100 --critic-warmup 2 \
  2>&1 | tee runs/kill-jax-next-001.log
```

加载前验证检查点记录的源文件哈希、权重哈希、参数树、形状、类型与有限性。
优化器、环境和 RNG 重新初始化，这属于热启动。源文件发生变化时需要审查迁移，不能直接跳过校验。
首次完整编译可能较长，稳定吞吐应取后续更新的中位数：
`global_samples / seconds` 为全链路环境步/秒（推荐配置的分子为 1048576），
`collect_seconds` 与 `ppo_seconds` 区分采样和优化，不含周期保存等少量主机开销。

外层 watchdog 为 5 小时，内部 4.95 小时，包含初始化和编译；清理最多额外 90 秒。
本次续训保持原始 11:44 UTC 内部截止与 11:47 UTC watchdog，没有重新追加 5 小时。
每卡 128 环境的验证轮次 `runs/kill-jax-8card-v2-20261115` 已完成 30 次更新、
196.6 万环境步和 1024 个完整对局，稳定吞吐约 8040 步/秒。
随后从其 u000030 热启动 `runs/kill-jax-8card-e512-20261115`，
每卡 512 环境、共 4096 环境、每轮 262144 样本。实测稳定更新约 13 秒，
约 2 万环境步/秒，其中采样约 8 秒、PPO 约 5 秒；每卡显存约 8.2 GiB。
16 次、间隔 3 秒的设备利用率样本平均约 43%，高于 128 环境配置的约 16%；
短内核及调度仍有开销，不能把吞吐提升解释为算力已经满载。
详细测量窗口和已验证权重见 [八卡 JAX 验证报告](../reports/dcu_jax_kill_validation_20261115.json)。

```bash
tail -f runs/kill-jax-8card-e512-20261115/updates.jsonl
cat runs/kill-jax-8card-e512-20261115/LATEST.json
# 仅请求本轮在完成当前设备操作后保存退出
touch runs/kill-jax-8card-e512-20261115/USER_STOP
```

## 检查点、磁盘与结果边界

每份保存 `actor.pt`、`critic.pt`、`CHECKPOINT.json`，不保存优化器。
初始化、首轮、Critic 预热完成、每 5 次更新或距上次保存 120 秒后的完整更新，以及正常退出时保存。
写入前检查八卡参数摘要一致，fsync/原子发布，失败保留此前 LATEST。
每轮保留初始及最新检查点，最多 64 份/4 GiB；每对权重约 58 MiB。
仅淘汰当前轮次自己的旧检查点，保留历史运行。前轮已占约 2.1 GiB，
128 环境验证轮次占约 0.51 GiB，本轮最多 4 GiB，检查点合计预算约 6.6 GiB。
按用户给出的剩余 40 GB 额度规划，不把共享 `df` 当作个人额度。
编译或单轮阻塞期间不能保证两分钟内生成新权重。

`PLAN.json` 记录实际配置和源码哈希；`updates.jsonl` 分列采样/PPO/总耗时、损失与对战计数；
`LATEST.json` 指向最后持久保存的权重；`STORAGE.json` 记录保留/淘汰；
`RESULT.json` / `STOPPED.json` 记录最终状态。没有 RESULT 不能当作训练已完成。
损失数组为总损失、策略项、Critic、熵、KL、clip 比例、价值 MSE、梯度范数。
metrics 为己方/对方两行：击杀、死亡、自死、放泡、被困 tick、整局积分。

当前 JAX 入口不含周期性固定对手评估，训练统计随对手变化，不能单独证明对拼能力提升。
完成后须做固定对手整局对照再决定模型发布；旧 JS 入口的严格源码检查不会直接接受新的
JAX 检查点，需要显式评估适配。训练不会自动替换网页模型。
完整 K4 Actor 网页导出仍须用 `bun_mode_zero_actor` 去除零模式输入行并保留 R3 元数据。
