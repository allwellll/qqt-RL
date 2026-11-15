# 运包抗拦截训练交接（2026-11-15）

本次开发了训练代码，并按用户要求执行单卡 DCU 性能诊断（包含短轮临时 PPO 更新）。
**没有启动持续训练或保存新训练权重**，没有更新线上模型。
`training_launch_policy.json` 保持关闭；用户之后明确要求开训时再处理停训门禁。
不要把这里的启动示例当成恢复训练授权。

初版分段采样器的历史 CPU 回归 **36 项通过**（152.41 秒），包含两阶段实际采样、GAE、PPO 更新与
采样/独立推理一致性检查。[验证记录](../reports/dcu_carry_cpu_validation_20261115.json)
保留该版代码 SHA256、模型身份和验证边界。当前 fused 采样器单卡运包阶段含 PPO 实测约 9,691 环境步/秒（4096 环境），
拦截阶段约 9,337 步/秒（2048 环境）；实机性能和验证边界见
[单卡效率实测](dcu_single_card_performance_20261115.md)，不能把旧版 CPU 结果当作新版全部验证。

## 为什么先适应运包地图

u930 在空笼、清场、击杀计分和随机复活的环境里练成，直接放进完整运包地图，
会遇到砖块、包子、携包减速、正常出生复活和交付即结束等变化。因此增加一个短的
拦截适应阶段是合理的，但未经训练评测，不能断言一定强于直接用原始 u930。

仅取消冻结对手的运包奖励不会改变行为。第一阶段实际更新 u930；第二阶段再冻结
选定的拦截模型，更新运包模型。拦截者不获得任何运输收益，且在规则层禁用
偷笼内包子、拾地面包子、击杀直接接包、交付四个入口。它仍可移动、放泡、使用和
拾取成长/技能道具，没有动作老师或额外寻路器。

**不禁止接近包子笼**：守笼或在笼附近伏击可能是有效的拦截。没有运输权限和奖励，
不等于数学上保证它永远不走向某个位置。是否主动追击，要通过实际对局评估。

## 两阶段设置

| 设置 | A：`interceptor` | B：`courier` |
|---|---|---|
| 唯一更新的 Actor | 从 u930 继续训练的拦截者 | 从 SP_web u23302 继续训练的运包者 |
| 对手 | 100% 冻结 SP_web u23302 | 默认 75% 冻结拦截者、25% 冻结 SP_web |
| 正向奖励 | 仅有效归因击杀 +5 | 运包交付为主，保留安全和战斗辅助奖励 |
| 负向奖励 | 死亡 −5，自己击杀另 −2 | 死亡、丢包、自爆等 |
| 运包权限 | 学习方禁运，对手可运 | 拦截对手禁运；学习方和普通 SP 对手可运 |
| 建议首轮时间 | 上限 45 分钟 | 上限 4 小时 |

默认全部对手整轮冻结，不做在线双边更新，也不自动刷新当前/历史对手。
第二阶段的 75% 可用 `--interceptor-fraction` 调整，实际比例按环境对数取整，
`PLAN.json` 记录生效比例。每类对手都均分两侧席位，类别在环境生命周期内固定。
保留普通 SP 对手，是为了检查正常运包对局的能力，避免只适应不能取包的对手。

两阶段首次进入都初始化独立 Critic 和 Adam，并默认先做 10 次 Critic 更新。
可用 `--checkpoint CHECKPOINT_DIRECTORY` 加载**同阶段**的 Actor 和 Critic；
优化器、随机数、环境和对局进度仍从头开始，不是精确续训。

运包奖励取自历史 full-game 事件奖励的主要项，整体除以 5：交付 +6，终局胜利 +4，
偷包 +0.8、带敌包跨回本方 +1.6，另有归还、救包、进度、安全和击杀项。
拦截奖励不包含这些运输、位置、交付或终局奖励。
这是新的 PPO 课程，**没有声称复现旧 SP 的 SIL/PFSP 训练协议**。

## 模型身份与文件

运包起点选用已经发布到网页的 SP_web u23302，未把更新号更大的 SP_latest
u24155 直接当成更好模型。

| 模型 | 默认文件 | SHA256 |
|---|---|---|
| SP_web u23302 | `data/portable-training-20261107/models/SP_web/policy.pt` | `39b0dea7327221891bdd4cfba93ab917c897aa86fd78d83401241ad945a982cf` |
| 原始 u930 | `data/carry-training-20261115/u930_actor.pt` | `1b1079d213a971ed7013739868d3b9848484ecf48ade25418142e16fb3a06d27` |

模型加载先验 SHA256，再反序列化并检查树结构、形状、类型和有限值。
u930 原始训练权重随 [训练输入 Release](https://github.com/allwellll/qqt-RL/releases/tag/dcu-carry-training-20261115)
发布，不放入 Git 历史；下载脚本校验固定 SHA256 和大小。SP_web 和完整校验所需数据
复用既有 `training-bundle-20261107`。新机器先复用环境配置 `.venv`，再执行：

```bash
.venv/bin/python -m scripts.fetch_dcu_carry_assets
.venv/bin/python -m scripts.fetch_dcu_carry_assets --check
```

第一条会下载缺失的既有运行包和 u930，已有文件先验证，不覆盖错误权重；全程不导入 JAX、
不加载 pickle、不占卡、不启动训练。第二条只检查，不联网或写入。已有原始 u930 文件时，
可用 `--u930-source runs/kill-jax-8card-e512-20261115/u000930/actor.pt` 安装同一权重。
训练仍可用 `--u930 actor.pt` 指定可信位置，预期哈希固定。第二阶段用 `--interceptor` 指定本入口阶段 A 保存的
检查点目录，并校验来源、源码与 Actor 哈希；省略则使用原始 u930，便于保留基线。
不要使用此前严重退化的 u1030 作为默认来源。

## 八卡启动

复用 `.venv` 和 `scripts/dcu_env.sh`，不升级系统或训练依赖。按用户最新要求，默认每卡 **2048** 个环境、
rollout 64 步、每卡 minibatch 512，使用 `fused`、`throughput` 编译配置，
`policy-chunk` 和 `value-chunk` 均为 512。
八卡全局 **16384** 个环境、每次采样 **1048576** 步，全局 minibatch 为 4096。当前只有单卡性能实测，
**不能把单卡速度乘八当成实测八卡吞吐，也不能直接沿用空笼约 2 万步/秒的结论**。
单卡配置推荐和拒绝项见单卡效率文档。
第一次正式启动前，先在已授权的 DCU 上用新目录执行
`--max-updates 1 --critic-warmup 0 --hours 0.25` 的短轮，核对
`INFERENCE_CHECK.json`、有限值、Actor/Critic 变化和检查点，再扩大预算。
本次有界单卡诊断未执行正式训练入口的检查点写盘短轮，八卡设备验证仍需执行。

纯读取计划，不导入 JAX、不占 DCU、不加载 pickle、不创建输出目录：

```bash
cd /root/private_data/qqt_rl
.venv/bin/python -m scripts.train_dcu_jax_carry plan --stage interceptor
.venv/bin/python -m scripts.train_dcu_jax_carry plan --stage courier
```

以下是**获得明确开训授权、解除对应停训门禁后**的示例。脚本要求新输出目录，
可重复使用既有源码和可信模型；不会恢复历史任务，也不自动衔接下一阶段。

```bash
# 两阶段共用同一个绝对截止时间，编译和人工检查也计入五小时预算。
qqt_deadline="$(date -u -d '+5 hours' +%Y-%m-%dT%H:%M:%SZ)"
bash scripts/launch_dcu_jax_carry_8card.sh interceptor \
  runs/carry-interceptor-NEW_DATE --deadline-utc "$qqt_deadline"
```

检查阶段 A 后，人工选定一个检查点，例如 `u000050`，**它只是路径示例，不代表最优**：

```bash
bash scripts/launch_dcu_jax_carry_8card.sh courier \
  runs/carry-courier-NEW_DATE \
  --interceptor runs/carry-interceptor-NEW_DATE/u000050 \
  --deadline-utc "$qqt_deadline"
```

两个命令应在同一 shell 保留 `qqt_deadline`；重新连接需使用原绝对截止，不能重新加五小时。
默认阶段上限 45 分钟 + 4 小时，预留约 15 分钟检查；这只是初始资源安排，
不是已测得的收敛时长。超过绝对截止会停止采样，编译本身可能延迟响应信号；外层
watchdog 最多五小时并给予清理时间。需要严格的整机截止时同时使用平台任务时限。

## 存储与停止

两阶段分别最多 32 份、2 GiB，合计模型检查点约 4 GiB，另保留 8 GiB 空闲余量。
每次只写 `actor.pt`、`critic.pt` 和 `CHECKPOINT.json`，不保存 Adam、环境和 RNG。
初始和最新检查点保留；每 5 次更新或距上次保存 120 秒后的下一次更新保存，
第一次更新、Critic 预热结束和正常退出也保存。**120 秒不是编译期间的硬保证**。
目录原子发布、fsync 和 `LATEST.json` 避免读取半写入文件；只裁剪本轮同 schema 的目录。
已有约 40 GB 磁盘预算不用于重复复制历史模型。

创建本轮 `USER_STOP`，或发送 TERM/INT 可停止，入口在可安全退出的位置保存最近一次
健康更新。不自动启动下一轮。异常导致非有限参数时保留此前的有效检查点。
不删除旧训练或他人的文件，不改变依赖和文件权限。

## 如何判断有效

```bash
.venv/bin/python -m scripts.summarize_dcu_carry runs/carry-interceptor-NEW_DATE
.venv/bin/python -m scripts.summarize_dcu_carry runs/carry-courier-NEW_DATE
```

读取 `updates.jsonl`，不使用 DCU。按对手类别输出交付率、胜率、击杀/死亡、携包者击杀
和掉包数。原始数据轴是 `[对手类别: interceptor/SP_web][学习方/对手][metric_names]`。
阶段 A 的 SP_web 分组里，学习方就是拦截者，所以重点看 **opponent_delivery_rate**
是否下降，携包者击杀是否增加，自死是否失控。阶段 B 重点看 **learner_delivery_rate**
在拦截和普通 SP 两组是否改善。
拦截者没有交付权限，因此阶段 A 的游戏胜率不适合作为主要指标；它成功阻止交付时，
正常结果常常是超时平局。不要因拦截者胜率低而误判它没有完成拦截任务。

训练累计指标包含不同版本学习方及尚未结束对局的事件，不能当作严格独立胜率评测，
也不要仅按总奖励或最新更新号选模型。正式采用适应后的拦截者前，应把原始 u930
与候选拦截者分别固定，对同一批冻结运包模型、相同 seeds、双席位做完整对局评估；
比较运包方交付率、交付耗时、携包者死亡与自死，并检查录像是否有效追击。
当前新增入口和汇总脚本提供训练及观测，**尚未新增独立评测/录像导出器，也没有
实际对战结果证明阶段 A 有收益**。若适应后不能降低交付率，可直接回用原始 u930，
无需为了两阶段流程强行采用退化模型。

## CPU 验证与接手入口

```bash
source scripts/dcu_env.sh
export HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAXBOMB_RULE=bun
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=false
taskset -c 112-119 python -m pytest \
  tests/test_dcu_jax_carry.py tests/test_kill_checkpoints.py tests/test_dcu_jax_kill.py
```

CPU 编号应按新机器允许的 affinity 调整；此命令显式隐藏 DCU。
测试覆盖完整地图观测、双方席位禁运、交付、击杀直接转移与掉落守恒、独立重置、
冻结对手和学习方独立更新，以及停止门禁先于 JAX 初始化。
用小型随机 Transformer 检查真实设备采样 → GAE → PPO 接口；另已在 CPU 用真实
SP_web/u930 对完整运包观测做前向检查，移动/能力维度均为 5/3 且结果有限。
这些验证不等于完成八卡实机训练，也不能预测训练收益。

开发早期用小型模型在 CPU 测试大融合图时出现数值异常，因此保留原分段采样器
`--collector baseline` 供诊断。当前新增 `jax_carry_fast.py`：真实模型单卡实测后，
默认改为设备端 `pmap + lax.scan` 整段采样，Critic 延后集中计算，reset 使用逐叶等价的
完整地图模板。所有观测、动作、奖励、GAE 和 PPO 数据留在设备上，无 Node 回调，
无每 tick 的 NumPy/主机读取。生产前向仍为原有 BF16，没有修改封存网络源码。
更大 batch 不一定数值合格，不能省略独立前向检查；详见单卡效率实测。

入口每次采样检查 value/logp/reward 有限性，第一次 PPO 前还把采样得到的 logp/value
与独立重推理结果比较，写入 `INFERENCE_CHECK.json`。最大 logp 差超过 0.03、value 差
超过 0.1 或出现非有限值会拒绝更新。接手后先完成设备短轮；若不通过需继续检查，
不能移除保护强行训练。

主要文件：`qqt_rl/training/jax_carry_env.py`、`jax_carry_collect.py`、`jax_carry_fast.py`、
`scripts/train_dcu_jax_carry.py`、`scripts/launch_dcu_jax_carry_8card.sh`。
历史源码指纹闭包保持原样；对底层环境的角色和动态席位兼容修改仅在独立运行时
函数内生效。网页相关已有工作区修改没有被本次任务覆盖。
