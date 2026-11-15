# DCU 训练交接记录（2026-11-15）

接手时先阅读本文，再检查实时产物。本记录的状态快照采集于 **2026-11-15 07:39 UTC**
（北京时间 15:39）；更新数和检查点会继续变化。本文本身不代表任务已结束。

## 1. 本次任务和当前结论

用户要求在海光八卡上训练 main 新增的“包子笼没有包子”对战模式，专门强化对拼，
整轮墙钟预算约 5 小时。每次只保存 **Actor + Critic**，不保存优化器，合理使用用户给出的
40 GB 可用磁盘额度。当前任务已实际启动，接手时优先检查这轮既有训练。

- 项目：`/root/private_data/qqt_rl`。
- 远程：`https://github.com/allwellll/qqt-RL.git`，分支 `dcu`，跟踪 `origin/dcu`。
- 训练实现提交：`ffaa89228e068ec68dfe99ab200b888a23157373`，已推送。
- Git 身份：`allwellll <1035628914@qq.com>`；文档、日志和仓库均不存放访问令牌。
- main 的新模式已合并；历史训练的停训策略文件保留，新入口使用显式本轮训练确认。
- 当前环境、JAX 规则 bot、自对弈推理、GAE 和 PPO 全部运行在 DCU 上。
- 稳定吞吐约 **2 万环境步/秒**，比此前 CPU/JS 路径约 1300 步/秒高约 15 倍。
- 已验证 Actor/Critic 联合更新、八卡参数摘要一致、保存权重哈希，以及完整对局重置。

## 2. 当前任务定位与截止时间

| 项目 | 值 |
|---|---|
| 运行目录 | `/root/private_data/qqt_rl/runs/kill-jax-8card-e512-20261115` |
| 主日志 | `runs/kill-jax-8card-e512-20261115.log` |
| 启动记录 | `runs/kill-jax-8card-e512-20261115.job.json` |
| watchdog / Python PID（快照） | `83381` / `83385`，使用前核对完整命令，PID 可能过期或复用 |
| 本进程开始 | 2026-11-15 07:27:38 UTC / 北京时间 15:27:38 |
| **内部停止采样截止** | **2026-11-15 11:44:03 UTC / 北京时间 19:44:03** |
| 外部 watchdog | 2026-11-15 11:47:03 UTC / 北京时间 19:47:03，最多再给 90 秒清理 |
| 自动后继 | 关闭；到期后不自动启动下一轮 |

原始整轮从 06:47 UTC 开始。此前纠正采样路径和扩大批量均保留原始截止，
**没有重新追加 5 小时**。系统部分命令可能显示北京时间，检查时间统一用 `date -u`。

07:39 UTC 快照：完成 **32 次更新、8,388,608 环境步、4096 个完整对局**，
最新已保存 `u000030`；这是当前 e512 轮次的计数，不含前两轮热启动来源的步数。
`RESULT.json`、`STOPPED.json`、`USER_STOP` 当时均不存在，进程仍运行。
接手时以实时文件为准，不按本文数字判断卡住或结束。

## 3. 接手后先执行的只读检查

以下检查不启动训练或额外 GPU 计算，也不停止当前任务。

```bash
cd /root/private_data/qqt_rl
date -u
git status --short --branch
git log -1 --oneline
ps -ww -p 83381,83385 -o pid,ppid,etime,pcpu,rss,args
rocm-smi --showuse --showmeminfo vram
tail -n 10 runs/kill-jax-8card-e512-20261115.log
tail -n 3 runs/kill-jax-8card-e512-20261115/updates.jsonl
cat runs/kill-jax-8card-e512-20261115/STORAGE.json
```

PID 不存在时，结合 `.job.json` 和下列只读搜索确认实际进程，不直接认定异常：

```bash
ps -eo pid,ppid,etime,args | rg '[s]cripts.train_dcu_jax_kill'
```

检查最新持久检查点、源文件哈希和最终状态，下面只用标准库，不加载 JAX 或 pickle：

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path('/root/private_data/qqt_rl')
run = root / 'runs/kill-jax-8card-e512-20261115'
for name in ('RESULT.json', 'STOPPED.json'):
    file = run / name
    print(name, file.read_text() if file.exists() else 'ABSENT')
print('USER_STOP exists:', (run / 'USER_STOP').exists())
latest = json.loads((run / 'LATEST.json').read_text())
directory = run / latest['checkpoint']
meta = json.loads((directory / 'CHECKPOINT.json').read_text())
plan = json.loads((run / 'PLAN.json').read_text())
assert meta['optimizer_saved'] is False
assert meta['cards'] == 8 and meta['replica_summaries_equal'] is True
assert {p.name for p in directory.iterdir()} == {'actor.pt', 'critic.pt', 'CHECKPOINT.json'}
assert meta['source_sha256'] == plan['source_sha256']
for branch in ('actor', 'critic'):
    digest = hashlib.sha256((directory / (branch + '.pt')).read_bytes()).hexdigest()
    assert digest == meta[branch + '_sha256'] == latest[branch + '_sha256']
for file, expected in meta['source_sha256'].items():
    assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected, file
print('VERIFIED', directory, 'update=', meta['update'],
      'saved_environment_steps=', meta['global_environment_steps'])
PY
```

判断要点：

- 日志持续产生新 update，通常一轮约 13 秒，周期保存稍增加耗时；间隔 30–60 秒对照两次。
- 每轮 `262144 / seconds` 为近似训练吞吐，不含保存等少量主机开销。
- 稳态 `collect_seconds` 约 8 秒、`ppo_seconds` 约 4.7 秒，每卡显存约 8.2 GiB。
- 16 次间隔 3 秒采样的平均利用率约 43%；瞬时 0% 不能单独判定训练卡住。
- 首次采样约 189 秒、首次 PPO 约 45 秒，均包含编译；新进程可能重复编译。
- CPU 负责初始化、编译、设备调度、统计和落盘；CPU 使用率不等于环境已回到 CPU。
- `.job.json` 的 status 是启动/验收快照，进程与实时结果文件才反映当前状态。
- 日志出现 Traceback、nonfinite、queue overflow、replicas differ、写盘报错时，保留日志和最后有效检查点，先定位原因。
- 正在训练时不另跑设备通信检查、GPU smoke 或第二个训练任务。

## 4. 实际训练参数与规则

| 参数 | 当前值 |
|---|---:|
| DCU | 8 张 BW，设备 0–7，各 65520 MiB |
| 每卡 / 全局环境 | 512 / 4096 |
| rollout | 64 tick，262144 全局样本 |
| 每卡 / 全局 minibatch | 512 / 4096 |
| 每轮优化 | 遍历一次，64 个 minibatch |
| Actor / Critic 学习率 | 3e-5 / 1e-4 |
| PPO clip / 梯度裁剪 | .15 / .5 |
| gamma / GAE lambda | .997 / .95 |
| 当前进程 Critic 单独预热 | 前 2 次更新；之后 Actor/Critic 联合训练 |

对手环境占比：hunter 25%、hunter_hard 25%、当前模型 25%、历史模型 12.5%、
固定 SP 6.25%、固定 G 6.25%。座位均衡，模型对手整局冻结，在 1800 tick 整局边界刷新。
`pmap` 分配八卡，`lax.scan` 执行设备内采样与 GAE；没有逐 tick CPU 回调或 Node 采样。
八卡梯度先平均再裁剪和 Adam，策略损失按全局有效动作数归一化。

地图 806，无包子和运输得分；初始 24 个镜像成长道具，死亡额外掉落 8 个，
30 tick 复活、10 tick 保护，初始/复活 HP=5，一局 1800 tick。
奖励：击杀 +5、死亡 -5、自死额外 -2、新困敌/被困 ±.25、整局胜负 ±2。
死亡不切断 GAE，整局结束才切断。输入 obs25/global37 加固定零模式行至 global38。

`updates.jsonl` 的 `metrics` 为己方/对方两行，各列是：
击杀、死亡、自死、放泡、被困 tick、整局积分（胜 1、平 .5、负 0）。
`loss` 各列是：总损失、策略项、Critic、熵、KL、clip 比例、价值 MSE、梯度范数。
对手会变化，训练中的整局积分不能直接当作固定基准胜率或棋力提升证明。

## 5. 检查点与磁盘

初始化、首轮、Critic 预热结束、每 5 次更新或距上次保存 120 秒后的完整更新，
以及正常退出时保存。单次编译/设备操作阻塞期间无法保证硬性两分钟内落盘。

- 每份仅有 `actor.pt`、`critic.pt`、`CHECKPOINT.json`，一对约 58 MiB。
- `LATEST.json` 指向已发布的持久检查点；`STORAGE.json` 记录数量、大小和淘汰。
- 当前轮最多 64 份、4 GiB，保留初始与最新，仅淘汰本轮自己的旧检查点。
- 旧 CPU 轮约 2.1 GiB、128 环境验证轮约 .51 GiB，连同本轮总检查点预算约 6.6 GiB。
- 写入使用 fsync 与原子发布；没有保存 Adam、环境或 RNG，因此只能热启动，不能逐位恢复。
- 用户给出的 40 GB 是本轮规划依据，`df` 的共享文件系统总余量不是个人额度。
- 不擅自删除历史运行、团队文件，不把权重或大型产物加入 Git。

## 6. 热启动链与需要避免的旧入口

| 轮次 | 结果与用途 |
|---|---|
| `runs/kill-8card-5h-20261115` | 旧 CPU/JS 路径已停止；u000173，709952 步，是迁移来源 |
| `runs/kill-jax-8card-v2-20261115` | 八卡 JAX 验证已正常停止；u000030，1966080 步、1024 整局，约 8040 步/秒 |
| `runs/kill-jax-8card-e512-20261115` | 当前训练，从上行 u000030 热启动；使用本文实时检查步骤 |

不要恢复旧 CPU 轮或把 `scripts/train_dcu_kill.py`、`launch_dcu_kill_8card.sh`
当作当前推荐训练入口。当前代码如下：

- `scripts/train_dcu_jax_kill.py`：训练、热启动、截止与保存。
- `scripts/launch_dcu_jax_kill_8card.sh`：推荐的新轮次八卡启动器。
- `qqt_rl/training/jax_kill_env.py`：原生 JAX 模拟器的空笼击杀适配。
- `qqt_rl/training/jax_kill_collect.py`：规则 bot/自对弈、设备 rollout 与 GAE。
- `qqt_rl/training/kill_ppo.py`：同步 PPO；`checkpoints.py`：权重保存与滚动保留。

## 7. 环境、已验证范围与剩余事项

DTK 26.04，Python 3.11.9，海光 JAX/JAXlib `0.6.0+das.opt1.dtk2604`；
项目 `.venv` 补充 Optax/Chex，`source scripts/dcu_env.sh` 激活并预加载 DTK OpenMPI。
保留现有驱动、系统 Python/Conda 和训练依赖。模型数据在 `data/portable-training-20261107`，
19 个资产约 263 MiB；不要修改封存源文件或在系统盘积累大缓存。
本入口禁用持久编译缓存；保守 XLA 参数由 `qqt_rl/training/dcu.py` 设置，当前进程源码哈希见 PLAN。
训练运行期间不修改其指纹覆盖的源码，修改后续训需要审查来源迁移。

已通过：物理八卡通信与同步更新；44 项配置、运行时和检查点检查；
32 个种子的开局规则、初始网页输入对照、延迟击杀/自死、掉落与复活；
60 tick 网页/JAX 固定动作物理轨迹对照。随机掉落/复活坐标不逐位对齐，检查不是全部规则的穷尽证明。

**待完成：到期验收及固定对手整局评估。** 当前 JAX 训练器不含周期性固定对手评估；
旧 JS 评估器因严格检查源码来源不能直接接受新 JAX 检查点，需要先做显式评估适配。
不把训练 loss、训练胜率或吞吐当作对拼能力已提升的证据，不自动替换网页模型。

到期后检查 `RESULT.json` 的 `status=PASS_FUSED_JAX_DCU_UPDATE`、`cards=8`、更新数，
核对 `STOPPED.json`、最终 LATEST、权重哈希与日志。`STOPPED.json` 在异常退出时也可能存在，
不能单独当成功；只有进程消失而无 RESULT 时应读日志定位。检查完成后汇报最终更新数、
环境步数、吞吐、最终权重目录和是否异常，保留原始权重及元数据。

只在用户要求提前停止时，对确认属于本轮的目录执行：

```bash
touch /root/private_data/qqt_rl/runs/kill-jax-8card-e512-20261115/USER_STOP
```

之后等待当前设备操作结束、保存退出并检查结果。普通接手检查不执行此命令。
本轮预算结束后不自动延长或另开 5 小时；后续训练按用户的新要求和资源分配执行。
明确需要续训时，选择 LATEST 指向的可信检查点、新输出目录、已分配的空闲八卡：

```bash
cd /root/private_data/qqt_rl
# 将 uXXXXXX 替换成实时 LATEST 指向的目录；这会启动新一轮训练
bash scripts/launch_dcu_jax_kill_8card.sh runs/kill-jax-next-NEW \
  --checkpoint runs/kill-jax-8card-e512-20261115/uXXXXXX --critic-warmup 2 \
  2>&1 | tee runs/kill-jax-next-NEW.log
```

读取检查点前验证源码/权重哈希和参数 schema；不跳过来源校验。网页发布还需
`bun_mode_zero_actor` 去除零模式行并保留 R3 输入元数据，原始 Actor/Critic 应保留。

进一步说明：[JAX 训练方法](dcu_jax_kill_training.md)、
[八卡验证报告](../reports/dcu_jax_kill_validation_20261115.json)、
[环境迁移](dcu_migration.md)。接手本轮时，优先使用本文第 2、3、7 节与实时产物。

## 8. 后续补充：新模型录像导出

2026-11-15 08:52 UTC 左右冻结 u000330，已在 CPU 上完成 12 局固定 DUEL/SP/G 对战，
双方座位、每局 1800 tick，7 胜、3 平、2 负；每对手 4 局是展示样本，不是大规模棋力结论。
按用户最新要求只展示新模型，没有训练前后对照。这是检查点重新对弈录像，非训练现场轨迹。

观看入口：<https://allwellll.github.io/qqt-RL/dcu-replays/>。
离线文件：`runs/kill-replays-20261115-01/watch.html`；
动作、统计与验收在同目录 `replays.json` / `RESULT.json` / `BROWSER_CHECK.json`。
Pages 实时发布状态看 `PAGES_PUBLISH.json`，不要把仅提交状态当成部署完成。

新增 `scripts/export_dcu_kill_replays.py` 和 `scripts/verify_dcu_kill_replays.js`，
可直接读取已验证的新 JAX 检查点，不用旧 JS 训练入口，也不跳过源码/权重校验。
因此第 7 节的“尚需评估适配”已完成本次 CPU 回放适配；大样本固定对手正式评估仍待做。
导出不会改变当前八卡训练、截止或检查点策略。复现、统计与发布说明见
[新模型录像说明](dcu_kill_replays_20261115.md)。
