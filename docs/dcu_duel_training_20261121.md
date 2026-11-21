# DCU 纯击杀对拼训练（2026-11-21）

本轮目标是纯击杀地图的对拼能力。起点为固定评测表现较好的 u930；不接着退步的 u1030。
训练使用 JAX 原生 map806 击杀模拟器和同步 PPO，八张 DCU，每轮 rollout 64 tick。
每卡环境数、学习方前向 batch 和 PPO minibatch 的默认值统一为 2048。
全局 16384 环境，每轮 1048576 样本，64 次 PPO 参数更新。
前一段每卡 512 的运行已安全停止在 u000213（55836672 环境步），
新段从该检查点的 Actor/Critic 接续，重新初始化 Adam、环境和 RNG，Critic 不再额外预热。
5 小时以内的总预算包含初始化和编译；续训没有重新追加 5 小时。
Actor/Critic 保存，不保存 Adam、环境或 RNG。当前运行状态与实时更新数以本轮 `runs/` 目录为准。

## 当前接续任务（每卡 2048）

- 当前目录：`runs/duel-pool-8card-e2048-mb2048-20261121`。
- 父检查点：`runs/duel-pool-8card-20261121/u000213`，旧进程 PID 5951 已退出。
- 原定截止时间：`2026-11-21T08:09:12+00:00`，从旧进程实际启动时间与 4.95 小时预算计算。
- 启动回执：`runs/duel-pool-8card-e2048-mb2048-20261121.switch.json`。
- 新段步数与更新号从零计数；累计环境步 = 55836672 + 新段 `environment_steps`。
- `PLAN.json` 的 `initial_actor_sha256` 及新段 u000000 的 Actor/Critic SHA 应与父检查点一致。

热启动只接纳代码中明确列出的旧启动入口和采样器哈希组合，其余源文件逐一校验。
前一次采样前向 2048、PPO minibatch 512 的尝试在首轮重推理检查中被拒绝，最大 logp 误差 0.1472。
目前把两边的计算形状统一为 2048，保留相同的 logp/value 检查阈值。
实机首轮检查通过，最大 logp 误差 1.97e-5、value 误差为 0，首轮已保存 `u000001`。
排除首次编译，第 2–3 轮约 1.98 万环境步/秒。旧 512 轮次末尾 25 轮中位数约 1.45 万环境步/秒。

## 冻结对手联赛

每卡环境按座位均衡、整局冻结抽样，名义占比为：

| 对手 | 全部环境占比 |
|---|---:|
| 规则 bot | 10% |
| 当前策略 / 最近冻结策略 | 各 9% |
| 强历史 u930 / u840 | 各 31.5% |
| SP_latest / G_frozen | 各 4.5% |

历史模型 SHA256 固定：u930 `1b1079d213a971ed7013739868d3b9848484ecf48ade25418142e16fb3a06d27`，
u840 `dcc148a17acea599555c468c4295c8578bd515ff584035397b9920d684bda7f5`。
加载前校验 SHA、树结构、形状、dtype 和有限值。学习方每局结束时更新当前/最近冻结槽；
u930、u840、SP 和 G 槽在本轮固定，不自动把候选加入历史池。

此分布降低了旧入口 50% 的规则 bot 占比，并增加强历史版本覆盖。高自身历史占比是待验证的训练配置，
不保证一定消除遗忘或后期退化。对手表现之外还须通过固定网页规则、相同种子、双方席位的完整局评测。

第一次 PPO 更新前，脚本会对 rollout 的 logp/value 作独立重算；超出 logp 0.03、value 0.1 或出现非有限值，
则拒绝更新并保留初始检查点。检查点每 5 次更新或间隔达到 120 秒后的完整更新保存，初始和退出时也保存。

## 接手

```bash
cd /root/private_data/qqt_rl
source scripts/dcu_env.sh
.venv/bin/python -m scripts.train_dcu_jax_kill plan \
  --devices 0,1,2,3,4,5,6,7 --ticks 64

tail -F runs/duel-pool-8card-e2048-mb2048-20261121/updates.jsonl
cat runs/duel-pool-8card-e2048-mb2048-20261121/LATEST.json
```

本轮停止需创建 `runs/duel-pool-8card-e2048-mb2048-20261121/USER_STOP`，程序会在安全边界保存并退出。
训练入口不自动发布或替换网页模型。若停止或完成后要选最好点，应先作同规则固定对手评测，
不能仅凭训练 K/D、loss 或更新号选模型。

对局回放由独立的 CPU 进程每半小时导出，并在训练保存最终检查点后再导出一批。
每批固定种子，使用双方席位对战 DUEL、SP、G，共 12 局。
观看入口为 [GitHub Pages 回放](https://allwellll.github.io/qqt-RL/dcu-replays/)，
发布状态见 `runs/duel-replays-publish-20261121/publisher.log`。
