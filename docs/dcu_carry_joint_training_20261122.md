# 运包者与拦截者交替协同训练（2026-11-22）

固定 u104 会让运包者只适应单一防守策略。本轮改为交替更新：先更新运包 Actor，冻结当前拦截检查点；再更新拦截 Actor，冻结刚保存的运包检查点；如此循环。每个 PPO 段内对手保持不变，段与段之间刷新，因此仍可复现每段指标，同时避免长期固定对手。

起点是运包 `u000587` 和拦截 `u000104`。每段 20 次更新，6 个运包段与 6 个拦截段交替，共 12 段。每段使用 8 卡、每卡 2048 环境、64 tick、前向分块 2048、PPO minibatch 2048、fused collector；每段最多 45 分钟。每个段只保存 Actor、Critic 和元数据，不保存优化器。新任务目录下的 `courier-s01`、`interceptor-s01` 等分别记录阶段；下一个段只读取上一段的 Actor/Critic，Adam 重新初始化。

启动：

```bash
bash scripts/launch_dcu_jax_carry_joint_8card.sh \
  runs/carry-joint-u587-u104-20261122
```

`interceptor` 阶段使用 `--courier-opponent` 接收上一段运包检查点；该对手在当前段冻结。`courier` 阶段使用 `--interceptor` 接收上一段拦截检查点。每段完成后才切换下一方，任何一段异常都会停止整个流程，不会自动跳过坏检查点。

本次已启动的任务目录为 `runs/carry-joint-u587-u104-20261122b`（8 卡 DCU，首段 `courier-s01`）。实际启动参数固定为 `--envs 2048 --minibatch 2048 --policy-chunk 2048 --value-chunk 2048`；首个更新包含一次性编译开销，后续实测约 48k 全局环境步/秒。检查点目录只保存 `actor.pt`、`critic.pt` 和 `CHECKPOINT.json`，不保存优化器状态；`LATEST.json`、`updates.jsonl` 是接手和监控入口。

当前执行状态：6 个运包阶段与 6 个拦截阶段均已完成，最新为 `courier-s06/u000020` 与 `interceptor-s06/u000020`。12 个正式阶段的 `RESULT.json` 合计记录 239 次更新、250,609,664 环境步；首个拦截阶段因指标聚合维度错误拆为部分重启，健康检查点保留，修复后完成续训。末期稳态约 48.5k 全局环境步/秒。所有更新只保存 Actor/Critic 与元数据，不保存优化器。

录像发布器 `scripts/publish_dcu_carry_training_replays.py` 在 CPU 上每 30 分钟导出并发布最新运包/拦截模型的 4 局交换座位录像，导出通过完整局与有限值检查后才替换 latest。最新页面：[运包协同训练录像](https://allwellll.github.io/qqt-RL/carry-replays/latest.html)，归档列表：[录像目录](https://allwellll.github.io/qqt-RL/carry-replays/)。首批已验证发布使用 `courier-s04/u000020` 对 `interceptor-s04/u000020`；更晚检查点的 CPU 完整对局导出遇到非有限 logits，发布器拒绝覆盖有效录像。训练结束后发布器退出，最新线上录像仍是首批验证版。

主要观察指标是交付胜率、携包者击杀、掉包和自死。最终评估必须使用没有进入交替训练的冻结拦截模型和新种子，且只有把对方包子交回己方笼子才算运包胜利；包子掉在外面或超时未交付都算平局。

后续续训的运包阶段将使用 60% 拦截者与 40% 上一轮最新运包者作为冻结对手（`--interceptor-fraction 0.60 --courier-self-opponent`）。运包者对手所在环境的 `can_transport` 为真，因此它可以主动抢包、携包和交付；拦截阶段则继续使用 100% 最新运包者作为对手。拦截者学习者自己的环境 `can_transport` 为假，所以拦截者本身不会触发拿包，这是角色约束而非模型失效。
