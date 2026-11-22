# 八卡运包训练记录（2026-11-21）

本轮继续训练 SP_web u23302，让它在被追击时仍能拿到包子、脱离拦截并交付。对手池固定不更新。每卡 2048 个环境，其中 1536 个对阵仅能拦截的 u104，512 个对阵可以正常运包的 SP_web。u104 从对拼模型 u244 热启，已经在完整运包地图上完成 104 次更新。

训练前用 JAX 完整运包环境重新对弈。在两个固定种子上交换出生侧，共录制四局。模型前向采用现有网页导出实现的 FP32 计算。运包方四局均取到包子，两局完成交付，耗时 101.2 秒和 169.1 秒。另两局打满 240 秒仍未交付，按交付胜负口径记平局。u104 共击杀 17 次，其中 6 次击杀携包者。运包方必须将对方包子送回己方笼子才算胜。包子掉在外面或仍被携带时终局均算平，库存差不会改变这个统计。对不能运包的拦截者，交付胜率直接取日志中的 `captures / games`。

- [四局录像](https://allwellll.github.io/qqt-RL/carry-replays/)
- 拦截者权重来自 `runs/carry-defender-u244-20261121/u000104`。
- 录像导出入口为 `.tools/web-export-env/bin/python -m scripts.export_carry_match_replay`，默认从以上权重与 SP_web 重跑四局。

八卡共 16,384 个环境，每轮 64 tick、1,048,576 环境步。融合采样和 PPO 在 DCU 上运行，策略、价值和 PPO 批大小均为 512。前 10 次更新只训练 Critic，上限 4 小时，包含编译。以下入口的输出目录必须是新的，实际运行配置记录在该目录的 `PLAN.json`。

```bash
bash scripts/launch_dcu_jax_carry_8card.sh courier \
  runs/carry-courier-u244-defender-u104-20261121 \
  --interceptor runs/carry-defender-u244-20261121/u000104
```

检查点保存 Actor、Critic 和元数据，省略优化器。保留初始与最新版本，按每 5 轮或超过 120 秒后的下一次完整更新保存，首次更新、Critic 预热完成与正常结束也保存。最多保留 32 份，上限 2 GiB。`LATEST.json` 指向当前完整检查点，`updates.jsonl` 按对手记录交付、携包者击杀、掉包和吞吐。可用以下命令查看汇总。

```bash
.venv/bin/python -m scripts.summarize_dcu_carry \
  runs/carry-courier-u244-defender-u104-20261121
```

评估时用固定新种子、两侧出生位和完整对局，分别对阵 u104 与正常 SP_web。优先比较实际交付率和交付耗时，同时检查普通 SP 对局的表现。

## 上线与启动验收

GitHub Pages 部署和线上文件哈希校验已通过。[部署记录](https://github.com/allwellll/qqt-RL/actions/runs/44906458468)

八卡训练完成第一轮 Critic 更新并保存 `u000001`，累计 1,048,576 环境步。首轮 logp 和 value 重推理误差均为 0。日志为 `runs/carry-courier-u244-defender-u104-20261121.launch.log`。首轮包含采样与更新编译，测稳定吞吐看后续 `updates.jsonl`。
