# CX-35 至 CX-45 主动击杀研究结论

截至 2026-10-04，CX-35 至 CX-45 已完成。没有模型在真实网页 `bun_hunter_bot.js` hard/normal 上形成跨训练 seed 的 surviving non-trade kill 提升，因此没有新模型晋升；historical `phase2_it4000.pt` 仍是已知主动击杀基线。

## 主要实验结论

- 单步动作干预不足：直接补放泡、修正新泡 `FUSE+1` 时序，都没有产生稳定 hard/normal 击杀。
- 局部几何代理不足：近距 start-state、`attackSeeds` 邻近搜索、`threatMap` 路径代价以及静态压力缩减奖励，最多改善部分 Tactical/安全指标，没有迁移为真实网页击杀。
- JAX 训练 Bot 与真实 JS Hunter 仍存在迁移鸿沟；代理 Bot 的机制触发或专项表现不能替代真实网页评测。
- 真实网页动作监督数据曾发现输入尺度错误；修复是必要正确性工作，但修复后短训和提高拟合预算均未改善 hard/normal。
- CX-45 两个 seed 的训练动作准确率约 99%、验证移动准确率 65%–67%，但 hard/normal/easy 均 0/64；Tactical 从 GPU 基线 27/64 降至 14/64、12/64，hard 自炸从 14/64 升至 51/64、52/64。离线动作拟合没有转化为闭环策略能力。
- 成功示范本身稀缺：真实 hard Hunter 作为 teacher 对 normal/easy 的 64 局筛选仅得到 1 条可用成功轨迹，不足以构造独立训练/验证集。

## 当前判断

主要制约不是训练吞吐，而是目标事件稀疏、长序列信用分配、训练代理到真实网页的迁移鸿沟，以及离线监督后的闭环分布偏移。后续不应继续盲调同类局部 reward 或在同一小规模 teacher 数据上增加拟合步数。

64 局只用于快筛；本轮没有值得进入独立 seed、512 局确认的正信号。以上结论仅覆盖已测试配置，不代表所有交互式监督、序列搜索或更广成功示范方案均无效。

## 详细证据

详细代码、逐局 CSV、summary JSON、GPU 审计和 CX-35 至 CX-45 报告保留在分支：

`agent/cx-active-kill-v2-20261003`

关键入口：

- `reports/active_kill_gpu_phase_20261004.md`
- `reports/cx35_attack_decoder_20261003.md`
- `reports/cx36_timing_20261003.md`
- `reports/cx37_combat_curriculum_20261003.md`
- `reports/cx39_attack_search_20261004.md`
- `reports/cx40_threat_map_20261004.md`
- `reports/cx41_pressure_delta_20261004.md`
- `reports/cx42_web_teacher_bc_20261004.md`
- `reports/cx43_web_success_bc_20261004.md`
- `reports/cx44_web_bc_scale_20261004.md`
- `reports/cx45_web_bc_fit_20261004.md`

实验代码没有合入 main。该分支当前完整测试存在一个已知、且 main 同样存在的基线失败：`tests/test_aux_heads.py::test_aux_heads_present_only_when_requested` 期望辅助头 15 维、实现为 10 维；因此分支代码只用于实验复盘，不视为已验收生产实现。
