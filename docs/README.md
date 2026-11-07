# 文档索引

当前状态以 [20261107 网页、模型与暂停说明](current_web_and_training_status_20261107.md) 为准：[模型交付](../reports/cx_web_specialists_items_v3_20261107.md) · [手机触控](../reports/cx_mobile_controls_20261107.md) · [报告索引](../reports/README.md)。当前 v4-facing、G_frozen τ0.3 默认、DUEL/SP可选，训练立即暂停、全部后继撤销；旧日期文档中的运行状态是历史快照。

- [训练代码与封存索引](training_code_map_20261107.md)：已review工具、暂停策略、外部冻结输入与旧入口禁用边界。
- [20261101 R1–R4 训练、评估与奖励快照](r1_r4_training_guide_20261101.md)：历史入口、版本和证据边界；不代表当前网页规则或训练授权。
- [实验总账](training_experiment_ledger.md)：包含历史负结果、协议修订、停止与恢复记录；按实验ID检索。
- [本次 main 同步验收](../reports/main_sync_20261031.md)：同步、排除、测试及同 SHA Pages 验收。
- [网页模型交付](github_pages_model_delivery.md)、[Bun392 导出](bun392_model_release_20261012.md)：模型SHA、Release资产、冷加载和回放。
- [R4 冻结协议](protocols/r4/r4_combat_v1.json)、[糖浆复活修订](protocols/r4/r4_combat_v1_amendment01_sugar_respawn.json)、[持包势差修订](protocols/r4/r4_combat_v1_amendment02_carrier_conservative.json)、[K1/K3 独立放行](protocols/r4/r4_combat_v1_amendment03_independent_activation.json)。这些是冻结文档副本，不代替运行收据。
- [JE10 长训总结](je10_control_route_training_summary.md)、[CX 经验](cx_experiment_lessons.md)、[主动击杀历史结论](active_kill_experiment_conclusions_20261004.md)：保留旧协议的限制，不作为 R3/R4 晋升证据。

`runs/`、完整 state/Adam/env/RNG、原始逐局记录及大模型不进入 Git。下列指南明确哪些复现需要外部产物；干净 checkout 不具备这些数据时，不应把缺数据称为测试通过。
