# 道具物理对齐与训练迁移（20261107）

## 最新立即停止状态（历史内容保留）

用户最新要求立即停止已执行，七臂worker/训练owner退出，GPU0–7均0MiB，无后继授权。下文“运行/继续1200/自动后继”是历史时间线，不可作为恢复指令。最终checkpoint与真实物理（含同步线后继失败回退旧物理）见 `reports/cx_training_stopped_20261107.md`；入口索引 `docs/training_code_map_20261107.md`。


本轮收尾：七臂覆盖物理 GPU0–7，全部已迁移至独立 bun-native-v3-items namespace；smoke、完整输入状态迁移、fresh2/fresh8、真实正式 finite 更新与实际保存状态检查均通过。各臂继续预声明的 1200 更新预算。逐卡最新保存点/吞吐见文末“八卡实际新物理验收”；以下迁移事件为历史时间线，不代表当前仍待准入。未改网页、未 push、未调用 CC。

网页唯一契约已就绪：`<ITEM_RULE_READY>`，指向独立网页工作树；原规模训练七条策略覆盖八张物理卡均继续finite。旧物理为死亡位置曼哈顿最近合法格；现行806炸砖概率0.582011，上限10/8/2.4。

新契约为原炸砖率不变，在reset按初始地面箱+有效砖×原概率估计B，补ceil(1.5×(B或10)-B)个全球不放回道具。806为81砖，B=47.142891，补24，预期71.142891；DUEL明确清除旧专项三个人工箱，补15，DUEL未来炸砖率0。保留数量、类型、超级属性、胶水包量、溢出墓地与回收和属性上限。

实现均为私有源副本，旧run/协议/录像不改。当前阶段：CPU真实canonical引擎对拍和供给/掉落/数量/边界测试；完成后先GPU6 DUEL保存边界完整迁移、smoke与fresh，再逐臂迁移。尚未宣称GPU新规则准入。临时证据目录 `runs/items_alignment_20261107/v2/`，现行训练快照 `OLD_RULE_LIVE_BASELINE.json`。不CC、不push、不改网页默认、不新增周期监工。

迁移事件：`{"arm": "L2_DUEL", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794021564.4225202, "old_protocol": "runs/cc_train_20261105/training/protocols/CX_L2_DUEL_20261107_V3.json"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794021581.0964181, "source_update": 19589, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel/PROTOCOL_V1.json", "physical_gpus": [6]}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794021583.4622526, "phase": "smoke2", "config": "local_config/20261107/parameters_14337.json", "compile_budget_seconds": 900}`

## CPU规则准入与迁移语义

真实canonical引擎SHA `24c5bcc0f12088a11f24bc0f581d4b3cfd25094148a144afa573bf7af88a7c40`；冻结契约SHA `572c7e8dc31fd481ef86666636365e02fe9e6afc3058e070c7876f31920b5bb8`。8组共享随机数带对拍逐项一致，覆盖全阻塞/部分不足、超大原堆栈与重试、实际AABB、供给禁用、战术类型与超级类型。131072次全图抽样，χ²=221.66455078125，通过预声明8σ门；不是JAX与JS的PRNG位级一致，而是相同随机带结果精确一致、独立种子分布一致。额外通过原生数量守恒、死亡属性回退、三种属性上限与溢出、空投落地燃烧/活人占格退墓地测试。网页npm全测状态以冻结ready回执为准：PASS。

原始checkpoint状态树ABI保持不变。迁移时Actor/Critic/Adam/env/aux/逐环境与优化器RNG/league逐位保持；在途局不追补开局道具，只在实际reset对最终地图与出生位置补充，保留genuine bank历史，bank不重复补给。DUEL新reset替换原3个手放成长箱为15个按契约抽样的箱，未来炸砖率0；完整局炸砖率仍0.582011。专项只按原真实kill/death结算，吃道具不另奖；完整局原运包、道具信用与保护保持。新评估请求写入各新namespace且标v3，旧录像继续按旧引擎解释。

GPU6迁移边界已原子保存u019589；实际新worker载入收据与完整输入状态digest验证均PASS，尚在首个collector编译。新protocol `runs/cc_train_20261105/training/items_v3_20261107_l2_duel/PROTOCOL_V1.json`；新旧状态/协议映射在 `v2/transitions/L2_DUEL/`。首编译预算900秒、单验收phase总预算1200秒，出错回旧配方，不关闭SHA校验。其余卡保持旧规则有效更新，DUEL准入后最多同时编译两个策略。

## 可选模型证据边界

旧物理首批DUEL完整局独立holdout已完成，argmax/τ0.3×双座，每模型16局，仅作早期线索：

|checkpoint|W-L-D|交付|因果杀|自死|
|---|---|---:|---:|---:|
|G_frozen|4-5-7|2|2|4|
|L2_S05g_u019302|10-3-3|5|3|2|
|L2_DUEL_u019310|12-1-3|9|4|3|

DUEL样本有运包/胜负正信号，但dev与座位/采样结果混合，样本不足且为旧物理，暂不宣称鲜明长处、不擅改网页默认。完整证据 `training/duel_valid_20261107/full_eval_initial/SUMMARY.json`；原专项Review继续只读封存。以后若新物理独立评测形成稳定专项长处，将给出checkpoint与局限供网页CX选用。

GPU6首个真实新物理PPO u19590已finite，ActorΔ=0.8460062146186829、CriticΔ=1.8363591432571411；262144实际envsteps、32次Adam，首轮含编译291.52秒，不能作稳态吞吐。实际rollout reset道具总数9426；DUEL reset=478，专项包事件0。`FIRST_REAL_UPDATE.json`已落，smoke2/fresh仍在实际执行。

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794021928.479512, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794021930.9823241, "phase": "fresh", "config": "local_config/20261107/parameters_14468.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794022053.9928558, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022125.254289, "phase": "smoke2", "config": "local_config/20261107/parameters_fd38ff824f2c4525a5fc6565ba27dd3a.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794022248.26471, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022248.2944179, "phase": "fresh", "config": "local_config/20261107/parameters_caa3ac57b98249ceba1827a6f7bebd27.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794022371.3028505, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022371.3345492, "phase": "to8", "config": "local_config/20261107/parameters_2e9f76914ec84b0a90e2891db3bf9cf5.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794022599.3459344, "phase": "to8"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022599.3749723, "phase": "fresh", "config": "local_config/20261107/parameters_ea8af79a02dd4f09bb3ec8fa339fd944.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794022719.3831751, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022719.4256082, "phase": "formal", "config": "local_config/20261107/parameters_77fa99eb3d61484187f4fd0b9fffde31.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794022836.5795767, "update": 19598, "delta": {"actor": 0.8496674299240112, "critic": 1.9462921619415283}, "envsteps_s": 2424.088767735693, "source_update": 19589, "target": 20789}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794022836.6082313, "old_protocol": "runs/cc_train_20261105/training/protocols/CX_L2_DUEL75_20261107_V3.json"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794022836.608664, "old_protocol": "runs/cc_train_20261105/training/protocols/CX_L2_DUEL25_20261107_V3.json"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794022855.2803395, "source_update": 19644, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel25/PROTOCOL_V1.json", "physical_gpus": [2]}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022855.2982323, "phase": "smoke2", "config": "local_config/20261107/parameters_37e35b9258124f9eac939d93488626d2.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794022857.3459918, "source_update": 19655, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel75/PROTOCOL_V1.json", "physical_gpus": [3]}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794022857.366186, "phase": "smoke2", "config": "local_config/20261107/parameters_5423e794086e49eaae54c25f96681990.json", "compile_budget_seconds": 900}`

GPU6准入完成：u019589→u019591/u019597两个COMMITTED保存点，fresh2/fresh8的完整下一更新digest均一致；正式预算首更新u019598已finite，ActorΔ0.8496674/CriticΔ1.9462922。该正式首轮约108.14秒为冷加载，稳态前述17.6–18.1秒分列，不能冒称首轮吞吐。GPU2/3已分别在u019644/u019655完整保存、归档旧规则并实际起新规则smoke。其余线继续旧规则，不统一假报新物理。

迁移事件：`{"arm": "L2_DUEL25", "stage": "PHASE_PASS", "unix": 1794023212.3132942, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023212.3446696, "phase": "fresh", "config": "local_config/20261107/parameters_f8642fc0cfa44e3c8ca73332b7af862e.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "PHASE_PASS", "unix": 1794023214.3817117, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023214.4057622, "phase": "fresh", "config": "local_config/20261107/parameters_7d9592c9c8a74a9da074a64a4531c5ea.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "PHASE_PASS", "unix": 1794023335.35302, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023335.3830588, "phase": "to8", "config": "local_config/20261107/parameters_c72b7db4dac9435eb76223598cb58ae1.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "PHASE_PASS", "unix": 1794023337.4140577, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023337.4383082, "phase": "to8", "config": "local_config/20261107/parameters_31a3b57b3c864c64bf402ea387d5c2e7.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "PHASE_PASS", "unix": 1794023572.3952994, "phase": "to8"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023572.4273477, "phase": "fresh", "config": "local_config/20261107/parameters_db45ba7dd6fc4f92b0222e45ad02edde.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "PHASE_PASS", "unix": 1794023574.4496832, "phase": "to8"}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023574.474047, "phase": "fresh", "config": "local_config/20261107/parameters_e1c5659eb3ad4cee8f11fbebb9a1bc9d.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "PHASE_PASS", "unix": 1794023697.4822476, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023697.5267775, "phase": "formal", "config": "local_config/20261107/parameters_77859a0d6354426b97df6cc30b609f04.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "PHASE_PASS", "unix": 1794023698.4366026, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023698.4682782, "phase": "formal", "config": "local_config/20261107/parameters_d856b15b86c64eab8897827a219d963a.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794023817.7040582, "update": 19664, "delta": {"actor": 0.8837059736251831, "critic": 2.079662799835205}, "envsteps_s": 2353.7713036533564, "source_update": 19655, "target": 20855}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794023817.7247577, "old_protocol": "runs/cc_train_20261105/training/protocols/CX_L2_DUEL_SYNC2_20261107_V3.json"}`

迁移事件：`{"arm": "L2_DUEL25", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794023818.641734, "update": 19653, "delta": {"actor": 0.9070122838020325, "critic": 1.8396657705307007}, "envsteps_s": 2358.3932619748302, "source_update": 19644, "target": 20844}`

迁移事件：`{"arm": "L2_FTN2", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794023818.6618788, "old_protocol": "runs/cc_train_20261105/training/protocols/CX_L2_FTN2_20261107_V4.json"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794023828.7945619, "source_update": 19788, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel_sync2/PROTOCOL_V1.json", "physical_gpus": [0, 1]}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023828.820333, "phase": "smoke2", "config": "local_config/20261107/parameters_ed5992965247455598b0baa7bdd18ca2.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794023829.3867323, "source_update": 21049, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_ftn2/PROTOCOL_V1.json", "physical_gpus": [4]}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794023829.4032724, "phase": "smoke2", "config": "local_config/20261107/parameters_b0a4a725028242deb966b5334198aadc.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "PHASE_PASS", "unix": 1794024156.4182084, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024156.4477549, "phase": "fresh", "config": "local_config/20261107/parameters_fd0b048c35284bb6b4117969dcc97715.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "PHASE_PASS", "unix": 1794024273.4558916, "phase": "fresh"}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024273.4781945, "phase": "to8", "config": "local_config/20261107/parameters_49f69886cfe24facab5bb755e65259eb.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "PHASE_PASS", "unix": 1794024320.8409348, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024320.8669477, "phase": "fresh", "config": "local_config/20261107/parameters_377b3d5d982d4ecb8a2476c26fa68ca5.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "PHASE_PASS", "unix": 1794024440.875318, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024440.9027092, "phase": "to8", "config": "local_config/20261107/parameters_52abeab7e3dd4c9ebcd02b2cb6ed3daa.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "PHASE_PASS", "unix": 1794024492.4894388, "phase": "to8"}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024492.5145004, "phase": "fresh", "config": "local_config/20261107/parameters_7a43bd44f65848f4990c85b215952e6c.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "PHASE_PASS", "unix": 1794024609.5225902, "phase": "fresh"}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024609.5678499, "phase": "formal", "config": "local_config/20261107/parameters_fe6d4a388f5e4e6d8356f84f853d05e8.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794024720.6948411, "update": 21058, "delta": {"actor": 0.8345624208450317, "critic": 1.7589679956436157}, "envsteps_s": 2539.638380938125, "source_update": 21049, "target": 22249}`

迁移事件：`{"arm": "L2_ATK", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794024720.7167919, "old_protocol": "runs/cc_train_20261105/training/protocols/CC_LEAGUE2_L2_ATK_V6.json"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "PHASE_PASS", "unix": 1794024740.917594, "phase": "to8"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024740.944185, "phase": "fresh", "config": "local_config/20261107/parameters_4e977ddfaaec4059b3ba8cd46c748ecf.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_ATK", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794024815.5133286, "source_update": 24423, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_atk/PROTOCOL_V1.json", "physical_gpus": [5]}`

迁移事件：`{"arm": "L2_ATK", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024815.5364997, "phase": "smoke2", "config": "local_config/20261107/parameters_78ac4ae2f3054803af1b94837ae0827e.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "PHASE_PASS", "unix": 1794024866.953668, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794024866.9912183, "phase": "formal", "config": "local_config/20261107/parameters_f91113f43fc2457b8b8fa27e67b89039.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794024984.1237137, "update": 19797, "delta": {"actor": 0.8806919455528259, "critic": 1.9541224241256714}, "envsteps_s": 2508.4287609556195, "source_update": 19788, "target": 20988}`

迁移事件：`{"arm": "L2_SP", "stage": "SAVE_BOUNDARY_REQUESTED", "unix": 1794024984.1489122, "old_protocol": "runs/cc_train_20261105/training/protocols/CC_LEAGUE2_L2_SP_V4.json"}`

迁移事件：`{"arm": "L2_SP", "stage": "SAVED_NEW_PROTOCOL", "unix": 1794025000.0658855, "source_update": 23207, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_sp/PROTOCOL_V1.json", "physical_gpus": [7]}`

迁移事件：`{"arm": "L2_SP", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025000.0875952, "phase": "smoke2", "config": "local_config/20261107/parameters_54a3bcd7e30949ea8ef2d0288f7b30ec.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_ATK", "stage": "PHASE_PASS", "unix": 1794025121.5505064, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_ATK", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025121.583487, "phase": "fresh", "config": "local_config/20261107/parameters_0ef04f52d65045c6a7e992f7188782fe.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_ATK", "stage": "PHASE_PASS", "unix": 1794025232.5925465, "phase": "fresh"}`

迁移事件：`{"arm": "L2_ATK", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025232.6173701, "phase": "to8", "config": "local_config/20261107/parameters_b87aa07b867e4695860d68ff1dd78297.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_SP", "stage": "PHASE_PASS", "unix": 1794025321.1017792, "phase": "smoke2"}`

迁移事件：`{"arm": "L2_SP", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025321.1247897, "phase": "fresh", "config": "local_config/20261107/parameters_802dc62a9fd843b19ba4578ae551b357.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_SP", "stage": "PHASE_PASS", "unix": 1794025438.1329083, "phase": "fresh"}`

迁移事件：`{"arm": "L2_SP", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025438.1594121, "phase": "to8", "config": "local_config/20261107/parameters_02e14cdfa5bb4355add8337d7c4b4cd7.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_ATK", "stage": "PHASE_PASS", "unix": 1794025442.6278589, "phase": "to8"}`

迁移事件：`{"arm": "L2_ATK", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025442.6527212, "phase": "fresh", "config": "local_config/20261107/parameters_b193cb8033054201905d17ca3d9028f3.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_ATK", "stage": "PHASE_PASS", "unix": 1794025553.660301, "phase": "fresh"}`

迁移事件：`{"arm": "L2_ATK", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025553.7008135, "phase": "formal", "config": "local_config/20261107/parameters_d2cd1e504b8d47d6becba3651d6605ba.json", "compile_budget_seconds": 900}`

网页ready最新状态已为published-verified；与训练冻结契约的物理源SHA映射和精确定义完全一致，变化为发布状态元数据。保持已封存协议/契约SHA，不重写旧protocol；复核凭据 `runs/items_alignment_20261107/v2/CANONICAL_LATEST_RECHECK.json`。网页发布由网页CX完成，本训练owner未发布或push。

迁移事件：`{"arm": "L2_ATK", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794025658.8281791, "update": 24432, "delta": {"actor": 0.8694620728492737, "critic": 1.7019387483596802}, "envsteps_s": 2720.5964427713216, "source_update": 24423, "target": 25623}`

迁移事件：`{"arm": "L2_SP", "stage": "PHASE_PASS", "unix": 1794025684.1716168, "phase": "to8"}`

迁移事件：`{"arm": "L2_SP", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025684.1961453, "phase": "fresh", "config": "local_config/20261107/parameters_4d9efe4e65004d05ac108ae280b6f435.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_SP", "stage": "PHASE_PASS", "unix": 1794025798.2041118, "phase": "fresh"}`

迁移事件：`{"arm": "L2_SP", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794025798.244394, "phase": "formal", "config": "local_config/20261107/parameters_c29dc45cf57b4592b56020a53e983072.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_SP", "stage": "ADMITTED_REAL_FORMAL_UPDATE", "unix": 1794025909.3770306, "update": 23216, "delta": {"actor": 0.8709498643875122, "critic": 1.5477468967437744}, "envsteps_s": 3875.837816740628, "source_update": 23207, "target": 24407}`

## 八卡实际新物理验收

七个策略覆盖物理0–7全部通过完整输入状态逐位迁移、两个保存点fresh完整下一更新、真实formal finite；原状态、旧物理协议/录像保留，单worker物理锁均实核持有，ps父子进程保持generic入口。

|GPU|臂|旧保存点→新目标|最新finite|最新COMMITTED|env/mb|旧采样envsteps/s|新稳态envsteps/s|
|---|---|---|---:|---|---|---:|---:|
|0,1|L2_DUEL_SYNC2|19788→20988|19857|u019822|2048/8192|18647|18348|
|2|L2_DUEL25|19644→20844|19767|u019742|2048/8192|14533|14171|
|3|L2_DUEL75|19655→20855|19782|u019782|2048/8192|14822|14723|
|4|L2_FTN2|21049→22249|21130|u021102|2048/8192|15464|15304|
|5|L2_ATK|24423→25623|24450|u024431|2048/8192|15801|15626|
|6|L2_DUEL|19589→20789|19774|u019742|2048/8192|15190|14863|
|7|L2_SP|23207→24407|23219|u023215|3072/12288|18530|18315|

吞吐为相同batch规模的实际稳态观察（各最近最多8个<60秒更新中位数），非同轨迹因果基准；每次独立进程冷加载和fresh诊断额外墙钟不隐藏，first_cold_cycle_s见FINAL_ACCEPTANCE。fresh诊断不计保留envsteps/Adam。SP实际3072/12288，每更新393216envsteps，1200更新471859200fresh ticks；初版预算元数据误用common2048，其原文件保留，独立BUDGET_ACTUAL_SCALE_CORRECTION明确更正，未改变真实worker或1200更新预算。同步是单策略真实pmean，global2048/8192，每卡1024/4096，未冒充两个独立run。

每臂规则均为旧bun-native-v2→bun-native-v3-items。真实自然806开局新增24、砖率0.582011不变、预期总71.142891；DUEL新reset15、未来砖率0。迁移在途局及genuine bank历史未追补：未把旧状态凭空改成新reset，实际reset地面量在items_reset_ground/natural/bank/duel中逐更新记账。新规则随机死亡掉落和落地检查从迁移后的真实step生效。专项reward逐更新符合原kill/death/self-death公式，包事件0；FTN2原奖励字段交叉验收通过。上限、运包和原完整局奖励保持。

各新namespace的PROTOCOL_V1、BUDGET、EFFECTIVE_CONFIG、TECHNICAL_ADMISSION、FIRST_FORMAL_UPDATE以及各checkpoint COMMITTED/RESTORE或FRESH为证；FINAL_ACCEPTANCE含每臂协议SHA、来源与实际累计envsteps/Adam。继续执行预声明1200固定预算及同配置后继。新物理独立能力评估尚未建立，本轮不改网页默认或可选模型目录，不把旧物理小样本成绩当新物理效果。

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794042290.5055604, "phase": "fresh", "config": "local_config/20261107/parameters_59c0d98f81fb49de9db0578ad75b4cb0.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "PHASE_PASS", "unix": 1794042407.5179396, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794042407.5489783, "phase": "formal", "config": "local_config/20261107/parameters_ca8e4643329e4b899b147984105091a5.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "SAME_NAMESPACE_FIXED1200_SUCCESSOR", "unix": 1794042407.5533197, "start": 20988, "target": 22188, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel_sync2/PROTOCOL_BUDGET_20988.json"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "FORMAL_TECHNICAL_FAILURE", "unix": 1794042417.5775113, "latest": 20988, "target": 22188}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "TECHNICAL_REJECTION", "unix": 1794042417.5839329, "error": "formal exited before budget target"}`

迁移事件：`{"arm": "L2_DUEL_SYNC2", "stage": "OLD_RECIPE_ROLLBACK_STARTED", "unix": 1794042417.612363, "config": "local_config/20261107/parameters_c2c18b2701c844edb01a4a62312ed102.json", "worker": 47750}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794043987.7392066, "phase": "fresh", "config": "local_config/20261107/parameters_a265acf7477345b1aa223bd6502d7794.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "PHASE_PASS", "unix": 1794044107.7524652, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794044107.7867541, "phase": "formal", "config": "local_config/20261107/parameters_d37e25da794e43fd90649adeaa314cf1.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL", "stage": "SAME_NAMESPACE_FIXED1200_SUCCESSOR", "unix": 1794044107.7909644, "start": 20789, "target": 21989, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel/PROTOCOL_BUDGET_20789.json"}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794045177.8884108, "phase": "fresh", "config": "local_config/20261107/parameters_9cf31a271e1e44d896fdec1111fbce8b.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "PHASE_PASS", "unix": 1794045303.9003308, "phase": "fresh"}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794045303.9342651, "phase": "formal", "config": "local_config/20261107/parameters_1e49f6a425244b8a8feaab5b733031cf.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_DUEL75", "stage": "SAME_NAMESPACE_FIXED1200_SUCCESSOR", "unix": 1794045303.938825, "start": 20855, "target": 22055, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_duel75/PROTOCOL_BUDGET_20855.json"}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794045303.9751148, "phase": "fresh", "config": "local_config/20261107/parameters_aab0c83accd143f7a1edeb5a7d7de014.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "PHASE_PASS", "unix": 1794045420.9852214, "phase": "fresh"}`

迁移事件：`{"arm": "L2_FTN2", "stage": "ACTUAL_WORKER_STARTED", "unix": 1794045421.0174863, "phase": "formal", "config": "local_config/20261107/parameters_8c3b060d2b4b49b392d5b9716fd775cf.json", "compile_budget_seconds": 900}`

迁移事件：`{"arm": "L2_FTN2", "stage": "SAME_NAMESPACE_FIXED1200_SUCCESSOR", "unix": 1794045421.022222, "start": 22249, "target": 23449, "protocol": "runs/cc_train_20261105/training/items_v3_20261107_l2_ftn2/PROTOCOL_BUDGET_22249.json"}`
