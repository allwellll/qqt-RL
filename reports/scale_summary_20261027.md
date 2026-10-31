# 规模实验汇总（20261027）

结论：env32 是当前最有依据的吞吐候选，暖周期平均吞吐约提升六成，三seed自然采样交付也改善；但死亡增加、greedy 第二seed回落，且缺生产JS/Hunter和Tactical独立评估，**尚不能称“吞吐提升且整体能力/安全不降”**。保守兼容配置仍为 env16、batch16384、minibatch512、lr7.5e-5。lr5e-5 的能力结果值得单独保留，但它没有吞吐收益，不能与env32未测组合后推荐。

所有训练/checkpoint与GPU采样均为已有固定预算结果，本次没有重训或新增GPU任务；不择峰，报告固定终点与全部dev保存点。

## 配置与可比性

- SCALE共同父为各自R1 s2751/52/53 u064完整Adam/env/aux/RNG；三配对seed为2026103051/52/53；每run新增5,242,880真实环境步。固定legacy奖励、网络、rule3、Bank比例.25、DAgger/KL=0、rollout128、epoch1。
- env32仅并行环境16→32/卡，global batch16384→32768，终点u160；其余u320。实际优化步以原始记录为准，env32与control均10,240；mb256为20,480。
- minibatch实验仅512→256；LR实验仅7.5e-5→5e-5；不能合并两变量解释。各组control策略hash须分别核验，不能把重复相同轨迹当额外独立seed。
- MULTIGPU固定global128环境、batch16384、minibatch512，仅重分片，1/2/4/8卡各131,072步、8更新、单seed。不是强扩展长训能力对照，shuffle/浮点归约未承诺逐位一致。

## 吞吐与设备观测

env/s使用排除前两更新的有效样本数/周期耗时；周期不包含全部冷启动、保存恢复墙钟。功率/util是u2完成至终点之间的离散采样均值（包含周期内host与保存空档），不是峰值kernel利用率或能耗积分。

| 组/臂 | seed尾号 | env/s | 暖周期中位s | active卡平均W/util% | 全8卡平均W | 采样数 |
|---|---:|---:|---:|---:|---:|---:|
| scale_env_v1/control | 51 | 2574.4 | 6.246 | 126.2/51.2 | 1010.0 | 831 |
| scale_env_v1/env32 | 51 | 4279.8 | 7.503 | 129.3/49.9 | 1034.7 | 478 |
| scale_env_v1/control | 52 | 2609.5 | 6.191 | 126.2/52.9 | 1009.5 | 780 |
| scale_env_v1/env32 | 52 | 4619.3 | 7.011 | 130.0/55.1 | 1039.7 | 443 |
| scale_env_v1/control | 53 | 2961.9 | 5.463 | 127.4/51.7 | 1019.4 | 714 |
| scale_env_v1/env32 | 53 | 4728.8 | 6.899 | 130.7/56.2 | 1045.8 | 433 |
| scale_minibatch_v1/control | 51 | 2663.1 | 6.165 | 126.8/54.1 | 1014.6 | 793 |
| scale_minibatch_v1/mb256 | 51 | 2642.3 | 6.143 | 127.8/53.8 | 1022.1 | 804 |
| scale_minibatch_v1/control | 52 | 2731.6 | 5.979 | 126.7/54.3 | 1013.9 | 781 |
| scale_minibatch_v1/mb256 | 52 | 2974.6 | 5.448 | 129.4/54.2 | 1035.5 | 725 |
| scale_minibatch_v1/control | 53 | 2940.0 | 5.530 | 127.3/51.7 | 1018.7 | 731 |
| scale_minibatch_v1/mb256 | 53 | 2753.6 | 5.983 | 128.2/54.7 | 1025.8 | 792 |
| scale_lr_v1/control | 51 | 2732.6 | 5.999 | 126.7/54.3 | 1013.3 | 809 |
| scale_lr_v1/lr5e5 | 51 | 2671.1 | 6.124 | 126.3/54.0 | 1010.8 | 826 |
| scale_lr_v1/control | 52 | 2726.2 | 6.005 | 126.7/53.9 | 1013.5 | 808 |
| scale_lr_v1/lr5e5 | 52 | 2704.6 | 6.044 | 126.2/55.5 | 1009.3 | 809 |
| scale_lr_v1/control | 53 | 2982.3 | 5.430 | 126.8/51.8 | 1014.7 | 742 |
| scale_lr_v1/lr5e5 | 53 | 2732.8 | 5.988 | 126.6/54.2 | 1012.8 | 793 |
| multigpu_v1/cards8 | 51 | 2654.5 | 6.169 | 123.8/46.7 | 990.1 | 16 |
| multigpu_v1/cards4 | 51 | 2582.3 | 6.375 | 129.5/59.5 | 820.6 | 16 |
| multigpu_v1/cards2 | 51 | 2387.7 | 6.902 | 136.7/68.7 | 728.0 | 16 |
| multigpu_v1/cards1 | 51 | 2634.5 | 6.222 | 150.3/74.4 | 682.1 | 15 |

| 组 | control均值 env/s | 实验臂均值 env/s | 配对均值变化 |
|---|---:|---:|---:|
| scale_env_v1 | 2715.3 | 4542.6 | +67.6% |
| scale_minibatch_v1 | 2778.2 | 2790.2 | +0.6% |
| scale_lr_v1 | 2813.7 | 2702.8 | -3.8% |

## 固定终点的独立能力评估

每个seed/臂dev64局/动作模式，holdout304局/动作模式；u1/中点dev另列在后表。所有为自然完整native JAX JE07、layout0、P0、固定rule3、老师关闭，评估CPU独立于训练；这些结果不能替代JS Hunter或Python Tactical。D/父点的其他引擎评估不是SCALE候选评估。表内“死亡”是事件数/局数，可大于1；交付是一局至少一次。

| 组/臂 | seed尾号 | dev categorical 交付/死亡 | dev greedy 交付/死亡 | holdout categorical 交付/死亡 | holdout greedy 交付/死亡 |
|---|---:|---|---|---|---|
| scale_env_v1/control | 51 | 64/64；8/64 | 34/64；2/64 | 296/304；43/304 | 158/304；9/304 |
| scale_env_v1/env32 | 51 | 63/64；12/64 | 32/64；5/64 | 301/304；49/304 | 201/304；10/304 |
| scale_env_v1/control | 52 | 53/64；30/64 | 35/64；38/64 | 274/304；67/304 | 139/304；144/304 |
| scale_env_v1/env32 | 52 | 62/64；8/64 | 34/64；1/64 | 293/304；68/304 | 111/304；5/304 |
| scale_env_v1/control | 53 | 0/64；4/64 | 0/64；0/64 | 2/304；22/304 | 0/304；1/304 |
| scale_env_v1/env32 | 53 | 63/64；12/64 | 7/64；0/64 | 292/304；90/304 | 19/304；3/304 |
| scale_minibatch_v1/control | 51 | 64/64；8/64 | 34/64；2/64 | 296/304；43/304 | 158/304；9/304 |
| scale_minibatch_v1/mb256 | 51 | 60/64；26/64 | 0/64；0/64 | 280/304；139/304 | 0/304；0/304 |
| scale_minibatch_v1/control | 52 | 53/64；30/64 | 35/64；38/64 | 274/304；67/304 | 139/304；144/304 |
| scale_minibatch_v1/mb256 | 52 | 0/64；6/64 | 0/64；15/64 | 0/304；50/304 | 0/304；58/304 |
| scale_minibatch_v1/control | 53 | 0/64；4/64 | 0/64；0/64 | 2/304；22/304 | 0/304；1/304 |
| scale_minibatch_v1/mb256 | 53 | 13/64；20/64 | 0/64；0/64 | 59/304；107/304 | 0/304；0/304 |
| scale_lr_v1/control | 51 | 64/64；8/64 | 34/64；2/64 | 296/304；43/304 | 158/304；9/304 |
| scale_lr_v1/lr5e5 | 51 | 63/64；10/64 | 61/64；1/64 | 296/304；57/304 | 283/304；7/304 |
| scale_lr_v1/control | 52 | 53/64；30/64 | 35/64；38/64 | 274/304；67/304 | 139/304；144/304 |
| scale_lr_v1/lr5e5 | 52 | 64/64；12/64 | 15/64；0/64 | 304/304；65/304 | 77/304；16/304 |
| scale_lr_v1/control | 53 | 0/64；4/64 | 0/64；0/64 | 2/304；22/304 | 0/304；1/304 |
| scale_lr_v1/lr5e5 | 53 | 62/64；14/64 | 35/64；1/64 | 285/304；86/304 | 180/304；10/304 |

三seed逐seed差值比把912局看作912个训练seed更重要；相同游戏seed用于配对，只有3个独立训练seed，不能凭均值或窄逐局CI证明泛化/安全非劣。

| 组 | holdout模式 | 实验臂−control交付局数（seed51/52/53，各304局） | 死亡事件差（同序） |
|---|---|---|---|
| scale_env_v1 | categorical | [5, 19, 290] | [6, 1, 68] |
| scale_env_v1 | greedy | [43, -28, 19] | [1, -139, 2] |
| scale_minibatch_v1 | categorical | [-16, -274, 57] | [96, -17, 85] |
| scale_minibatch_v1 | greedy | [-158, -139, 0] | [-9, -86, -1] |
| scale_lr_v1 | categorical | [0, 30, 283] | [14, -2, 64] |
| scale_lr_v1 | greedy | [125, -62, 180] | [-2, -128, 9] |

## Async eval与多卡建议

- 同run异步评估扩容校准：低并发2675.7 env/s，扩容2667.4 env/s，变化-0.31%。这是调度短窗校准，非能力A/B；之后滚动回落触发了自动缩容，故推荐保留5%吞吐保护与资源分区，而非无限加CPU worker。
- 1卡吞吐接近8卡；当前工作负载主要host/collector受限，不能声称8卡有线性加速。1/2/4卡可作为独立实验分组候选，但本次没有并发组训练验证，没有multiGPU checkpoint dev/holdout独立能力评估，不推荐直接以短probe替换生产训练。
- mb256吞吐无稳定收益且交付明显下降；不推荐。lr5e-5自然采样尾部改善、greedy不同seed有得失，平均吞吐略低；只能作为能力候选，非效率推荐。
- env32可以作为后续优先性能配置候选，配置为8卡×32env、rollout128、batch32768、minibatch512、epoch1、lr7.5e-5；当前不为它新增训练。死亡增量及生产JS缺口使“效果不降”的证据不足。

## 终点holdout窗口/返程与胜负（各seed分列）

窗口取包率分母为完整闭合的死亡/原生糖浆窗口；携包交付分母为已闭合custody，censored另列于机器可读记录；net win由完整局胜负计算，不能等同包子交付。

| 组/臂/seed尾号/动作模式 | 死亡窗口取包 | 原生糖浆窗口取包 | 同custody交付 | 净胜/局 |
|---|---|---|---|---|
| scale_env_v1/control/51/categorical | 290/369 | 5/378 | 296/317 | 303/304 |
| scale_env_v1/control/51/greedy | 169/187 | 1/193 | 158/162 | 170/304 |
| scale_env_v1/env32/51/categorical | 290/354 | 9/366 | 301/333 | 304/304 |
| scale_env_v1/env32/51/greedy | 175/292 | 21/295 | 201/202 | 203/304 |
| scale_env_v1/control/52/categorical | 252/338 | 23/352 | 274/296 | 279/304 |
| scale_env_v1/control/52/greedy | 118/268 | 18/273 | 139/155 | 147/304 |
| scale_env_v1/env32/52/categorical | 290/407 | 5/419 | 293/311 | 300/304 |
| scale_env_v1/env32/52/greedy | 108/185 | 3/187 | 111/111 | 111/304 |
| scale_env_v1/control/53/categorical | 1/59 | 0/62 | 2/21 | 2/304 |
| scale_env_v1/control/53/greedy | 0/86 | 0/88 | 0/0 | 0/304 |
| scale_env_v1/env32/53/categorical | 280/433 | 13/440 | 292/306 | 299/304 |
| scale_env_v1/env32/53/greedy | 21/107 | 1/109 | 19/21 | 22/304 |
| scale_minibatch_v1/control/51/categorical | 290/369 | 5/378 | 296/317 | 303/304 |
| scale_minibatch_v1/control/51/greedy | 169/187 | 1/193 | 158/162 | 170/304 |
| scale_minibatch_v1/mb256/51/categorical | 248/359 | 19/366 | 280/325 | 290/304 |
| scale_minibatch_v1/mb256/51/greedy | 0/0 | 0/0 | 0/0 | 0/304 |
| scale_minibatch_v1/control/52/categorical | 252/338 | 23/352 | 274/296 | 279/304 |
| scale_minibatch_v1/control/52/greedy | 118/268 | 18/273 | 139/155 | 147/304 |
| scale_minibatch_v1/mb256/52/categorical | 0/19 | 0/19 | 0/0 | 0/304 |
| scale_minibatch_v1/mb256/52/greedy | 0/42 | 0/42 | 0/0 | 0/304 |
| scale_minibatch_v1/control/53/categorical | 1/59 | 0/62 | 2/21 | 2/304 |
| scale_minibatch_v1/control/53/greedy | 0/86 | 0/88 | 0/0 | 0/304 |
| scale_minibatch_v1/mb256/53/categorical | 64/170 | 3/182 | 59/78 | 73/304 |
| scale_minibatch_v1/mb256/53/greedy | 0/0 | 0/0 | 0/0 | 0/304 |
| scale_lr_v1/control/51/categorical | 290/369 | 5/378 | 296/317 | 303/304 |
| scale_lr_v1/control/51/greedy | 169/187 | 1/193 | 158/162 | 170/304 |
| scale_lr_v1/lr5e5/51/categorical | 279/385 | 7/392 | 296/305 | 297/304 |
| scale_lr_v1/lr5e5/51/greedy | 257/377 | 21/381 | 283/285 | 283/304 |
| scale_lr_v1/control/52/categorical | 252/338 | 23/352 | 274/296 | 279/304 |
| scale_lr_v1/control/52/greedy | 118/268 | 18/273 | 139/155 | 147/304 |
| scale_lr_v1/lr5e5/52/categorical | 246/438 | 48/444 | 304/310 | 304/304 |
| scale_lr_v1/lr5e5/52/greedy | 53/171 | 24/174 | 77/78 | 79/304 |
| scale_lr_v1/control/53/categorical | 1/59 | 0/62 | 2/21 | 2/304 |
| scale_lr_v1/control/53/greedy | 0/86 | 0/88 | 0/0 | 0/304 |
| scale_lr_v1/lr5e5/53/categorical | 277/421 | 8/430 | 285/298 | 296/304 |
| scale_lr_v1/lr5e5/53/greedy | 172/319 | 6/325 | 180/180 | 184/304 |

## 全部dev检查点（禁止择峰）

| 条件 | categorical交付/死亡 | greedy交付/死亡 |
|---|---|---|
| scale_env_v1_control_s2026103051_u000001_dev | 64/64；33/64 | 32/64；3/64 |
| scale_env_v1_control_s2026103051_u000160_dev | 63/64；19/64 | 1/64；1/64 |
| scale_env_v1_control_s2026103051_u000320_dev | 64/64；8/64 | 34/64；2/64 |
| scale_env_v1_control_s2026103052_u000001_dev | 62/64；34/64 | 25/64；1/64 |
| scale_env_v1_control_s2026103052_u000160_dev | 62/64；15/64 | 58/64；5/64 |
| scale_env_v1_control_s2026103052_u000320_dev | 53/64；30/64 | 35/64；38/64 |
| scale_env_v1_control_s2026103053_u000001_dev | 55/64；19/64 | 6/64；0/64 |
| scale_env_v1_control_s2026103053_u000160_dev | 2/64；3/64 | 3/64；0/64 |
| scale_env_v1_control_s2026103053_u000320_dev | 0/64；4/64 | 0/64；0/64 |
| scale_env_v1_env32_s2026103051_u000001_dev | 63/64；25/64 | 35/64；7/64 |
| scale_env_v1_env32_s2026103051_u000080_dev | 63/64；24/64 | 25/64；5/64 |
| scale_env_v1_env32_s2026103051_u000160_dev | 63/64；12/64 | 32/64；5/64 |
| scale_env_v1_env32_s2026103052_u000001_dev | 63/64；25/64 | 23/64；8/64 |
| scale_env_v1_env32_s2026103052_u000080_dev | 63/64；22/64 | 27/64；0/64 |
| scale_env_v1_env32_s2026103052_u000160_dev | 62/64；8/64 | 34/64；1/64 |
| scale_env_v1_env32_s2026103053_u000001_dev | 60/64；13/64 | 8/64；0/64 |
| scale_env_v1_env32_s2026103053_u000080_dev | 55/64；16/64 | 11/64；1/64 |
| scale_env_v1_env32_s2026103053_u000160_dev | 63/64；12/64 | 7/64；0/64 |
| scale_lr_v1_control_s2026103051_u000001_dev | 64/64；33/64 | 32/64；3/64 |
| scale_lr_v1_control_s2026103051_u000160_dev | 63/64；19/64 | 1/64；1/64 |
| scale_lr_v1_control_s2026103051_u000320_dev | 64/64；8/64 | 34/64；2/64 |
| scale_lr_v1_control_s2026103052_u000001_dev | 62/64；34/64 | 25/64；1/64 |
| scale_lr_v1_control_s2026103052_u000160_dev | 62/64；15/64 | 58/64；5/64 |
| scale_lr_v1_control_s2026103052_u000320_dev | 53/64；30/64 | 35/64；38/64 |
| scale_lr_v1_control_s2026103053_u000001_dev | 55/64；19/64 | 6/64；0/64 |
| scale_lr_v1_control_s2026103053_u000160_dev | 2/64；3/64 | 3/64；0/64 |
| scale_lr_v1_control_s2026103053_u000320_dev | 0/64；4/64 | 0/64；0/64 |
| scale_lr_v1_lr5e5_s2026103051_u000001_dev | 64/64；32/64 | 40/64；8/64 |
| scale_lr_v1_lr5e5_s2026103051_u000160_dev | 63/64；15/64 | 37/64；3/64 |
| scale_lr_v1_lr5e5_s2026103051_u000320_dev | 63/64；10/64 | 61/64；1/64 |
| scale_lr_v1_lr5e5_s2026103052_u000001_dev | 60/64；26/64 | 17/64；6/64 |
| scale_lr_v1_lr5e5_s2026103052_u000160_dev | 64/64；8/64 | 55/64；0/64 |
| scale_lr_v1_lr5e5_s2026103052_u000320_dev | 64/64；12/64 | 15/64；0/64 |
| scale_lr_v1_lr5e5_s2026103053_u000001_dev | 59/64；15/64 | 12/64；1/64 |
| scale_lr_v1_lr5e5_s2026103053_u000160_dev | 63/64；12/64 | 7/64；0/64 |
| scale_lr_v1_lr5e5_s2026103053_u000320_dev | 62/64；14/64 | 35/64；1/64 |
| scale_minibatch_v1_control_s2026103051_u000001_dev | 64/64；33/64 | 32/64；3/64 |
| scale_minibatch_v1_control_s2026103051_u000160_dev | 63/64；19/64 | 1/64；1/64 |
| scale_minibatch_v1_control_s2026103051_u000320_dev | 64/64；8/64 | 34/64；2/64 |
| scale_minibatch_v1_control_s2026103052_u000001_dev | 62/64；34/64 | 25/64；1/64 |
| scale_minibatch_v1_control_s2026103052_u000160_dev | 62/64；15/64 | 58/64；5/64 |
| scale_minibatch_v1_control_s2026103052_u000320_dev | 53/64；30/64 | 35/64；38/64 |
| scale_minibatch_v1_control_s2026103053_u000001_dev | 55/64；19/64 | 6/64；0/64 |
| scale_minibatch_v1_control_s2026103053_u000160_dev | 2/64；3/64 | 3/64；0/64 |
| scale_minibatch_v1_control_s2026103053_u000320_dev | 0/64；4/64 | 0/64；0/64 |
| scale_minibatch_v1_mb256_s2026103051_u000001_dev | 63/64；28/64 | 34/64；4/64 |
| scale_minibatch_v1_mb256_s2026103051_u000160_dev | 60/64；24/64 | 3/64；1/64 |
| scale_minibatch_v1_mb256_s2026103051_u000320_dev | 60/64；26/64 | 0/64；0/64 |
| scale_minibatch_v1_mb256_s2026103052_u000001_dev | 58/64；34/64 | 20/64；4/64 |
| scale_minibatch_v1_mb256_s2026103052_u000160_dev | 0/64；5/64 | 0/64；1/64 |
| scale_minibatch_v1_mb256_s2026103052_u000320_dev | 0/64；6/64 | 0/64；15/64 |
| scale_minibatch_v1_mb256_s2026103053_u000001_dev | 57/64；21/64 | 11/64；1/64 |
| scale_minibatch_v1_mb256_s2026103053_u000160_dev | 28/64；34/64 | 0/64；0/64 |
| scale_minibatch_v1_mb256_s2026103053_u000320_dev | 13/64；20/64 | 0/64；0/64 |

## 复算与执行状态

复算：`.venv/bin/python -m scripts.campaign_scale_summary`。机器可读数据及全部原始SHA：`runs/authorized_campaign_20261026/SCALE_RECOMPUTABLE_SUMMARY_20261027.json`；每个终点state/policy SHA与逐更新日志在生成时实际核验。
GPU空闲原因：已登记GPU预算全部完成，无已登记GPU后继；剩余三项原入口存在输出自等待，修复后只启动CPU生产者。BANK-RATIO正文与CPU入口/GPU限制冲突单列，不伪造训练完成；不自动恢复旧队列。
