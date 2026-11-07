# L2_DUEL 多样性击杀补充（20261107）

## 最新立即停止状态（历史内容保留）

用户最新要求立即停止已执行，七臂worker/训练owner退出，GPU0–7均0MiB，无后继授权。下文“运行/继续1200/自动后继”是历史时间线，不可作为恢复指令。最终checkpoint与真实物理（含同步线后继失败回退旧物理）见 `reports/cx_training_stopped_20261107.md`；入口索引 `docs/training_code_map_20261107.md`。


实验ID：CX-L2-DUEL-20261107-V1。当前实现与CPU验证进行中，尚未将排队/编译记为成功。只授权GPU6；GPU4/5/7、线上及FLOW停止线不变，不push、不调用CC。

冻结方案：健康L2_S05g u019302继承Actor/Critic/Adam，新namespace `runs/cc_train_20261105/training/duel_20261107/L2_DUEL`，新seed2026727302；2048envs×128ticks，8192真实minibatch，固定1200更新到u020502。1200结束按同配方预声明后继，除技术故障或明显reward hack外不因少数点评估波动停。

50%完整抢包使用原生真实bank和原R3/运输/道具奖励；50%合法空手短局专项，最长300tick。已有global38末列mode0/1作为显式任务标记，输入维度/参数/Adam无需扩展。完整局mode0导出只移除末行，不改线上。专项清空包库存/散包/携包/复活计时，保留原生陷阱受控、接触击杀与救援规则；首次真实死亡或超时即reset，不允许受害者复活重复刷分。trap_enter不等于kill。

专项奖励：唯一存活归因真实kill +16，真实死亡−16，自死或同tick互死另−8；困泡/放泡/空爆/存量/idle/取包交付/超时均0。无DAgger、无未过吞吐掩码；完整局继承原SIL .5，专项仅PPO。每个实际8192批按mode归一化signed GAE，并记录两mode的ability0/1计数、raw与normalized均值。

人工场景192条：train128/dev32/holdout32，八类覆盖开阔追击、窄道、十字、绕柱、破砖、多出口、错时连锁和香蕉/胶水道具；不同距离、翻转、双方交换、属性与障碍密度。预置hazard只归对手，不能给学习者免费根源击杀信用；每座至少两个初始移动出口，初始泡远离双方且fuse≥24tick。课程逐步解锁较难variant，简单局始终保留。训练/评估seed分离，近距起点明确为人工课程。

联赛目标95%模型：G当前35%/自身ring20%/健康fixed10%，C活跃peer20%/peer ring5%/G_frozen5%；猎手/困难bot各2.5%。G每COMMITTED推进、40更新换代；C读其他健康训练快照，不改其worker。

代码：`scripts/cx_duel_scenarios_v1.py`、`cx_duel_reward_v1.py`、`cx_duel_mixed_v1.py`、`cx_duel_updater_v1.py`、`cx_duel_lib_v1.py`、`cx_duel_train_v1.py`。正式协议将保留上游SHA校验并增加当前私有源码、场景SHA；不修改旧protocol。

## GPU4 FTN2接续实证（20261107）

用户最新授权覆盖此前GPU4保留限制。无持久暂停闸门，GPU4/worker锁为空，u020502三资产和完整fresh验收有效；同namespace/seed/奖励/规模接新1200更新至u021702。新协议 `training/protocols/CX_L2_FTN2_20261107_V4.json` 保留旧SHA校验及兼容父checkpoint协议。实际首更新u020503 finite、ActorΔ0.7886841/CriticΔ1.6332899，98.7607秒含首次编译、2654envsteps/s不作稳态；奖励回执与rollout中f2_variant/no_sync_chain/frontier/clawback交叉核验PASS。预算及首更新见 `training/ftn2_20261106/SUCCESSOR_20261107_BUDGET.json`、`SUCCESSOR_20261107_FIRST_UPDATE.json`。GPU5/7未动，GPU6继续DUEL开发。

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794014723.6760917, "phase": "smoke2", "config": "local_config/20261107/parameters_12141.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794014877.0919676, "phase": "smoke2", "config": "local_config/20261107/parameters_12211.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 2, "unix": 1794015168.1479492, "phase": "smoke2", "config": "local_config/20261107/parameters_12287.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 3, "unix": 1794015170.422376, "phase": "smoke2", "config": "local_config/20261107/parameters_12288.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 6, "unix": 1794015192.1713624, "phase": "smoke2", "update": 19303, "delta": {"actor": 1.052401065826416, "critic": 1.237987756729126}, "envsteps_s": 901.9819891333993, "mode_ticks": [131072.0, 131072.0]}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794015232.1988475, "phase": "smoke2", "latest": 19304}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794015234.4054794, "phase": "fresh2", "config": "local_config/20261107/parameters_12318.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794015349.4728167, "phase": "fresh2", "latest": 19304}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794015351.397575, "phase": "to8", "config": "local_config/20261107/parameters_12338.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 6, "unix": 1794015466.467037, "phase": "to8", "update": 19305, "delta": {"actor": 0.8926133513450623, "critic": 1.132287859916687}, "envsteps_s": 2493.1132998704497, "mode_ticks": [131072.0, 131072.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 2, "unix": 1794015513.2225673, "phase": "smoke2", "update": 19303, "delta": {"actor": 1.1147451400756836, "critic": 1.0106215476989746}, "envsteps_s": 851.7316358835866, "mode_ticks": [196608.0, 65536.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 3, "unix": 1794015515.5078278, "phase": "smoke2", "update": 19303, "delta": {"actor": 1.018176555633545, "critic": 1.3949214220046997}, "envsteps_s": 853.7393013598687, "mode_ticks": [65536.0, 196608.0]}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 3, "unix": 1794015813.765824, "phase": "smoke2", "config": "local_config/20261107/parameters_12513.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 2, "unix": 1794015813.796747, "phase": "smoke2", "config": "local_config/20261107/parameters_12514.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794015813.7973633, "phase": "smoke2", "config": "local_config/20261107/parameters_12515.json", "compile_budget_seconds": 900}`

## 合法生产版本重启与范围收据（20261107）

重要更正：旧 `duel_v2_20261107`、`duel25_20261107`、`duel75_20261107` 和旧同步尝试的人工属性低于本生产地图初值，全部拒绝准入，其smoke/fresh仅为工程证据，不能计本轮有效训练。已保留停止标记及原产物，未延续其PPO。新合法catalog SHA `49c1093db8b57528b981bf6099660cad783658c645a413357913dc5803f5fde2`，采用生产初值2泡/2火力/1.3速度及真实成长格点；192场景的边界、出生出口与实际native构造状态检查PASS。原真实kill、防刷分、mode输入隔离与最终collector schema测试沿用，新增生产构造gate的来源与SHA在收据中分列，不声称重新跑过全部旧测试。

四个合法新worker已实际启动，入口及子进程均为generic配置：GPU6 `duel_valid_20261107/L2_DUEL`，GPU2 `duel25_valid_20261107/L2_DUEL25`，GPU3 `duel75_valid_20261107/L2_DUEL75`，GPU0/1 `duel_sync2_valid_20261107/L2_DUEL_SYNC2`。均重新从健康u019302起步；单卡global2048env×128、8192minibatch、32Adam/update；双卡仍同global规模，每卡1024env/4096minibatch，全局mode moments及pmean梯度后Adam。每臂预声明1200更新至u020502。当前首轮编译，尚无合法新版本finite/COMMITTED/fresh验收，不以占显存视为成功。GPU4 FTN2已恢复且稳态更新；5 ATK、7 SP保持，未停止未知进程，无新增定时任务、CC或push。

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 6, "unix": 1794016123.842855, "phase": "smoke2", "update": 19303, "delta": {"actor": 1.0569920539855957, "critic": 1.5034114122390747}, "envsteps_s": 897.5393992754138, "mode_ticks": [131072.0, 131072.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 3, "unix": 1794016128.8267477, "phase": "smoke2", "update": 19303, "delta": {"actor": 0.9567255973815918, "critic": 1.7121963500976562}, "envsteps_s": 873.4303327467121, "mode_ticks": [65536.0, 196608.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 2, "unix": 1794016128.8478441, "phase": "smoke2", "update": 19303, "delta": {"actor": 1.1946829557418823, "critic": 1.1913602352142334}, "envsteps_s": 878.5520254562894, "mode_ticks": [196608.0, 65536.0]}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794016163.865099, "phase": "smoke2", "latest": 19304}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794016165.772025, "phase": "fresh2", "config": "local_config/20261107/parameters_12638.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 3, "unix": 1794016173.8522048, "phase": "smoke2", "latest": 19304}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 2, "unix": 1794016173.8715916, "phase": "smoke2", "latest": 19304}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 3, "unix": 1794016175.7136004, "phase": "fresh2", "config": "local_config/20261107/parameters_12639.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 2, "unix": 1794016175.713831, "phase": "fresh2", "config": "local_config/20261107/parameters_12640.json", "compile_budget_seconds": 900}`

双卡同步首个真实PPO更新已完成并原子COMMITTED u019303：finite，ActorΔ1.0887068510055542、CriticΔ1.5472880601882935，两个策略副本逐位相等；实际262144 envsteps、32次聚合后Adam。首轮314.25秒包含编译，不作稳态加速依据。收据 `training/duel_sync2_valid_20261107/FIRST_REAL_UPDATE.json`。单卡6/2/3 smoke2均完成且u019304 COMMITTED，正在fresh2；第2更新稳态约15120/14407/15029envsteps/s。同步仍待第二更新吞吐及两个完整fresh验收。

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794016280.8306627, "phase": "fresh2", "latest": 19304}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794016282.8979409, "phase": "to8", "config": "local_config/20261107/parameters_12684.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 2, "unix": 1794016295.7750583, "phase": "fresh2", "latest": 19304}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 3, "unix": 1794016295.775196, "phase": "fresh2", "latest": 19304}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 3, "unix": 1794016297.6861415, "phase": "to8", "config": "local_config/20261107/parameters_12694.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 2, "unix": 1794016297.686702, "phase": "to8", "config": "local_config/20261107/parameters_12695.json", "compile_budget_seconds": 900}`

首批原生专项Review已生成：`reports/cx_duel_review_20261107.html`，16局含健康配对与失败，非挑胜录像。回执 `runs/cc_train_20261105/training/duel_valid_20261107/initial_review/REVIEW_PUBLICATION.json`。外网 https://jdh-nlp.s3.cn-north-1.jdcloud-oss.com/agent_model/qqt_diagnostics/20261107/duel_fa225ec5b3b196c5.html

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 6, "unix": 1794016397.9577186, "phase": "to8", "update": 19305, "delta": {"actor": 0.9165453910827637, "critic": 1.1766128540039062}, "envsteps_s": 2498.749601632887, "mode_ticks": [131072.0, 131072.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 2, "unix": 1794016412.7466252, "phase": "to8", "update": 19305, "delta": {"actor": 0.8992071747779846, "critic": 1.2624249458312988}, "envsteps_s": 2446.22099267514, "mode_ticks": [196608.0, 65536.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 3, "unix": 1794016412.7479513, "phase": "to8", "update": 19305, "delta": {"actor": 1.1054630279541016, "critic": 1.0656934976577759}, "envsteps_s": 2467.0537995136688, "mode_ticks": [65536.0, 196608.0]}`

首批原生Review已实际完成并外网逐字节读回：16局（8固定dev人工场景×候选/健康同批），非筛选胜局；候选4胜0负4平、健康2胜0负6平，无死亡事件，4/6个超时平局均保留。只作首批运行诊断，argmax/P1与τ0.3/P2在本小批中耦合，不能据此比较决策方式、座位或宣称能力提升；后续完整局评估按双座×模式交叉。原生实际状态重跑digest逐位一致，专项包事件0。公开录像 https://jdh-nlp.s3.cn-north-1.jdcloud-oss.com/agent_model/qqt_diagnostics/20261107/duel_fa225ec5b3b196c5.html，本地 `reports/cx_duel_review_20261107.html`。

同步fresh2已PASS：独立新进程完整下一次collector/GAE/32次全局pmean-Adam及分片env/aux/RNG/league，digest逐位一致、maxabs0，见 `training/duel_sync2_valid_20261107/L2_DUEL_SYNC2/FRESH_u019304.json`。三个单卡fresh2也全部PASS，均继续to8/fresh8，未以smoke作能力验收。

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794016503.0419364, "phase": "to8", "latest": 19310}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794016518.2904606, "phase": "fresh8", "config": "local_config/20261107/parameters_12750.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 3, "unix": 1794016522.8365443, "phase": "to8", "latest": 19310}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 2, "unix": 1794016527.8386555, "phase": "to8", "latest": 19310}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 3, "unix": 1794016541.3269887, "phase": "fresh8", "config": "local_config/20261107/parameters_12800.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 2, "unix": 1794016543.8115227, "phase": "fresh8", "config": "local_config/20261107/parameters_12803.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794016633.414315, "phase": "fresh8", "latest": 19310}`

DUEL运行事件：`{"stage": "ADMITTED_FIXED1200", "owner": "CX", "gpu": 6, "unix": 1794016633.4296396, "start": 19302, "target": 20502, "new_updates": 1200}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 6, "unix": 1794016635.824839, "phase": "formal_fixed1200", "config": "local_config/20261107/parameters_12845.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 3, "unix": 1794016661.4369197, "phase": "fresh8", "latest": 19310}`

DUEL运行事件：`{"stage": "ADMITTED_FIXED1200", "owner": "CX", "gpu": 3, "unix": 1794016661.4490938, "start": 19302, "target": 20502, "new_updates": 1200}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 3, "unix": 1794016663.8937857, "phase": "formal_fixed1200", "config": "local_config/20261107/parameters_12850.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 2, "unix": 1794016663.9195187, "phase": "fresh8", "latest": 19310}`

DUEL运行事件：`{"stage": "ADMITTED_FIXED1200", "owner": "CX", "gpu": 2, "unix": 1794016663.930809, "start": 19302, "target": 20502, "new_updates": 1200}`

DUEL运行事件：`{"stage": "WORKER_LAUNCHED", "owner": "CX", "gpu": 2, "unix": 1794016666.3197594, "phase": "formal_fixed1200", "config": "local_config/20261107/parameters_12851.json", "compile_budget_seconds": 900}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 6, "unix": 1794016750.9435544, "phase": "formal_fixed1200", "update": 19311, "delta": {"actor": 0.869989812374115, "critic": 1.4923909902572632}, "envsteps_s": 2507.2613611377465, "mode_ticks": [131072.0, 131072.0]}`

### 同global规模吞吐实测与缓存判断修正

同步早期19304..19306约98–102秒，实时native栈确在XLA缓存可执行文件加载；随后19307..19310已稳定，不是持续死锁或每步永久重载。身份检查阻止了对已切入fresh8的worker执行旧phase停止，未终止任何进程，也未启用新修订。只保留已有显式CompiledOperator的CPU双设备接口验证，暂不改已稳定路径。

|臂|19307–19310中位秒/update|采集更新envsteps/s|相邻记录墙钟envsteps/s|
|---|---:|---:|---:|
|L2_DUEL|17.14|15265|15234|
|L2_DUEL25|17.99|14545|14506|
|L2_DUEL75|17.37|15077|15041|
|L2_DUEL_SYNC2|13.92|18830|18811|

单卡和双卡均global2048×128、8192 minibatch、32Adam/update，双卡每卡1024/4096；同步计算吞吐比50%单卡1.234倍。四更新短窗口，种子不同，未含冷启动/周期存盘/重启成本，不是线性加速或能力提升证明。原始回执 `.hermes/EIGHT_GPU_THROUGHPUT_20261107.json`；正式段将继续留下逐更新时序。

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 3, "unix": 1794016779.005778, "phase": "formal_fixed1200", "update": 19311, "delta": {"actor": 0.8904064297676086, "critic": 1.2926026582717896}, "envsteps_s": 2449.8044119187102, "mode_ticks": [65536.0, 196608.0]}`

DUEL运行事件：`{"stage": "REAL_FINITE_UPDATE", "owner": "CX", "gpu": 2, "unix": 1794016781.4413881, "phase": "formal_fixed1200", "update": 19311, "delta": {"actor": 0.8391517400741577, "critic": 1.5706487894058228}, "envsteps_s": 2431.917045021744, "mode_ticks": [196608.0, 65536.0]}`

## 固定预算准入、口径与后继

- GPU6 L2_DUEL：smoke2与fresh2/fresh8全部PASS；完整Actor/Critic/Adam/env/aux/RNG/league下一更新逐位恢复。当前已实际进入正式预算，首正式更新u019311 finite，ActorΔ0.869990、CriticΔ1.492391；`training/duel_valid_20261107/TECHNICAL_ADMISSION.json`。
- GPU2 L2_DUEL25：smoke2与fresh2/fresh8全部PASS；完整Actor/Critic/Adam/env/aux/RNG/league下一更新逐位恢复。当前已实际进入正式预算，首正式更新u019311 finite，ActorΔ0.839152、CriticΔ1.570649；`training/duel25_valid_20261107/TECHNICAL_ADMISSION.json`。
- GPU3 L2_DUEL75：smoke2与fresh2/fresh8全部PASS；完整Actor/Critic/Adam/env/aux/RNG/league下一更新逐位恢复。当前已实际进入正式预算，首正式更新u019311 finite，ActorΔ0.890406、CriticΔ1.292603；`training/duel75_valid_20261107/TECHNICAL_ADMISSION.json`。

四个新DUEL臂总预算均从u019302计1200次唯一更新至u020502，技术验证保留的8次计入预算；fresh诊断更新不保留、不计budget。新单卡臂已接线同配方下一1200至u021702；同步后继预算收据 `training/duel_sync2_valid_20261107/SUCCESSOR_BUDGET.json`，同一run的接续owner仅阻塞等待现owner锁释放，不占GPU，不设定时器；必须终态20502完整fresh和无用户/技术停止才接续。旧协议及SHA验证保留，新预算使用独立协议，未关闭校验。

GPU0/1同步实际runtime来自SYNC_RECEIPT（global2048/8192、perdevice1024/4096、32Adam、全局mode归一化和梯度pmean），协议继承的common.devices=1/envs_per_device=2048字段不用于该私有同步worker，不能据此把两个独立run当同步；源码固定了实际规模并逐更新记录。

真实8192 minibatch按mode与ability0/1分组的live计数、raw signed GAE及归一化signed GAE均逐批记录，每更新32条；`.hermes/EIGHT_GPU_EXECUTION_RECEIPT_20261107.json`含按真实行数加权的汇总。mode1放泡行在u019304总体raw GAE为正，完整mode0放泡行仍略负，只是支持观察，不作安全专家或效果结论；不据泡量晋升。专项安全爆炸遥测的定义是爆炸后仍活且未受困，并非预动作可生还证明，也没有据此支付奖励。完整局奖励与运包/道具机制原样保留。

首批录像实际Chromium桌面/移动端检查PASS（16局均可末帧定位、播放、无JS错误及横向溢出），回执 `training/duel_valid_20261107/initial_review/BROWSER_CHECK.json`。完整局新队列已启动：dev与holdout不同布局索引和独立新seed，双座×argmax/τ0.3交叉，健康及G_frozen同批参照，共96局，最多2个CPU worker，状态在 `training/duel_valid_20261107/full_eval_initial/`；专项Review使用最多4核且已结束。没有新增周期监工或调用CC、修改线上、push。

同步fresh8亦已PASS_FULL_PARAMS_ADAM_SHARDED_ENV_AUX_RNG_LEAGUE，bitwise_exact=true、maxabs0；两保存点u019304/u019310的完整下一更新恢复均通过，`TECHNICAL_ADMISSION.json`已签发，正式固定1200 worker已启动（generic配置parameters_12902）。同一global batch的实测加速为1.234倍，首次正式更新仍待落逐更新行，不能把启动当更新；最新实际状态随后以资源表为准。

## 本轮实际收尾资源表

全部物理0–7均有唯一有效训练worker持卡锁；CPU评估限额，未启动新增定时监工。表中finite更新与已原子COMMITTED点分列，当前算子可在启动预热或采集/Adam阶段；不把显存等同成功。

|物理GPU|run|实际阶段|最新finite|最新COMMITTED|最新cycle秒|最新envsteps/s|
|---|---|---|---:|---|---:|---:|
|0/1|L2_DUEL_SYNC2|SYNCHRONIZED_COLLECTOR|19311|u019310|98.88|2651|
|2|L2_DUEL25|COLLECTOR|19324|u019310|18.14|14451|
|3|L2_DUEL75|COLLECTOR|19324|u019310|17.56|14928|
|4|L2_FTN2|COLLECTOR|20648|u020622|16.80|15605|
|5|L2_ATK|COLLECTOR|23961|u023942|16.57|15825|
|6|L2_DUEL|OPTIMIZER|19326|u019310|17.15|15282|
|7|L2_SP|COLLECTOR|22848|u022822|21.09|18647|

0/1最新cycle若为冷启动98秒，不能用它覆盖前述四更新稳态窗口，也不能隐去冷启动成本。同步及三个独立新DUEL臂均有 `TECHNICAL_ADMISSION.json`、两个完整fresh、u019310 COMMITTED及 `FIRST_FORMAL_UPDATE.json`；固定预算已实际运行至至少u019311。实际进程argv（含worker）核验generic入口通过，0–7物理锁均持有。最终快照 `.hermes/EIGHT_GPU_EXECUTION_RECEIPT_20261107.json`。专项首批录像已可用，完整局96局评估继续以最多2核运行，未据小样本晋升。

最终正式段复核：同步u019311/u019312为缓存预热约98.88/96.79秒，随后u019313/u019314已回到14.07/13.98秒，均finite且副本逐位相等，`FORMAL_STEADY_RECEIPT.json`已落。此时两卡均实做PPO，八张卡实时利用率均有GPU计算；最新逐卡finite/COMMITTED及利用率已刷新 `.hermes/EIGHT_GPU_EXECUTION_RECEIPT_20261107.json`，不再停留首编译。完整局CPU评估当前12/96，继续限2核；能力结论仍未建立。

完整局首批CPU配对dev/holdout已完成96局（每模型32），双座/P1、argmax/τ0.3分列；起点健康及G_frozen为同批参照。明细与发呆/喷泉回放指标见 `runs/cc_train_20261105/training/duel_valid_20261107/full_eval_initial/SUMMARY.json`。样本较小，不据此晋升或停止固定预算。

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 6, "unix": 1794021580.7138655, "phase": "formal_fixed1200", "latest": 19589}`

DUEL运行事件：`{"stage": "SAVED_USER_STOP", "owner": "CX", "gpu": 6, "unix": 1794021580.7329345}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 2, "unix": 1794022855.5187745, "phase": "formal_fixed1200", "latest": 19644}`

DUEL运行事件：`{"stage": "SAVED_USER_STOP", "owner": "CX", "gpu": 2, "unix": 1794022855.5349658}`

DUEL运行事件：`{"stage": "PHASE_PASS", "owner": "CX", "gpu": 3, "unix": 1794022856.6745572, "phase": "formal_fixed1200", "latest": 19655}`

DUEL运行事件：`{"stage": "SAVED_USER_STOP", "owner": "CX", "gpu": 3, "unix": 1794022856.68914}`
