# 八卡训练执行记录（20261107）

## 最新立即停止状态（历史内容保留）

用户最新要求立即停止已执行，七臂worker/训练owner退出，GPU0–7均0MiB，无后继授权。下文“运行/继续1200/自动后继”是历史时间线，不可作为恢复指令。最终checkpoint与真实物理（含同步线后继失败回退旧物理）见 `reports/cx_training_stopped_20261107.md`；入口索引 `docs/training_code_map_20261107.md`。


用户明确授权物理GPU0–7，旧范围限制被替代；仅CX训练owner，不调用CC、不push、不改线上、不终止未知进程。资源allowed_gpus已更新，旧status字符串仅为封存worker兼容保留，实际范围以allowed_gpus及authorization_scope为准。

## GPU4 FTN2接续实证（20261107）

用户最新授权覆盖此前GPU4保留限制。无持久暂停闸门，GPU4/worker锁为空，u020502三资产和完整fresh验收有效；同namespace/seed/奖励/规模接新1200更新至u021702。新协议 `training/protocols/CX_L2_FTN2_20261107_V4.json` 保留旧SHA校验及兼容父checkpoint协议。实际首更新u020503 finite、ActorΔ0.7886841/CriticΔ1.6332899，98.7607秒含首次编译、2654envsteps/s不作稳态；奖励回执与rollout中f2_variant/no_sync_chain/frontier/clawback交叉核验PASS。预算及首更新见 `training/ftn2_20261106/SUCCESSOR_20261107_BUDGET.json`、`SUCCESSOR_20261107_FIRST_UPDATE.json`。GPU5/7未动，GPU6继续DUEL开发。

0/1计划验证真正pmap同步：固定全局2048envs、8192minibatch，双卡各1024envs/4096minibatch，pmean梯度与全局按mode归一化；相同32次Adam/update。2/3按限时工程结果使用有价值的独立混合/seed备用，不将独立run当同步加速。当前尚未同步启动，不虚报finite或吞吐。

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794015147.4032643, "phase": "smoke2", "config": "local_config/20261107/parameters_12280.json", "budget_seconds": 900}`

同步运行事件：`{"stage": "TECHNICAL_FAILURE", "physical_gpus": [0, 1], "unix": 1794015187.4099495, "phase": "smoke2", "code": 1}`

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794015215.1453955, "phase": "smoke2", "config": "local_config/20261107/parameters_12308.json", "budget_seconds": 900}`

同步运行事件：`{"stage": "TECHNICAL_FAILURE", "physical_gpus": [0, 1], "unix": 1794015530.1553457, "phase": "smoke2", "code": -15}`

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794015812.6744392, "phase": "smoke2", "config": "local_config/20261107/parameters_12512.json", "budget_seconds": 900}`

## 合法生产版本重启与范围收据（20261107）

重要更正：旧 `duel_v2_20261107`、`duel25_20261107`、`duel75_20261107` 和旧同步尝试的人工属性低于本生产地图初值，全部拒绝准入，其smoke/fresh仅为工程证据，不能计本轮有效训练。已保留停止标记及原产物，未延续其PPO。新合法catalog SHA `49c1093db8b57528b981bf6099660cad783658c645a413357913dc5803f5fde2`，采用生产初值2泡/2火力/1.3速度及真实成长格点；192场景的边界、出生出口与实际native构造状态检查PASS。原真实kill、防刷分、mode输入隔离与最终collector schema测试沿用，新增生产构造gate的来源与SHA在收据中分列，不声称重新跑过全部旧测试。

四个合法新worker已实际启动，入口及子进程均为generic配置：GPU6 `duel_valid_20261107/L2_DUEL`，GPU2 `duel25_valid_20261107/L2_DUEL25`，GPU3 `duel75_valid_20261107/L2_DUEL75`，GPU0/1 `duel_sync2_valid_20261107/L2_DUEL_SYNC2`。均重新从健康u019302起步；单卡global2048env×128、8192minibatch、32Adam/update；双卡仍同global规模，每卡1024env/4096minibatch，全局mode moments及pmean梯度后Adam。每臂预声明1200更新至u020502。当前首轮编译，尚无合法新版本finite/COMMITTED/fresh验收，不以占显存视为成功。GPU4 FTN2已恢复且稳态更新；5 ATK、7 SP保持，未停止未知进程，无新增定时任务、CC或push。

双卡同步首个真实PPO更新已完成并原子COMMITTED u019303：finite，ActorΔ1.0887068510055542、CriticΔ1.5472880601882935，两个策略副本逐位相等；实际262144 envsteps、32次聚合后Adam。首轮314.25秒包含编译，不作稳态加速依据。收据 `training/duel_sync2_valid_20261107/FIRST_REAL_UPDATE.json`。单卡6/2/3 smoke2均完成且u019304 COMMITTED，正在fresh2；第2更新稳态约15120/14407/15029envsteps/s。同步仍待第二更新吞吐及两个完整fresh验收。

同步运行事件：`{"stage": "PHASE_PASS", "physical_gpus": [0, 1], "unix": 1794016297.6892235, "phase": "smoke2", "update": 19304}`

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794016299.6646552, "phase": "fresh", "config": "local_config/20261107/parameters_12696.json", "budget_seconds": 900}`

同步运行事件：`{"stage": "PHASE_PASS", "physical_gpus": [0, 1], "unix": 1794016414.6721988, "phase": "fresh", "update": 19304}`

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794016417.1670644, "phase": "to8", "config": "local_config/20261107/parameters_12723.json", "budget_seconds": 900}`

首批原生Review已实际完成并外网逐字节读回：16局（8固定dev人工场景×候选/健康同批），非筛选胜局；候选4胜0负4平、健康2胜0负6平，无死亡事件，4/6个超时平局均保留。只作首批运行诊断，argmax/P1与τ0.3/P2在本小批中耦合，不能据此比较决策方式、座位或宣称能力提升；后续完整局评估按双座×模式交叉。原生实际状态重跑digest逐位一致，专项包事件0。公开录像 https://jdh-nlp.s3.cn-north-1.jdcloud-oss.com/agent_model/qqt_diagnostics/20261107/duel_fa225ec5b3b196c5.html，本地 `reports/cx_duel_review_20261107.html`。

同步fresh2已PASS：独立新进程完整下一次collector/GAE/32次全局pmean-Adam及分片env/aux/RNG/league，digest逐位一致、maxabs0，见 `training/duel_sync2_valid_20261107/L2_DUEL_SYNC2/FRESH_u019304.json`。三个单卡fresh2也全部PASS，均继续to8/fresh8，未以smoke作能力验收。

同步运行事件：`{"stage": "PHASE_PASS", "physical_gpus": [0, 1], "unix": 1794016707.1794317, "phase": "to8", "update": 19310}`

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794016709.2220542, "phase": "fresh", "config": "local_config/20261107/parameters_12855.json", "budget_seconds": 900}`

### 同global规模吞吐实测与缓存判断修正

同步早期19304..19306约98–102秒，实时native栈确在XLA缓存可执行文件加载；随后19307..19310已稳定，不是持续死锁或每步永久重载。身份检查阻止了对已切入fresh8的worker执行旧phase停止，未终止任何进程，也未启用新修订。只保留已有显式CompiledOperator的CPU双设备接口验证，暂不改已稳定路径。

|臂|19307–19310中位秒/update|采集更新envsteps/s|相邻记录墙钟envsteps/s|
|---|---:|---:|---:|
|L2_DUEL|17.14|15265|15234|
|L2_DUEL25|17.99|14545|14506|
|L2_DUEL75|17.37|15077|15041|
|L2_DUEL_SYNC2|13.92|18830|18811|

单卡和双卡均global2048×128、8192 minibatch、32Adam/update，双卡每卡1024/4096；同步计算吞吐比50%单卡1.234倍。四更新短窗口，种子不同，未含冷启动/周期存盘/重启成本，不是线性加速或能力提升证明。原始回执 `.hermes/EIGHT_GPU_THROUGHPUT_20261107.json`；正式段将继续留下逐更新时序。

同步运行事件：`{"stage": "PHASE_PASS", "physical_gpus": [0, 1], "unix": 1794016829.2308846, "phase": "fresh", "update": 19310}`

同步运行事件：`{"stage": "ADMITTED_FIXED1200", "physical_gpus": [0, 1], "unix": 1794016829.2533228, "target": 20502}`

同步运行事件：`{"stage": "STARTED", "physical_gpus": [0, 1], "unix": 1794016831.7292762, "phase": "formal", "config": "local_config/20261107/parameters_12902.json", "budget_seconds": 40000}`

## 固定预算准入、口径与后继

- GPU6 L2_DUEL：smoke2与fresh2/fresh8全部PASS；完整Actor/Critic/Adam/env/aux/RNG/league下一更新逐位恢复。当前已实际进入正式预算，首正式更新u019311 finite，ActorΔ0.869990、CriticΔ1.492391；`training/duel_valid_20261107/TECHNICAL_ADMISSION.json`。
- GPU2 L2_DUEL25：smoke2与fresh2/fresh8全部PASS；完整Actor/Critic/Adam/env/aux/RNG/league下一更新逐位恢复。当前已实际进入正式预算，首正式更新u019311 finite，ActorΔ0.839152、CriticΔ1.570649；`training/duel25_valid_20261107/TECHNICAL_ADMISSION.json`。
- GPU3 L2_DUEL75：smoke2与fresh2/fresh8全部PASS；完整Actor/Critic/Adam/env/aux/RNG/league下一更新逐位恢复。当前已实际进入正式预算，首正式更新u019311 finite，ActorΔ0.890406、CriticΔ1.292603；`training/duel75_valid_20261107/TECHNICAL_ADMISSION.json`。

四个新DUEL臂总预算均从u019302计1200次唯一更新至u020502，技术验证保留的8次计入预算；fresh诊断更新不保留、不计budget。新单卡臂已接线同配方下一1200至u021702；同步后继预算收据 `training/duel_sync2_valid_20261107/SUCCESSOR_BUDGET.json`，同一run的接续owner仅阻塞等待现owner锁释放，不占GPU，不设定时器；必须终态20502完整fresh和无用户/技术停止才接续。旧协议及SHA验证保留，新预算使用独立协议，未关闭校验。

GPU0/1同步实际runtime来自SYNC_RECEIPT（global2048/8192、perdevice1024/4096、32Adam、全局mode归一化和梯度pmean），协议继承的common.devices=1/envs_per_device=2048字段不用于该私有同步worker，不能据此把两个独立run当同步；源码固定了实际规模并逐更新记录。

真实8192 minibatch按mode与ability0/1分组的live计数、raw signed GAE及归一化signed GAE均逐批记录，每更新32条；`.hermes/EIGHT_GPU_EXECUTION_RECEIPT_20261107.json`含按真实行数加权的汇总。mode1放泡行在u019304总体raw GAE为正，完整mode0放泡行仍略负，只是支持观察，不作安全专家或效果结论；不据泡量晋升。专项安全爆炸遥测的定义是爆炸后仍活且未受困，并非预动作可生还证明，也没有据此支付奖励。完整局奖励与运包/道具机制原样保留。

首批录像实际Chromium桌面/移动端检查PASS（16局均可末帧定位、播放、无JS错误及横向溢出），回执 `training/duel_valid_20261107/initial_review/BROWSER_CHECK.json`。完整局新队列已启动：dev与holdout不同布局索引和独立新seed，双座×argmax/τ0.3交叉，健康及G_frozen同批参照，共96局，最多2个CPU worker，状态在 `training/duel_valid_20261107/full_eval_initial/`；专项Review使用最多4核且已结束。没有新增周期监工或调用CC、修改线上、push。

同步fresh8亦已PASS_FULL_PARAMS_ADAM_SHARDED_ENV_AUX_RNG_LEAGUE，bitwise_exact=true、maxabs0；两保存点u019304/u019310的完整下一更新恢复均通过，`TECHNICAL_ADMISSION.json`已签发，正式固定1200 worker已启动（generic配置parameters_12902）。同一global batch的实测加速为1.234倍，首次正式更新仍待落逐更新行，不能把启动当更新；最新实际状态随后以资源表为准。

冷启动与验收墙钟另计：从各自实际smoke启动至两点fresh准入，50%单卡约820秒，双卡约1017秒，包含重启与诊断不保留更新。稳态1.234倍不能替代端到端加速结论。各臂 `EFFECTIVE_CONFIG.json` 明确实际1200更新对应314572800新envsteps、38400次Adam；继承的提案级environment_steps_per_run等未使用字段不作为实际预算口径。

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

道具物理新版本接续已完成：七臂覆盖0–7通过新namespace完整状态迁移、fresh2/fresh8和真实formal finite，保持固定1200预算。逐臂checkpoint、实际规模和新旧吞吐见 `reports/cx_items_training_align_20261107.md`，总收据 `runs/items_alignment_20261107/v2/FINAL_ACCEPTANCE.json`；不得再按旧物理解释新保存点。
