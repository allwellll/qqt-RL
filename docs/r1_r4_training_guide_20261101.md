# R1–R4 现行代码指南（2026-11-01）

> 本文是20261101代码/规则快照。“当前”仅指当日；现行网页为v4-facing、G_frozen τ0.3默认与DUEL/SP可选，训练已按用户指令立即暂停、撤销全部后继。[最新状态与有效入口](current_web_and_training_status_20261107.md)覆盖本文旧运行状态；下方训练命令不获恢复授权。

本页整理现行实现，不启动、停止或改变已有训练。权威实验事件见[总账](training_experiment_ledger.md)，运行事实以对应 namespace 的原子 checkpoint、updates 和评估分片为准。线上默认模型以 `web/models.json` 首项为准；本次同步保持 R3-a u4002、`bun-native-v2`、默认对手不变。

## 版本与入口

|阶段|目的与语义|入口与关键实现|
|---|---|---|
|R1|进攻/吃道具；原生死亡属性重置与类型回收；速度步长 1.1/7、超级箱 +5；high-water 不因死亡回退|`scripts/bomb_threat_r1_items_schedule.py`、`scripts/bomb_threat_r1_items_runtime.py`、`scripts/bomb_threat_r1_item_progress.py`、`jax_bomb/je05_core_env_r1_items.py`|
|R2|从库存窗口切到自由原生搬包；b/c 用分级守恒包归属势差|`scripts/bun_native_v2_owner.py`、`scripts/bun_native_v2_custody_train.py`、`scripts/bun_native_v2_custody_fast2.py`、`scripts/bun_native_v2_custody_reward.py`|
|R3|真实道具/通路/连泡事件与可见观测补充；本侧出生、复活；保留自然取包链|`scripts/r3_items_owner.py`、`scripts/r3_items_collector.py`、`scripts/r3_items_reward.py`、`scripts/r3_items_observation.py`、`scripts/r3_items_inference.js`|
|R3-sp|独立种子、自对弈比例与吞吐/安全护栏；历史强点评估|`scripts/r3_selfplay_owner_20261031.py`、`scripts/r3_sp_seed2_owner_20261031.py`、`scripts/r3_sp_fixed_history_owner_20261031.py`|
|R4 K1/K3|K1 增加真实对拼课程；K3 在 K1 上提高 selfplay，奖励不变|`scripts/r4_combat_owner.py`、`scripts/r4_combat_train.py`、`scripts/r4_combat_collector.py`、`scripts/r4_combat_watchdog.py`|

这些历史入口消费本地冻结协议、外部父 checkpoint 和持久队列，不能在干净 checkout 随意执行 `owner` 来“复现”已有任务。先检查已有 owner、暂停闸门、谱系和预算；完整恢复须包含 Actor/Critic、Adam、env、aux、ring 和 RNG，不能只载 policy 冒充续训。旧 O5/continuous/R2 停止授权不因代码合入失效。

`jax_bomb/je07_bun_env_v2.py` 保留独立规则路径；网页 `web/sim-bun-native-v2.js` 为当前默认，`web/sim-stock-window-v1.js` 与旧 Review simulator 保留历史回放。旧清单曾把玩家放在对侧，相关旧 v2 选型已作废；本侧协议同时约束出生和复活，不把随机位置等同于本侧覆盖。 原生规则第五轮 G1 的 fresh2 为199/206一致、规则型首分歧0；转移类覆盖14/16，另外2类窗口未补抽。该 PASS 只覆盖已比较的 Sim.step 字段，不代表真人 frameStep 已逐帧一致。

## 评估与分母

- **生产 JS**：`scripts/v2_fresh_selection_20261031.py` 及同系列分片/验收脚本负责冻结本侧 fresh hard/normal/easy；按地图、席位、seed、checkpoint 和动作模式配对。采样与 argmax 分列，网页使用 argmax。独立 holdout 和多 seed 尾部/最差点保留，不择峰晋升。
- **原生 JAX 与 Python Tactical**：各自报告环境、对手和规则；不把 JAX 规则 bot 当成 JS Hunter。完整自然局与 Bank/Bank-P 分开，真实 Bank 来源与历史单图/P0/allfar 局限不得合并成泛化结论。
- **完整链**：创造死亡/原生糖浆窗口→真实库存取包→同次持包交付；慢慢胶不算原生糖浆窗口。持续持包返程不因对手复活切回空手目标。掉包、接触转移、己包回收分别记。
- **道具**：机会/库存/合法释放概率/实际放置/自己放置后自吃或因果敌命中/全轨迹净收益/自伤/交付按阶段与来源拆分母。香蕉主语义为持包时自踩加速、缩短交付 tick；观察性命中不能当因果收益。
- **KILL-EVAL-V1（实验中）**：206 张真实地图，按 canonical SHA 分 train144/dev31/holdout31；无笼、原生糖浆与复活、120秒计击杀。`jax_bomb/maps/kill_maps.json` 为地图源数据；同步仅压缩 JSON 空白，解析对象不变。JS/JAX 完整 arena parity 未过，不进入正式 K4。
- **运包压力集**：`scripts/r4_pressure_eval_owner.py`、`scripts/r4_candidate_pressure_owner.py`、`scripts/r4_pressure_metrics.js`。完整成功链与持包段成功率分开，不用整局胜率代替拦截能力；吞吐或进程存活不是训练更新证据。

## 奖励与防刷

R2-b/c 的守恒量：包在家 Φ=0；敌持包 Φ=−0.30−0.50×clip(返敌家进度,0,1)；散包/己方回收 Φ=−0.10−0.10×clip(回收进度,0,1)。采用 **Φ′−Φ**，不是 γΦ′−Φ；完整未终结循环原始净和≈0，不声称折扣 PBRS 的策略不变性。对手送达有对称 −20 终局，首 terminal 后不退款。

**R4 K2（实验中，opt-in）** 加深敌持我包势值为 −0.40−0.80×progress、散包 −0.10；拦截的即时回升来自同一势差，不再叠加持包者普通击杀奖。严格同批次强制封路的 +0.1 是预支，20 tick 内匹配击杀从势差扣回，否则失效/终局 clawback。不同批次、无因果来源、出生保护、家门口 farming 不能付款。奖励来自真实事件，合法请求、logit 或几何威胁本身不付款。

非持包者因果击杀、封路预支与收回由 `scripts/r4_k2_collector.py` 和 `scripts/r4_k2_carrier_reward.py` 实现。纯事件/守恒单测不等于完整优化器接线通过；当前 K2 仍等待 ability 梯度等验收，不能启动正式 K2。旧 D 的 minimal capture +2 对 legacy transport +24 且结算时序变化，是奖励包混杂，不能描述为“只删除辅助塑形”。

## 实验模块边界

`web/sim-kill-arena.js`、`web/bun_interceptor_bot.js`、K2 奖励/collector 与 R4 协议均为**实验中**，网页默认路径不引用。JS 拦截 timing-v2 在独立测试子类中，固定64局运包成功率43/66=65.15%，旧 v1为57/61=93.44%，hard为53/62=85.48%；这是诊断 dev，非独立确认。JS/JAX bot/action parity 与非致死自困归因缺失，不进入训练池。未验收的 JAX interceptor planner 草稿本次不入库。

K1/K3 从 R3-a ownside u4002 分叉，每臂640更新、167772160新 on-policy步，env2048×128、mb8192、epoch1、lr7.5e−5、entropy0.005、DAgger/KL0、真实Bank0.25、席位50/50。实际新更新与评估异步记录在总账所指 namespace；没有把未完成预算写成完成。

## 发布与回放

1. 冻结候选、基线、规则、本侧 fresh 清单与独立 holdout；按批准的 P1 主排序、各档/P0安全门槛选择。
2. 核验完整父 state 与 policy SHA；导出 JSON/ONNX 的 tensor/FP32/JS/WASM parity，模型元数据与 `r3_items_inference.js` ABI 对齐。
3. 生产 JS 唯一 Review 链录完整局，固定首局和 seed 保留成功/失败，标规则/模型/对手/席位，老师关闭。JAX 帧不可冒充 JS。
4. 数据库 `supabase/migrations/20261031074000_ruleset_boards.sql` 保留旧数据，v1/v2 分榜；数据库迁移与网页默认切换各有验收。
5. 获得该次发布授权才修改 catalog/默认规则，正常快进 push；同 head SHA 核验 Actions、Pages build-info、模型SHA与冷加载整局。浏览器验收屏蔽数据库写入。本次是代码同步，既不换模型，也不打模型 tag。

## 复现检查

```bash
python -m compileall -q jax_bomb qqt_rl scripts tests
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAXBOMB_RULE=bun OMP_NUM_THREADS=1 pytest -q
npm ci
npm test
npm run test:review
```

历史 run 依赖测试需外部协议/数据/完整checkpoint。缺数据、代码断言失败、技术超时分别报告。Review 历史 source hash 保持原值；新 evaluator 的源文件归档与精确可逆兼容记录不是改写旧证据。参见本次同步报告的实际测试计数和限制。
