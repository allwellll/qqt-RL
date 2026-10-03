# 训练优化实验总账

> 本文档是训练实验的唯一长期索引。任何会改变训练分布、优化目标、对手、场景、样本规模、训练规模、超参数或评估协议的动作，都必须在实验启动时新增记录，并在结果产生后原位回填。聊天结论、run 日志和单独报告不能替代本总账。

## 1. 记录规则

### 1.1 每次实验必须记录

1. 实验 ID、日期、Owner、状态和变更类型。
2. 假设与唯一主要改动；组合实验必须列出所有改动及其已有单项证据。
3. 代码 commit、初始化 checkpoint、run 路径和完整关键配置。
4. 对手/场景/课程分布、环境数量、设备数量、训练步数或 iter、seed。
5. 固定评估协议：checkpoint、seed、局数、座位、出生、对手、最大步数和输出 JSON。
6. 训练吞吐与完整墙钟；失败重启、smoke 和正式训练分开统计。
7. 原始指标、相对基线变化、结论、是否晋升和下一步。

### 1.2 禁止事项

- 不得只写“效果更好”；必须附原始计数或率及评估产物路径。
- 不得把不同 seed、地图、对手、座位或局数的结果直接混算。
- 不得用训练分布内胜率代替跨对手泛化结论。
- 不得把场均放泡数增加单独解释为能力提升。
- 进行中的实验不得提前写成成功；未完成字段统一标为“待回填”。
- 新 checkpoint 只有同时通过专项能力、独立泛化和安全性门槛，才可标为“晋升”。

### 1.3 状态

`设计中` → `smoke` → `训练中` → `评估中` → `完成/中止`。

## 2. 统一评估口径

当前统一快速筛选协议固定为四类对手，每个 checkpoint、每类对手 64 局：

- `bun.tactical_v2`：冻结 Python Tactical v2；
- `bun.hunter@hard`：真实网页猎手困难；
- `bun.hunter@normal`：真实网页猎手普通；
- `bun.hunter@easy`：真实网页猎手简单。

固定 seed `20261001`、相同出生、greedy、`max_steps=300`，逐类报告 surviving non-trade kill、被Bot击杀、own-bomb death、mutual death、W-L 与 bombs/game。网页猎手必须运行 `web/sim.js` + `web/bun_hunter_bot.js`，禁止用设备端 JAX 同名 Bot 代替。Tactical 与网页猎手是不同模拟器/协议，只能在同一对手列内比较。

JAX Bot ladder 可作为训练专项能力的旁路诊断，但不能替代上述四Bot统一快筛。64局只用于快速筛点；晋升结论必须由预声明 seed 的更大固定样本确认。

## 3. 实验索引

| ID | 日期 | 主要优化动作 | 状态 | 关键结果 | 决策 |
|---|---|---|---|---|---|
| EXP-20260929-01 | 2026-09-29 | 危险监督、Reward 重构与安全 teacher 审计 | 完成（部分后续未回填） | 证明自炸主要是学习策略问题；旧 control 虽把自炸降至 0.8%，但放泡降至 0.43、可避免死亡升至 21.1%，属于被动崩溃 | 不晋升旧 control；安全提升必须同时保持进攻活性 |
| EXP-20260930-01 | 2026-09-30 | Reward V2、enemy-threat reward、逃生分析提速 | 完成 | 训练吞吐约 42.3k→121.2k SPS；短点可接近基线，但长训退化 | Reward V2方向保留；采用短训选点，不以最终点自动晋升 |
| EXP-20261001-01 | 2026-10-01 | 分难度设备端 JAX Bot 池 | 完成 | 新2500对JAX Bot平均p=0.939、强Bot p=0.878；历史it4000为0.658/0.190 | 专项能力显著提升，但需独立泛化门槛 |
| EXP-20261001-02 | 2026-10-01 | 出生位置多样化 | 已实现并进入长训 | 与Reward V2、Bot池组合进入8卡run；尚无单独消融 | 组合结果不可归因到该动作，后续需单项A/B |
| EXP-20261001-03 | 2026-10-01 | 8卡、2048环境、长训规模扩大 | 训练中/评估中 | 约20万SPS；截至15:09为12932/60000，最新完整ckpt it12500 | 继续训练；任何ckpt不得仅凭JAX Bot结果晋升 |
| EXP-20261001-04 | 2026-10-01 | 新增危险残局/随机场景族 | 实现完成，验证中 | 6类场景已接入；专项测试、4卡smoke和短A/B仍在进行 | 未评估，不得宣称改善泛化 |
| EXP-CX-20261003-29 | 2026-10-03 | trap16 路线目标场与四 Bot attack trace | 完成 | 两次 512 局确认的 Delta C CI 均大于零；it101 快筛 0/4 通过，hard/normal 未改善 | 不按原配置长训；先做对手混合的配对门控试验 |
| EXP-CX-20261003-30 | 2026-10-03 | trap16 对手混合：50% vs 75% JAX hunter_hard | 完成 pilot | mixed-hard 8/16 保存点通过、3/4 seed 有至少两个通过点；hard S 仍为 0，normal S 仅少量出现，末端 Tactical 退化 | 不启动 401；若继续只做新 seed 的 512 局确认 |

## 4. 详细记录

### EXP-20260929-01：危险监督与安全进攻联合修复

- 变更类型：Reward / 辅助监督 / 评估与teacher审计。
- 背景：学习策略出现自炸、被动和不逃生。
- 动作：danger特征进obs；actor auxiliary supervision；奖励只鼓励非trade击杀、逃生与安全结算，惩罚自炸、trade和可避免死亡；锁定Python/JS安全teacher parity。
- 关键证据：训练分布4747个存活态中逃生动作可用率100%，平均可生存动作8.91；规则Bot在清砖/带砖各128局均0自炸，说明主要问题不是teacher无路可逃。
- 旧实验对照（128局）：

| arm | surviving causal kill | own-bomb death | bombs/game | avoidable danger death |
|---|---:|---:|---:|---:|
| base | 5.5% | 28.9% | 1.02 | 8.6% |
| control | 4.7% | 0.8% | 0.43 | 21.1% |
| safe_medium | 3.9% | 0.8% | 0.31 | 12.5% |
| safe_high | 3.9% | 11.7% | 0.54 | 9.4% |

- 结论：单纯压低自炸会导致“不放泡也不逃”的被动崩溃，不能作为晋升标准。
- 来源：canonical工作区中的`reports/danger_curriculum_optimization_20260929.md`（canonical工作区历史未提交报告；后续应迁入版本库）。

### EXP-20260930-01：Reward V2 + threat

- 变更类型：Reward / 性能优化。
- commit：`5d991b1`（Reward与逃生分析），报告 `c0c4b43`。
- 初始化：历史 `runs/overnight_tf_v2/phase2_it4000.pt`。
- 公共训练：Transformer 192×4，1024 env，128 step，flee ratio 0.3，danger_arena=1。
- Reward V2：安全放泡base 0.06、tactical placement 0.6、forced kill 2.0、resolution 1.0、enemy threat 0.4；自炸/换命时屏蔽正向战术结算。
- 性能：V2路径优化前42,340 SPS，优化后121,202 SPS（2.86×）；V2+threat约116,899 SPS。
- 固定评估：Tactical v2，seed 20261001，1024局，greedy，max_steps=300。

| checkpoint | surviving kill | own-bomb | mutual | bombs/game | 相对it4000结论 |
|---|---:|---:|---:|---:|---|
| 历史it4000 | 41.8% | 7.3% | 0.7% | 16.6 | 基线 |
| sweep threat0.4 it100 | 41.3% | 6.2% | 1.7% | 25.8 | 击杀持平、进攻更活跃，换命略升 |
| long threat0.4 lr1e-4 it200 | 43.8% | 6.2% | 1.7% | 30.0 | 点估计更高但在噪声内 |
| long threat0.4 lr1e-4 it1500 | 25.0% | 7.4% | 1.6% | 22.8 | 长训显著退化 |
| long threat0.4 lr3e-4 it1500 | 11.5% | 25.7% | 1.2% | 22.8 | 严重退化 |

- 结论：Reward V2能防止攻击性塌缩，但未证明比历史it4000更强；lr 1e-4、≤300 iter并密集评估优于盲目长训。
- 来源：`docs/reward_v2_experiment_report.md`。

### EXP-20261001-01：分难度JAX Bot池

- 变更类型：新Bot / 对手数量与难度增加。
- commit：`75619e0`。
- Bot档位：dodge_easy、dodge、bomber_easy、hunter、hunter_hard，并保留legacy_flee。
- 训练分布：后续8卡run使用70%同参自对弈、30%分难度JAX Bot池。
- 固定评估：seed 20261001，每档1024局，danger_arena，max_steps=300。

| checkpoint | 六档平均p | hunter p | hunter_hard p | 强Bot平均p |
|---|---:|---:|---:|---:|
| 历史it4000 | 0.658 | 0.239 | 0.140 | 0.190 |
| dp8 it2500 | 0.939 | 0.925 | 0.831 | 0.878 |
| dp8 it6000 | 0.930 | 0.923 | 0.795 | 0.859 |
| dp8 it8000 | 0.938 | 0.927 | 0.844 | 0.886 |
| dp8 it10000 | 0.930 | 0.932 | 0.786 | 0.859 |
| dp8 it12000 | 0.916 | 0.884 | 0.752 | 0.818 |

- 结论：专项对战能力大幅提升，it8000是当前强Bot点估计最佳；但12k已有回落，说明增加训练数量并非单调收益。
- 结果：`runs/v2_jaxbot_dp8_20261001_102237_job-t3vswpexou-master-0/eval_live_20261001_new/jax_bot_ladder_1024.json`。

### EXP-20261001-02：出生位置多样化

- 变更类型：场景/出生分布。
- commit：`870b7dd`；8卡入口接入commit `6946024`。
- 动作：增加danger_arena出生位置采样桶，降低固定出生关系过拟合。
- 当前证据：native路径回归已通过；动作与Reward V2、Bot池一起进入8卡run。
- 缺口：尚无“只改出生分布”的固定seed消融，不能将组合训练效果单独归因于该动作。
- 决策：保留实现；后续若场景优化有效，需设计spawn-only对照。

### EXP-20261001-03：8卡长训规模扩大

- 变更类型：训练规模 / 对手池 / Reward / 出生多样化组合。
- 入口：`scripts/train_v2_jaxbot_dp8.sh`。
- 相关commit：`634c31e`（8卡入口）、`6946024`（出生多样化接入）、`036c2f4`（解释器修复）、`1e323bc`（前台持有训练任务）。
- 初始化：历史 `phase2_it4000.pt`。
- 配置：8设备、2048 env、minibatch 4096、60000 iter、每500 iter保存；70%同参自对弈 + 30% JAX Bot池；Reward V2；出生多样化。
- run：`runs/v2_jaxbot_dp8_20261001_102237_job-t3vswpexou-master-0`。
- 训练状态快照（2026-10-01 15:09 CST）：12932/60000，约1.30–1.33秒/iter，约19.7万–20.2万SPS；最新完整checkpoint `final_it12500.pt`。本条为进行中快照，最终墙钟待回填。
- Tactical v2探索评估：seed 20261001，每点256局，固定相同出生，greedy，max_steps=300。

| checkpoint | surviving kill | own-bomb | danger→death | mutual | safeDet | tactResolution | bombs/game |
|---|---:|---:|---:|---:|---:|---:|---:|
| 历史it4000 | 37.1% | 5.1% | 7.0% | 0.4% | 99.5% | 91.9% | 16.0 |
| dp8 it2500 | 22.7% | 23.0% | 28.9% | 1.2% | 95.7% | 90.4% | 12.4 |
| dp8 it6000 | 16.0% | 15.6% | 33.6% | 1.2% | 93.3% | 85.6% | 24.2 |
| dp8 it8000 | 16.4% | 22.3% | 41.8% | 2.7% | 92.4% | 91.6% | 34.8 |
| dp8 it10000 | 21.9% | 22.3% | 40.6% | 2.7% | 92.9% | 88.7% | 27.8 |
| dp8 it12000 | 26.2% | 16.4% | 37.9% | 2.3% | 93.4% | 82.5% | 40.7 |

- 当前解释：256局探索结果显示所有新增点的Tactical surviving kill仍低于历史基线，且自炸/危险死亡明显更高。该次启动误将`--device gpu`当作设备值，JAX无可用CUDA后退到CPU；因此只用于组内筛点，不能与历史GPU结果混算。
- 1024局GPU确认：固定seed 20261001、相同出生，仅评估历史it4000、探索候选it12000和明显回归点it8000。历史it4000 surviving kill 41.8%、自炸7.3%、danger→death 9.4%；it8000分别15.4%、13.1%、33.2%；it12000分别17.9%、20.6%、50.5%。相对基线的配对差均显著：it8000 kill −26.4pp [−30.1,−22.7]；it12000 kill −23.9pp [−27.8,−20.0]。
- 决策：当前无可晋升checkpoint；1024局确认已足以判定“训练Bot专项进步、独立泛化与安全退化”，无需继续放大该比较样本。
- 探索结果：`.../eval_live_20261001_new/tactical_256/summary.json`。
- 确认结果：评估worktree下`runs/eval_dp8_confirm_20261001/tactical_confirm_1024_gpu/summary.json`；完整checkpoint路径、SHA256和审计摘要在同目录上级`audit_report.json`。

### EXP-20261001-04：危险残局与随机场景族

- 变更类型：新场景 / 训练分布多样化。
- 分支：`agent/train-scene-opt`；commit `568609b`。
- 假设：专项Bot池导致对手过拟合；通过预生成并验证的多种危险局面，让模型学习逃生、走廊处理、连锁爆炸、主动攻击和攻防冲突。
- 场景：native、escape、corridor、chain、attack、conflict。
- 实现：设备端JAX/JIT生成与采样；接入reset/auto-reset和DP训练；场景标签不进入模型观测；新增固定seed场景评估器和A/B入口。
- 当前状态：实现、合法性审计、4卡DP smoke、两个独立训练seed的短A/B与评估均完成。
- 4卡 smoke：control与scenario均退出0；warmup约111秒；control中位约110.4k SPS，scenario中位约109.5k SPS。
- 大样本seed-1结论：scenario显著降低自炸与danger→death，但Tactical v2存活击杀下降；scenario final仅6.3%存活击杀（历史it4000为41.8%），虽自炸为0%、danger→death为2.8%，但属于过度保守和攻击塌缩。
- 第二seed的64局场景复核方向一致：scenario final在留出场景对hunter/hunter_hard存活约84.1%/83.8%，高于control约65.0%/64.4%，但native对hunter存活击杀仅7.8%（control 18.8%），对hunter_hard为0%。
- 统一四Bot 64局快速矩阵进一步确认：scenario两seed自炸普遍较低，但对网页猎手困难/普通几乎无存活击杀；结果见`reports/four_bot_matrix_64_20261001/`。64局只作筛点，不替代既有1024局证据。
- 决策：不晋升当前scenario checkpoint，不启动当前配置长训。下一轮降低scenario比例，并加入native/JAX Bot进攻能力保持约束。
- 晋升门槛：场景合法性与可解性通过；旧native路径不回归；4卡吞吐可接受；Tactical v2/held-out场景安全与击杀改善，同时JAX Bot专项能力不显著下降。

### EXP-CX-20261003-29：trap16 路线目标场

- 状态：完成短程训练、独立复现和四 Bot 确认；未启动长程训练。
- 变更类型：训练对手策略 / 评估协议。
- 假设：在半数 JAX `hunter_hard` P1 槽位中，为路线目标加入距敌三格内、至多 16 个可达安全候选，可提升安全进攻。
- 训练：control/trap16 各 4 seed，101 updates；起点 it4000，projected critic、kill reward 24、reward shaping 0.6、EMA 0.95、512 env x 64、top 25%、32 PPO epoch、lr 3e-4。全部正式任务 `rc=0` 且 checkpoint 有限。
- 快筛：固定 seed 20261001、每 Bot 64 局、真实 JS hunter、300 tick。trap16 it50 为 2/4 通过，it101 为 0/4；两条已通过轨迹从 C=0.547 到 0.422、C=0.516 到 0.250。
- 确认：每 Bot 512 局、eval seed 20264011、bootstrap seed 20264012、10,000 paired bootstrap。原候选 Delta C +0.142578，95% CI [0.060498, 0.226562]；独立重训 +0.101562，95% CI [0.009766, 0.193359]。
- 行为：tactical surviving-kill 为 201/230/229（基线/原候选/重训），自炸为 28/8/1；hard 为 4/0/0，normal 均为 0，收益不能解释为高难攻击转化。
- 决策：不批准原配置长训。后续先做四 seed、两臂、201-update 的对手混合配对试验，保存 50/100/150/200/201，按真实四 Bot 快筛和 hard/normal 不退化门控决定是否扩至 401 updates。
- 证据：`runs/cx20261003_29_trap_routes/`、`runs/cx20261003_29_screen/screen.json`、`runs/cx20261003_29_confirm_eval/`。运行产物不纳入版本控制；完整设计见 `reports/cx29_trap_routes_20261003.md`。

### EXP-CX-20261003-30：对手混合主动击杀 pilot

- 状态：完成四 seed、两臂、201-update 训练和 64 局四 Bot 快筛；未启动 401-update 或 512 局确认。
- 假设与唯一变量：固定 CX-29 trap16 及全部训练配置，只把 P1 JAX `hunter_hard` 比例从 50% 提高到 75%；reference 保持 50% 作为配对参照。
- 训练：两臂各 4 个固定 seed，初始化为 historical it4000，保存 it50/100/150/200/201；8 个任务均 `rc=0` 且 checkpoint 有限。
- 快筛：固定 seed 20261001、每 Bot 64 局、真实网页 hard/normal/easy、Tactical v2、同出生、300 tick 和 attack trace。reference 通过 1/16，mixed-hard 通过 8/16；mixed-hard 的 seed 20265002/3/4 各有至少两个通过点。
- 行为限制：mixed-hard 在 it50/100/150/201 的 hard+normal S 总计为 0/1/4/3（512 局合计），hard S 各保存点均为 0；it201 Tactical S/B/P 下降到 82/14.520/0.746。结果支持改变对手混合再评估，但不支持把它写成 hard 攻击改善。
- 决策：不启动 401-update 长训，不合入一次性训练入口；若继续，按预注册聚合规则固定一个 mixed-hard checkpoint，用新的训练 seed 做每 Bot 512 局确认和 10,000 次 paired bootstrap。详细结果见 `reports/cx30_active_kill_20261003.md`。

## 5. 新实验追加模板

复制本节到“详细记录”末尾，并同步更新“实验索引”。

```text
### EXP-YYYYMMDD-NN：实验名

- 状态：设计中 / smoke / 训练中 / 评估中 / 完成 / 中止
- Owner：
- 变更类型：新Bot / Reward / 场景 / 数量规模 / 超参数 / 评估协议 / 其他
- 假设：
- 唯一主要改动：
- 代码commit：
- 初始化checkpoint及SHA256：
- run路径：
- 训练配置：设备；env；step；iter；minibatch；lr；seed；保存频率
- 对手/场景/课程分布：
- 墙钟：smoke；失败/重启；有效训练；评估；总计
- 吞吐：
- 评估协议：checkpoint；seed；局数；座位；出生；对手；max_steps；输出JSON

| checkpoint/arm | 专项能力 | 泛化能力 | 自炸 | trade | danger→death | bombs | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | | | | | | | |
| treatment | | | | | | | |

- 相对基线变化：
- 结论与归因：
- 决策：晋升 / 不晋升 / 需要更大样本 / 中止
- 下一步：
- 已知限制：
```

## 6. 维护约定

- 每次启动新实验：先创建实验ID和“设计中”条目，再启动训练。
- 每次保存评估结果：立即追加指标和产物路径，不等聊天收尾。
- 每次停止、失败或重启：记录原因和墙钟，不能只保留成功段。
- 每次提交训练相关代码：PR/commit说明中引用实验ID。
- 每次模型晋升：在对应条目中写明被替代checkpoint、依据和可回滚路径。
