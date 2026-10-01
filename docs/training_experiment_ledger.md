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

当前至少保留以下两条独立轴：

- JAX Bot ladder：固定 seed，逐档报告 kill、death、self、trade、bombs、`kill/(kill+death)`；必须单列 hunter 与 hunter_hard。
- Tactical v2：固定 seed 与相同出生，报告 surviving non-trade kill、own-bomb death、danger-to-death、mutual death、safe detonation、tactical resolution 和 bombs/game。

探索性小样本只用于筛点；晋升结论必须由预声明 seed 的更大固定样本确认。

## 3. 实验索引

| ID | 日期 | 主要优化动作 | 状态 | 关键结果 | 决策 |
|---|---|---|---|---|---|
| EXP-20260929-01 | 2026-09-29 | 危险监督、Reward 重构与安全 teacher 审计 | 完成（部分后续未回填） | 证明自炸主要是学习策略问题；旧 control 虽把自炸降至 0.8%，但放泡降至 0.43、可避免死亡升至 21.1%，属于被动崩溃 | 不晋升旧 control；安全提升必须同时保持进攻活性 |
| EXP-20260930-01 | 2026-09-30 | Reward V2、enemy-threat reward、逃生分析提速 | 完成 | 训练吞吐约 42.3k→121.2k SPS；短点可接近基线，但长训退化 | Reward V2方向保留；采用短训选点，不以最终点自动晋升 |
| EXP-20261001-01 | 2026-10-01 | 分难度设备端 JAX Bot 池 | 完成 | 新2500对JAX Bot平均p=0.939、强Bot p=0.878；历史it4000为0.658/0.190 | 专项能力显著提升，但需独立泛化门槛 |
| EXP-20261001-02 | 2026-10-01 | 出生位置多样化 | 已实现并进入长训 | 与Reward V2、Bot池组合进入8卡run；尚无单独消融 | 组合结果不可归因到该动作，后续需单项A/B |
| EXP-20261001-03 | 2026-10-01 | 8卡、2048环境、长训规模扩大 | 训练中/评估中 | 约20万SPS；截至15:09为12932/60000，最新完整ckpt it12500 | 继续训练；任何ckpt不得仅凭JAX Bot结果晋升 |
| EXP-20261001-04 | 2026-10-01 | 新增危险残局/随机场景族 | 实现完成，验证中 | 6类场景已接入；专项测试、4卡smoke和短A/B仍在进行 | 未评估，不得宣称改善泛化 |

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
