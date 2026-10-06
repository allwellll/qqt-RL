# 主要训练与主动击杀方法实验总结（截至 2026-10-06）

本文按“方法族”归纳当前已经实际实验或完成工程验证的主要路线，记录核心做法、真实结果、结论、实验分支和日期，便于避免重复试错。整理目标分支为 `main`，资料截止日期为 2026-10-06。它是面向决策的总览，不替代逐轮实验总账、原始报告和 run 证据。

## 1. 证据口径

- 当前基准模型仍为历史 Transformer `phase2_it4000.pt`。
- `S` 指 surviving non-trade kill（主动、存活、非换命击杀）；安全指标需同时看自炸、被杀、trade、danger-to-death 和 Tactical。
- JAX 训练 Bot、课程内成功、离线 loss、threat/pressure、放泡数、微场景得分都只是诊断，不能替代真实 `web/sim.js + bun_hunter_bot.js` Hunter 结果。
- 32/64 局只用于发现和快筛；通过后才允许独立训练 seed 与更大样本确认。不同 seed、出生、座位和协议不得合并择优。
- 截至本文日期，没有新 checkpoint 在真实网页 Hunter hard/normal 上形成跨训练 seed、同时不损伤 Tactical/安全性的稳定提升，因此没有替代 it4000 的模型。

## 2. 方法总表

| 方法族 | 日期 | 主要做法 | 主要结果 | 结论 | 对应分支 |
|---|---|---|---|---|---|
| 危险监督与安全 Reward | 2026-09-29 | danger 特征、actor 辅助监督；奖励非 trade 击杀/逃生/安全结算，惩罚自炸、trade、可避免死亡 | 旧 control 自炸降至 0.8%，但放泡降至 0.43/局、可避免死亡升至 21.1% | 单纯压自炸会形成被动崩溃；安全与攻击必须联合门控 | `main`；早期证据见 `reports/danger_curriculum_optimization_20260929.md` |
| Reward V2 + enemy threat | 2026-09-30 | 安全泡、战术落泡、forced kill、延迟结算和 enemy-threat shaping；缩短逃生 BFS | 吞吐约 42.3k→121.2k SPS；短点最高 kill 0.438，与 it4000 0.418 差异在噪声内；1500 iter 降至 0.250/0.115 | 能维持攻击活性但未变强；适合 lr 1e-4、≤300 iter 密集选点，不宜盲目长训 | `experiment/reward-v2`；主要实现已入 `main` |
| 分难度 JAX Bot 池与 8 卡长训 | 2026-10-01 | dodge/bomber/hunter/hunter_hard 梯度对手；70%同参自对弈+30% Bot 池；2048 env、8卡长训 | it2500 对 JAX Bot 平均 p=0.939，但 Tactical 1024局 kill 低于 it4000；it8000/12000 分别约15.4%/17.9%，基线41.8%，且安全退化 | 专项 Bot 能力明显过拟合，不能当网页泛化；新增 checkpoint 均不晋升 | `agent/train-jax-bot`、`integrate/v2-jax-bot`、`agent/eval-dp8-ckpts`；实现已入 `main` |
| 出生多样化与危险场景课程 | 2026-10-01 | 多出生桶；native/escape/corridor/chain/attack/conflict 六类设备端场景 | 场景模型明显降低自炸，但 Tactical surviving kill 最终仅6.3%；网页 hard/normal 几乎无 S | 过度安全导致攻击塌缩；出生多样化尚缺单独消融，当前场景比例不长训 | `agent/train-scene-opt`；通用实现已入 `main` |
| Reward、信用和价值路径消融（CX-01/02/05/06/10/11/20/22/23/25/28） | 2026-10-02～03 | GAE λ、删安全结算、pressure/shaping/kill reward、shared/detached value、EMA、HL-Gauss target、逃路 margin | 多数只在原点或综合分有信号；CX-25 两次512局 ΔC +.205/+.156，但 hard S 均0/512 | 加奖励、延长信用或价值稳定都未带来真实 hard 攻击；CX-25只证明综合安全收益 | `agent/eval-guided-8gpu-optimization`、`agent/cx-eval-bot-8gpu-opt`；经验文档已入 `main` |
| 对手、采样、优化器与参数稳定性（CX-03/04/07/12～19/21/24/26/27） | 2026-10-02～03 | current/frozen 对手、JAX hard 占比、top25/均匀/分组采样、大 batch、梯度裁剪、EMA、参数平均、冻结特征、greedy/sample 解码 | 多次出现单 seed/单点综合分改善，但独立重训 Tactical 退化或 CI 跨零；hard 基本无转化 | 训练稳定性或综合 C 不等于泛化；固定候选、跨 seed 和逐 Bot 门槛不能放宽 | 同上；汇总见 `main:docs/cx_experiment_lessons.md` |
| trap16、对手混合与局部 trap-search（CX-29～34） | 2026-10-03 | 安全候选 trap 目标场、50%→75% JAX hard、forced 几何窗口、对手 ladder、局部候选格搜索 | CX-29 两次512局 ΔC CI>0、Tactical S 201→230/229，但 hard 4→0/0；CX-30 快筛8/16点通过但 hard仍0；CX-31～34无跨 seed网页击杀 | 可复现的是安全/Tactical收益，不是高难攻击；不按原配置长训 | `agent/cx-eval-bot-8gpu-opt`、`agent/cx-active-kill-20261003`、`agent/active-kill-research-20261003`；报告已入 `main` |
| 单步干预、几何代理与网页 BC（CX-35～45） | 2026-10-03～04 | 补泡/FUSE时序、近距/搜索/threatMap/pressure；真实网页动作监督和成功示范 BC | 修复输入尺度后，训练动作准确率约99%、验证移动65%–67%，但 Hunter三档均0/64；Tactical 27→14/12，hard自炸14→51/52 | 离线拟合没有转化为闭环能力；目标稀疏、长信用、代理迁移和分布偏移仍是瓶颈 | `agent/cx-active-kill-v2-20261003`；总结已入 `main`，细节留分支 |
| Oracle-DAgger、特征与结构扩展（CX-46～50） | 2026-10-04 | Oracle-DAgger；24→28观测通道（逃路、轨迹、连锁、引信）；192×4→384×8；8卡扩量 | 28通道 hard S 4→0、normal 21→19；大模型遇 JAX/Triton 编译失败；8卡扩量因24/28通道环境变量错配而无效 | 输入/容量不是已证实瓶颈；先验证可学性与编译/shape合同，工程失败不得算算法负结果 | `agent/cx-arch-feature-opt-20261004`、`bot_cx_opt` |
| 真实 JS 关键时刻课程与动作链监督（C04～C21） | 2026-10-04～05 | 从隔离开发轨迹找最早分歧；attack/escape/seal/phase、接近/转向/慢移动/威胁窗口；完整链、paired/timing contrastive 辅助监督；每轮约1800有效秒 | 多轮课程内机制可达128/128，但冻结 Hunter hard 通常0、normal仅零星1/64；C21 final normal 1/64、Tactical 26/26/7，低于29/29/6基线 | 受控子技能与监督 loss 可学习，但未迁移到真实多步闭环；不再机械迭代同类局部课程 | `agent/cx-codex-opt-20261004`；细节在该分支 `docs/codex_research_20261004/05.md` |
| 短 A/B：历史 league、frontier credit/entropy、成功 replay | 2026-10-05 | 同初始化小步真实更新；历史对手 league；threat后策略/value加权或熵；成功局4×重放 | 所有 Hunter hard/normal S=0/64；Tactical最多持平局部指标，常有退化 | loss、熵、threat funnel 的小改善不能证明攻击转化；这些单变量均快速否定 | `agent/cx-codex-opt-20261004` |
| 权威 JS 反事实与逆向搜索 | 2026-10-05 | pressure continuation、contact-start、目标承诺；beam3×depth2及beam6×depth4反向 frontier，固定RNG并要求完整300tick clean witness | 24+36个完整候选均0 witness/0 clean S；局部 pressure 或少死亡未形成闭环击杀 | 当前短/深局部搜索都未找到可训练因果见证；不能仅扩大同类视界 | `agent/cx-codex-opt-20261004` |
| 开放种群、自对弈、PFSP/QD（JS在线、JP01） | 2026-10-05～06 | 8独立分支、terminal-only reward、PFSP历史对手、QD archive/繁殖、双向换位；真实JS或JAX设备端在线更新 | JS阶段多次暴露跨lane聚合、采样器、Adam/诊断等工程问题；修复后阶段B早期/中期 Hunter S仍为0且曾安全崩溃。JP01完成约1亿fresh ticks，其最终文档/评测仍保留在实验分支，未合入main | 种群多样性和镜像胜率不能替代 Hunter；终局稀疏信号下仍未形成可晋升证据，工程有效性必须逐项审计 | `agent/cx-codex-opt-20261004` |
| JAX 攻击课程 JA01（100m→500m） | 2026-10-06 | 8独立 learner；control/static/moving/mixed ×2 seed；近距课程退火到全地图；仅完整结算 clean kill 正奖，精确续训到500,170,752 ticks | 五个长训点课程 arm 的 easy clean 均未超过 it4000 3/32；500m hard/normal 全0，Tactical低于基线，death/self更差 | 完整状态长训与更大预算未产生预定义迁移信号；不晋升，不继续同配置扩量 | `agent/cx-jax-attack-opt-20261006` |
| 版本化微场景能力评测 | 2026-10-06 | 37题、4类，权威JS CPU执行；分解躲避、近敌起攻、远距接敌与组合链；逐tick replay/schema/hash | it4000 15/37；JA01 g19/g96/g191为11/12/12，g763回到15/37但easy仅1/32；所有模型持续压力/击杀均0 | 能定位“接近、起攻、压力、击杀”断链，但不替代 Hunter/Tactical；微场景持平不等于迁移恢复 | `agent/eval-framework-20261006` |

## 3. 已稳定得到的结论

1. 当前最强的稳定基线仍是 `phase2_it4000.pt`；尚无新模型满足真实 Hunter 与 Tactical/安全的联合晋升门槛。
2. 代理任务过拟合是最反复出现的问题：JAX Bot胜率、课程成功率、离线准确率、threat/pressure、放泡数和微场景分数都可能上升，但真实 JS Hunter 不随之上升。
3. “少自炸”不能单独视为进步。多条路线通过降低放泡或回避接触取得安全改善，同时造成 Tactical/主动击杀塌缩。
4. 训练越久并不越好。Reward V2、dp8、trap16和JA01均出现早期局部信号不稳定、后期回落或安全恶化。
5. 单 seed、单 checkpoint、综合 C 或快筛偶发 S 只能触发复训，不能触发晋升；跨 seed 复现和逐 Bot 指标必须保留。
6. 当前核心瓶颈更像是：真实网页目标事件稀疏、长序列信用分配、闭环分布偏移，以及 JAX/受控课程到真实 JS Hunter 的迁移鸿沟，而不是单纯吞吐、模型容量或观测通道数量。

## 4. 后续方法选择原则

- 新方法先在未参与训练的开发状态上证明完整因果链：接近/接触→起攻→压力/补泡→撤离→clean kill→完全结算存活；仅局部 threat 或安全见证不够。
- 每次只改变一个主变量，固定初始化、seed namespace、出生、对手、训练预算和评估协议；工程失败与科学负结果分开。
- 先做小规模、跨两个训练 seed、多个保存点的真实 Hunter hard/normal 信号筛查；没有信号不扩量。
- JAX 仅用于训练吞吐和受控机制；最终效果必须回到真实网页 JS Hunter。微场景用于定位能力断点，不替代晋升门槛。
- 同参比较应报告训练前后 unique、熵、Jaccard、S/T/D/B、Tactical及完整墙钟；不能用代理分数或总墙钟冒充有效训练收益。

## 5. 详细资料索引

- 权威总账：[`training_experiment_ledger.md`](training_experiment_ledger.md)
- Reward V2：[`reward_v2_experiment_report.md`](reward_v2_experiment_report.md)
- CX-01～30：[`cx_experiment_lessons.md`](cx_experiment_lessons.md)、[`cx01_29_reproduction_lessons_20261003.md`](cx01_29_reproduction_lessons_20261003.md)
- CX-35～45：[`active_kill_experiment_conclusions_20261004.md`](active_kill_experiment_conclusions_20261004.md)
- CX-29～34原始报告：`reports/cx29_trap_routes_20261003.md` 至 `reports/cx34_trapsearch_20261003.md`
- C04～C21、种群与JP01：分支 `agent/cx-codex-opt-20261004` 的 `docs/codex_research_20261004.md`、`codex_research_20261005.md`、`codex_research_20261006.md`
- CX-48～50：分支 `agent/cx-arch-feature-opt-20261004` 的 `docs/cx48_cx49_cx50_failures.md`
- JA01：分支 `agent/cx-jax-attack-opt-20261006` 的 `docs/jax_attack_final_review_20261006.md`
- 微场景：分支 `agent/eval-framework-20261006` 的 `docs/micro_eval_20261006.md`

> 分支名用于定位对应实验快照。未合入 main 的实验代码和 run 只能作为该分支上的证据，不应被误认为当前 main 的生产实现。